#pragma once
#include "split512_resident.h"
#include "c32_resident.h"
#include "swin_t.hip"
#include "upsample48_prefix.hip"
#include "spatial256_window.hip"
#include "c256_ffn_candidate.hip"
#include "c256_attention_candidate.hip"
#include "tiled_gemm.hip"
#include "encoder256_downsample.hip"
#include <stdexcept>

namespace c256_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

#define C256_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+255)/256), dim3(256), 0, c512_resident::stream, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while (0)

inline bool check(const std::string& dir, const char* name, const float* device,
                  size_t count, bool verify, size_t& comparisons) {
    if (!verify) return true;
    ++comparisons;
    traffic.d2h_bytes += count*sizeof(float);
    return compare((dir+": "+name).c_str(), const_cast<float*>(device), read(dir, name, count));
}
// A lazy buffer is only touched (allocated) when verifying.
inline bool check(const std::string& dir, const char* name, const c512_resident::Buffer::Lazy& device,
                  size_t count, bool verify, size_t& comparisons) {
    return !verify || check(dir, name, (const float*)device.get(), count, verify, comparisons);
}

class Prefix {
    static int checked(int w, int h) {
        if (w <= 0 || h <= 0 || w % 2 || h % 2) throw std::invalid_argument("block48 input must be a positive multiple of 2");
        return w;
    }
    int width, height, out_w, out_h;
    size_t n, out_n;
    std::string dir;
    Buffer weights, scale, low, merged;
    bool ready = false;
    std::unique_ptr<Buffer> merged8;  // byte_out: the merge as E4M3 bytes
    bool wrote8 = false;
public:
    size_t comparisons = 0;
    // byte_out: the fast path reads byte inputs/skips and writes the merge as bytes.
    bool byte_out = false;
    // ow x oh (0: twice the input): the output and skip extent, at most twice the input.
    Prefix(int w, int h, const std::string& fixture, int ow = 0, int oh = 0) : width(checked(w,h)), height(h),
        out_w(ow > 0 ? ow : 2*w), out_h(oh > 0 ? oh : 2*h),
        n(size_t(w)*h*512), out_n(size_t(out_w)*out_h*256), dir(fixture), weights(dir,"weights",256*512),
        scale(dir,"scale",256), low(n/2), merged(out_n) {
        if (out_w > 2*w || out_h > 2*h) throw std::invalid_argument("block48 output larger than twice its input");
    }
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        wrote8 = byte_out && !verify && !c512_resident::exact_math;
        if ((input.fmt || skip.fmt) && (verify || c512_resident::exact_math)) return false;
        tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,input.fmt?nullptr:input.data,512,weights.data,512,nullptr,0,nullptr,
            low.data,nullptr,256,width*height,256,1,0,0,0,1,input.fmt==1?reinterpret_cast<const unsigned char*>(input.data):nullptr);  // k_upsample48_project
        if (wrote8 && !merged8) merged8.reset(new Buffer((out_n+3)/4));
        C256_LAUNCH(k_upsample48_merge, out_n, low.data,skip.data,scale.data,wrote8 ? merged8->data.get() : merged.data.get(),width,height,out_w,out_h,skip.fmt,int(wrote8));
        if (!check(dir,"low",low.data,n/2,verify,comparisons) ||
            !check(dir,"merged",merged.data,out_n,verify,comparisons)) return false;
        ready = true;
        return true;
    }
    DeviceTensor low_view() const { return ready ? DeviceTensor{low.data,n/2} : DeviceTensor{}; }
    DeviceTensor final_view() const {
        return !ready ? DeviceTensor{} : wrote8 ? DeviceTensor{merged8->data.get(),out_n,1} : DeviceTensor{merged.data.get(),out_n};
    }
};

struct Weights {
    std::string dir;
    int shift;
    Buffer w1,w2,w3,skip,qkv,bias,scales,projection,attention_skip;
    Weights(const std::string& d, int s) : dir(d),shift(s),
        w1(d+"/ffn","w1",1024*256),w2(d+"/ffn","w2",256*1024),
        w3(d+"/ffn","w3",256*256),skip(d+"/ffn","skip",256),
        qkv(d+"/attention","qkv_weights",3*256*256),bias(d+"/attention","bias",8*4096),
        scales(d+"/attention","scales",8),projection(d+"/attention","projection_weights",256*256),
        attention_skip(d+"/attention","attention_skip",256) {}
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks) {
        if (w <= 0 || h <= 0 || w % 4 || h % 4 || blocks.size() != 8)
            throw std::invalid_argument("resident C256 requires a positive multiple of 4 and eight blocks");
        for (const auto& b: blocks) if (b.second < 0 || b.second > 3) throw std::invalid_argument("invalid shift");
        return w;
    }
    int width,height,max_tokens;
    size_t n;
    bool encoder;
    Buffer ping,pong,windows,expanded,hidden,middle,feature,qkv,normalized,scores,
           exponents,probabilities,inverse,context,linear,residual;
    // Encoder22 also keeps its unquantized projection for the downsample pool.
    std::unique_ptr<Buffer> raw_windows,raw;
    std::vector<std::unique_ptr<Weights>> weights;
    float* result = nullptr;
    int result_fmt = 0;
    bool raw_was_half = false;
    std::unique_ptr<Buffer> ping8, pong8, raw_h;  // byte activations, half raw
public:
    size_t comparisons = 0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder_mode=false) :
        width(checked(w,h,blocks)),height(h),max_tokens((w+8)*(h+8)),n(size_t(w)*h*256),encoder(encoder_mode),
        ping(n),pong(n),windows(size_t(max_tokens)*256),expanded(size_t(max_tokens)*1024),
        hidden(expanded.count),middle(windows.count),feature(windows.count),qkv(windows.count*3),
        normalized(qkv.count),scores(size_t(max_tokens)*512),exponents(scores.count),
        probabilities(scores.count),inverse(scores.count/32),context(windows.count),linear(windows.count),residual(windows.count) {
        if (encoder) { raw_windows.reset(new Buffer(windows.count)); raw.reset(new Buffer(n)); }
        for (const auto& b:blocks) weights.emplace_back(new Weights(b.first,b.second));
    }
    // consume_source: the fast path may use the source as its second buffer
    // (one buffer less; set when nothing reads the source afterwards).
    bool consume_source = false;
    // byte_out: the fast path passes E4M3 bytes between blocks (input bytes or
    // halves, raw output as halves) and returns them.
    bool byte_out = false;
    bool run_from_device(DeviceTensor source,bool verify=false) {
        result = nullptr; result_fmt = 0; raw_was_half = false;
        if (!source.data || source.count != n) return false;
        const bool fast = !verify && !c512_resident::exact_math;
        if (source.fmt && !fast) return false;
        if (fast && (byte_out || source.fmt)) {
            // Outputs as bytes (byte_out) or floats; a byte-sized source cannot
            // take float outputs, so the second buffer is then this chain's own.
            const size_t words = byte_out ? (n+3)/4 : n;
            const bool use_source = consume_source && (byte_out || !source.fmt);
            if (!pong8) pong8.reset(new Buffer(words));
            if (!use_source && !ping8) ping8.reset(new Buffer(words));
            if (encoder && !raw_h) raw_h.reset(new Buffer((n+1)/2));
            const void* in = source.data; int in_fmt = source.fmt;
            void* out = pong8->data.get();
            void* other = use_source ? const_cast<float*>(source.data) : static_cast<void*>(ping8->data.get());
            for (size_t index = 0; index < weights.size(); ++index) {
                const auto& w = *weights[index];
                const bool last_raw = encoder && index+1 == weights.size();
                swin_t::SwinIo sio; sio.in_fmt = in_fmt; sio.out_fmt = byte_out ? 1 : 0; sio.raw_fmt = 2;
                if (!swin_t::run<256>(static_cast<const float*>(in), static_cast<float*>(out), last_raw ? raw_h->data.get() : nullptr,
                        width, height, w.shift, {w.w1.data,w.w2.data,w.w3.data,w.skip.data,w.qkv.data,w.scales.data,w.bias.data,
                     w.projection.data,w.attention_skip.data}, middle.data, sio)) return false;
                const void* prev = in; in = out; in_fmt = sio.out_fmt; out = index == 0 ? other : const_cast<void*>(prev);
            }
            result = static_cast<float*>(const_cast<void*>(in)); result_fmt = byte_out ? 1 : 0; raw_was_half = encoder;
            return true;
        }
        float *input = const_cast<float*>(source.data), *output = pong.data;
        // The first block reads the source in place (no copy into ping), the
        // rest ping-pong between pong and ping: the source is never written.
        auto advance = [&] { float* old = input; input = output;
            output = old == ping.data.p || old == pong.data.p ? old
                   : consume_source && !verify && !c512_resident::exact_math ? const_cast<float*>(source.data) : ping.data.get(); };
        for (size_t index=0;index<weights.size();++index) {
            const auto& w=*weights[index];
            bool last_raw=encoder && index+1==weights.size();
            int px=(w.shift&1)?4:0,py=(w.shift&2)?4:0;
            int tokens=((width+px+7)/8)*8*((height+py+7)/8)*8;
            int wn=tokens/64;
            size_t count=size_t(tokens)*256, sc=size_t(wn)*8*4096;
            if (!check(w.dir+"/spatial","input",input,n,verify,comparisons)) return false;
            if (!verify && swin_t::run<256>(input,output,last_raw?raw->data:nullptr,width,height,w.shift,
                    {w.w1.data,w.w2.data,w.w3.data,w.skip.data,w.qkv.data,w.scales.data,w.bias.data,
                     w.projection.data,w.attention_skip.data},middle.data)) {
                advance();
                continue;
            }
            C256_LAUNCH(k_spatial256_gather,count,input,windows.data,width,height,w.shift);
            // Tiled, bit-identical forms of the w1+gate, w2, w3 and QKV kernels.
            tiled::gemm<tiled::Split,tiled::GATE,false>(c512_resident::stream,windows.data,256,w.w1.data,256,
                nullptr,0,nullptr,verify?expanded.data:nullptr,hidden.data,1024,tokens,1024);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,hidden.data,1024,w.w2.data,1024,
                nullptr,0,nullptr,middle.data,nullptr,256,tokens,256);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,middle.data,256,w.w3.data,256,
                windows.data,256,w.skip.data,feature.data,nullptr,256,tokens,256);
            tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,feature.data,256,w.qkv.data,256,
                nullptr,0,nullptr,qkv.data,nullptr,768,tokens,768);
            HIP_CHECK(hipGetLastError());
            if (verify || !c512_resident::fused_attention(qkv.data, w.scales.data, w.bias.data, context.data, tokens, 256)) {
                C256_LAUNCH(k_split512_qknorm_inv,size_t(tokens)*2*8,qkv.data,inverse.data,tokens,256);
                C256_LAUNCH(k_split512_qknorm_apply,size_t(tokens)*3*256,qkv.data,w.scales.data,inverse.data,normalized.data,tokens,256);
                C256_LAUNCH(k_c256_scores,sc,normalized.data,w.bias.data,scores.data,wn);
                C256_LAUNCH(k_split512_exp,sc,scores.data,exponents.data,int(sc));
                C256_LAUNCH(k_split512_inv_rows,sc/64,exponents.data,inverse.data,int(sc/64));
                C256_LAUNCH(k_split512_prob_rows,sc,exponents.data,inverse.data,probabilities.data,int(sc));
                C256_LAUNCH(k_c256_context,count,probabilities.data,normalized.data,context.data,wn);
            }
            // The linear projection is a diagnostic output; it runs only when verifying.
            if (verify) C256_LAUNCH(k_c256_projection_linear,count,context.data,w.projection.data,linear.data,tokens);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,context.data,256,w.projection.data,256,
                feature.data,256,w.attention_skip.data,residual.data,last_raw?raw_windows->data:nullptr,256,tokens,256);
            HIP_CHECK(hipGetLastError());
            C256_LAUNCH(k_spatial256_scatter,n,residual.data,output,width,height,w.shift);
            if (last_raw) {
                C256_LAUNCH(k_spatial256_scatter,n,raw_windows->data,raw->data,width,height,w.shift);
            }
            if (!check(w.dir+"/spatial","windows",windows.data,count,verify,comparisons) ||
                !check(w.dir+"/ffn","expanded",expanded.data,4*count,verify,comparisons) ||
                !check(w.dir+"/ffn","hidden",hidden.data,4*count,verify,comparisons) ||
                !check(w.dir+"/ffn","middle",middle.data,count,verify,comparisons) ||
                !check(w.dir+"/ffn","feature",feature.data,count,verify,comparisons) ||
                !check(w.dir+"/attention","qkv",qkv.data,3*count,verify,comparisons) ||
                !check(w.dir+"/attention","normalized",normalized.data,3*count,verify,comparisons) ||
                !check(w.dir+"/attention","scores",scores.data,sc,verify,comparisons) ||
                !check(w.dir+"/attention","exponents",exponents.data,sc,verify,comparisons) ||
                !check(w.dir+"/attention","probabilities",probabilities.data,sc,verify,comparisons) ||
                !check(w.dir+"/attention","context",context.data,count,verify,comparisons) ||
                !check(w.dir+"/attention","projection_linear",linear.data,count,verify,comparisons) ||
                !check(w.dir+"/attention","projection_residual",residual.data,count,verify,comparisons) ||
                !check(w.dir+"/output","output",output,n,verify,comparisons)) return false;
            if (last_raw && (!check(w.dir+"/attention","projection_raw",raw_windows->data,count,verify,comparisons) ||
                             !check(w.dir+"/raw_output","output",raw->data,n,verify,comparisons))) return false;
            advance();
        }
        result=input;
        return true;
    }
    DeviceTensor final_view() const { return result ? DeviceTensor{result,n,result_fmt} : DeviceTensor{}; }
    DeviceTensor raw_view() const {
        return !(result && encoder) ? DeviceTensor{} : raw_was_half ? DeviceTensor{raw_h->data.get(),n,2} : DeviceTensor{raw->data,n};
    }
};

// Encoder22 raw body -> rounded 2x2 pool -> FP8 C512 projection.
class Downsample {
    static int checked(int w,int h) {
        if (w <= 0 || h <= 0 || w % 2 || h % 2) throw std::invalid_argument("downsample22 extent must be a positive multiple of 2");
        return w;
    }
    int width,height;
    size_t n,pool_n,out_n;
    std::string dir;
    Buffer matrix,pool,output;
    bool ready=false, wrote8=false;
    std::unique_ptr<Buffer> pool8, output8;
public:
    size_t comparisons=0;
    // byte_out: the fast path writes the output as E4M3 bytes (a half raw input
    // is read either way).
    bool byte_out = false;
    Downsample(int w,int h,const std::string& fixture) : width(checked(w,h)),height(h),
        n(size_t(w)*h*256),pool_n(n/4),out_n(n/2),dir(fixture),matrix(dir,"matrix",512*256),
        pool(pool_n),output(out_n) {}
    bool run_from_device(DeviceTensor raw,bool verify=false) {
        ready=false;
        if (!raw.data || raw.count != n) return false;
        if (!check(dir,"raw",raw.data,n,verify,comparisons)) return false;
        const bool fast = !verify && !c512_resident::exact_math;
        if (raw.fmt && !fast) return false;
        wrote8 = fast && byte_out;
        if (fast && (raw.fmt || byte_out)) {
            // The pool as E4M3 bytes (it holds E4M3 values) feeding the GEMM's byte input.
            if (!pool8) pool8.reset(new Buffer((pool_n+3)/4));
            if (wrote8 && !output8) output8.reset(new Buffer((out_n+3)/4));
            C256_LAUNCH(k_encoder256_pool,pool_n,raw.data,pool8->data.get(),width,height,raw.fmt,1);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,nullptr,256,matrix.data,256,nullptr,0,nullptr,
                wrote8 ? nullptr : output.data.get(),nullptr,512,width*height/4,512,1,0,0,0,1,
                reinterpret_cast<const unsigned char*>(pool8->data.get()),
                wrote8 ? reinterpret_cast<unsigned char*>(output8->data.get()) : nullptr);  // k_encoder256_downsample
        } else {
            C256_LAUNCH(k_encoder256_pool,pool_n,raw.data,pool.data,width,height);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,pool.data,256,matrix.data,256,nullptr,0,nullptr,
                output.data,nullptr,512,width*height/4,512);  // k_encoder256_downsample
        }
        if (!check(dir,"pool",pool.data,pool_n,verify,comparisons) ||
            !check(dir,"output",output.data,out_n,verify,comparisons)) return false;
        ready=true;
        return true;
    }
    DeviceTensor pool_view() const { return ready ? DeviceTensor{pool.data,pool_n} : DeviceTensor{}; }
    DeviceTensor final_view() const {
        return !ready ? DeviceTensor{} : wrote8 ? DeviceTensor{output8->data.get(),out_n,1} : DeviceTensor{output.data.get(),out_n};
    }
};
#undef C256_LAUNCH
} // namespace c256_resident
