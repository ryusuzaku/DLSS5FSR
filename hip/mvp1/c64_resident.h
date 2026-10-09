#pragma once
// Fixed-shape decoder62-65 candidate; unchanged arithmetic kernels below.
// Borrowed same-device/default-stream views expire at the next submission.
#include "split512_resident.h"
#include "c32_resident.h"
#include "swin_t.hip"
#include "upsample62_prefix.hip"
#include "spatial64_window.hip"
#include "c64_ffn_candidate.hip"
#include "c64_attention_candidate.hip"
#include "tiled_gemm.hip"
#include "encoder64_downsample.hip"
#include <stdexcept>

namespace c64_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

#define C64_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+63)/64), dim3(64), 0, c512_resident::stream, __VA_ARGS__); \
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
        if (w <= 0 || h <= 0 || w % 8 || h % 8) throw std::invalid_argument("block62 input must be a positive multiple of 8");
        return w;
    }
    int width, height;
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
    Prefix(int w, int h, const std::string& fixture) : width(checked(w,h)), height(h),
        n(size_t(w)*h*128), out_n(n*2), dir(fixture), weights(dir,"weights",64*128),
        scale(dir,"scale",64), low(n/2), merged(out_n) {}
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        wrote8 = byte_out && !verify && !c512_resident::exact_math;
        if ((input.fmt || skip.fmt) && (verify || c512_resident::exact_math)) return false;
        tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,input.fmt?nullptr:input.data,128,weights.data,128,nullptr,0,nullptr,
            low.data,nullptr,64,width*height,64,1,0,0,0,1,input.fmt==1?reinterpret_cast<const unsigned char*>(input.data):nullptr);  // k_upsample62_project
        if (wrote8 && !merged8) merged8.reset(new Buffer((out_n+3)/4));
        C64_LAUNCH(k_upsample62_merge, out_n, low.data,skip.data,scale.data,wrote8 ? merged8->data.get() : merged.data.get(),width,height,skip.fmt,int(wrote8));
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
        w1(d+"/ffn","w1",256*64),w2(d+"/ffn","w2",64*256),
        w3(d+"/ffn","w3",64*64),skip(d+"/ffn","skip",64),
        qkv(d+"/attention","qkv_weights",3*64*64),bias(d+"/attention","bias",2*4096),
        scales(d+"/attention","scales",2),projection(d+"/attention","projection_weights",64*64),
        attention_skip(d+"/attention","attention_skip",64) {}
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder=false) {
        (void)encoder;
        if (w <= 0 || h <= 0 || w % 8 || h % 8 || blocks.size() != 4)
            throw std::invalid_argument("resident C64 requires a positive multiple of 8 and four blocks");
        for (const auto& b: blocks) if (b.second < 0 || b.second > 3) throw std::invalid_argument("invalid shift");
        return w;
    }
    int width,height,max_tokens;
    size_t n;
    bool encoder;
    Buffer ping,pong,windows,expanded,hidden,middle,feature,qkv,normalized,scores,
           exponents,probabilities,inverse,context,linear,residual;
    // Encoder mode also keeps the last block's unquantized projection for its pool.
    std::unique_ptr<Buffer> raw_windows,raw;
    std::vector<std::unique_ptr<Weights>> weights;
    float* result = nullptr;
    int result_fmt = 0;
    bool raw_was_half = false;
    std::unique_ptr<Buffer> ping8, pong8, raw_h;  // byte activations, half raw
public:
    size_t comparisons = 0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder_mode=false) :
        width(checked(w,h,blocks,encoder_mode)),height(h),max_tokens((w+8)*(h+8)),n(size_t(w)*h*64),encoder(encoder_mode),
        ping(n),pong(n),windows(size_t(max_tokens)*64),expanded(size_t(max_tokens)*256),
        hidden(expanded.count),middle(windows.count),feature(windows.count),qkv(windows.count*3),
        normalized(qkv.count),scores(size_t(max_tokens)*128),exponents(scores.count),
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
                if (!swin_t::run<64>(static_cast<const float*>(in), static_cast<float*>(out), last_raw ? raw_h->data.get() : nullptr,
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
            size_t count=size_t(tokens)*64, sc=size_t(wn)*2*4096;
            if (!check(w.dir+"/spatial","input",input,n,verify,comparisons)) return false;
            if (!verify && swin_t::run<64>(input,output,last_raw?raw->data:nullptr,width,height,w.shift,
                    {w.w1.data,w.w2.data,w.w3.data,w.skip.data,w.qkv.data,w.scales.data,w.bias.data,
                     w.projection.data,w.attention_skip.data},middle.data)) {
                advance();
                continue;
            }
            C64_LAUNCH(k_spatial64_gather,count,input,windows.data,width,height,w.shift);
            // Tiled, bit-identical forms of the w1+gate, w2, w3 and QKV kernels.
            tiled::gemm<tiled::Split,tiled::GATE,false>(c512_resident::stream,windows.data,64,w.w1.data,64,
                nullptr,0,nullptr,verify?expanded.data:nullptr,hidden.data,256,tokens,256);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,hidden.data,256,w.w2.data,256,
                nullptr,0,nullptr,middle.data,nullptr,64,tokens,64);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,middle.data,64,w.w3.data,64,
                windows.data,64,w.skip.data,feature.data,nullptr,64,tokens,64);
            tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,feature.data,64,w.qkv.data,64,
                nullptr,0,nullptr,qkv.data,nullptr,192,tokens,192);
            HIP_CHECK(hipGetLastError());
            if (verify || !c512_resident::fused_attention(qkv.data, w.scales.data, w.bias.data, context.data, tokens, 64)) {
                C64_LAUNCH(k_split512_qknorm_inv,size_t(tokens)*2*2,qkv.data,inverse.data,tokens,64);
                C64_LAUNCH(k_split512_qknorm_apply,size_t(tokens)*3*64,qkv.data,w.scales.data,inverse.data,normalized.data,tokens,64);
                C64_LAUNCH(k_c64_scores,sc,normalized.data,w.bias.data,scores.data,wn);
                C64_LAUNCH(k_split512_exp,sc,scores.data,exponents.data,int(sc));
                C64_LAUNCH(k_split512_inv_rows,sc/64,exponents.data,inverse.data,int(sc/64));
                C64_LAUNCH(k_split512_prob_rows,sc,exponents.data,inverse.data,probabilities.data,int(sc));
                C64_LAUNCH(k_c64_context,count,probabilities.data,normalized.data,context.data,wn);
            }
            // The linear projection is a diagnostic output; it runs only when verifying.
            if (verify) C64_LAUNCH(k_c64_projection_linear,count,context.data,w.projection.data,linear.data,tokens);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,context.data,64,w.projection.data,64,
                feature.data,64,w.attention_skip.data,residual.data,last_raw?raw_windows->data:nullptr,64,tokens,64);
            HIP_CHECK(hipGetLastError());
            C64_LAUNCH(k_spatial64_scatter,n,residual.data,output,width,height,w.shift);
            if (last_raw) {
                C64_LAUNCH(k_spatial64_scatter,n,raw_windows->data,raw->data,width,height,w.shift);
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

// Encoder8 raw body -> rounded 2x2 pool -> FP8 C128 projection.
class Downsample {
    static int checked(int w,int h) {
        if (w <= 0 || h <= 0 || w % 16 || h % 16) throw std::invalid_argument("downsample8 extent must be a positive multiple of 16");
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
        n(size_t(w)*h*64),pool_n(n/4),out_n(n/2),dir(fixture),matrix(dir,"matrix",128*64),
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
            C64_LAUNCH(k_encoder64_pool,pool_n,raw.data,pool8->data.get(),width,height,raw.fmt,1);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,nullptr,64,matrix.data,64,nullptr,0,nullptr,
                wrote8 ? nullptr : output.data.get(),nullptr,128,width*height/4,128,1,0,0,0,1,
                reinterpret_cast<const unsigned char*>(pool8->data.get()),
                wrote8 ? reinterpret_cast<unsigned char*>(output8->data.get()) : nullptr);  // k_encoder64_downsample
        } else {
            C64_LAUNCH(k_encoder64_pool,pool_n,raw.data,pool.data,width,height);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,pool.data,64,matrix.data,64,nullptr,0,nullptr,
                output.data,nullptr,128,width*height/4,128);  // k_encoder64_downsample
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
#undef C64_LAUNCH
} // namespace c64_resident
