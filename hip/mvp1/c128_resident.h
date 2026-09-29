#pragma once
// Fixed-shape decoder56-61 candidate; unchanged arithmetic kernels below.
// Borrowed same-device/default-stream views expire at the next submission.
#include "split512_resident.h"
#include "upsample56_prefix.hip"
#include "spatial128_window.hip"
#include "c128_ffn_candidate.hip"
#include "c128_attention_candidate.hip"
#include "tiled_gemm.hip"
#include "encoder128_downsample.hip"
#include <stdexcept>

namespace c128_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

#define C128_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+127)/128), dim3(128), 0, c512_resident::stream, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while (0)

inline bool check(const std::string& dir, const char* name, const float* device,
                  size_t count, bool verify, size_t& comparisons) {
    if (!verify) return true;
    ++comparisons;
    traffic.d2h_bytes += count*sizeof(float);
    return compare((dir+": "+name).c_str(), const_cast<float*>(device), read(dir, name, count));
}

class Prefix {
    static int checked(int w, int h) {
        if (w <= 0 || h <= 0 || w % 8 || h % 8) throw std::invalid_argument("block56 input must be a positive multiple of 8");
        return w;
    }
    int width, height;
    size_t n, out_n;
    std::string dir;
    Buffer weights, scale, low, merged;
    bool ready = false;
public:
    size_t comparisons = 0;
    Prefix(int w, int h, const std::string& fixture) : width(checked(w,h)), height(h),
        n(size_t(w)*h*256), out_n(n*2), dir(fixture), weights(dir,"weights",128*256),
        scale(dir,"scale",128), low(n/2), merged(out_n) {}
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        C128_LAUNCH(k_upsample56_project, n/2, input.data,weights.data,low.data,width*height);
        C128_LAUNCH(k_upsample56_merge, out_n, low.data,skip.data,scale.data,merged.data,width,height);
        if (!check(dir,"low",low.data,n/2,verify,comparisons) ||
            !check(dir,"merged",merged.data,out_n,verify,comparisons)) return false;
        ready = true;
        return true;
    }
    DeviceTensor low_view() const { return ready ? DeviceTensor{low.data,n/2} : DeviceTensor{}; }
    DeviceTensor final_view() const { return ready ? DeviceTensor{merged.data,out_n} : DeviceTensor{}; }
};

struct Weights {
    std::string dir;
    int shift;
    Buffer w1,w2,w3,skip,qkv,bias,scales,projection,attention_skip;
    Weights(const std::string& d, int s) : dir(d),shift(s),
        w1(d+"/ffn","w1",512*128),w2(d+"/ffn","w2",128*512),
        w3(d+"/ffn","w3",128*128),skip(d+"/ffn","skip",128),
        qkv(d+"/attention","qkv_weights",3*128*128),bias(d+"/attention","bias",4*4096),
        scales(d+"/attention","scales",4),projection(d+"/attention","projection_weights",128*128),
        attention_skip(d+"/attention","attention_skip",128) {}
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder=false) {
        (void)encoder;
        if (w <= 0 || h <= 0 || w % 8 || h % 8 || blocks.size() != 6)
            throw std::invalid_argument("resident C128 requires a positive multiple of 8 and six blocks");
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
public:
    size_t comparisons = 0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder_mode=false) :
        width(checked(w,h,blocks,encoder_mode)),height(h),max_tokens((w+8)*(h+8)),n(size_t(w)*h*128),encoder(encoder_mode),
        ping(n),pong(n),windows(size_t(max_tokens)*128),expanded(size_t(max_tokens)*512),
        hidden(expanded.count),middle(windows.count),feature(windows.count),qkv(windows.count*3),
        normalized(qkv.count),scores(size_t(max_tokens)*256),exponents(scores.count),
        probabilities(scores.count),inverse(scores.count/32),context(windows.count),linear(windows.count),residual(windows.count) {
        if (encoder) { raw_windows.reset(new Buffer(windows.count)); raw.reset(new Buffer(n)); }
        for (const auto& b:blocks) weights.emplace_back(new Weights(b.first,b.second));
    }
    bool run_from_device(DeviceTensor source,bool verify=false) {
        result = nullptr;
        if (!source.data || source.count != n) return false;
        if (source.data != ping.data) {
            HIP_CHECK(hipMemcpyAsync(ping.data,source.data,n*sizeof(float),hipMemcpyDeviceToDevice,c512_resident::stream));
            traffic.d2d_bytes += n*sizeof(float);
        }
        float *input=ping.data,*output=pong.data;
        for (size_t index=0;index<weights.size();++index) {
            const auto& w=*weights[index];
            bool last_raw=encoder && index+1==weights.size();
            int px=(w.shift&1)?4:0,py=(w.shift&2)?4:0;
            int tokens=((width+px+7)/8)*8*((height+py+7)/8)*8;
            int wn=tokens/64;
            size_t count=size_t(tokens)*128, sc=size_t(wn)*4*4096;
            if (!check(w.dir+"/spatial","input",input,n,verify,comparisons)) return false;
            C128_LAUNCH(k_spatial128_gather,count,input,windows.data,width,height,w.shift);
            // Tiled, bit-identical forms of the w1+gate, w2, w3 and QKV kernels.
            tiled::gemm<tiled::Split,tiled::GATE,false>(c512_resident::stream,windows.data,128,w.w1.data,128,
                nullptr,0,nullptr,verify?expanded.data:nullptr,hidden.data,512,tokens,512);
            tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream,hidden.data,512,w.w2.data,512,
                nullptr,0,nullptr,middle.data,nullptr,128,tokens,128);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,middle.data,128,w.w3.data,128,
                windows.data,128,w.skip.data,feature.data,nullptr,128,tokens,128);
            tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,feature.data,128,w.qkv.data,128,
                nullptr,0,nullptr,qkv.data,nullptr,384,tokens,384);
            HIP_CHECK(hipGetLastError());
            C128_LAUNCH(k_split512_qknorm_inv,size_t(tokens)*2*4,qkv.data,inverse.data,tokens,128);
            C128_LAUNCH(k_split512_qknorm_apply,size_t(tokens)*3*128,qkv.data,w.scales.data,inverse.data,normalized.data,tokens,128);
            C128_LAUNCH(k_c128_scores,sc,normalized.data,w.bias.data,scores.data,wn);
            C128_LAUNCH(k_split512_exp,sc,scores.data,exponents.data,int(sc));
            C128_LAUNCH(k_split512_inv_rows,sc/64,exponents.data,inverse.data,int(sc/64));
            C128_LAUNCH(k_split512_prob_rows,sc,exponents.data,inverse.data,probabilities.data,int(sc));
            C128_LAUNCH(k_c128_context,count,probabilities.data,normalized.data,context.data,wn);
            // The linear projection is a diagnostic output; it runs only when verifying.
            if (verify) C128_LAUNCH(k_c128_projection_linear,count,context.data,w.projection.data,linear.data,tokens);
            tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream,context.data,128,w.projection.data,128,
                feature.data,128,w.attention_skip.data,residual.data,last_raw?raw_windows->data:nullptr,128,tokens,128);
            HIP_CHECK(hipGetLastError());
            C128_LAUNCH(k_spatial128_scatter,n,residual.data,output,width,height,w.shift);
            if (last_raw) {
                C128_LAUNCH(k_spatial128_scatter,n,raw_windows->data,raw->data,width,height,w.shift);
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
            std::swap(input,output);
        }
        result=input;
        return true;
    }
    DeviceTensor final_view() const { return result ? DeviceTensor{result,n} : DeviceTensor{}; }
    DeviceTensor raw_view() const { return result && encoder ? DeviceTensor{raw->data,n} : DeviceTensor{}; }
};

// Encoder14 raw body -> rounded 2x2 pool -> FP8 C256 projection.
class Downsample {
    static int checked(int w,int h) {
        if (w <= 0 || h <= 0 || w % 16 || h % 16) throw std::invalid_argument("downsample14 extent must be a positive multiple of 16");
        return w;
    }
    int width,height;
    size_t n,pool_n,out_n;
    std::string dir;
    Buffer matrix,pool,output;
    bool ready=false;
public:
    size_t comparisons=0;
    Downsample(int w,int h,const std::string& fixture) : width(checked(w,h)),height(h),
        n(size_t(w)*h*128),pool_n(n/4),out_n(n/2),dir(fixture),matrix(dir,"matrix",256*128),
        pool(pool_n),output(out_n) {}
    bool run_from_device(DeviceTensor raw,bool verify=false) {
        ready=false;
        if (!raw.data || raw.count != n) return false;
        if (!check(dir,"raw",raw.data,n,verify,comparisons)) return false;
        C128_LAUNCH(k_encoder128_pool,pool_n,raw.data,pool.data,width,height);
        C128_LAUNCH(k_encoder128_downsample,out_n,pool.data,matrix.data,output.data,width*height/4);
        if (!check(dir,"pool",pool.data,pool_n,verify,comparisons) ||
            !check(dir,"output",output.data,out_n,verify,comparisons)) return false;
        ready=true;
        return true;
    }
    DeviceTensor pool_view() const { return ready ? DeviceTensor{pool.data,pool_n} : DeviceTensor{}; }
    DeviceTensor final_view() const { return ready ? DeviceTensor{output.data,out_n} : DeviceTensor{}; }
};
#undef C128_LAUNCH
} // namespace c128_resident
