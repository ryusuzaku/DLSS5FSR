#pragma once
// Fixed-shape decoder62-65 candidate; unchanged arithmetic kernels below.
// Borrowed same-device/default-stream views expire at the next submission.
#include "split512_resident.h"
#include "upsample62_prefix.hip"
#include "spatial64_window.hip"
#include "c64_ffn_candidate.hip"
#include "c64_attention_candidate.hip"
#include "encoder64_downsample.hip"
#include <stdexcept>

namespace c64_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

#define C64_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+63)/64), dim3(64), 0, c512_resident::stream, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); } while (0)

inline bool check(const std::string& dir, const char* name, const float* device,
                  size_t count, bool verify, size_t& comparisons) {
    if (!verify) return true;
    ++comparisons;
    traffic.d2h_bytes += count*sizeof(float);
    return compare((dir+": "+name).c_str(), const_cast<float*>(device), read(dir, name, count));
}

class Prefix {
    static int checked(int w, int h) {
        if ((w != 32 && w != 128) || h != 32) throw std::invalid_argument("block62 requires 32x32 or 128x32 input");
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
        n(size_t(w)*h*128), out_n(n*2), dir(fixture), weights(dir,"weights",64*128),
        scale(dir,"scale",64), low(n/2), merged(out_n) {}
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        C64_LAUNCH(k_upsample62_project, n/2, input.data,weights.data,low.data,width*height);
        C64_LAUNCH(k_upsample62_merge, out_n, low.data,skip.data,scale.data,merged.data,width,height);
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
        w1(d+"/ffn","w1",256*64),w2(d+"/ffn","w2",64*256),
        w3(d+"/ffn","w3",64*64),skip(d+"/ffn","skip",64),
        qkv(d+"/attention","qkv_weights",3*64*64),bias(d+"/attention","bias",2*4096),
        scales(d+"/attention","scales",2),projection(d+"/attention","projection_weights",64*64),
        attention_skip(d+"/attention","attention_skip",64) {}
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder=false) {
        if (encoder ? ((w != 64 && w != 256) || h != 64 || blocks.size() != 4)
                    : ((w != 64 && w != 256) || h != 64 || blocks.size() != 4))
            throw std::invalid_argument(encoder ? "resident encoder C64 requires 64x64 or 256x64 and 4 blocks"
                                                : "resident decoder C64 requires 64x64 or 256x64 and four blocks");
        for (const auto& b: blocks) if (b.second < 0 || b.second > 3) throw std::invalid_argument("invalid shift");
        return w;
    }
    int width,height,max_tokens;
    size_t n;
    bool encoder;
    Buffer ping,pong,windows,expanded,hidden,middle,feature,qkv,normalized,scores,
           exponents,probabilities,context,linear,residual;
    // Encoder mode also keeps the last block's unquantized projection for its pool.
    std::unique_ptr<Buffer> raw_windows,raw;
    std::vector<std::unique_ptr<Weights>> weights;
    float* result = nullptr;
public:
    size_t comparisons = 0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks,bool encoder_mode=false) :
        width(checked(w,h,blocks,encoder_mode)),height(h),max_tokens((w+8)*(h+8)),n(size_t(w)*h*64),encoder(encoder_mode),
        ping(n),pong(n),windows(size_t(max_tokens)*64),expanded(size_t(max_tokens)*256),
        hidden(expanded.count),middle(windows.count),feature(windows.count),qkv(windows.count*3),
        normalized(qkv.count),scores(size_t(max_tokens)*128),exponents(scores.count),
        probabilities(scores.count),context(windows.count),linear(windows.count),residual(windows.count) {
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
            size_t count=size_t(tokens)*64, sc=size_t(wn)*2*4096;
            if (!check(w.dir+"/spatial","input",input,n,verify,comparisons)) return false;
            C64_LAUNCH(k_spatial64_gather,count,input,windows.data,width,height,w.shift);
            C64_LAUNCH(k_c64_w1,4*count,windows.data,w.w1.data,expanded.data,tokens);
            C64_LAUNCH(k_c64_gate,4*count,expanded.data,hidden.data,int(4*count));
            C64_LAUNCH(k_c64_w2,count,hidden.data,w.w2.data,middle.data,tokens);
            C64_LAUNCH(k_c64_w3,count,middle.data,windows.data,w.w3.data,w.skip.data,feature.data,tokens);
            C64_LAUNCH(k_c64_qkv,3*count,feature.data,w.qkv.data,qkv.data,tokens);
            C64_LAUNCH(k_c64_qknorm,3*count,qkv.data,w.scales.data,normalized.data,wn);
            C64_LAUNCH(k_c64_scores,sc,normalized.data,w.bias.data,scores.data,wn);
            C64_LAUNCH(k_split512_exp,sc,scores.data,exponents.data,int(sc));
            C64_LAUNCH(k_split512_prob,sc,exponents.data,probabilities.data,int(sc));
            C64_LAUNCH(k_c64_context,count,probabilities.data,normalized.data,context.data,wn);
            C64_LAUNCH(k_c64_projection_linear,count,context.data,w.projection.data,linear.data,tokens);
            C64_LAUNCH(k_c64_projection_residual,count,context.data,feature.data,w.projection.data,
                        w.attention_skip.data,residual.data,tokens);
            C64_LAUNCH(k_spatial64_scatter,n,residual.data,output,width,height,w.shift);
            if (last_raw) {
                C64_LAUNCH(k_c64_projection_residual,count,context.data,feature.data,w.projection.data,
                            w.attention_skip.data,raw_windows->data,tokens,true);
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
            std::swap(input,output);
        }
        result=input;
        return true;
    }
    DeviceTensor final_view() const { return result ? DeviceTensor{result,n} : DeviceTensor{}; }
    DeviceTensor raw_view() const { return result && encoder ? DeviceTensor{raw->data,n} : DeviceTensor{}; }
};

// Encoder8 raw body -> rounded 2x2 pool -> FP8 C128 projection.
class Downsample {
    static int checked(int w,int h) {
        if ((w != 64 && w != 256) || h != 64) throw std::invalid_argument("downsample8 requires 64x64 or 256x64");
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
        n(size_t(w)*h*64),pool_n(n/4),out_n(n/2),dir(fixture),matrix(dir,"matrix",128*64),
        pool(pool_n),output(out_n) {}
    bool run_from_device(DeviceTensor raw,bool verify=false) {
        ready=false;
        if (!raw.data || raw.count != n) return false;
        if (!check(dir,"raw",raw.data,n,verify,comparisons)) return false;
        C64_LAUNCH(k_encoder64_pool,pool_n,raw.data,pool.data,width,height);
        C64_LAUNCH(k_encoder64_downsample,out_n,pool.data,matrix.data,output.data,width*height/4);
        if (!check(dir,"pool",pool.data,pool_n,verify,comparisons) ||
            !check(dir,"output",output.data,out_n,verify,comparisons)) return false;
        ready=true;
        return true;
    }
    DeviceTensor pool_view() const { return ready ? DeviceTensor{pool.data,pool_n} : DeviceTensor{}; }
    DeviceTensor final_view() const { return ready ? DeviceTensor{output.data,out_n} : DeviceTensor{}; }
};
#undef C64_LAUNCH
} // namespace c64_resident
