#pragma once
#include "split512_resident.h"
#include "upsample48_prefix.hip"
#include "spatial256_window.hip"
#include "c256_ffn_candidate.hip"
#include "c256_attention_candidate.hip"
#include <stdexcept>

namespace c256_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

#define C256_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+255)/256), dim3(256), 0, 0, __VA_ARGS__); \
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
        if ((w != 8 && w != 32) || h != 8) throw std::invalid_argument("block48 requires 8x8 or 32x8 input");
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
        n(size_t(w)*h*512), out_n(n*2), dir(fixture), weights(dir,"weights",256*512),
        scale(dir,"scale",256), low(n/2), merged(out_n) {}
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        C256_LAUNCH(k_upsample48_project, n/2, input.data,weights.data,low.data,width*height);
        C256_LAUNCH(k_upsample48_merge, out_n, low.data,skip.data,scale.data,merged.data,width,height);
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
        w1(d+"/ffn","w1",1024*256),w2(d+"/ffn","w2",256*1024),
        w3(d+"/ffn","w3",256*256),skip(d+"/ffn","skip",256),
        qkv(d+"/attention","qkv_weights",3*256*256),bias(d+"/attention","bias",8*4096),
        scales(d+"/attention","scales",8),projection(d+"/attention","projection_weights",256*256),
        attention_skip(d+"/attention","attention_skip",256) {}
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks) {
        if ((w != 16 && w != 64) || h != 16 || blocks.size() != 8)
            throw std::invalid_argument("resident decoder C256 requires 16x16 or 64x16 and eight blocks");
        for (const auto& b: blocks) if (b.second < 0 || b.second > 3) throw std::invalid_argument("invalid shift");
        return w;
    }
    int width,height,max_tokens;
    size_t n;
    Buffer ping,pong,windows,expanded,hidden,middle,feature,qkv,normalized,scores,
           exponents,probabilities,context,linear,residual;
    std::vector<std::unique_ptr<Weights>> weights;
    float* result = nullptr;
public:
    size_t comparisons = 0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks) :
        width(checked(w,h,blocks)),height(h),max_tokens((w+8)*(h+8)),n(size_t(w)*h*256),
        ping(n),pong(n),windows(size_t(max_tokens)*256),expanded(size_t(max_tokens)*1024),
        hidden(expanded.count),middle(windows.count),feature(windows.count),qkv(windows.count*3),
        normalized(qkv.count),scores(size_t(max_tokens)*512),exponents(scores.count),
        probabilities(scores.count),context(windows.count),linear(windows.count),residual(windows.count) {
        for (const auto& b:blocks) weights.emplace_back(new Weights(b.first,b.second));
    }
    bool run_from_device(DeviceTensor source,bool verify=false) {
        result = nullptr;
        if (!source.data || source.count != n) return false;
        if (source.data != ping.data) {
            HIP_CHECK(hipMemcpyAsync(ping.data,source.data,n*sizeof(float),hipMemcpyDeviceToDevice));
            traffic.d2d_bytes += n*sizeof(float);
        }
        float *input=ping.data,*output=pong.data;
        for (const auto& owned:weights) {
            const auto& w=*owned;
            int px=(w.shift&1)?4:0,py=(w.shift&2)?4:0;
            int tokens=((width+px+7)/8)*8*((height+py+7)/8)*8;
            int wn=tokens/64;
            size_t count=size_t(tokens)*256, sc=size_t(wn)*8*4096;
            if (!check(w.dir+"/spatial","input",input,n,verify,comparisons)) return false;
            C256_LAUNCH(k_spatial256_gather,count,input,windows.data,width,height,w.shift);
            C256_LAUNCH(k_c256_w1,4*count,windows.data,w.w1.data,expanded.data,tokens);
            C256_LAUNCH(k_c256_gate,4*count,expanded.data,hidden.data,int(4*count));
            C256_LAUNCH(k_c256_w2,count,hidden.data,w.w2.data,middle.data,tokens);
            C256_LAUNCH(k_c256_w3,count,middle.data,windows.data,w.w3.data,w.skip.data,feature.data,tokens);
            C256_LAUNCH(k_c256_qkv,3*count,feature.data,w.qkv.data,qkv.data,tokens);
            C256_LAUNCH(k_c256_qknorm,3*count,qkv.data,w.scales.data,normalized.data,wn);
            C256_LAUNCH(k_c256_scores,sc,normalized.data,w.bias.data,scores.data,wn);
            C256_LAUNCH(k_split512_exp,sc,scores.data,exponents.data,int(sc));
            C256_LAUNCH(k_split512_prob,sc,exponents.data,probabilities.data,int(sc));
            C256_LAUNCH(k_c256_context,count,probabilities.data,normalized.data,context.data,wn);
            C256_LAUNCH(k_c256_projection_linear,count,context.data,w.projection.data,linear.data,tokens);
            C256_LAUNCH(k_c256_projection_residual,count,context.data,feature.data,w.projection.data,
                        w.attention_skip.data,residual.data,tokens);
            C256_LAUNCH(k_spatial256_scatter,n,residual.data,output,width,height,w.shift);
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
            std::swap(input,output);
        }
        result=input;
        return true;
    }
    DeviceTensor final_view() const { return result ? DeviceTensor{result,n} : DeviceTensor{}; }
};
#undef C256_LAUNCH
} // namespace c256_resident
