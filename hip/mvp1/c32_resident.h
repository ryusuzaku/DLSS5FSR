#pragma once
// Resident scalar C32 candidate in the audited public QMMA transition basis.
// Reuses existing arithmetic; this does not establish original-kernel parity.
#include "split512_resident.h"
#include "upsample66_prefix.hip"
#include "spatial32_peer.hip"
#include "head70.hip"
#include "swin_1h_chain_c32.hip"
#include "head70_normalized.hip"
#include <stdexcept>

namespace c32_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;
#define C32_LAUNCH(kernel,count,...) do { \
    hipLaunchKernelGGL(kernel,dim3(((count)+255)/256),dim3(256),0,c512_resident::stream,__VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); } while(0)
inline bool check(const std::string& dir,const char* name,const float* data,
                  size_t count,bool verify,size_t& comparisons) {
    if (!verify) return true;
    ++comparisons;traffic.d2h_bytes+=count*sizeof(float);
    return compare((dir+": "+name).c_str(),const_cast<float*>(data),read(dir,name,count));
}
// Same elementwise operations as the standalone C32/head test drivers.
__global__ void quantize_body(const float* body,float* output,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=h70_f(body[i]);
}
__global__ void native_to_peer(const float* native,float* peer,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)peer[i]=native[(i&~31)+peer_to_native32(i&31)];
}
__global__ void peer_to_native(const float* peer,float* native,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)native[(i&~31)+peer_to_native32(i&31)]=peer[i];
}
struct Weights {
    std::string dir;
    float scale;
    Buffer data;
    explicit Weights(const std::string& d):dir(d),scale(read(d,"weights",16449)[16384]),data(d,"weights",16449) {}
};
// Shared retained workspace for sequential C32 body dispatches.
class Body {
    int capacity;
    Buffer expanded,hidden,ffn,qkv,norm,scores,ex,den,prob,context,body,output;
public:
    explicit Body(int rows):capacity(rows),expanded(size_t(rows)*128),hidden(expanded.count),
        ffn(size_t(rows)*32),qkv(size_t(rows)*96),norm(qkv.count),scores(size_t(rows)*64),
        ex(scores.count),den(rows),prob(scores.count),context(ffn.count),body(ffn.count),output(ffn.count) {}
    bool run(const Weights& w,const float* input,int rows,bool verify,size_t& comparisons) {
        if (!input || rows<=0 || rows>capacity || rows%64) return false;
        auto dw=w.data.data;size_t n=size_t(rows)*32;
        C32_LAUNCH(k_h70_expand,size_t(rows)*128,input,dw,expanded.data,rows);
        C32_LAUNCH(k_h70_hidden,size_t(rows)*128,expanded.data,hidden.data,rows*128);
        C32_LAUNCH(k_h70_ffn,n,input,hidden.data,dw+4096,dw+16385,ffn.data,rows);
        C32_LAUNCH(k_h70_qkv,size_t(rows)*96,ffn.data,dw+8192,qkv.data,rows);
        C32_LAUNCH(k_h70_qknorm,size_t(rows)*96,qkv.data,norm.data,rows,w.scale);
        C32_LAUNCH(k_h70_scores,size_t(rows)*64,norm.data,dw+12288,scores.data,rows/64);
        C32_LAUNCH(k_h70_exp,size_t(rows)*64,scores.data,ex.data,rows*64);
        C32_LAUNCH(k_h70_den,rows,ex.data,den.data,rows);
        C32_LAUNCH(k_h70_prob,size_t(rows)*64,ex.data,den.data,prob.data,rows*64);
        C32_LAUNCH(k_h70_context,n,prob.data,norm.data,context.data,rows/64);
        C32_LAUNCH(k_h70_projection,n,context.data,ffn.data,dw+11264,dw+16417,body.data,rows);
        C32_LAUNCH(quantize_body,n,body.data,output.data,int(n));
        struct Stage {const char* name;float* data;size_t count;};
        Stage stages[]={{"expanded",expanded.data,size_t(rows)*128},{"hidden",hidden.data,size_t(rows)*128},
          {"ffn",ffn.data,n},{"qkv",qkv.data,size_t(rows)*96},{"qknorm",norm.data,size_t(rows)*96},
          {"scores",scores.data,size_t(rows)*64},{"exp",ex.data,size_t(rows)*64},{"den",den.data,size_t(rows)},
          {"prob",prob.data,size_t(rows)*64},{"context",context.data,n},{"body",body.data,n},{"output",output.data,n}};
        for (auto& s:stages) if(!check(w.dir,s.name,s.data,s.count,verify,comparisons))return false;
        return true;
    }
    const float* raw() const {return body.data;}
    const float* quantized() const {return output.data;}
};
class Prefix {
    static int checked(int w, int h) {
        if ((w != 64 && w != 256) || h != 64) throw std::invalid_argument("block66 requires 64x64 or 256x64 input");
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
        n(size_t(w)*h*64), out_n(n*2), dir(fixture), weights(dir,"weights",32*64),
        scale(dir,"scale",32), low(n/2), merged(out_n) {}
    bool run_from_device(DeviceTensor input, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!input.data || input.count != n || !skip.data || skip.count != out_n) return false;
        if (!check(dir,"input",input.data,n,verify,comparisons) ||
            !check(dir,"skip",skip.data,out_n,verify,comparisons)) return false;
        C32_LAUNCH(k_upsample66_project, n/2, input.data,weights.data,low.data,width*height);
        C32_LAUNCH(k_upsample66_merge, out_n, low.data,skip.data,scale.data,merged.data,width,height);
        if (!check(dir,"low",low.data,n/2,verify,comparisons) ||
            !check(dir,"merged",merged.data,out_n,verify,comparisons)) return false;
        ready = true;
        return true;
    }
    DeviceTensor low_view() const { return ready ? DeviceTensor{low.data,n/2} : DeviceTensor{}; }
    DeviceTensor final_view() const { return ready ? DeviceTensor{merged.data,out_n} : DeviceTensor{}; }
};

class Chain {
    static int checked(int w,int h,const std::vector<std::pair<std::string,int>>& blocks) {
        if ((w!=128 && w!=512)||h!=128||blocks.size()!=4)throw std::invalid_argument("C32 requires 128x128 or 512x128 and four blocks");
        for(auto& b:blocks)if(b.second<0||b.second>3)throw std::invalid_argument("invalid shift");
        return w;
    }
    int width,height;
    size_t n;
    Buffer ping,pong,windows;
    Body body;
    std::vector<std::unique_ptr<Weights>> weights;
    std::vector<int> shifts;
    float* result=nullptr;
public:
    size_t comparisons=0;
    Chain(int w,int h,const std::vector<std::pair<std::string,int>>& blocks):width(checked(w,h,blocks)),height(h),
        n(size_t(w)*h*32),ping(n),pong(n),windows(size_t(w+8)*(h+8)*32),body((w+8)*(h+8)) {
        for(auto& b:blocks){weights.emplace_back(new Weights(b.first+"/body"));shifts.push_back(b.second);}
    }
    bool run_from_device(DeviceTensor source,bool verify=false) {
        result=nullptr;if(!source.data||source.count!=n)return false;
        if(source.data!=ping.data){
            HIP_CHECK(hipMemcpyAsync(ping.data,source.data,n*sizeof(float),hipMemcpyDeviceToDevice,c512_resident::stream));traffic.d2d_bytes+=n*sizeof(float);
        }
        float* input=ping.data;float* output=pong.data;
        for(size_t i=0;i<weights.size();++i){
            auto& w=*weights[i];int shift=shifts[i],px=(shift&1)?4:0,py=(shift&2)?4:0;
            int rows=((width+px+7)/8)*8*((height+py+7)/8)*8;size_t count=size_t(rows)*32;
            std::string dir=w.dir.substr(0,w.dir.size()-5);
            if(!check(dir+"/spatial","input",input,n,verify,comparisons))return false;
            C32_LAUNCH(k_spatial32_peer_gather,count,input,windows.data,width,height,shift);
            if(!check(dir+"/spatial","windows",windows.data,count,verify,comparisons)||
               !body.run(w,windows.data,rows,verify,comparisons))return false;
            C32_LAUNCH(k_spatial32_peer_scatter,n,body.quantized(),output,width,height,shift);
            if(!check(dir+"/output","output",output,n,verify,comparisons))return false;
            std::swap(input,output);
        }
        result=input;return true;
    }
    DeviceTensor final_view()const{return result?DeviceTensor{result,n}:DeviceTensor{};}
};

class Head {
    static int checked(int w,int h){
        if((w!=256&&w!=1024)||h!=256)throw std::invalid_argument("head requires 256x256 or 1024x256");return w;
    }
    int width,height;
    size_t n,nrgb;
    std::string dir;
    Buffer sm,ss,coeff,merged,peer,native,rgb_native,rgb_public;
    Weights weights;
    Body body;
    bool ready=false;
public:
    size_t comparisons=0;
    Head(int w,int h,const std::string& fixture):width(checked(w,h)),height(h),n(size_t(w)*h*32),nrgb(size_t(w)*h*3),dir(fixture),
        sm(dir,"sm",32),ss(dir,"ss",32),coeff(dir,"coeff",96),merged(n),peer(n),native(n),rgb_native(nrgb),rgb_public(nrgb),
        weights(dir+"/body"),body(w*h) {}
    bool run_from_device(DeviceTensor main,DeviceTensor skip,DeviceTensor color,bool verify=false){
        ready=false;
        if(!main.data||main.count!=n/4||!skip.data||skip.count!=n||!color.data||color.count!=nrgb)return false;
        if(!check(dir,"main",main.data,n/4,verify,comparisons)||!check(dir,"skip",skip.data,n,verify,comparisons)||
           !check(dir,"color",color.data,nrgb,verify,comparisons))return false;
        C32_LAUNCH(k_head70_merge,n,main.data,skip.data,sm.data,ss.data,merged.data,width,height);
        C32_LAUNCH(native_to_peer,n,merged.data,peer.data,int(n));
        if(!check(dir,"merged",merged.data,n,verify,comparisons)||!check(dir,"peer",peer.data,n,verify,comparisons)||
           !body.run(weights,peer.data,width*height,verify,comparisons))return false;
        C32_LAUNCH(peer_to_native,n,body.raw(),native.data,int(n));
        C32_LAUNCH(k_head70_finish,nrgb,native.data,coeff.data,color.data,rgb_native.data,width,height,.03125f);
        C32_LAUNCH(k_head70_finish,nrgb,native.data,coeff.data,color.data,rgb_public.data,width,height,1.f);
        if(!check(dir,"native",native.data,n,verify,comparisons)||!check(dir,"rgb_native",rgb_native.data,nrgb,verify,comparisons)||
           !check(dir,"rgb_public",rgb_public.data,nrgb,verify,comparisons))return false;
        ready=true;return true;
    }
    DeviceTensor merged_view()const{return ready?DeviceTensor{merged.data,n}:DeviceTensor{};}
    DeviceTensor peer_view()const{return ready?DeviceTensor{peer.data,n}:DeviceTensor{};}
    DeviceTensor body_view()const{return ready?DeviceTensor{body.raw(),n}:DeviceTensor{};}
    DeviceTensor native_view()const{return ready?DeviceTensor{native.data,n}:DeviceTensor{};}
    DeviceTensor final_view()const{return ready?DeviceTensor{rgb_native.data,nrgb}:DeviceTensor{};}
    DeviceTensor public_view()const{return ready?DeviceTensor{rgb_public.data,nrgb}:DeviceTensor{};}
};
#undef C32_LAUNCH
} // namespace c32_resident
