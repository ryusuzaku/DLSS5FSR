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

#include "tiled_gemm.hip"
#include "c32_fused.hip"
#include "c32_fp8.hip"
#include "c32_t.hip"

namespace c32_resident {
// Tiled-GEMM rounding policies of the head70/C32 kernels. The expand and QKV
// kernels quantize their input to FP8 on every product; the tile does it once
// per staged value, which is the same value. FFN inputs are used as stored.
struct H70 {
    // Same RNE value as h70_h (exhaustively checked); this form stays fast in tiles.
    __device__ static float half(float x) { return split512_half(x); }
    __device__ static float fp8(float x) { return h70_f(x); }
    __device__ static float load(float x) { return h70_f(x); }
    __device__ static float hd(double x) { return h70_hd(x); }
};
struct H70Raw : H70 {
    __device__ static float load(float x) { return x; }
};
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;
#define C32_LAUNCH(kernel,count,...) do { \
    hipLaunchKernelGGL(kernel,dim3(((count)+255)/256),dim3(256),0,c512_resident::stream,__VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while(0)
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
    Buffer halves;  // first 12288 weights as half (6144 floats of storage) for k_c32_wmma
    Buffer fp8;     // the same as E4M3 bytes (3072 floats of storage) for k_c32_fp8
    explicit Weights(const std::string& d):dir(d),scale(read(d,"weights",16449)[16384]),data(d,"weights",16449),halves(6144),fp8(3072) {
        hipLaunchKernelGGL(k_c32_half_weights,dim3(48),dim3(256),0,c512_resident::stream,
                           data.data,reinterpret_cast<_Float16*>(halves.data),12288);
        hipLaunchKernelGGL(k_c32_fp8_weights,dim3(48),dim3(256),0,c512_resident::stream,
                           data.data,reinterpret_cast<unsigned char*>(fp8.data),12288);
        HIP_CHECK(hipGetLastError());
    }
    // RESIDENT_C32_F16=1: the half WMMA body instead of the FP8 one.
    // Always in the half-activation build (H70_NO_FP8), which has no E4M3 operands.
#if defined(H70_NO_FP8)
    static inline const bool c32_f16 = true;
#else
    static inline const bool c32_f16 = [] { const char* v = getenv("RESIDENT_C32_F16"); return v && *v == '1'; }();
#endif
    // The fused full-resolution blocks (k_c32_t MODE 1 / 2; see C32Fuse):
    // FP8 build, not exact_math.
    // RESIDENT_C32_FUSE=0 keeps the separate stem / pool / merge / finish passes.
    static inline const bool fuse = [] { const char* v = getenv("RESIDENT_C32_FUSE"); return !(v && *v == '0'); }();
    static bool fusable() { return fuse && !c512_resident::exact_math && !c32_f16; }
    template<int MODE> void fused(int rows, const C32Fuse& fz) const {
        hipLaunchKernelGGL((k_c32_t<1, MODE>), dim3(rows/64), dim3(128), 0, c512_resident::stream,
                           nullptr, data.data, reinterpret_cast<const unsigned char*>(fp8.data), scale,
                           nullptr, nullptr, C32Io(), fz);
        HIP_CHECK(hipGetLastError());
        if (c512_resident::launch_hook) c512_resident::launch_hook(MODE == 1 ? "k_c32_t<pre>" : "k_c32_t<head>");
    }
    // Fused body of one launch: WMMA unless c512_resident::exact_math.
    void body(const float* input,int rows,float* raw_out,float* quant_out,const C32Io& io=C32Io()) const {
        if (c512_resident::exact_math && !io.in_hwc && !io.out_hwc && !io.permute) {
            hipLaunchKernelGGL(k_c32_fused,dim3(rows/64),dim3(256),0,c512_resident::stream,
                               input,data.data,scale,raw_out,quant_out);
        } else if (c32_f16) {
            // The half WMMA form (operands through h70_f: E4M3 values, or
            // half with H70_NO_FP8).
            hipLaunchKernelGGL(k_c32_wmma,dim3(rows/64),dim3(256),0,c512_resident::stream,
                               input,data.data,reinterpret_cast<const _Float16*>(halves.data),scale,raw_out,quant_out,io);
        } else {
            // FP8 WMMA form, transposed: bit-identical to k_c32_fp8 / k_c32_wmma,
            // 2.9x faster than k_c32_fp8 (S336).
            hipLaunchKernelGGL(k_c32_t<1>,dim3(rows/64),dim3(128),0,c512_resident::stream,
                               input,data.data,reinterpret_cast<const unsigned char*>(fp8.data),scale,raw_out,quant_out,io);
        }
        HIP_CHECK(hipGetLastError());
        if (c512_resident::launch_hook) c512_resident::launch_hook(c512_resident::exact_math?"k_c32_fused":"k_c32_t");
    }
};
// Shared retained workspace for sequential C32 body dispatches.
class Body {
    int capacity;
    Buffer expanded,hidden,ffn,qkv,norm,ninv,scores,ex,den,prob,context,body,output;
public:
    explicit Body(int rows):capacity(rows),expanded(size_t(rows)*128),hidden(expanded.count),
        ffn(size_t(rows)*32),qkv(size_t(rows)*96),norm(qkv.count),ninv(size_t(rows)*2),scores(size_t(rows)*64),
        ex(scores.count),den(rows),prob(scores.count),context(ffn.count),body(ffn.count),output(ffn.count) {}
    bool run(const Weights& w,const float* input,int rows,bool verify,size_t& comparisons) {
        if (!input || rows<=0 || rows>capacity || rows%64) return false;
        if (!verify) {
            // Fused one-window-per-workgroup body; stage outputs are not kept.
            w.body(input,rows,body.data,output.data);
            return true;
        }
        auto dw=w.data.data;size_t n=size_t(rows)*32;
        tiled::gemm<H70,tiled::GATE,false>(c512_resident::stream,input,32,dw,32,nullptr,0,nullptr,
            verify?expanded.data:nullptr,hidden.data,128,rows,128);
        tiled::gemm<H70Raw,tiled::RAW,true>(c512_resident::stream,hidden.data,128,dw+4096,128,input,32,dw+16385,
            ffn.data,nullptr,32,rows,32);
        tiled::gemm<H70,tiled::RAW,false>(c512_resident::stream,ffn.data,32,dw+8192,32,nullptr,0,nullptr,
            qkv.data,nullptr,96,rows,96);
        HIP_CHECK(hipGetLastError());
        C32_LAUNCH(k_h70_qknorm_inv,size_t(rows)*2,qkv.data,ninv.data,rows);
        C32_LAUNCH(k_h70_qknorm_apply,size_t(rows)*96,qkv.data,ninv.data,norm.data,rows,w.scale);
        C32_LAUNCH(k_h70_scores,size_t(rows)*64,norm.data,dw+12288,scores.data,rows/64);
        C32_LAUNCH(k_h70_exp,size_t(rows)*64,scores.data,ex.data,rows*64);
        C32_LAUNCH(k_h70_den,rows,ex.data,den.data,rows);
        C32_LAUNCH(k_h70_prob,size_t(rows)*64,ex.data,den.data,prob.data,rows*64);
        C32_LAUNCH(k_h70_context,n,prob.data,norm.data,context.data,rows/64);
        tiled::gemm<H70,tiled::DOUBLE_RES,true>(c512_resident::stream,context.data,32,dw+11264,32,ffn.data,32,dw+16417,
            body.data,nullptr,32,rows,32);
        HIP_CHECK(hipGetLastError());
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
    int rows_capacity() const {return capacity;}
    // Any number of windows in capacity-sized chunks, written straight into
    // caller buffers (raw half body, optional FP8 output). Windows are
    // independent, so chunking at multiples of 64 rows keeps the arithmetic.
    bool run_into(const Weights& w,const float* input,int rows,float* raw_out,float* quant_out) {
        if (!input || !raw_out || rows<=0 || rows%64 || capacity%64) return false;
        {
            w.body(input,rows,raw_out,quant_out);
            return true;
        }
        auto dw=w.data.data;
        for (int start=0;start<rows;start+=capacity) {
            int r=rows-start<capacity?rows-start:capacity;
            size_t n=size_t(r)*32;
            const float* in=input+size_t(start)*32;
            float* out=raw_out+size_t(start)*32;
            tiled::gemm<H70,tiled::GATE,false>(c512_resident::stream,in,32,dw,32,nullptr,0,nullptr,
                nullptr,hidden.data,128,r,128);
            tiled::gemm<H70Raw,tiled::RAW,true>(c512_resident::stream,hidden.data,128,dw+4096,128,in,32,dw+16385,
                ffn.data,nullptr,32,r,32);
            tiled::gemm<H70,tiled::RAW,false>(c512_resident::stream,ffn.data,32,dw+8192,32,nullptr,0,nullptr,
                qkv.data,nullptr,96,r,96);
            HIP_CHECK(hipGetLastError());
            C32_LAUNCH(k_h70_qknorm_inv,size_t(r)*2,qkv.data,ninv.data,r);
        C32_LAUNCH(k_h70_qknorm_apply,size_t(r)*96,qkv.data,ninv.data,norm.data,r,w.scale);
            C32_LAUNCH(k_h70_scores,size_t(r)*64,norm.data,dw+12288,scores.data,r/64);
            C32_LAUNCH(k_h70_exp,size_t(r)*64,scores.data,ex.data,r*64);
            C32_LAUNCH(k_h70_den,r,ex.data,den.data,r);
            C32_LAUNCH(k_h70_prob,size_t(r)*64,ex.data,den.data,prob.data,r*64);
            C32_LAUNCH(k_h70_context,n,prob.data,norm.data,context.data,r/64);
            tiled::gemm<H70,tiled::DOUBLE_RES,true>(c512_resident::stream,context.data,32,dw+11264,32,ffn.data,32,
                dw+16417,out,nullptr,32,r,32);
            HIP_CHECK(hipGetLastError());
            if (quant_out) C32_LAUNCH(quantize_body,n,out,quant_out+size_t(start)*32,int(n));
        }
        return true;
    }
};
class Prefix {
    static int checked(int w, int h) {
        if (w <= 0 || h <= 0 || w % 8 || h % 8) throw std::invalid_argument("block66 input must be a positive multiple of 8");
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
        tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream,input.data,64,weights.data,64,nullptr,0,nullptr,
            low.data,nullptr,32,width*height,32);  // k_upsample66_project
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
        if (w<=0||h<=0||w%8||h%8||blocks.size()!=4)throw std::invalid_argument("C32 requires a positive multiple of 8 and four blocks");
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
        float *input = const_cast<float*>(source.data), *output = pong.data;
        // The first block reads the source in place (no copy into ping), the
        // rest ping-pong between pong and ping: the source is never written.
        auto advance = [&] { float* old = input; input = output; output = old == ping.data || old == pong.data ? old : ping.data; };
        for(size_t i=0;i<weights.size();++i){
            auto& w=*weights[i];int shift=shifts[i],px=(shift&1)?4:0,py=(shift&2)?4:0;
            int rows=((width+px+7)/8)*8*((height+py+7)/8)*8;size_t count=size_t(rows)*32;
            std::string dir=w.dir.substr(0,w.dir.size()-5);
            if(!check(dir+"/spatial","input",input,n,verify,comparisons))return false;
            if(!verify&&!c512_resident::exact_math){
                C32Io io;io.in_hwc=io.out_hwc=io.permute=1;io.width=width;io.height=height;io.px=px;io.py=py;
                io.pw=((width+px+7)/8)*8;
                w.body(input,rows,nullptr,output,io);
                advance();
                continue;
            }
            C32_LAUNCH(k_spatial32_peer_gather,count,input,windows.data,width,height,shift);
            if(!check(dir+"/spatial","windows",windows.data,count,verify,comparisons)||
               !body.run(w,windows.data,rows,verify,comparisons))return false;
            C32_LAUNCH(k_spatial32_peer_scatter,n,body.quantized(),output,width,height,shift);
            if(!check(dir+"/output","output",output,n,verify,comparisons))return false;
            advance();
        }
        result=input;return true;
    }
    DeviceTensor final_view()const{return result?DeviceTensor{result,n}:DeviceTensor{};}
};

class Head {
    static int checked(int w,int h){
        if(w<=0||h<=0||w%8||h%8)throw std::invalid_argument("head requires a positive multiple of 8");return w;
    }
    int width,height;
    size_t n,nrgb;
    std::string dir;
    Buffer sm,ss,coeff,merged,peer,native,rgb_native,rgb_public;
    Weights weights;
    Body body;
    std::unique_ptr<Buffer> chunked;  // raw body when rows exceed the body capacity
    bool ready=false;
    bool coeff_half=false;  // half-valued coefficients allow the fused finish pass
public:
    bool keep_native=true;  // the fused head writes the native body only when set (native_view)
    size_t comparisons=0;
    // max_rows caps the body scratch; larger frames run in chunks without stage checks.
    Head(int w,int h,const std::string& fixture,int max_rows=0):width(checked(w,h)),height(h),n(size_t(w)*h*32),nrgb(size_t(w)*h*3),dir(fixture),
        sm(dir,"sm",32),ss(dir,"ss",32),coeff(dir,"coeff",96),merged(n),peer(n),native(n),rgb_native(nrgb),rgb_public(nrgb),
        weights(dir+"/body"),body(max_rows>0&&max_rows<w*h?max_rows:w*h) {
        if (body.rows_capacity()<w*h) chunked.reset(new Buffer(n));
        coeff_half=true;
        for (float c:read(dir,"coeff",96)) coeff_half=coeff_half&&float(_Float16(c))==c;
    }
    bool run_from_device(DeviceTensor main,DeviceTensor skip,DeviceTensor color,bool verify=false){
        ready=false;
        if(!main.data||main.count!=n/4||!skip.data||skip.count!=n||!color.data||color.count!=nrgb)return false;
        if(!check(dir,"main",main.data,n/4,verify,comparisons)||!check(dir,"skip",skip.data,n,verify,comparisons)||
           !check(dir,"color",color.data,nrgb,verify,comparisons))return false;
        if(!verify&&coeff_half&&Weights::fusable()){
            // Merge, body and finish in one launch (k_c32_t MODE 2).
            C32Fuse fz;fz.width=width;fz.height=height;fz.main=main.data;fz.skip=skip.data;fz.sm=sm.data;fz.ss=ss.data;
            fz.coeff=coeff.data;fz.color=color.data;fz.rgb_native=rgb_native.data;fz.rgb_public=rgb_public.data;
            fz.native=keep_native?native.data:nullptr;fz.native_scale=.03125f;
            weights.fused<2>(width*height,fz);
            ready=true;return true;
        }
        C32_LAUNCH(k_head70_merge,n,main.data,skip.data,sm.data,ss.data,merged.data,width,height);
        if(!verify&&!c512_resident::exact_math){
            // The body reads the native merge in peer order and writes native directly.
            C32Io io;io.permute=1;
            weights.body(merged.data,width*height,native.data,nullptr,io);
        }else{
            C32_LAUNCH(native_to_peer,n,merged.data,peer.data,int(n));
            if(!check(dir,"merged",merged.data,n,verify,comparisons)||!check(dir,"peer",peer.data,n,verify,comparisons))return false;
            if(chunked){
                if(verify||!body.run_into(weights,peer.data,width*height,chunked->data,nullptr))return false;
            }else if(!body.run(weights,peer.data,width*height,verify,comparisons))return false;
            C32_LAUNCH(peer_to_native,n,chunked?chunked->data:body.raw(),native.data,int(n));
        }
        if(coeff_half){
            C32_LAUNCH(k_head70_finish_pair,size_t(width)*height,native.data,coeff.data,color.data,
                       rgb_native.data,rgb_public.data,width,height,.03125f);
        }else{
            C32_LAUNCH(k_head70_finish,nrgb,native.data,coeff.data,color.data,rgb_native.data,width,height,.03125f);
            C32_LAUNCH(k_head70_finish,nrgb,native.data,coeff.data,color.data,rgb_public.data,width,height,1.f);
        }
        if(!check(dir,"native",native.data,n,verify,comparisons)||!check(dir,"rgb_native",rgb_native.data,nrgb,verify,comparisons)||
           !check(dir,"rgb_public",rgb_public.data,nrgb,verify,comparisons))return false;
        ready=true;return true;
    }
    DeviceTensor merged_view()const{return ready?DeviceTensor{merged.data,n}:DeviceTensor{};}
    DeviceTensor peer_view()const{return ready?DeviceTensor{peer.data,n}:DeviceTensor{};}
    DeviceTensor body_view()const{return ready?DeviceTensor{chunked?chunked->data:body.raw(),n}:DeviceTensor{};}
    DeviceTensor native_view()const{return ready?DeviceTensor{native.data,n}:DeviceTensor{};}
    DeviceTensor final_view()const{return ready?DeviceTensor{rgb_native.data,nrgb}:DeviceTensor{};}
    DeviceTensor public_view()const{return ready?DeviceTensor{rgb_public.data,nrgb}:DeviceTensor{};}
};
#undef C32_LAUNCH
} // namespace c32_resident
