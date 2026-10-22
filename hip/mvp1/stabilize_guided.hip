#pragma once
#include <hip/hip_runtime.h>
#include <cmath>

namespace stabilize_guided {
__device__ inline float decode(float x) { return x <= 0.04045f ? x/12.92f : powf((x+0.055f)/1.055f,2.4f); }
__global__ void guide(const float* input, float4* out, int w, int h, int stride) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    const float* c=input+(size_t(i/w)*stride+i%w)*3;
    out[i]=make_float4(log2f(fmaxf(decode(c[0]),0.0f)+0.005f),
                      log2f(fmaxf(decode(c[1]),0.0f)+0.005f),
                      log2f(fmaxf(decode(c[2]),0.0f)+0.005f),0);
}
__device__ inline float3 rgb(float4 v) { return make_float3(v.x,v.y,v.z); }
__device__ inline float3 add(float3 a,float3 b,float k=1) { return make_float3(a.x+k*b.x,a.y+k*b.y,a.z+k*b.z); }
__device__ inline float3 minimum(float3 a,float3 b) { return make_float3(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z)); }
__device__ inline float3 maximum(float3 a,float3 b) { return make_float3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z)); }
__device__ inline bool finite(float3 a) { return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z); }
__device__ inline float3 clip(float3 a,float3 lo,float3 hi,float3 center,float limit) {
    // Small margin lets flat low-amplitude stochastic changes accumulate.
    const float margin=fminf(limit,1.0f/1024);
    lo=add(lo,make_float3(margin,margin,margin),-1);
    hi=add(hi,make_float3(margin,margin,margin));
    const float3 spatial=minimum(maximum(a,lo),hi);
    // The low-band adjustment may move the full-band center outside this
    // neighborhood. Apply the change cap last; intersecting disjoint ranges
    // would otherwise exceed it. The cap wins when both cannot be satisfied.
    return minimum(maximum(spatial,add(center,make_float3(limit,limit,limit),-1)),
                   add(center,make_float3(limit,limit,limit)));
}
__global__ void apply(const float4* map,const float4* low,const float4* old_low,float4* next_low,float4* out,
                      const float2* motion,const float* depth,const float* old_depth,int w,int h,float2 ms,
                      float amount,float limit,int valid,const float4* old_full,float4* next_full,
                      float still_amount,float still_limit,const float4* current_guide,const float4* old_guide) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    const int x=i%w,y=i/w;
    const float4 c=map[i];
    const float3 base=rgb(low[i]), high=add(rgb(c),base,-1);
    float3 l=base, full=rgb(c);
    const float d=depth[i];
    const float2 mv=motion[i];
    const float vx=mv.x*ms.x,vy=mv.y*ms.y,px=x+vx,py=y+vy;
    if(valid && finite(rgb(current_guide[i])) && isfinite(px)&&isfinite(py)&&px>=0&&py>=0&&px<=w-1&&py<=h-1&&isfinite(d)&&d>=0) {
        const int x0=int(floorf(px)),y0=int(floorf(py)),x1=min(x0+1,w-1),y1=min(y0+1,h-1);
        const float fx=px-x0,fy=py-y0;
        const int ids[4]={y0*w+x0,y0*w+x1,y1*w+x0,y1*w+x1};
        const float weights[4]={(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy};
        float sum=0; float3 hl=make_float3(0,0,0),hf=hl,hg=hl;
        bool still=hypotf(vx,vy)<0.5f;
        for(int k=0;k<4;++k) if(weights[k]>0) {
            const float dp=old_depth[ids[k]],diff=fabsf(dp-d),denom=fmaxf(d,1e-6f);
            const float3 g=rgb(old_guide[ids[k]]),a=rgb(old_low[ids[k]]),b=rgb(old_full[ids[k]]);
            if(isfinite(dp)&&dp>=0&&diff<=0.06f*denom&&dp<=d*1.015f+1e-7f&&finite(g)&&finite(a)&&finite(b)) {
                hl=add(hl,a,weights[k]); hf=add(hf,b,weights[k]); hg=add(hg,g,weights[k]); sum+=weights[k];
                still=still&&diff<=0.02f*denom;
            } else still=false;
        }
        if(sum>0) {
            hl=make_float3(hl.x/sum,hl.y/sum,hl.z/sum); hf=make_float3(hf.x/sum,hf.y/sum,hf.z/sum); hg=make_float3(hg.x/sum,hg.y/sum,hg.z/sum);
            float3 glo=rgb(current_guide[i]),ghi=glo,llo=base,lhi=base,flo=rgb(c),fhi=flo;
            for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
                const int at=min(max(y+dy,0),h-1)*w+min(max(x+dx,0),w-1);
                const float q=depth[at];
                if(isfinite(q)&&fabsf(q-d)<=0.02f*fmaxf(d,1e-6f)) {
                    const float3 g=rgb(current_guide[at]),a=rgb(low[at]),b=rgb(map[at]);
                    if(finite(g)&&finite(a)&&finite(b)) {
                        glo=minimum(glo,g); ghi=maximum(ghi,g); llo=minimum(llo,a); lhi=maximum(lhi,a); flo=minimum(flo,b); fhi=maximum(fhi,b);
                    }
                }
            }
            const float3 error=maximum(maximum(add(glo,hg,-1),add(hg,ghi,-1)),make_float3(0,0,0));
            const float mismatch=fmaxf(error.x,fmaxf(error.y,error.z));
            const float t=fminf(fmaxf((mismatch-0.1f)/0.2f,0),1),confidence=(1-t*t*(3-2*t))*sum;
            const float3 clipped=clip(hl,llo,lhi,base,limit);
            l=add(base,add(clipped,base,-1),amount*confidence);
            full=add(l,high);
            if(still&&still_amount>0) full=add(full,add(clip(hf,flo,fhi,full,still_limit),full,-1),still_amount*confidence);
        }
    }
    next_low[i]=make_float4(l.x,l.y,l.z,0);
    next_full[i]=make_float4(full.x,full.y,full.z,0);
    out[i]=make_float4(full.x,full.y,full.z,c.w);
}
} // namespace stabilize_guided
