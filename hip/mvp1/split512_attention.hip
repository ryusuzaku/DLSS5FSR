#pragma once
// Scalar 16-head C512 attention baseline for one 8x8 window.
// Uses the arithmetic in native_split_reference.attention.
#include "split512_block.hip"

// Per-head inverse norm; every channel of a head uses the same factor.
__device__ inline float split512_norm_inverse(const float* v) {
    float parts[16];
    for (int j=0;j<16;++j) {
        float upper=split512_half(v[j+16]*v[j+16]);
        _Float16 fused=(_Float16)(double(v[j])*double(v[j])+double(upper));
        parts[j]=(float)fused;
    }
    for (int j=0;j<8;++j) parts[j]=split512_half(parts[2*j]+parts[2*j+1]);
    for (int width=4;width; width/=2)
        for (int j=0;j<width;++j) parts[j]=split512_half(parts[j]+parts[j+width]);
    return split512_half(1.0f/sqrtf(fmaxf(parts[0],.00006198883056640625f)));
}

__device__ inline float split512_norm(const float* v, int channel) {
    return split512_half(v[channel]*split512_norm_inverse(v));
}

// split512_norm_inverse without volatile temporaries (same rounding; see
// h70_normalize_inverse_fast).
__device__ inline float split512_norm_inverse_fast(const float* v) {
    float parts[16];
#pragma unroll
    for (int j=0;j<16;++j) {
        float upper=split512_half(v[j+16]*v[j+16]);
        _Float16 fused=(_Float16)(double(v[j])*double(v[j])+double(upper));
        parts[j]=(float)fused;
    }
#pragma unroll
    for (int j=0;j<8;++j) parts[j]=split512_half(parts[2*j]+parts[2*j+1]);
#pragma unroll
    for (int width=4;width; width/=2)
#pragma unroll
        for (int j=0;j<width;++j) parts[j]=split512_half(parts[j]+parts[j+width]);
    return split512_half(1.0f/sqrtf(fmaxf(parts[0],.00006198883056640625f)));
}

// Two-phase C64-C512 qknorm: per-head inverses, then coalesced elements.
__global__ void k_split512_qknorm_inv(const float* qkv,float* inv,int tokens,int channels) {
    int heads=channels/32;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*2*heads)return;
    int token=i/(2*heads),matrix=i/heads%2,head=i%heads;
    inv[i]=split512_norm_inverse_fast(qkv+size_t(token)*3*channels+matrix*channels+head*32);
}

__global__ void k_split512_qknorm_apply(const float* qkv,const float* scales,const float* inv,
                                        float* normalized,int tokens,int channels) {
    int heads=channels/32;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*3*channels)return;
    int token=i/(3*channels),matrix=i/channels%3,channel=i%channels,head=channel/32;
    if(matrix==2){normalized[i]=split512_fp8(qkv[i]);return;}
    float value=split512_half(qkv[i]*inv[(token*2+matrix)*heads+head]);
    if(matrix==0)value=split512_half(value*split512_half(scales[head]));
    normalized[i]=split512_fp8(value);
}

// Row form of the C64-C512 qknorm kernels: one thread per token, matrix and
// 32-channel head computes the inverse once and writes the head's 32 values.
__global__ void k_split512_qknorm_rows(const float* qkv,const float* scales,
                                       float* normalized,int tokens,int channels) {
    int heads=channels/32;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*3*heads)return;
    int token=i/(3*heads),matrix=i/heads%3,head=i%heads;
    size_t base=size_t(token)*3*channels+matrix*channels+head*32;
    const float* v=qkv+base;
    float* out=normalized+base;
    if(matrix==2){for(int c=0;c<32;++c)out[c]=split512_fp8(v[c]);return;}
    float inv=split512_norm_inverse_fast(v);
    float scale=split512_half(scales[head]);
    for(int c=0;c<32;++c){
        float value=split512_half(v[c]*inv);
        if(matrix==0)value=split512_half(value*scale);
        out[c]=split512_fp8(value);
    }
}

__global__ void k_split512_qknorm(const float* qkv,const float* scales,
                                   float* normalized,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*64*3*512)return;
    int channel=i%512,matrix=i/512%3;
    if(matrix==2) {normalized[i]=split512_fp8(qkv[i]);return;}
    int head=channel/32;
    float value=split512_norm(qkv+i-channel+head*32,channel%32);
    if(matrix==0)value=split512_half(value*split512_half(scales[head]));
    normalized[i]=split512_fp8(value);
}

__global__ void k_split512_scores(const float* normalized,const float* bias,
                                   float* scores,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*16*4096)return;
    int win=i/(16*4096),head=i/4096%16,q=i/64%64,k=i%64;
    const float* Q=normalized+(win*64+q)*1536+head*32;
    const float* K=normalized+(win*64+k)*1536+512+head*32;
    float sum=0;
    for(int c=0;c<32;++c)sum+=Q[c]*K[c];
    scores[i]=split512_half(sum+bias[(head*64+q)*64+k]);
}

__global__ void k_split512_exp(const float* scores,float* exp,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float a=scores[i]*.044921875f;
    float b=a+1.30078125f;
    float clipped=fminf(1.5693359375f,fmaxf(1.03125f,split512_half(b)));
    _Float16 half=split512_to_half(clipped);
    unsigned short bits;
    memcpy(&bits,&half,2);
    bits=(unsigned short)(((unsigned)bits<<5)+0x8000u);
    memcpy(&half,&bits,2);
    exp[i]=(float)half;
}

__device__ inline float split512_den(const float* e) {
    float parity[2];const int lanes[4]={0,2,8,10};const int offsets[3]={4,32,36};
    for(int odd=0;odd<2;++odd) {
        float partial[4];
        for(int l=0;l<4;++l) {
            int b=odd+lanes[l];float value=split512_half(e[b]+e[b+16]);
            for(int t=0;t<3;++t) {
                int off=offsets[t];
                value=split512_half(value+split512_half(e[b+off]+e[b+off+16]));
            }
            partial[l]=value;
        }
        float total=partial[0];
        for(int l=1;l<4;++l)total=split512_half(total+partial[l]);
        parity[odd]=total;
    }
    return split512_half(parity[0]+parity[1]);
}

// Row form of k_split512_prob: the half inverse denominator once per 64 keys.
__global__ void k_split512_inv_rows(const float* exp,float* inv,int rows) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<rows)inv[r]=split512_half(1.0f/split512_den(exp+size_t(r)*64));
}

__global__ void k_split512_prob_rows(const float* exp,const float* inv,float* prob,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)prob[i]=split512_fp8(split512_half(exp[i]*inv[i/64]));
}

__global__ void k_split512_prob(const float* exp,float* prob,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float den=split512_den(exp+(i/64)*64);
    float inv=split512_half(1.0f/den);
    prob[i]=split512_fp8(split512_half(exp[i]*inv));
}

__global__ void k_split512_context(const float* prob,const float* normalized,
                                    float* context,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*64*512)return;
    int win=i/(64*512),q=i/512%64,head=i/32%16,c=i%32;
    const float* p=prob+(win*16+head)*4096+q*64;
    float acc=0;
    for(int step=0;step<2;++step) {
        float part=0;
        for(int k=0;k<32;++k)
            part+=p[step*32+k]*normalized[(win*64+step*32+k)*1536+1024+head*32+c];
        acc=split512_half(acc+part);
    }
    context[i]=split512_fp8(acc);
}
