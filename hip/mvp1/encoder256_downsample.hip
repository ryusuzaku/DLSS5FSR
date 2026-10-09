#pragma once
// Candidate C256 block22 raw body -> 2x2 pool -> C512 projection.
#include "split512_block.hip"

// raw_fmt 2: raw as halves; pool_bytes: the pool as E4M3 bytes (S336 storage formats).
__global__ void k_encoder256_pool(const float* raw,float* pool,int width,int height,int raw_fmt=0,int pool_bytes=0) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int pw=width/2,ph=height/2;
    if(i>=pw*ph*256)return;
    int c=i%256,x=i/256%pw,y=i/256/pw;
    int top=(2*y*width+2*x)*256+c;
    int bottom=((2*y+1)*width+2*x)*256+c;
    auto at=[&](int k){return raw_fmt==2?float(reinterpret_cast<const _Float16*>(raw)[k]):raw[k];};
    float a=split512_half(at(top)+at(top+256));
    float b=split512_half(at(bottom)+at(bottom+256));
    const float v=split512_fp8(split512_half(split512_half(a+b)*.25f));
    if(pool_bytes) reinterpret_cast<unsigned char*>(pool)[i]=(unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(v,0.0f,0,false)&0xFF);
    else pool[i]=v;
}

__global__ void k_encoder256_downsample(const float* pool,const float* matrix,
                                          float* output,int pixels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pixels*512)return;
    int pixel=i/512,row=i%512;
    const float* x=pool+pixel*256;
    const float* w=matrix+row*256;
    float acc=0.0f;
    for(int base=0;base<256;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[i]=split512_fp8(acc);
}
