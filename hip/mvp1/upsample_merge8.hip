#pragma once
#include "split512_block.hip"
// The fast form of k_upsample62_merge / k_upsample56_merge (C channels,
// S340): low as halves (the RAW projection writes them, wmma_gemm out16),
// skip and merge as E4M3 bytes; one thread per eight channels of an output
// pixel, the same per-element arithmetic.
template<int C>
__global__ void k_upsample_merge8(const _Float16* low,const unsigned char* skip,
                                    const float* scale,unsigned char* merged,int width,int height) {
    typedef _Float16 h8 __attribute__((ext_vector_type(8)));
    const int i=blockIdx.x*blockDim.x+threadIdx.x;  // (pixel, channel octet)
    const int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*(C/8))return;
    const int c0=(i%(C/8))*8,p=i/(C/8),x=p%outwidth,y=p/outwidth;
    const h8 main=*reinterpret_cast<const h8*>(low+((y/2)*width+x/2)*C+c0);
    const uint2 sk=*reinterpret_cast<const uint2*>(skip+size_t(p)*C+c0);
    float v[8];
    for(int e=0;e<8;++e) {
        const float sv=__builtin_amdgcn_cvt_f32_fp8(int((e<4?sk.x:sk.y)>>(8*(e%4))),0);
        const float total=float(main[e])+sv*scale[c0+e];
        v[e]=fminf(448.0f,fmaxf(-448.0f,split512_half(total)));
    }
    unsigned lo=0,hi=0;
    lo=__builtin_amdgcn_cvt_pk_fp8_f32(v[0],v[1],lo,false); lo=__builtin_amdgcn_cvt_pk_fp8_f32(v[2],v[3],lo,true);
    hi=__builtin_amdgcn_cvt_pk_fp8_f32(v[4],v[5],hi,false); hi=__builtin_amdgcn_cvt_pk_fp8_f32(v[6],v[7],hi,true);
    *reinterpret_cast<uint2*>(merged+size_t(p)*C+c0)=uint2{lo,hi};
}
