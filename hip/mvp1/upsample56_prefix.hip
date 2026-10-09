// Standalone block56 C256->C128 projection and nearest upsample/skip merge.
// The C128 Swin body and original encoder14 skip are outside this fixture.
#include "split512_block.hip"

__global__ void k_upsample56_project(const float* input,const float* weights,
                                     float* low,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*128)return;
    int token=i/128,row=i%128;
    const float* x=input+token*256;
    const float* w=weights+row*256;
    float acc=0;
    for(int base=0;base<256;base+=32) {
        float part=0;
        for(int c=0;c<32;++c)part+=x[base+c]*w[base+c];
        acc=split512_half(acc+part);
    }
    low[i]=acc;
}

__global__ void k_upsample56_merge(const float* low,const float* skip,
                                   const float* scale,float* merged,
                                   int width,int height,int skip_fmt=0,int out_bytes=0) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*128)return;
    int c=i%128,x=i/128%outwidth,y=i/128/outwidth;
    float main=low[((y/2)*width+x/2)*128+c];
    // skip_fmt 1: the skip as E4M3 bytes; out_bytes: the merge as E4M3 bytes.
    const float sv=skip_fmt==1?__builtin_amdgcn_cvt_f32_fp8(int(reinterpret_cast<const unsigned char*>(skip)[i]),0):skip[i];
    float product=sv*scale[c];
    float total=main+product;
    const float v=split512_fp8(split512_half(total));
    if(out_bytes) reinterpret_cast<unsigned char*>(merged)[i]=(unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(v,0.0f,0,false)&0xFF);
    else merged[i]=v;
}
