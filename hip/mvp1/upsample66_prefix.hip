// Standalone block66 C64->C32 projection and nearest upsample/skip merge.
// The C32 Swin body and original encoder4 skip are outside this fixture.
#include "split512_block.hip"

__global__ void k_upsample66_project(const float* input,const float* weights,
                                     float* low,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*32)return;
    int token=i/32,row=i%32;
    const float* x=input+token*64;
    const float* w=weights+row*64;
    float acc=0;
    for(int base=0;base<64;base+=32) {
        float part=0;
        for(int c=0;c<32;++c)part+=x[base+c]*w[base+c];
        acc=split512_half(acc+part);
    }
    low[i]=acc;
}

// skip_peer: the skip in the peer channel order (the front's block4 image)
// instead of native (its k_front_native32 copy): read through the inverse
// of peer_to_native32, the same value.
// skip_fmt 1: the skip as E4M3 bytes; out_half: merged as halves (it is
// half-valued; S336 storage formats).
__global__ void k_upsample66_merge(const float* low,const float* skip,
                                   const float* scale,float* merged,
                                   int width,int height,int skip_peer=0,int skip_fmt=0,int out_half=0) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*32)return;
    int c=i%32,x=i/32%outwidth,y=i/32/outwidth;
    float main=low[((y/2)*width+x/2)*32+c];
    const int sc=skip_peer?((c&~15)|(c&1)|((c&2)<<2)|((c&4)>>1)|((c&8)>>1)):c;
    const size_t si=(i&~31)+sc;
    const float sv=skip_fmt==1?__builtin_amdgcn_cvt_f32_fp8(int(reinterpret_cast<const unsigned char*>(skip)[si]),0):skip[si];
    float product=sv*scale[c];
    float total=main+product;
    if(out_half) reinterpret_cast<_Float16*>(merged)[i]=split512_to_half(total);
    else merged[i]=split512_half(total);
}
