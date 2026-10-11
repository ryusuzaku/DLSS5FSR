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

// k_upsample66_merge's fast form (S340): low as halves (the RAW projection
// writes them, wmma_gemm out16), the skip as E4M3 bytes, merged as halves;
// one thread per eight channels of an output pixel (vector loads and
// stores), the same per-element arithmetic.
__global__ void k_upsample66_merge8(const _Float16* low,const unsigned char* skip,
                                    const float* scale,_Float16* merged,
                                    int width,int height,int skip_peer) {
    typedef _Float16 h8 __attribute__((ext_vector_type(8)));
    const int i=blockIdx.x*blockDim.x+threadIdx.x;  // (pixel, channel octet)
    const int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*4)return;
    const int c0=(i%4)*8,p=i/4,x=p%outwidth,y=p/outwidth;
    const h8 main=*reinterpret_cast<const h8*>(low+((y/2)*width+x/2)*32+c0);
    const uint4 group=*reinterpret_cast<const uint4*>(skip+size_t(p)*32+(c0&~15));  // the channel's 16-byte group
    const unsigned char* g=reinterpret_cast<const unsigned char*>(&group);
    h8 out;
    for(int e=0;e<8;++e) {
        const int c=c0+e;
        const int sc=skip_peer?((c&~15)|(c&1)|((c&2)<<2)|((c&4)>>1)|((c&8)>>1)):c;
        const float sv=__builtin_amdgcn_cvt_f32_fp8(int(g[sc&15]),0);
        const float product=sv*scale[c];
        const float total=float(main[e])+product;
        out[e]=split512_to_half(total);
    }
    *reinterpret_cast<h8*>(merged+size_t(p)*32+c0)=out;
}
