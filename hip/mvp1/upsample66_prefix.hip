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

__global__ void k_upsample66_merge(const float* low,const float* skip,
                                   const float* scale,float* merged,
                                   int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*32)return;
    int c=i%32,x=i/32%outwidth,y=i/32/outwidth;
    float main=low[((y/2)*width+x/2)*32+c];
    volatile float product=skip[i]*scale[c];
    volatile float total=main+product;
    merged[i]=split512_half(total);
}
