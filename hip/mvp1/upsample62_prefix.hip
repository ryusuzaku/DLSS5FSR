// Standalone block62 C128->C64 projection and nearest upsample/skip merge.
// The C64 Swin body and original encoder8 skip are outside this fixture.
#include "split512_block.hip"

__global__ void k_upsample62_project(const float* input,const float* weights,
                                     float* low,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*64)return;
    int token=i/64,row=i%64;
    const float* x=input+token*128;
    const float* w=weights+row*128;
    float acc=0;
    for(int base=0;base<128;base+=32) {
        float part=0;
        for(int c=0;c<32;++c)part+=x[base+c]*w[base+c];
        acc=split512_half(acc+part);
    }
    low[i]=acc;
}

__global__ void k_upsample62_merge(const float* low,const float* skip,
                                   const float* scale,float* merged,
                                   int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*64)return;
    int c=i%64,x=i/64%outwidth,y=i/64/outwidth;
    float main=low[((y/2)*width+x/2)*64+c];
    volatile float product=skip[i]*scale[c];
    volatile float total=main+product;
    merged[i]=split512_fp8(split512_half(total));
}
