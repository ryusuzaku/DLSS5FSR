// Standalone block48 C512->C256 projection and nearest upsample/skip merge.
// The later fused C256 Swin body is outside this fixture.
#include "split512_block.hip"

__global__ void k_upsample48_project(const float* input,const float* weights,
                                     float* low,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*256)return;
    int token=i/256,row=i%256;
    const float* x=input+token*512;
    const float* w=weights+row*512;
    float acc=0;
    for(int base=0;base<512;base+=32) {
        float part=0;
        for(int c=0;c<32;++c)part+=x[base+c]*w[base+c];
        acc=split512_half(acc+part);
    }
    low[i]=acc;
}

__global__ void k_upsample48_merge(const float* low,const float* skip,
                                   const float* scale,float* merged,
                                   int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*256)return;
    int c=i%256,x=i/256%outwidth,y=i/256/outwidth;
    float main=low[((y/2)*width+x/2)*256+c];
    volatile float product=skip[i]*scale[c];
    volatile float total=main+product;
    merged[i]=split512_fp8(split512_half(total));
}
