// Standalone block39 arithmetic on a declared source-linear ViT/HWC control.
#include "split512_block.hip"
#include <cstdint>

__global__ void k_decoder39_inverse(const float* vit,const int32_t* inverse,
                                    float* main,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)main[i]=vit[inverse[i]];
}

__global__ void k_decoder39_project(const float* main,const float* weights,
                                    float* projected,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*512)return;
    int token=i/512,row=i%512;
    const float* x=main+token*1024;
    const float* w=weights+row*1024;
    float result=0;
    for(int part=0;part<4;++part) {
        float acc=0;
        for(int base=part*256;base<(part+1)*256;base+=32) {
            float sum=0;
            for(int col=0;col<32;++col)sum+=x[base+col]*w[base+col];
            acc=split512_half(acc+sum);
        }
        result=part==0?acc:split512_half(result+acc);
    }
    projected[i]=result;
}

__global__ void k_decoder39_upsample_skip(const float* projected,const float* skip,
                                          const float* scale,float* output,
                                          int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int outwidth=2*width,outheight=2*height;
    if(i>=outwidth*outheight*512)return;
    int c=i%512,x=i/512%outwidth,y=i/512/outwidth;
    float main=projected[((y/2)*width+x/2)*512+c];
    volatile float product=skip[i]*scale[c];
    volatile float total=main+product;
    output[i]=split512_fp8(split512_half(total));
}
