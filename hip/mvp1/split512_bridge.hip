// Encoder block30: raw final projection -> 2x2 pool -> 512-to-1024 head.
#include "split512_block.hip"

__global__ void k_split512_pool(const float* raw,float* pool,int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int pw=width/2,ph=height/2;
    if(i>=pw*ph*512)return;
    int c=i%512,x=i/512%pw,y=i/512/pw;
    int top=(2*y*width+2*x)*512+c;
    int bottom=((2*y+1)*width+2*x)*512+c;
    float a=split512_half(raw[top]+raw[top+512]);
    float b=split512_half(raw[bottom]+raw[bottom+512]);
    pool[i]=split512_fp8(split512_half(split512_half(a+b)*.25f));
}

__global__ void k_split512_head(const float* pool,const float* matrix,
                                float* output,int pixels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pixels*1024)return;
    int pixel=i/1024,row=i%1024;
    const float* x=pool+pixel*512;
    const float* w=matrix+row*512;
    float acc=0;
    for(int base=0;base<512;base+=32) {
        float part=0;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[i]=split512_fp8(acc);
}
