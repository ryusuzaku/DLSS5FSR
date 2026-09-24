// Candidate block14 raw body -> 2x2 pool -> measured C128-to-C256 projection.
#include "split512_block.hip"

__global__ void k_encoder128_pool(const float* raw,float* pool,int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int pw=width/2,ph=height/2;
    if(i>=pw*ph*128)return;
    int c=i%128,x=i/128%pw,y=i/128/pw;
    int top=(2*y*width+2*x)*128+c;
    int bottom=((2*y+1)*width+2*x)*128+c;
    float a=split512_half(raw[top]+raw[top+128]);
    float b=split512_half(raw[bottom]+raw[bottom+128]);
    pool[i]=split512_fp8(split512_half(split512_half(a+b)*.25f));
}

__global__ void k_encoder128_downsample(const float* pool,const float* matrix,
                                         float* output,int pixels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pixels*256)return;
    int pixel=i/256,row=i%256;
    const float* x=pool+pixel*128;
    const float* w=matrix+row*128;
    float acc=0.0f;
    for(int base=0;base<128;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[i]=split512_fp8(acc);
}
