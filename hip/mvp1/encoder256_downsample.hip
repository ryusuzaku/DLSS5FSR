// Candidate C256 block22 raw body -> 2x2 pool -> C512 projection.
#include "split512_block.hip"

__global__ void k_encoder256_pool(const float* raw,float* pool,int width,int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int pw=width/2,ph=height/2;
    if(i>=pw*ph*256)return;
    int c=i%256,x=i/256%pw,y=i/256/pw;
    int top=(2*y*width+2*x)*256+c;
    int bottom=((2*y+1)*width+2*x)*256+c;
    float a=split512_half(raw[top]+raw[top+256]);
    float b=split512_half(raw[bottom]+raw[bottom+256]);
    pool[i]=split512_fp8(split512_half(split512_half(a+b)*.25f));
}

__global__ void k_encoder256_downsample(const float* pool,const float* matrix,
                                          float* output,int pixels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pixels*512)return;
    int pixel=i/512,row=i%512;
    const float* x=pool+pixel*256;
    const float* w=matrix+row*256;
    float acc=0.0f;
    for(int base=0;base<256;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[i]=split512_fp8(acc);
}
