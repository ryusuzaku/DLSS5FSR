// Scalar C256 FFN candidate with the upstream 32-product half boundaries.
// Weight coordinates are supplied by the host; this does not validate their
// original-kernel layout or the later attention stages.
#include "split512_block.hip"

__global__ void k_c256_w1(const float* input,const float* weight,float* expanded,
                          int tokens) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*1024)return;
    int token=id/1024,row=id%1024;
    const float* x=input+token*256;
    const float* w=weight+row*256;
    float acc=0.0f;
    for(int base=0;base<256;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    expanded[id]=acc;
}

__global__ void k_c256_gate(const float* expanded,float* hidden,int count) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count)return;
    float acc=expanded[id];
    float gate=fminf(4.0f,fmaxf(-4.0f,acc));
    float inner=split512_half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
    float poly=split512_half(gate*inner+0.89453125f);
    hidden[id]=split512_fp8(split512_half(acc*poly));
}

__global__ void k_c256_w2(const float* hidden,const float* weight,float* middle,
                          int tokens) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*256)return;
    int token=id/256,row=id%256;
    const float* x=hidden+token*1024;
    const float* w=weight+row*1024;
    float acc=0.0f;
    for(int base=0;base<1024;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    middle[id]=split512_fp8(acc);
}

__global__ void k_c256_w3(const float* middle,const float* input,
                          const float* weight,const float* skip,float* feature,
                          int tokens) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*256)return;
    int token=id/256,row=id%256;
    const float* x=middle+token*256;
    const float* w=weight+row*256;
    float acc=split512_half(input[id]*skip[row]);
    for(int base=0;base<256;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    feature[id]=split512_fp8(acc);
}
