// C128 four-head arithmetic using the measured address-bit matrix/bias rules.
// These kernels check the upstream logical reference, not original PTX output.
#include "split512_attention.hip"

__global__ void k_c128_qkv(const float* feature,const float* weights,
                           float* output,int tokens) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*3*128)return;
    int token=id/(3*128),matrix=id/(128)%3,row=id%128;
    const float* x=feature+token*128;
    const float* w=weights+(matrix*128+row)*128;
    float acc=0.0f;
    for(int base=0;base<128;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[id]=acc;
}

__global__ void k_c128_qknorm(const float* qkv,const float* scales,
                              float* normalized,int windows) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=windows*64*3*128)return;
    int channel=id%128,matrix=id/128%3;
    if(matrix==2){normalized[id]=split512_fp8(qkv[id]);return;}
    int head=channel/32;
    float value=split512_norm(qkv+id-channel+head*32,channel%32);
    if(matrix==0)value=split512_half(value*split512_half(scales[head]));
    normalized[id]=split512_fp8(value);
}

__global__ void k_c128_scores(const float* normalized,const float* bias,
                               float* scores,int windows) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=windows*4*4096)return;
    int win=id/(4*4096),head=id/4096%4,q=id/64%64,k=id%64;
    const float* Q=normalized+(win*64+q)*384+head*32;
    const float* K=normalized+(win*64+k)*384+128+head*32;
    float sum=0.0f;
    for(int c=0;c<32;++c)sum+=Q[c]*K[c];
    scores[id]=split512_half(sum+bias[(head*64+q)*64+k]);
}

__global__ void k_c128_context(const float* prob,const float* normalized,
                                float* context,int windows) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=windows*64*128)return;
    int win=id/(64*128),q=id/128%64,head=id/32%4,c=id%32;
    const float* p=prob+(win*4+head)*4096+q*64;
    float acc=0.0f;
    for(int step=0;step<2;++step) {
        float part=0.0f;
        for(int k=0;k<32;++k)
            part+=p[step*32+k]*normalized[(win*64+step*32+k)*384+256+head*32+c];
        acc=split512_half(acc+part);
    }
    context[id]=split512_fp8(acc);
}

__global__ void k_c128_projection_linear(const float* context,const float* weight,
                                          float* linear,int tokens) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*128)return;
    int token=id/128,row=id%128;
    const float* x=context+token*128,*w=weight+row*128;
    float acc=0.0f;
    for(int base=0;base<128;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    linear[id]=acc;
}

__global__ void k_c128_projection_residual(const float* context,const float* feature,
                                            const float* weight,const float* skip,
                                            float* output,int tokens,bool raw_output=false) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=tokens*128)return;
    int token=id/128,row=id%128;
    const float* x=context+token*128,*w=weight+row*128;
    float acc=split512_half(feature[id]*skip[row]);
    for(int base=0;base<128;base+=32) {
        float part=0.0f;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    output[id]=raw_output?acc:split512_fp8(acc);
}
