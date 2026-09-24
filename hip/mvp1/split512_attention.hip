// Scalar 16-head C512 attention baseline for one 8x8 window.
// Uses the arithmetic in native_split_reference.attention.
#include "split512_block.hip"

__device__ inline float split512_norm(const float* v, int channel) {
    float parts[16];
    for (int j=0;j<16;++j) {
        float upper=split512_half(v[j+16]*v[j+16]);
        volatile _Float16 fused=(_Float16)(double(v[j])*double(v[j])+double(upper));
        parts[j]=(float)fused;
    }
    for (int j=0;j<8;++j) parts[j]=split512_half(parts[2*j]+parts[2*j+1]);
    for (int width=4;width; width/=2)
        for (int j=0;j<width;++j) parts[j]=split512_half(parts[j]+parts[j+width]);
    float inv=split512_half(1.0f/sqrtf(fmaxf(parts[0],.00006198883056640625f)));
    return split512_half(v[channel]*inv);
}

__global__ void k_split512_qknorm(const float* qkv,const float* scales,
                                   float* normalized,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*64*3*512)return;
    int channel=i%512,matrix=i/512%3;
    if(matrix==2) {normalized[i]=split512_fp8(qkv[i]);return;}
    int head=channel/32;
    float value=split512_norm(qkv+i-channel+head*32,channel%32);
    if(matrix==0)value=split512_half(value*split512_half(scales[head]));
    normalized[i]=split512_fp8(value);
}

__global__ void k_split512_scores(const float* normalized,const float* bias,
                                   float* scores,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*16*4096)return;
    int win=i/(16*4096),head=i/4096%16,q=i/64%64,k=i%64;
    const float* Q=normalized+(win*64+q)*1536+head*32;
    const float* K=normalized+(win*64+k)*1536+512+head*32;
    float sum=0;
    for(int c=0;c<32;++c)sum+=Q[c]*K[c];
    scores[i]=split512_half(sum+bias[(head*64+q)*64+k]);
}

__global__ void k_split512_exp(const float* scores,float* exp,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    volatile float a=scores[i]*.044921875f;
    volatile float b=a+1.30078125f;
    float clipped=fminf(1.5693359375f,fmaxf(1.03125f,split512_half(b)));
    _Float16 half=(_Float16)clipped;
    unsigned short bits;
    memcpy(&bits,&half,2);
    bits=(unsigned short)(((unsigned)bits<<5)+0x8000u);
    memcpy(&half,&bits,2);
    exp[i]=(float)half;
}

__device__ inline float split512_den(const float* e) {
    float parity[2];const int lanes[4]={0,2,8,10};const int offsets[3]={4,32,36};
    for(int odd=0;odd<2;++odd) {
        float partial[4];
        for(int l=0;l<4;++l) {
            int b=odd+lanes[l];float value=split512_half(e[b]+e[b+16]);
            for(int t=0;t<3;++t) {
                int off=offsets[t];
                value=split512_half(value+split512_half(e[b+off]+e[b+off+16]));
            }
            partial[l]=value;
        }
        float total=partial[0];
        for(int l=1;l<4;++l)total=split512_half(total+partial[l]);
        parity[odd]=total;
    }
    return split512_half(parity[0]+parity[1]);
}

__global__ void k_split512_prob(const float* exp,float* prob,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float den=split512_den(exp+(i/64)*64);
    float inv=split512_half(1.0f/den);
    prob[i]=split512_fp8(split512_half(exp[i]*inv));
}

__global__ void k_split512_context(const float* prob,const float* normalized,
                                    float* context,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=windows*64*512)return;
    int win=i/(64*512),q=i/512%64,head=i/32%16,c=i%32;
    const float* p=prob+(win*16+head)*4096+q*64;
    float acc=0;
    for(int step=0;step<2;++step) {
        float part=0;
        for(int k=0;k<32;++k)
            part+=p[step*32+k]*normalized[(win*64+step*32+k)*1536+1024+head*32+c];
        acc=split512_half(acc+part);
    }
    context[i]=split512_fp8(acc);
}
