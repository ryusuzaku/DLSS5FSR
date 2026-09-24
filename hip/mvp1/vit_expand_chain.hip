// First ViT 1D stage: C1024 -> C4096 expansion and gated FP8 activation.
#include "split512_block.hip"

__global__ void k_vit_expand(const float* input,const float* weight,
                             float* expanded,float* hidden,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*4096)return;
    int token=i/4096,row=i%4096;
    const float* x=input+token*1024;
    const float* w=weight+row*1024;
    float acc=0;
    for(int base=0;base<1024;base+=32) {
        float part=0;
        for(int col=0;col<32;++col)part+=x[base+col]*w[base+col];
        acc=split512_half(acc+part);
    }
    expanded[i]=acc;
    float gate=fminf(4.0f,fmaxf(-4.0f,acc));
    float poly=split512_half(gate*split512_half(fabsf(gate)*
               (-.055908203125f)+.447265625f)+.89453125f);
    hidden[i]=split512_fp8(split512_half(acc*poly));
}

// Four independent projections, each rounded every 32 products. The residual
// enters only the first part; the four results then add in half.
__global__ void k_vit_residual_projection(const float* input,const float* residual,
                                          const float* weight,const float* skip,
                                          float* output,int tokens,int input_channels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*1024)return;
    int token=i/1024,row=i%1024;
    const float* x=input+token*input_channels;
    const float* w=weight+row*input_channels;
    float result=0;
    int size=input_channels/4;
    for(int part=0;part<4;++part) {
        float acc=part==0?split512_half(residual[i]*skip[row]):0;
        for(int base=part*size;base<(part+1)*size;base+=32) {
            float sum=0;
            for(int col=0;col<32;++col)sum+=x[base+col]*w[base+col];
            acc=split512_half(acc+sum);
        }
        result=part==0?acc:split512_half(result+acc);
    }
    output[i]=split512_fp8(result);
}

// ViT QKV uses two separately rounded 512-input halves per matrix row.
__global__ void k_vit_qkv_projection(const float* input,const float* weights,
                                     float* projected,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*3*1024)return;
    int token=i/(3*1024),matrix=i/1024%3,row=i%1024;
    const float* x=input+token*1024;
    const float* w=weights+(matrix*1024+row)*1024;
    float halves[2];
    for(int part=0;part<2;++part) {
        float acc=0;
        for(int base=part*512;base<(part+1)*512;base+=32) {
            float sum=0;
            for(int col=0;col<32;++col)sum+=x[base+col]*w[base+col];
            acc=split512_half(acc+sum);
        }
        halves[part]=acc;
    }
    projected[i]=split512_half(halves[0]+halves[1]);
}

__device__ inline int vit_qkv_inverse_order(int j) {
    return ((j>>1)&1)|((j&1)<<1)|(((j>>4)&1)<<2)|
           (((j>>2)&1)<<3)|(((j>>3)&1)<<4);
}

__global__ void k_vit_qkv_normalize(const float* projected,const float* scales,
                                    float* output,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*3*1024)return;
    int channel=i%1024,matrix=i/1024%3;
    if(matrix==2){output[i]=split512_fp8(projected[i]);return;}
    int base=i-channel+(channel/32)*32;
    float ordered[32];
    for(int j=0;j<32;++j)
        ordered[j]=projected[base+vit_qkv_inverse_order(j)];
    float parts[16];
    for(int j=0;j<16;++j) {
        float upper=split512_half(ordered[j+16]*ordered[j+16]);
        volatile _Float16 fused=(_Float16)(double(ordered[j])*double(ordered[j])+double(upper));
        parts[j]=(float)fused;
    }
    for(int j=0;j<8;++j)parts[j]=split512_half(parts[2*j]+parts[2*j+1]);
    for(int width=4;width;width/=2)
        for(int j=0;j<width;++j)parts[j]=split512_half(parts[j]+parts[j+width]);
    float inv=split512_half(1.0f/sqrtf(fmaxf(parts[0],.00006198883056640625f)));
    float norm=split512_half(projected[i]*inv);
    if(matrix==0) {
        norm=split512_half(norm*5.65625f);
        norm=split512_half(norm*split512_half(scales[channel/32]));
    }
    output[i]=split512_fp8(norm);
}

// Verified upstream attention arithmetic is available for 64 tokens.
__global__ void k_vit_scores(const float* qkv,float* scores,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=32*tokens*tokens)return;
    int head=i/(tokens*tokens),query=i/tokens%tokens,key=i%tokens;
    const float* q=qkv+query*3072+head*32;
    const float* k=qkv+key*3072+1024+head*32;
    float sum=0;
    for(int c=0;c<32;++c)sum+=q[c]*k[c];
    scores[i]=split512_half(sum);
}

__global__ void k_vit_exponents(const float* scores,float* exponents,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    volatile float scaled=scores[i]*.08953857421875f;
    volatile float shifted=scaled+1.708984375f;
    float affine=fminf(1.9775390625f,fmaxf(1.439453125f,split512_half(shifted)));
    _Float16 half=(_Float16)affine;
    unsigned short bits;
    memcpy(&bits,&half,2);
    bits=(unsigned short)(((unsigned)bits<<4)+0x4000u);
    memcpy(&half,&bits,2);
    exponents[i]=(float)half;
}

__device__ inline int vit_key_inverse_order(int j) {
    return ((j>>4)&1)|((j&1)<<1)|(((j>>1)&1)<<2)|
           (((j>>3)&1)<<3)|(((j>>2)&1)<<4)|(j&32);
}

__device__ inline float vit_denominator(const float* e) {
    float parity[2];const int lanes[4]={0,2,8,10};const int offsets[3]={4,32,36};
    for(int odd=0;odd<2;++odd) {
        float partial[4];
        for(int l=0;l<4;++l) {
            int b=odd+lanes[l];
            float v=split512_half(e[vit_key_inverse_order(b)]+
                                  e[vit_key_inverse_order(b+16)]);
            for(int t=0;t<3;++t) {
                int off=offsets[t];
                float pair=split512_half(e[vit_key_inverse_order(b+off)]+
                                          e[vit_key_inverse_order(b+off+16)]);
                v=split512_half(v+pair);
            }
            partial[l]=v;
        }
        float sum=partial[0];
        for(int l=1;l<4;++l)sum=split512_half(sum+partial[l]);
        parity[odd]=sum;
    }
    return split512_half(parity[0]+parity[1]);
}

__global__ void k_vit_attention(const float* qkv,const float* exponents,
                                float* output,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*1024)return;
    int query=i/1024,head=i%1024/32,channel=i%32;
    const float* e=exponents+(head*tokens+query)*tokens;
    float den=vit_denominator(e);
    double first=0,second=0;
    for(int k=0;k<32;++k)
        first+=double(split512_fp8(e[k]))*double(qkv[k*3072+2048+head*32+channel]);
    for(int k=32;k<64;++k)
        second+=double(split512_fp8(e[k]))*double(qkv[k*3072+2048+head*32+channel]);
    float numerator=split512_half((float)first);
    numerator=split512_half((float)(double(numerator)+second));
    output[i]=split512_fp8(split512_half(numerator*split512_half(1.0f/den)));
}

// Candidate for the image's 16 valid tokens. The original 16-token reduction
// order has not been checked against an SM120 execution.
__global__ void k_vit_attention16_candidate(const float* qkv,const float* exponents,
                                             float* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=16*1024)return;
    int query=i/1024,head=i%1024/32,channel=i%32;
    const float* e=exponents+(head*16+query)*16;
    double den=0.0,numerator=0.0;
    for(int key=0;key<16;++key) {
        den+=double(e[key]);
        numerator+=double(split512_fp8(e[key]))*
                   double(qkv[key*3072+2048+head*32+channel]);
    }
    float half_den=split512_half(float(den));
    float half_num=split512_half(float(numerator));
    output[i]=split512_fp8(split512_half(half_num*split512_half(1.0f/half_den)));
}
