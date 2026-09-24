// S243: logical-weight C32 HEAD body, distinct from the shipped fused body.
// Arithmetic contract: Kien's MIT native_c32_reference.py, native_c32_normalize.py,
// native_c32_softmax_sum.py in ref/dlss5-port/Development. Scalar correctness port.
// Requires head70.hip and the existing chain's FP8 conversion helpers.

__device__ static float h70_h(float x) { return head70_half(x); }
__device__ static float h70_hd(double x) { return float(static_cast<_Float16>(x)); }
__device__ static float h70_f(float x) {
    return sw_e4m3_decode(sw_h2e4m3(__half_as_ushort(__float2half_rn(x))));
}

extern "C" __global__ void k_h70_expand(const float* x, const float* w, float* out, int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*128) return;
    int r=i/128,c=i%128; float sum=0;
    for(int k=0;k<32;k++) sum+=h70_f(x[r*32+k])*w[c*32+k];
    out[i]=h70_h(sum);
}

extern "C" __global__ void k_h70_hidden(const float* ex, float* out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float g=fminf(4.f,fmaxf(-4.f,ex[i]));
    volatile float a=fabsf(g)*(-.055908203125f);
    volatile float b=a+.447265625f;
    volatile float c=g*h70_h(b);
    volatile float d=c+.89453125f;
    volatile float e=ex[i]*h70_h(d);
    out[i]=h70_f(h70_h(e));
}

extern "C" __global__ void k_h70_ffn(const float* x,const float* hidden,
    const float* w,const float* gate,float* out,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*32) return;
    int r=i/32,c=i%32;
    volatile float residual=x[i]*gate[c]; float acc=h70_h(residual);
    for(int step=0;step<4;step++) {
        float sum=0;
        for(int k=0;k<32;k++) sum+=hidden[r*128+step*32+k]*w[c*128+step*32+k];
        volatile float next=acc+sum; acc=h70_h(next);
    }
    out[i]=acc;
}

extern "C" __global__ void k_h70_qkv(const float* x,const float* w,float* out,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*96) return;
    int r=i/96,c=i%96; float sum=0;
    for(int k=0;k<32;k++) sum+=h70_f(x[r*32+k])*w[c*32+k];
    out[i]=h70_h(sum);
}

__device__ static float h70_normalize(const float* v,int c) {
    float partial[16];
    for(int j=0;j<16;j++) {
        volatile float upper=v[j+16]*v[j+16];
        // Half fused multiply-add: do not lose the low addend at a half tie.
        partial[j]=h70_hd(double(v[j])*double(v[j])+double(h70_h(upper)));
    }
    for(int j=0;j<8;j++) partial[j]=h70_h(partial[2*j]+partial[2*j+1]);
    for(int width=4;width; width/=2)
        for(int j=0;j<width;j++) partial[j]=h70_h(partial[j]+partial[j+width]);
    volatile float root=sqrtf(fmaxf(partial[0],.00006198883056640625f));
    volatile float reciprocal=1.f/root;
    volatile float value=v[c]*h70_h(reciprocal);
    return h70_h(value);
}

extern "C" __global__ void k_h70_qknorm(const float* qkv,float* out,int rows,float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*96) return;
    int r=i/96,c=i%96;
    if(c>=64) {out[i]=h70_f(qkv[i]);return;}
    float value=h70_normalize(qkv+r*96+(c/32)*32,c%32);
    if(c<32) {volatile float scaled=value*h70_h(scale); value=h70_h(scaled);}
    out[i]=h70_f(value);
}

extern "C" __global__ void k_h70_scores(const float* qkv,const float* bias,float* out,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=windows*4096) return;
    int win=i/4096,q=(i%4096)/64,k=i%64; float sum=0;
    for(int c=0;c<32;c++) sum+=qkv[(win*64+q)*96+c]*qkv[(win*64+k)*96+32+c];
    volatile float score=sum+bias[q*64+k]; out[i]=h70_h(score);
}

extern "C" __global__ void k_h70_exp(const float* scores,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    volatile float a=scores[i]*.044921875f;
    volatile float b=a+1.30078125f;
    float clipped=fminf(1.5693359375f,fmaxf(1.03125f,h70_h(b)));
    unsigned short bits=__half_as_ushort(__float2half_rn(clipped));
    out[i]=__half2float(__ushort_as_half((unsigned short)((unsigned(bits)<<5)+0x8000u)));
}

extern "C" __global__ void k_h70_den(const float* ex,float* out,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows) return;
    const float* e=ex+i*64; float parity[2]; int lanes[4]={0,2,8,10};
    for(int odd=0;odd<2;odd++) {
        float partial[4];
        for(int l=0;l<4;l++) {
            int b=odd+lanes[l]; float value=h70_h(e[b]+e[b+16]);
            for(int off: {4,32,36}) value=h70_h(value+h70_h(e[b+off]+e[b+off+16]));
            partial[l]=value;
        }
        float total=partial[0];
        for(int j=1;j<4;j++) total=h70_h(total+partial[j]);
        parity[odd]=total;
    }
    out[i]=h70_h(parity[0]+parity[1]);
}

extern "C" __global__ void k_h70_prob(const float* ex,const float* den,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    volatile float inv=1.f/den[i/64];
    volatile float p=ex[i]*h70_h(inv); out[i]=h70_f(h70_h(p));
}

extern "C" __global__ void k_h70_context(const float* p,const float* qkv,float* out,int windows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=windows*2048) return;
    int win=i/2048,q=(i%2048)/32,c=i%32; float acc=0;
    for(int step=0;step<2;step++) {
        float sum=0;
        for(int k=0;k<32;k++) sum+=p[(win*64+q)*64+step*32+k]*qkv[(win*64+step*32+k)*96+64+c];
        volatile float next=acc+sum; acc=h70_h(next);
    }
    out[i]=acc;
}

extern "C" __global__ void k_h70_projection(const float* av,const float* feature,
    const float* w,const float* gate,float* out,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*32) return;
    int r=i/32,c=i%32;float sum=0;
    for(int k=0;k<32;k++) sum+=h70_f(av[r*32+k])*w[c*32+k];
    volatile float residual=feature[i]*gate[c];
    out[i]=h70_hd(double(sum)+double(h70_h(residual)));
}
