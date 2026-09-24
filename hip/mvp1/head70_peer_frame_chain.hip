// Connected candidate head70: native-basis merge -> peer-basis C32 body ->
// native-basis RGB. Standalone correctness executable, not production wiring.
#include "head70.hip"
#include "head70_test_common.h"
#include "swin_1h_chain_c32.hip"
#include "head70_normalized.hip"

static float* empty_buffer(size_t count) {
    return upload(std::vector<float>(count,std::numeric_limits<float>::quiet_NaN()));
}

#define CHAIN_RUN(kernel,n,...) do { \
    hipLaunchKernelGGL(kernel,dim3(((n)+255)/256),dim3(256),0,0,__VA_ARGS__); \
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize()); \
} while(0)

__device__ static int native_channel_for_peer(int p) {
    return (p&~15)|(p&1)|((p&8)>>2)|((p&2)<<1)|((p&4)<<1);
}

__global__ void k_native_to_peer(const float* native,float* peer,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)peer[i]=native[(i&~31)+native_channel_for_peer(i&31)];
}

__global__ void k_peer_to_native(const float* peer,float* native,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)native[(i&~31)+native_channel_for_peer(i&31)]=peer[i];
}

int main(int argc,char**argv) {
    if(argc!=4){fprintf(stderr,"usage: head70_peer_frame_chain fixture width height\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    if(width<=0||height<=0||width%8||height%8||width>4096||height>4096)return 2;
    int rows=width*height,window_count=rows/64,n=rows*32,nrgb=rows*3;
    auto weights=read(dir,"weights",16449);float qscale=weights[16384];
    auto main=upload(read(dir,"main",n/4)),skip=upload(read(dir,"skip",n));
    auto sm=upload(read(dir,"sm",32)),ss=upload(read(dir,"ss",32));
    auto dw=upload(weights),coeff=upload(read(dir,"coeff",96));
    auto color=upload(read(dir,"color",nrgb));
    auto merged=empty_buffer(n),entry=empty_buffer(n);
    auto expanded=empty_buffer(size_t(rows)*128),hidden=empty_buffer(size_t(rows)*128);
    auto ffn=empty_buffer(n),qkv=empty_buffer(size_t(rows)*96);
    auto norm=empty_buffer(size_t(rows)*96),scores=empty_buffer(size_t(rows)*64);
    auto ex=empty_buffer(size_t(rows)*64),den=empty_buffer(rows);
    auto prob=empty_buffer(size_t(rows)*64),context=empty_buffer(n);
    auto body=empty_buffer(n),native_body=empty_buffer(n);
    auto rgb_native=empty_buffer(nrgb),rgb_public=empty_buffer(nrgb);

    CHAIN_RUN(k_head70_merge,n,main,skip,sm,ss,merged,width,height);
    CHAIN_RUN(k_native_to_peer,n,merged,entry,n);
    CHAIN_RUN(k_h70_expand,size_t(rows)*128,entry,dw,expanded,rows);
    CHAIN_RUN(k_h70_hidden,size_t(rows)*128,expanded,hidden,rows*128);
    CHAIN_RUN(k_h70_ffn,n,entry,hidden,dw+4096,dw+16385,ffn,rows);
    CHAIN_RUN(k_h70_qkv,size_t(rows)*96,ffn,dw+8192,qkv,rows);
    CHAIN_RUN(k_h70_qknorm,size_t(rows)*96,qkv,norm,rows,qscale);
    CHAIN_RUN(k_h70_scores,size_t(rows)*64,norm,dw+12288,scores,window_count);
    CHAIN_RUN(k_h70_exp,size_t(rows)*64,scores,ex,rows*64);
    CHAIN_RUN(k_h70_den,rows,ex,den,rows);
    CHAIN_RUN(k_h70_prob,size_t(rows)*64,ex,den,prob,rows*64);
    CHAIN_RUN(k_h70_context,n,prob,norm,context,window_count);
    CHAIN_RUN(k_h70_projection,n,context,ffn,dw+11264,dw+16417,body,rows);
    CHAIN_RUN(k_peer_to_native,n,body,native_body,n);
    CHAIN_RUN(k_head70_finish,nrgb,native_body,coeff,color,rgb_native,width,height,.03125f);
    CHAIN_RUN(k_head70_finish,nrgb,native_body,coeff,color,rgb_public,width,height,1.0f);

    bool ok=true;
    ok=compare("merge",merged,read(dir,"merged",n))&&ok;
    ok=compare("body",body,read(dir,"body",n))&&ok;
    ok=compare("native-gain-rgb",rgb_native,read(dir,"rgb_native",nrgb))&&ok;
    ok=compare("public-gain-rgb",rgb_public,read(dir,"rgb_public",nrgb))&&ok;
    for(float* p:{main,skip,sm,ss,dw,coeff,color,merged,entry,expanded,hidden,ffn,
                  qkv,norm,scores,ex,den,prob,context,body,native_body,rgb_native,rgb_public})
        HIP_CHECK(hipFree(p));
    return ok?0:1;
}
