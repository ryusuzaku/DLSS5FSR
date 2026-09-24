#include "head70.hip"
#include "head70_test_common.h"
#include "swin_1h_chain_c32.hip"
#include "head70_normalized.hip"

static float* buffer(size_t n) {
    return upload(std::vector<float>(n,std::numeric_limits<float>::quiet_NaN()));
}
#define H70_RUN(kernel,n,...) do { \
 hipLaunchKernelGGL(kernel,dim3(((n)+255)/256),dim3(256),0,0,__VA_ARGS__); \
 HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize()); } while(0)

int main(int argc,char** argv) {
    if(argc!=4) {fprintf(stderr,"usage: head70_normalized_test fixture width height\n");return 2;}
    std::string dir=argv[1];int w=atoi(argv[2]),h=atoi(argv[3]);
    if(w<=0||h<=0||w%8||h%8||w>4096||h>4096)return 2;
    int rows=w*h,windows=rows/64,n=rows*32;
    auto weights=read(dir,"weights",16449);float scale=weights[16384];
    auto dw=upload(weights),dm=upload(read(dir,"main",n/4)),ds=upload(read(dir,"skip",n));
    auto sm=upload(read(dir,"sm",32)),ss=upload(read(dir,"ss",32));
    auto cf=upload(read(dir,"coeff",96)),color=upload(read(dir,"color",rows*3));
    auto merged=buffer(n),expanded=buffer(rows*128),hidden=buffer(rows*128),ffn=buffer(n);
    auto qkv=buffer(rows*96),norm=buffer(rows*96),scores=buffer(rows*64),ex=buffer(rows*64);
    auto den=buffer(rows),prob=buffer(rows*64),ctx=buffer(n),body=buffer(n),rgb=buffer(rows*3);
    H70_RUN(k_head70_merge,n,dm,ds,sm,ss,merged,w,h);
    H70_RUN(k_h70_expand,rows*128,merged,dw,expanded,rows);
    H70_RUN(k_h70_hidden,rows*128,expanded,hidden,rows*128);
    H70_RUN(k_h70_ffn,n,merged,hidden,dw+4096,dw+16385,ffn,rows);
    H70_RUN(k_h70_qkv,rows*96,ffn,dw+8192,qkv,rows);
    H70_RUN(k_h70_qknorm,rows*96,qkv,norm,rows,scale);
    H70_RUN(k_h70_scores,rows*64,norm,dw+12288,scores,windows);
    H70_RUN(k_h70_exp,rows*64,scores,ex,rows*64);
    H70_RUN(k_h70_den,rows,ex,den,rows);
    H70_RUN(k_h70_prob,rows*64,ex,den,prob,rows*64);
    H70_RUN(k_h70_context,n,prob,norm,ctx,windows);
    H70_RUN(k_h70_projection,n,ctx,ffn,dw+11264,dw+16417,body,rows);
    H70_RUN(k_head70_finish,rows*3,body,cf,color,rgb,w,h,.03125f);
    struct Output {const char* name;float* ptr;size_t count;};
    Output outputs[]={{"merged",merged,size_t(n)},{"expanded",expanded,size_t(rows)*128},
      {"hidden",hidden,size_t(rows)*128},{"ffn",ffn,size_t(n)},{"qkv",qkv,size_t(rows)*96},
      {"qknorm",norm,size_t(rows)*96},{"scores",scores,size_t(rows)*64},{"exp",ex,size_t(rows)*64},
      {"den",den,size_t(rows)},{"prob",prob,size_t(rows)*64},{"context",ctx,size_t(n)},
      {"body",body,size_t(n)},{"rgb",rgb,size_t(rows)*3}};
    bool ok=true;
    for(auto& o:outputs)ok=compare(o.name,o.ptr,read(dir,o.name,o.count))&&ok;
    // Isolate reduction operators with edge fixtures, independent of model data.
    auto ni=upload(read(dir,"norm_input",64*96)),no=buffer(64*96);
    H70_RUN(k_h70_qknorm,64*96,ni,no,64,scale);
    ok=compare("norm_edge",no,read(dir,"norm_expected",64*96))&&ok;
    auto di=upload(read(dir,"den_input",64*64)),dout=buffer(64);
    H70_RUN(k_h70_den,64,di,dout,64);
    ok=compare("den_edge",dout,read(dir,"den_expected",64))&&ok;
    for(auto p:std::vector<float*>{dw,dm,ds,sm,ss,cf,color,merged,expanded,hidden,ffn,qkv,norm,
        scores,ex,den,prob,ctx,body,rgb,ni,no,di,dout})HIP_CHECK(hipFree(p));
    return ok?0:1;
}
