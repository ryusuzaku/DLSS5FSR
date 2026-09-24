// Standalone C32 peer-basis body candidate. The input and weights are decoded
// from original records but original-kernel parity is not asserted here.
#include "head70.hip"
#include "head70_test_common.h"
#include "swin_1h_chain_c32.hip"
#include "head70_normalized.hip"

static float* buffer(size_t n) {
    return upload(std::vector<float>(n,std::numeric_limits<float>::quiet_NaN()));
}
#define C32_RUN(kernel,n,...) do { \
 hipLaunchKernelGGL(kernel,dim3(((n)+255)/256),dim3(256),0,0,__VA_ARGS__); \
 HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize()); } while(0)

__global__ void k_c32_peer_output(const float* body,float* output,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=h70_f(body[i]);
}

int main(int argc,char** argv) {
    if(argc!=3){fprintf(stderr,"usage: c32_peer_body_test dir windows\n");return 2;}
    std::string dir=argv[1];int windows=atoi(argv[2]);
    if(windows<=0)return 2;
    int rows=windows*64,n=rows*32;
    auto weights=read(dir,"weights",16449);float scale=weights[16384];
    auto dw=upload(weights),dx=upload(read(dir,"input",n));
    auto expanded=buffer(size_t(rows)*128),hidden=buffer(size_t(rows)*128),ffn=buffer(n);
    auto qkv=buffer(size_t(rows)*96),norm=buffer(size_t(rows)*96),scores=buffer(size_t(rows)*64);
    auto ex=buffer(size_t(rows)*64),den=buffer(rows),prob=buffer(size_t(rows)*64);
    auto ctx=buffer(n),body=buffer(n),output=buffer(n);
    C32_RUN(k_h70_expand,size_t(rows)*128,dx,dw,expanded,rows);
    C32_RUN(k_h70_hidden,size_t(rows)*128,expanded,hidden,rows*128);
    C32_RUN(k_h70_ffn,n,dx,hidden,dw+4096,dw+16385,ffn,rows);
    C32_RUN(k_h70_qkv,size_t(rows)*96,ffn,dw+8192,qkv,rows);
    C32_RUN(k_h70_qknorm,size_t(rows)*96,qkv,norm,rows,scale);
    C32_RUN(k_h70_scores,size_t(rows)*64,norm,dw+12288,scores,windows);
    C32_RUN(k_h70_exp,size_t(rows)*64,scores,ex,rows*64);
    C32_RUN(k_h70_den,rows,ex,den,rows);
    C32_RUN(k_h70_prob,size_t(rows)*64,ex,den,prob,rows*64);
    C32_RUN(k_h70_context,n,prob,norm,ctx,windows);
    C32_RUN(k_h70_projection,n,ctx,ffn,dw+11264,dw+16417,body,rows);
    C32_RUN(k_c32_peer_output,n,body,output,n);
    struct Stage {const char* name;float* ptr;size_t count;};
    Stage stages[]={{"expanded",expanded,size_t(rows)*128},{"hidden",hidden,size_t(rows)*128},
      {"ffn",ffn,size_t(n)},{"qkv",qkv,size_t(rows)*96},{"qknorm",norm,size_t(rows)*96},
      {"scores",scores,size_t(rows)*64},{"exp",ex,size_t(rows)*64},{"den",den,size_t(rows)},
      {"prob",prob,size_t(rows)*64},{"context",ctx,size_t(n)},{"body",body,size_t(n)},
      {"output",output,size_t(n)}};
    bool ok=true;
    for(auto& s:stages)ok=compare(s.name,s.ptr,read(dir,s.name,s.count))&&ok;
    if(ok) {
        std::vector<float> got(n);
        for(auto& stage:stages) {
            if(std::string(stage.name)!="body"&&std::string(stage.name)!="output")continue;
            HIP_CHECK(hipMemcpy(got.data(),stage.ptr,n*sizeof(float),hipMemcpyDeviceToHost));
            std::string path=dir+"/"+stage.name+"_device.f32";FILE* file=fopen(path.c_str(),"wb");
            if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
                fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
            }
            if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
        }
    }
    for(auto p:std::vector<float*>{dw,dx,expanded,hidden,ffn,qkv,norm,scores,ex,den,prob,ctx,body,output})
        HIP_CHECK(hipFree(p));
    return ok?0:1;
}
