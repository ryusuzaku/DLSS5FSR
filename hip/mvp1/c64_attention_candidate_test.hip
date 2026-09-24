#include "head70_test_common.h"
#include "c64_attention_candidate.hip"
#include <utility>

int main(int argc,char** argv) {
    if(argc!=3){fprintf(stderr,"usage: c64_attention_candidate_test dir windows\n");return 2;}
    std::string dir=argv[1];int windows=atoi(argv[2]);
    if(windows<=0)return 2;
    int tokens=windows*64;
    auto feature=read(dir,"feature",size_t(tokens)*64);
    auto weights=read(dir,"qkv_weights",3*64*64);
    auto bias=read(dir,"bias",2*64*64),scales=read(dir,"scales",2);
    auto projection_weights=read(dir,"projection_weights",64*64);
    auto attention_skip=read(dir,"attention_skip",64);
    auto qkv=read(dir,"qkv",size_t(tokens)*3*64);
    auto normalized=read(dir,"normalized",qkv.size());
    auto scores=read(dir,"scores",size_t(windows)*2*4096);
    auto exponents=read(dir,"exponents",scores.size());
    auto probabilities=read(dir,"probabilities",scores.size());
    auto context=read(dir,"context",feature.size());
    auto linear=read(dir,"projection_linear",feature.size());
    auto residual=read(dir,"projection_residual",feature.size());
    float *df=upload(feature),*dw=upload(weights),*db=upload(bias),*ds=upload(scales);
    float *dpw=upload(projection_weights),*dskip=upload(attention_skip),*dq,*dn,*dscore,*dexp,*dprob,*dc,*dl,*dr;
    HIP_CHECK(hipMalloc(&dq,qkv.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dn,normalized.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dscore,scores.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dexp,exponents.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dprob,probabilities.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dc,context.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dl,linear.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dr,residual.size()*sizeof(float)));
    hipLaunchKernelGGL(k_c64_qkv,dim3((qkv.size()+255)/64),dim3(64),0,0,df,dw,dq,tokens);
    hipLaunchKernelGGL(k_c64_qknorm,dim3((normalized.size()+255)/64),dim3(64),0,0,dq,ds,dn,windows);
    hipLaunchKernelGGL(k_c64_scores,dim3((scores.size()+255)/64),dim3(64),0,0,dn,db,dscore,windows);
    hipLaunchKernelGGL(k_split512_exp,dim3((exponents.size()+255)/64),dim3(64),0,0,
                       dscore,dexp,int(exponents.size()));
    hipLaunchKernelGGL(k_split512_prob,dim3((probabilities.size()+255)/64),dim3(64),0,0,
                       dexp,dprob,int(probabilities.size()));
    hipLaunchKernelGGL(k_c64_context,dim3((context.size()+255)/64),dim3(64),0,0,
                       dprob,dn,dc,windows);
    hipLaunchKernelGGL(k_c64_projection_linear,dim3((linear.size()+255)/64),dim3(64),0,0,
                       dc,dpw,dl,tokens);
    hipLaunchKernelGGL(k_c64_projection_residual,dim3((residual.size()+255)/64),dim3(64),0,0,
                       dc,df,dpw,dskip,dr,tokens);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("candidate C64 QKV",dq,qkv);
    ok=compare("candidate C64 QK norm",dn,normalized)&&ok;
    ok=compare("candidate C64 scores",dscore,scores)&&ok;
    ok=compare("candidate C64 exponents",dexp,exponents)&&ok;
    ok=compare("candidate C64 probabilities",dprob,probabilities)&&ok;
    ok=compare("candidate C64 context",dc,context)&&ok;
    ok=compare("candidate C64 projection linear",dl,linear)&&ok;
    ok=compare("candidate C64 projection residual",dr,residual)&&ok;
    if(ok) {
        for(const auto& item : {std::pair<const char*,float*>{"context_device.f32",dc},
                                {"projection_linear_device.f32",dl},
                                {"projection_residual_device.f32",dr}}) {
            std::vector<float> got(context.size());
            HIP_CHECK(hipMemcpy(got.data(),item.second,got.size()*sizeof(float),hipMemcpyDeviceToHost));
            std::string path=dir+"/"+item.first;
            FILE* file=fopen(path.c_str(),"wb");
            if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
                fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
            }
            if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
        }
    }
    HIP_CHECK(hipFree(df));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(ds));HIP_CHECK(hipFree(dpw));HIP_CHECK(hipFree(dskip));HIP_CHECK(hipFree(dq));
    HIP_CHECK(hipFree(dn));HIP_CHECK(hipFree(dscore));HIP_CHECK(hipFree(dexp));
    HIP_CHECK(hipFree(dprob));HIP_CHECK(hipFree(dc));HIP_CHECK(hipFree(dl));HIP_CHECK(hipFree(dr));
    return ok?0:1;
}
