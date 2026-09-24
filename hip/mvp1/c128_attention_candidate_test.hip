#include "head70_test_common.h"
#include "c128_attention_candidate.hip"
#include <utility>

int main(int argc,char** argv) {
    if(argc!=3&&argc!=4){fprintf(stderr,"usage: c128_attention_candidate_test dir windows [raw]\n");return 2;}
    std::string dir=argv[1];int windows=atoi(argv[2]);
    bool raw_mode=argc==4&&std::string(argv[3])=="raw";
    if(argc==4&&!raw_mode)return 2;
    if(windows<=0)return 2;
    int tokens=windows*64;
    auto feature=read(dir,"feature",size_t(tokens)*128);
    auto weights=read(dir,"qkv_weights",3*128*128);
    auto bias=read(dir,"bias",4*64*64),scales=read(dir,"scales",4);
    auto projection_weights=read(dir,"projection_weights",128*128);
    auto attention_skip=read(dir,"attention_skip",128);
    auto qkv=read(dir,"qkv",size_t(tokens)*3*128);
    auto normalized=read(dir,"normalized",qkv.size());
    auto scores=read(dir,"scores",size_t(windows)*4*4096);
    auto exponents=read(dir,"exponents",scores.size());
    auto probabilities=read(dir,"probabilities",scores.size());
    auto context=read(dir,"context",feature.size());
    auto linear=read(dir,"projection_linear",feature.size());
    auto residual=read(dir,"projection_residual",feature.size());
    std::vector<float> raw;
    if(raw_mode)raw=read(dir,"projection_raw",feature.size());
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
    hipLaunchKernelGGL(k_c128_qkv,dim3((qkv.size()+255)/128),dim3(128),0,0,df,dw,dq,tokens);
    hipLaunchKernelGGL(k_c128_qknorm,dim3((normalized.size()+255)/128),dim3(128),0,0,dq,ds,dn,windows);
    hipLaunchKernelGGL(k_c128_scores,dim3((scores.size()+255)/128),dim3(128),0,0,dn,db,dscore,windows);
    hipLaunchKernelGGL(k_split512_exp,dim3((exponents.size()+255)/128),dim3(128),0,0,
                       dscore,dexp,int(exponents.size()));
    hipLaunchKernelGGL(k_split512_prob,dim3((probabilities.size()+255)/128),dim3(128),0,0,
                       dexp,dprob,int(probabilities.size()));
    hipLaunchKernelGGL(k_c128_context,dim3((context.size()+255)/128),dim3(128),0,0,
                       dprob,dn,dc,windows);
    hipLaunchKernelGGL(k_c128_projection_linear,dim3((linear.size()+255)/128),dim3(128),0,0,
                       dc,dpw,dl,tokens);
    hipLaunchKernelGGL(k_c128_projection_residual,dim3((residual.size()+255)/128),dim3(128),0,0,
                       dc,df,dpw,dskip,dr,tokens);
    float* draw=nullptr;
    if(raw_mode){
        HIP_CHECK(hipMalloc(&draw,raw.size()*sizeof(float)));
        hipLaunchKernelGGL(k_c128_projection_residual,dim3((raw.size()+255)/128),dim3(128),0,0,
                           dc,df,dpw,dskip,draw,tokens,true);
    }
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("candidate C128 QKV",dq,qkv);
    ok=compare("candidate C128 QK norm",dn,normalized)&&ok;
    ok=compare("candidate C128 scores",dscore,scores)&&ok;
    ok=compare("candidate C128 exponents",dexp,exponents)&&ok;
    ok=compare("candidate C128 probabilities",dprob,probabilities)&&ok;
    ok=compare("candidate C128 context",dc,context)&&ok;
    ok=compare("candidate C128 projection linear",dl,linear)&&ok;
    ok=compare("candidate C128 projection residual",dr,residual)&&ok;
    if(raw_mode)ok=compare("candidate C128 raw projection",draw,raw)&&ok;
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
        if(raw_mode){
            std::vector<float> got(raw.size());
            HIP_CHECK(hipMemcpy(got.data(),draw,got.size()*sizeof(float),hipMemcpyDeviceToHost));
            std::string path=dir+"/projection_raw_device.f32";
            FILE* file=fopen(path.c_str(),"wb");
            if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size())return 1;
            if(fclose(file))return 1;
        }
    }
    HIP_CHECK(hipFree(df));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(ds));HIP_CHECK(hipFree(dpw));HIP_CHECK(hipFree(dskip));HIP_CHECK(hipFree(dq));
    HIP_CHECK(hipFree(dn));HIP_CHECK(hipFree(dscore));HIP_CHECK(hipFree(dexp));
    HIP_CHECK(hipFree(dprob));HIP_CHECK(hipFree(dc));HIP_CHECK(hipFree(dl));HIP_CHECK(hipFree(dr));
    if(draw)HIP_CHECK(hipFree(draw));
    return ok?0:1;
}
