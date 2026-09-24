#include "head70_test_common.h"
#include "vit_expand_chain.hip"

int main(int argc,char** argv) {
    if(argc!=3){fprintf(stderr,"usage: vit_expand_chain_test dir tokens\n");return 2;}
    std::string dir=argv[1];int tokens=atoi(argv[2]);
    if(tokens<=0||tokens>64)return 2;
    auto x=read(dir,"input",size_t(tokens)*1024);
    auto w=read(dir,"weights",4096*1024);
    auto expanded=read(dir,"expanded",size_t(tokens)*4096);
    auto hidden=read(dir,"hidden",size_t(tokens)*4096);
    auto contract_weights=read(dir,"contract_weights",1024*4096);
    auto contract_skip=read(dir,"contract_skip",1024);
    auto contract=read(dir,"contract",size_t(tokens)*1024);
    auto qkv_weights=read(dir,"qkv_weights",3*1024*1024);
    auto qkv_scales=read(dir,"qkv_scales",32);
    auto qkv_projected=read(dir,"qkv_projected",size_t(tokens)*3*1024);
    auto qkv=read(dir,"qkv",size_t(tokens)*3*1024);
    float *dx=upload(x),*dw=upload(w),*dcw=upload(contract_weights),
          *dcs=upload(contract_skip),*dqw=upload(qkv_weights),
          *dqs=upload(qkv_scales),*de,*dh,*dc,*dqp,*dq;
    HIP_CHECK(hipMalloc(&de,expanded.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dh,hidden.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dc,contract.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dqp,qkv_projected.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dq,qkv.size()*sizeof(float)));
    hipLaunchKernelGGL(k_vit_expand,dim3((expanded.size()+255)/256),dim3(256),0,0,
                       dx,dw,de,dh,tokens);
    hipLaunchKernelGGL(k_vit_residual_projection,dim3((contract.size()+255)/256),dim3(256),0,0,
                       dh,dx,dcw,dcs,dc,tokens,4096);
    hipLaunchKernelGGL(k_vit_qkv_projection,dim3((qkv.size()+255)/256),dim3(256),0,0,
                       dc,dqw,dqp,tokens);
    hipLaunchKernelGGL(k_vit_qkv_normalize,dim3((qkv.size()+255)/256),dim3(256),0,0,
                       dqp,dqs,dq,tokens);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("ViT expand",de,expanded);
    ok=compare("ViT gated hidden",dh,hidden)&&ok;
    ok=compare("ViT contract/residual",dc,contract)&&ok;
    ok=compare("ViT QKV projected",dqp,qkv_projected)&&ok;
    ok=compare("ViT QKV normalized",dq,qkv)&&ok;
    if(tokens==64) {
        auto scores=read(dir,"scores",32*64*64);
        auto exponents=read(dir,"exponents",32*64*64);
        auto attention=read(dir,"attention",size_t(tokens)*1024);
        auto projection_weights=read(dir,"projection_weights",1024*1024);
        auto projection_skip=read(dir,"projection_skip",1024);
        auto projection=read(dir,"projection",size_t(tokens)*1024);
        float *ds,*dex,*da,*dpw=upload(projection_weights),
              *dps=upload(projection_skip),*dp;
        HIP_CHECK(hipMalloc(&ds,scores.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&dex,exponents.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&da,attention.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&dp,projection.size()*sizeof(float)));
        hipLaunchKernelGGL(k_vit_scores,dim3((scores.size()+255)/256),dim3(256),0,0,
                           dq,ds,tokens);
        hipLaunchKernelGGL(k_vit_exponents,dim3((exponents.size()+255)/256),dim3(256),0,0,
                           ds,dex,int(exponents.size()));
        hipLaunchKernelGGL(k_vit_attention,dim3((attention.size()+255)/256),dim3(256),0,0,
                           dq,dex,da,tokens);
        hipLaunchKernelGGL(k_vit_residual_projection,dim3((projection.size()+255)/256),dim3(256),0,0,
                           da,dc,dpw,dps,dp,tokens,1024);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
        ok=compare("ViT attention scores",ds,scores)&&ok;
        ok=compare("ViT attention exponents",dex,exponents)&&ok;
        ok=compare("ViT attention output",da,attention)&&ok;
        ok=compare("ViT final projection",dp,projection)&&ok;
        if(ok) {
            std::vector<float> got(projection.size());
            HIP_CHECK(hipMemcpy(got.data(),dp,got.size()*sizeof(float),hipMemcpyDeviceToHost));
            std::string path=dir+"/projection_device.f32";
            FILE* file=fopen(path.c_str(),"wb");
            if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()) {
                fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
            }
            if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
        }
        HIP_CHECK(hipFree(ds));HIP_CHECK(hipFree(dex));HIP_CHECK(hipFree(da));
        HIP_CHECK(hipFree(dpw));HIP_CHECK(hipFree(dps));HIP_CHECK(hipFree(dp));
    }
    if(ok) {
        std::vector<float> got(contract.size());
        HIP_CHECK(hipMemcpy(got.data(),dc,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/contract_device.f32";
        FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()) {
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dx));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(dcw));
    HIP_CHECK(hipFree(dcs));HIP_CHECK(hipFree(de));HIP_CHECK(hipFree(dh));HIP_CHECK(hipFree(dc));
    HIP_CHECK(hipFree(dqw));HIP_CHECK(hipFree(dqs));HIP_CHECK(hipFree(dqp));HIP_CHECK(hipFree(dq));
    return ok?0:1;
}
