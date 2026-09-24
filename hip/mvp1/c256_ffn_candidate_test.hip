#include "head70_test_common.h"
#include "c256_ffn_candidate.hip"

int main(int argc,char** argv) {
    if(argc!=3){fprintf(stderr,"usage: c256_ffn_candidate_test dir tokens\n");return 2;}
    std::string dir=argv[1];int tokens=atoi(argv[2]);
    if(tokens<=0||tokens%64)return 2;
    auto input=read(dir,"input",size_t(tokens)*256);
    auto w1=read(dir,"w1",1024*256),w2=read(dir,"w2",256*1024),w3=read(dir,"w3",256*256);
    auto skip=read(dir,"skip",256);
    auto expanded=read(dir,"expanded",size_t(tokens)*1024);
    auto hidden=read(dir,"hidden",expanded.size());
    auto middle=read(dir,"middle",input.size()),feature=read(dir,"feature",input.size());
    float *di=upload(input),*d1=upload(w1),*d2=upload(w2),*d3=upload(w3),*ds=upload(skip);
    float *de,*dh,*dm,*df;
    HIP_CHECK(hipMalloc(&de,expanded.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dh,hidden.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dm,middle.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&df,feature.size()*sizeof(float)));
    hipLaunchKernelGGL(k_c256_w1,dim3((expanded.size()+255)/256),dim3(256),0,0,di,d1,de,tokens);
    hipLaunchKernelGGL(k_c256_gate,dim3((hidden.size()+255)/256),dim3(256),0,0,de,dh,int(hidden.size()));
    hipLaunchKernelGGL(k_c256_w2,dim3((middle.size()+255)/256),dim3(256),0,0,dh,d2,dm,tokens);
    hipLaunchKernelGGL(k_c256_w3,dim3((feature.size()+255)/256),dim3(256),0,0,dm,di,d3,ds,df,tokens);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("candidate C256 W1",de,expanded);
    ok=compare("candidate C256 gate",dh,hidden)&&ok;
    ok=compare("candidate C256 W2",dm,middle)&&ok;
    ok=compare("candidate C256 W3 residual",df,feature)&&ok;
    if(ok) {
        std::vector<float> got(feature.size());
        HIP_CHECK(hipMemcpy(got.data(),df,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/feature_device.f32";
        FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(di));HIP_CHECK(hipFree(d1));HIP_CHECK(hipFree(d2));
    HIP_CHECK(hipFree(d3));HIP_CHECK(hipFree(ds));HIP_CHECK(hipFree(de));
    HIP_CHECK(hipFree(dh));HIP_CHECK(hipFree(dm));HIP_CHECK(hipFree(df));
    return ok?0:1;
}
