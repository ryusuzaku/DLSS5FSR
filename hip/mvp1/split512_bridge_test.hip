#include "head70_test_common.h"
#include "split512_bridge.hip"

int main(int argc,char** argv) {
    if(argc!=4){fprintf(stderr,"usage: split512_bridge_test dir width height\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    if(width<4||height<4||width%2||height%2)return 2;
    int pixels=width*height/4;
    auto raw=read(dir,"raw",size_t(width)*height*512);
    auto weights=read(dir,"weights",1024*512);
    auto pool=read(dir,"pool",size_t(pixels)*512);
    auto head=read(dir,"head",size_t(pixels)*1024);
    float *dr=upload(raw),*dw=upload(weights),*dp,*dh;
    HIP_CHECK(hipMalloc(&dp,pool.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dh,head.size()*sizeof(float)));
    hipLaunchKernelGGL(k_split512_pool,dim3((pool.size()+255)/256),dim3(256),0,0,
                       dr,dp,width,height);
    hipLaunchKernelGGL(k_split512_head,dim3((head.size()+255)/256),dim3(256),0,0,
                       dp,dw,dh,pixels);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("split512 pooled",dp,pool);
    ok=compare("split512 C1024 head",dh,head)&&ok;
    if(ok) {
        std::vector<float> got(head.size());
        HIP_CHECK(hipMemcpy(got.data(),dh,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/head_device.f32";
        FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()) {
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dr));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(dp));HIP_CHECK(hipFree(dh));
    return ok?0:1;
}
