#include "head70_test_common.h"
#include "upsample56_prefix.hip"

int main(int argc,char** argv) {
    if(argc!=4){fprintf(stderr,"usage: upsample56_prefix_test dir width height\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    if(width<8||height<8||width>256||height>256||width%8||height%8)return 2;
    auto input=read(dir,"input",size_t(width)*height*256);
    auto weights=read(dir,"weights",128*256);
    auto scale=read(dir,"scale",128);
    auto skip=read(dir,"skip",size_t(width*2)*height*2*128);
    auto low=read(dir,"low",size_t(width)*height*128);
    auto merged=read(dir,"merged",skip.size());
    float *dx=upload(input),*dw=upload(weights),*ds=upload(skip),
          *dscale=upload(scale),*dl,*dm;
    HIP_CHECK(hipMalloc(&dl,low.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dm,merged.size()*sizeof(float)));
    hipLaunchKernelGGL(k_upsample56_project,dim3((low.size()+255)/256),dim3(256),0,0,
                       dx,dw,dl,width*height);
    hipLaunchKernelGGL(k_upsample56_merge,dim3((merged.size()+255)/256),dim3(256),0,0,
                       dl,ds,dscale,dm,width,height);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("block56 C256->C128 projection",dl,low);
    ok=compare("block56 upsample/skip merge",dm,merged)&&ok;
    if(ok) {
        std::vector<float> got(merged.size());
        HIP_CHECK(hipMemcpy(got.data(),dm,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/merged_device.f32";FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dx));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(ds));
    HIP_CHECK(hipFree(dscale));HIP_CHECK(hipFree(dl));HIP_CHECK(hipFree(dm));
    return ok?0:1;
}
