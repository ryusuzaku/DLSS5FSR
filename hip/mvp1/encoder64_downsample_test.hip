#include "head70_test_common.h"
#include "encoder64_downsample.hip"

int main(int argc,char** argv) {
    if(argc!=4){fprintf(stderr,"usage: encoder64_downsample_test dir width height\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    if(width<2||height<2||width%2||height%2)return 2;
    int pixels=width*height/4;
    auto raw=read(dir,"raw",size_t(width)*height*64);
    auto pool=read(dir,"pool",size_t(pixels)*64);
    auto matrix=read(dir,"matrix",128*64);
    auto output=read(dir,"output",size_t(pixels)*128);
    float* dr=upload(raw),*dm=upload(matrix),*dp,*do_;
    HIP_CHECK(hipMalloc(&dp,pool.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&do_,output.size()*sizeof(float)));
    hipLaunchKernelGGL(k_encoder64_pool,dim3((pool.size()+255)/256),dim3(256),0,0,
                       dr,dp,width,height);
    hipLaunchKernelGGL(k_encoder64_downsample,dim3((output.size()+255)/256),dim3(256),0,0,
                       dp,dm,do_,pixels);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("encoder8 raw pool",dp,pool);
    ok=compare("encoder8 C64 to C128",do_,output)&&ok;
    if(ok){
        for(const auto& item : {std::pair<const char*,float*>{"pool_device.f32",dp},
                                {"output_device.f32",do_}}) {
            size_t count=std::string(item.first)=="pool_device.f32"?pool.size():output.size();
            std::vector<float> got(count);
            HIP_CHECK(hipMemcpy(got.data(),item.second,count*sizeof(float),hipMemcpyDeviceToHost));
            std::string path=dir+"/"+item.first;
            FILE* file=fopen(path.c_str(),"wb");
            if(!file||fwrite(got.data(),sizeof(float),count,file)!=count)return 1;
            if(fclose(file))return 1;
        }
    }
    HIP_CHECK(hipFree(dr));HIP_CHECK(hipFree(dm));HIP_CHECK(hipFree(dp));HIP_CHECK(hipFree(do_));
    return ok?0:1;
}
