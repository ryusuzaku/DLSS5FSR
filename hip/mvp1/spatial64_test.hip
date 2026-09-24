#include "head70_test_common.h"
#include "spatial64_window.hip"

int main(int argc,char** argv) {
    if(argc!=5){fprintf(stderr,"usage: spatial64_window_test dir width height shift\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]),shift=atoi(argv[4]);
    if(width<8||height<8||width%8||height%8||shift<0||shift>3)return 2;
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int padded_width=((width+px+7)/8)*8;
    int padded_height=((height+py+7)/8)*8;
    auto input=read(dir,"input",size_t(width)*height*64);
    auto expected=read(dir,"windows",size_t(padded_width)*padded_height*64);
    float *di=upload(input),*dw,*do_;
    HIP_CHECK(hipMalloc(&dw,expected.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&do_,input.size()*sizeof(float)));
    hipLaunchKernelGGL(k_spatial64_gather,dim3((expected.size()+255)/64),dim3(64),0,0,
                       di,dw,width,height,shift);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("C64 window gather",dw,expected);
    hipLaunchKernelGGL(k_spatial64_scatter,dim3((input.size()+255)/64),dim3(64),0,0,
                       dw,do_,width,height,shift);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    ok=compare("C64 window scatter",do_,input)&&ok;
    if(ok) {
        std::vector<float> got(expected.size());
        HIP_CHECK(hipMemcpy(got.data(),dw,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/windows_device.f32";
        FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(di));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(do_));
    return ok?0:1;
}
