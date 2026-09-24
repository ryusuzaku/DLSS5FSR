#include "head70_test_common.h"
#include "split512_window.hip"

int main(int argc,char** argv) {
    if(argc!=6){fprintf(stderr,"usage: split512_window_test dir width height shift windows\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    int shift=atoi(argv[4]),count=atoi(argv[5]);
    if(width<4||height<4||width%4||height%4||shift<0||shift>3||count<=0)return 2;
    auto x=read(dir,"input",size_t(width)*height*512);
    auto expected=read(dir,"windows",size_t(count)*64*512);
    float *dx=upload(x),*dw,*dy;
    HIP_CHECK(hipMalloc(&dw,expected.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dy,x.size()*sizeof(float)));
    hipLaunchKernelGGL(k_split512_window_gather,dim3((expected.size()+255)/256),dim3(256),0,0,
                       dx,dw,width,height,shift);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("split512 window gather",dw,expected);
    hipLaunchKernelGGL(k_split512_window_scatter,dim3((x.size()+255)/256),dim3(256),0,0,
                       dw,dy,width,height,shift);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    ok=compare("split512 window scatter",dy,x)&&ok;
    HIP_CHECK(hipFree(dx));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(dy));
    return ok?0:1;
}
