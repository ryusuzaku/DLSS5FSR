#include "head70_test_common.h"
#include "spatial32_peer.hip"

int main(int argc,char** argv) {
    if(argc!=5){fprintf(stderr,"usage: spatial32_peer_output_test dir width height shift\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]),shift=atoi(argv[4]);
    if(width<8||height<8||width%8||height%8||shift<0||shift>3)return 2;
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int pw=((width+px+7)/8)*8,ph=((height+py+7)/8)*8;
    auto windows=read(dir,"windows",size_t(pw)*ph*32);
    auto expected=read(dir,"output",size_t(width)*height*32);
    float *dw=upload(windows),*do_;
    HIP_CHECK(hipMalloc(&do_,expected.size()*sizeof(float)));
    hipLaunchKernelGGL(k_spatial32_peer_scatter,dim3((expected.size()+255)/256),dim3(256),0,0,
                       dw,do_,width,height,shift);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("C32 peer output scatter",do_,expected);
    if(ok) {
        std::vector<float> got(expected.size());
        HIP_CHECK(hipMemcpy(got.data(),do_,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/output_device.f32";FILE* file=fopen(path.c_str(),"wb");
        if(!file||fwrite(got.data(),sizeof(float),got.size(),file)!=got.size()){
            fprintf(stderr,"cannot write %s\n",path.c_str());if(file)fclose(file);return 1;
        }
        if(fclose(file)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(do_));
    return ok?0:1;
}
