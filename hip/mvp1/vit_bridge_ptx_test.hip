#include "head70_test_common.h"
#include <cstdint>

__global__ void k_vit_bridge_gather(const float* hwc,const int32_t* indices,
                                    float* vit,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)vit[i]=hwc[indices[i]];
}

int main(int argc,char** argv) {
    if(argc!=4){fprintf(stderr,"usage: vit_bridge_ptx_test dir map-dir count\n");return 2;}
    std::string dir=argv[1],path=std::string(argv[2])+"/hwc-to-vit.i32";
    int count=atoi(argv[3]);if(count<=0||count>65536)return 2;
    auto x=read(dir,"input",count),want=read(dir,"expected",count);
    std::vector<int32_t> map(count);
    FILE* file=fopen(path.c_str(),"rb");
    if(!file||fread(map.data(),sizeof(int32_t),count,file)!=size_t(count)||fgetc(file)!=EOF){
        fprintf(stderr,"invalid map %s\n",path.c_str());if(file)fclose(file);return 2;
    }
    fclose(file);
    std::vector<unsigned char> seen(count);
    for(int v:map) {if(v<0||v>=count||seen[v]){fprintf(stderr,"non-bijective map\n");return 2;}seen[v]=1;}
    float *dx=upload(x),*dy;int32_t*dm;
    HIP_CHECK(hipMalloc(&dm,size_t(count)*sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&dy,size_t(count)*sizeof(float)));
    HIP_CHECK(hipMemcpy(dm,map.data(),size_t(count)*sizeof(int32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_vit_bridge_gather,dim3((count+255)/256),dim3(256),0,0,
                       dx,dm,dy,count);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("PTX source-linear -> ViT",dy,want);
    if(ok) {
        std::vector<float> got(count);
        HIP_CHECK(hipMemcpy(got.data(),dy,size_t(count)*sizeof(float),hipMemcpyDeviceToHost));
        std::string output=dir+"/device.f32";
        FILE* out=fopen(output.c_str(),"wb");
        if(!out||fwrite(got.data(),sizeof(float),count,out)!=size_t(count)) {
            fprintf(stderr,"cannot write %s\n",output.c_str());if(out)fclose(out);return 1;
        }
        if(fclose(out)){fprintf(stderr,"cannot close %s\n",output.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dx));HIP_CHECK(hipFree(dm));HIP_CHECK(hipFree(dy));
    return ok?0:1;
}
