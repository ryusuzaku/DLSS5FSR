#include "head70_test_common.h"
#include "decoder39_entry.hip"

int main(int argc,char** argv) {
    if(argc!=4){fprintf(stderr,"usage: decoder39_entry_test dir width height\n");return 2;}
    std::string dir=argv[1];int width=atoi(argv[2]),height=atoi(argv[3]);
    if(width!=16||height!=4)return 2;
    int tokens=width*height,count=tokens*1024;
    auto vit=read(dir,"vit",count),main=read(dir,"main",count);
    auto weights=read(dir,"weights",512*1024),scale=read(dir,"scale",512);
    auto skip=read(dir,"skip",size_t(width*2)*height*2*512);
    auto projected=read(dir,"projected",size_t(tokens)*512);
    auto output=read(dir,"output",skip.size());
    std::vector<int32_t> inverse(count);
    std::string path=dir+"/inverse.i32";FILE* file=fopen(path.c_str(),"rb");
    if(!file||fread(inverse.data(),sizeof(int32_t),count,file)!=size_t(count)||fgetc(file)!=EOF){
        fprintf(stderr,"invalid inverse %s\n",path.c_str());if(file)fclose(file);return 2;
    }
    fclose(file);
    std::vector<unsigned char> seen(count);
    for(int v:inverse){if(v<0||v>=count||seen[v]){fprintf(stderr,"non-bijective inverse\n");return 2;}seen[v]=1;}
    float *dv=upload(vit),*dw=upload(weights),*ds=upload(skip),*dscale=upload(scale),
          *dm,*dp,*do_;
    int32_t *di;
    HIP_CHECK(hipMalloc(&di,size_t(count)*sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(di,inverse.data(),size_t(count)*sizeof(int32_t),hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dm,main.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&dp,projected.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&do_,output.size()*sizeof(float)));
    hipLaunchKernelGGL(k_decoder39_inverse,dim3((count+255)/256),dim3(256),0,0,
                       dv,di,dm,count);
    hipLaunchKernelGGL(k_decoder39_project,dim3((projected.size()+255)/256),dim3(256),0,0,
                       dm,dw,dp,tokens);
    hipLaunchKernelGGL(k_decoder39_upsample_skip,dim3((output.size()+255)/256),dim3(256),0,0,
                       dp,ds,dscale,do_,width,height);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    bool ok=compare("decoder39 source-linear inverse",dm,main);
    ok=compare("decoder39 projection",dp,projected)&&ok;
    ok=compare("decoder39 upsample/skip",do_,output)&&ok;
    if(ok) {
        std::vector<float> got(output.size());
        HIP_CHECK(hipMemcpy(got.data(),do_,got.size()*sizeof(float),hipMemcpyDeviceToHost));
        path=dir+"/output_device.f32";FILE* out=fopen(path.c_str(),"wb");
        if(!out||fwrite(got.data(),sizeof(float),got.size(),out)!=got.size()){
            fprintf(stderr,"cannot write %s\n",path.c_str());if(out)fclose(out);return 1;
        }
        if(fclose(out)){fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    HIP_CHECK(hipFree(dv));HIP_CHECK(hipFree(dw));HIP_CHECK(hipFree(ds));
    HIP_CHECK(hipFree(dscale));HIP_CHECK(hipFree(di));HIP_CHECK(hipFree(dm));
    HIP_CHECK(hipFree(dp));HIP_CHECK(hipFree(do_));
    return ok?0:1;
}
