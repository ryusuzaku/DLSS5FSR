#include "candidate_input_256.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_CHECK(call) do { hipError_t e=(call); if(e!=hipSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(e)); return 1; } } while(0)

static std::vector<unsigned char> read_bytes(const char* path) {
    FILE* f=fopen(path,"rb");
    if(!f) { fprintf(stderr,"cannot open %s\n",path); exit(2); }
    fseek(f,0,SEEK_END); long length=ftell(f); rewind(f);
    if(length<0) { fprintf(stderr,"cannot size %s\n",path); exit(2); }
    std::vector<unsigned char> bytes((size_t)length);
    if(fread(bytes.data(),1,bytes.size(),f)!=bytes.size() || fclose(f)) {
        fprintf(stderr,"cannot read %s\n",path); exit(2);
    }
    return bytes;
}

static uint32_t u32(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) |
           ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

int main(int argc,char** argv) {
    if(argc!=4) {
        fprintf(stderr,"usage: candidate_input_256_test capture.bin expected.f32 output_device.f32\n");
        return 2;
    }
    auto capture=read_bytes(argv[1]);
    auto expected_bytes=read_bytes(argv[2]);
    if(capture.size()<36 || memcmp(capture.data(),"D5INP001",8)!=0) {
        fprintf(stderr,"bad capture header\n");return 2;
    }
    const int width=(int)u32(capture.data()+8);
    const int height=(int)u32(capture.data()+12);
    const int bpp=(int)u32(capture.data()+16);
    const int pitch=(int)u32(capture.data()+20);
    if(width<1 || width>4096 || height<1 || height>4096 ||
       (bpp!=4 && bpp!=8) || pitch<width*bpp ||
       capture.size()!=36+(size_t)pitch*height ||
       expected_bytes.size()!=256u*256u*3u*sizeof(float)) {
        fprintf(stderr,"capture/expected extent differs\n");return 2;
    }
    const size_t payload=capture.size()-36;
    const size_t output_bytes=expected_bytes.size();
    unsigned char* device_input=nullptr;
    float* device_output=nullptr;
    HIP_CHECK(hipMalloc(&device_input,payload));
    HIP_CHECK(hipMalloc(&device_output,output_bytes));
    HIP_CHECK(hipMemcpy(device_input,capture.data()+36,payload,hipMemcpyHostToDevice));
    dim3 grid((256*256+255)/256),block(256);
    for(int i=0;i<10;++i)
        hipLaunchKernelGGL(k_candidate_input_256,grid,block,0,0,
                           device_input,device_output,width,height,pitch,bpp);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start=nullptr,stop=nullptr;
    HIP_CHECK(hipEventCreate(&start));HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for(int i=0;i<200;++i)
        hipLaunchKernelGGL(k_candidate_input_256,grid,block,0,0,
                           device_input,device_output,width,height,pitch,bpp);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float total_ms=0;
    HIP_CHECK(hipEventElapsedTime(&total_ms,start,stop));
    std::vector<float> actual(256u*256u*3u);
    HIP_CHECK(hipMemcpy(actual.data(),device_output,output_bytes,hipMemcpyDeviceToHost));
    std::vector<float> expected(actual.size());
    memcpy(expected.data(),expected_bytes.data(),output_bytes);
    double sum_abs=0,max_abs=0;
    size_t above_1e6=0,above_5e6=0,nonfinite=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {++nonfinite;continue;}
        const double delta=fabs((double)actual[i]-(double)expected[i]);
        sum_abs+=delta;if(delta>max_abs)max_abs=delta;
        if(delta>1e-6)++above_1e6;
        if(delta>5e-6)++above_5e6;
    }
    printf("candidate input 256: %dx%d bpp=%d pitch=%d\n",width,height,bpp,pitch);
    printf("compare: values=%zu mae=%.9g max_abs=%.9g above_1e-6=%zu above_5e-6=%zu nonfinite=%zu\n",
           actual.size(),sum_abs/actual.size(),max_abs,above_1e6,above_5e6,nonfinite);
    printf("kernel only: %.6f ms/launch (200 warm-cache launches; excludes D3D/HIP sync and copies)\n",
           total_ms/200.0f);
    FILE* f=fopen(argv[3],"wb");
    if(!f || fwrite(actual.data(),1,output_bytes,f)!=output_bytes || fclose(f)) {
        fprintf(stderr,"cannot write %s\n",argv[3]);return 1;
    }
    HIP_CHECK(hipEventDestroy(start));HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(device_input));HIP_CHECK(hipFree(device_output));
    return nonfinite==0 && above_5e6==0 ? 0:1;
}
