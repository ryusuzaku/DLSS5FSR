// Usage: resident_engine_noise_test engine.dll dynamic_config static_config
// Configs must match except noise_static=0/1; use transfer=nvidia,
// motion=uv, external_controls=1. Synthetic fixed input, no captured assets.
// At history0, static warm frames must match bytes; dynamic frames must vary.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
static void hip_check(hipError_t result) { check(result==hipSuccess, hipGetErrorString(result)); }

int main(int argc, char** argv) {
    if (argc!=4) { fputs("usage: resident_engine_noise_test engine.dll dynamic_config static_config\n",stderr); return 2; }
    HMODULE dll=LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr,"cannot load engine (%lu)\n",GetLastError()); return 2; }
    auto create=(int (*)(const char*))GetProcAddress(dll,"re_create");
    auto destroy=(void (*)())GetProcAddress(dll,"re_destroy");
    auto passes=(int (*)(int))GetProcAddress(dll,"re_set_passes");
    auto history=(int (*)(float))GetProcAddress(dll,"re_set_history");
    auto controls=(int (*)(const float*,int))GetProcAddress(dll,"re_controls");
    auto motion=(int (*)(const void*,int,int,int))GetProcAddress(dll,"re_motion");
    auto submit=(int (*)(const void*,int,int,int,int,int))GetProcAddress(dll,"re_submit_full");
    auto completed=(unsigned long long (*)())GetProcAddress(dll,"re_completed");
    auto compose=(int (*)(void*,unsigned long long,int,int,int,float*))GetProcAddress(dll,"re_compose_device");
    auto error=(const char* (*)())GetProcAddress(dll,"re_last_error");
    if (!create||!destroy||!passes||!history||!controls||!motion||!submit||!completed||!compose||!error) {
        fputs("missing engine export\n",stderr); FreeLibrary(dll); return 2;
    }
    constexpr int w=320,h=320,pitch=w*8;
    constexpr size_t px=size_t(w)*h;
    void *input=nullptr,*output=nullptr,*mv=nullptr;
    int result=1;
    try {
        std::vector<_Float16> frame(px*4);
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            const size_t p=size_t(y)*w+x;
            frame[p*4]=(_Float16)(0.15f+0.45f*x/w+(((x/16+y/16)&1) ? 0.12f : 0.0f));
            frame[p*4+1]=(_Float16)(0.2f+0.4f*y/h);
            frame[p*4+2]=(_Float16)0.45f; frame[p*4+3]=(_Float16)1.0f;
        }
        hip_check(hipMalloc(&input,px*8)); hip_check(hipMalloc(&output,px*8)); hip_check(hipMalloc(&mv,px*8));
        hip_check(hipMemcpy(input,frame.data(),px*8,hipMemcpyHostToDevice));
        hip_check(hipMemset(mv,0,px*8));
        const float settings[6]={1,0,1,1,0.73f,1};
        auto run=[&](const char* config,int count) {
            check(create(config)==1,error());
            check(passes(count)==1,"set passes failed"); check(history(0)==1,"set history failed");
            check(controls(settings,6)==1,"config needs external_controls=1");
            std::vector<std::vector<_Float16>> frames;
            for (int n=0;n<4;++n) {
                hip_check(hipMemcpy(output,frame.data(),px*8,hipMemcpyHostToDevice));
                check(motion(mv,w,h,w*8)==1,error());
                const auto before=completed();
                check(submit(input,w,h,pitch,8,0)==1,error());
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while (completed()==before) {
                    check(std::chrono::steady_clock::now()<deadline,"frame timeout: "+std::string(error()));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                float ms=0; check(compose(output,pitch,w,h,8,&ms)==1,error());
                frames.emplace_back(px*4);
                hip_check(hipMemcpy(frames.back().data(),output,px*8,hipMemcpyDeviceToHost));
                for (size_t p=0;p<px;++p) for (int c=0;c<3;++c)
                    check(std::isfinite(float(frames.back()[p*4+c])),"nonfinite output");
            }
            destroy(); return frames;
        };
        bool ok=true;
        for (int count : {1,2}) {
            auto dynamic=run(argv[2],count), fixed=run(argv[3],count);
            const bool cold=memcmp(dynamic[0].data(),fixed[0].data(),px*8)==0;
            bool stable=true,varies=true;
            for (int n=2;n<4;++n) {
                stable &= memcmp(fixed[1].data(),fixed[n].data(),px*8)==0;
                varies &= memcmp(dynamic[1].data(),dynamic[n].data(),px*8)!=0;
            }
            printf("passes%d cold seed0 parity: %s\n",count,cold ? "PASS" : "FAIL");
            printf("passes%d fixed warm frames byte-identical: %s\n",count,stable ? "PASS" : "FAIL");
            printf("passes%d dynamic warm frames vary: %s\n",count,varies ? "PASS" : "FAIL");
            ok &= cold&&stable&&varies;
        }
        result=ok ? 0 : 1;
    } catch (const std::exception& e) { fprintf(stderr,"FAIL: %s\n",e.what()); }
    destroy(); if(input) hipFree(input); if(output) hipFree(output); if(mv) hipFree(mv);
    FreeLibrary(dll); return result;
}
