// A camera-cut reset must discard both network and output-repair history.
// Usage: resident_engine_reset_test engine.dll config.txt
// Use transfer=nvidia, motion=uv, uncover>0 and uncover_pass1=1.
// No captured images or original NVIDIA kernels are used. The synthetic
// old frame is nearer and moving; the new frame is still background.
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
static void hip_check(hipError_t result) {
    check(result == hipSuccess, hipGetErrorString(result));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fputs("usage: resident_engine_reset_test engine.dll config.txt\n", stderr);
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "cannot load engine (%lu)\n", GetLastError()); return 2; }
    auto create = (int (*)(const char*))GetProcAddress(dll, "re_create");
    auto destroy = (void (*)())GetProcAddress(dll, "re_destroy");
    auto reset = (int (*)())GetProcAddress(dll, "re_reset");
    auto passes = (int (*)(int))GetProcAddress(dll, "re_set_passes");
    auto controls = (int (*)(const float*, int))GetProcAddress(dll, "re_controls");
    auto motion = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_motion");
    auto depth = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_depth");
    auto submit = (int (*)(const void*, int, int, int, int, int))GetProcAddress(dll, "re_submit_full");
    auto completed = (unsigned long long (*)())GetProcAddress(dll, "re_completed");
    auto compose = (int (*)(void*, unsigned long long, int, int, int, float*))GetProcAddress(dll, "re_compose_device");
    auto error = (const char* (*)())GetProcAddress(dll, "re_last_error");
    if (!create || !destroy || !reset || !passes || !controls || !motion || !depth ||
        !submit || !completed || !compose || !error) {
        fputs("missing engine export\n", stderr); return 2;
    }
    constexpr int w = 320, h = 320, pitch = w*8;
    constexpr size_t px = size_t(w)*h;
    void *d_input = nullptr, *d_output = nullptr, *d_motion = nullptr, *d_depth = nullptr;
    int result = 1;
    try {
        hip_check(hipMalloc(&d_input, px*8));
        hip_check(hipMalloc(&d_output, px*8));
        hip_check(hipMalloc(&d_motion, px*sizeof(float2)));
        hip_check(hipMalloc(&d_depth, px*sizeof(float)));
        std::vector<_Float16> frame(px*4);
        std::vector<float2> mv(px);
        std::vector<float> z(px);
        const float settings[6] = {1.0f, 0.0f, 1.0f, 1.0f, 0.73f, 1.0f};
        auto step = [&](bool old) {
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const size_t p = size_t(y)*w+x;
                const float tile = ((x/16+y/16)&1) ? 0.12f : 0.0f;
                frame[p*4] = (_Float16)(0.15f+0.45f*x/w+tile);
                frame[p*4+1] = (_Float16)(0.2f+0.4f*y/h);
                frame[p*4+2] = (_Float16)(old ? 0.25f : 0.45f);
                frame[p*4+3] = (_Float16)1.0f;
                mv[p] = make_float2(old ? 8.0f/w : 0.0f, 0.0f);
                z[p] = old ? 0.6f : 0.2f;  // reversed Z
            }
            hip_check(hipMemcpy(d_input, frame.data(), px*8, hipMemcpyHostToDevice));
            hip_check(hipMemcpy(d_output, frame.data(), px*8, hipMemcpyHostToDevice));
            hip_check(hipMemcpy(d_motion, mv.data(), px*sizeof(float2), hipMemcpyHostToDevice));
            hip_check(hipMemcpy(d_depth, z.data(), px*sizeof(float), hipMemcpyHostToDevice));
            check(motion(d_motion,w,h,w*8)==1, error());
            check(depth(d_depth,w,h,w*4)==1, error());
            const auto before = completed();
            check(submit(d_input,w,h,pitch,8,0)==1, error());
            const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while (completed()==before) {
                check(std::chrono::steady_clock::now()<deadline, "frame timeout: "+std::string(error()));
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            float ms = 0;
            check(compose(d_output,pitch,w,h,8,&ms)==1, error());
            std::vector<_Float16> out(px*4);
            hip_check(hipMemcpy(out.data(), d_output, px*8, hipMemcpyDeviceToHost));
            return out;
        };
        auto run = [&](bool warm, bool cut) {
            check(create(argv[2])==1, error());
            check(passes(2)==1, "set passes failed");
            check(controls(settings,6)==1, "config needs external_controls=1");
            if (warm) (void)step(true);
            if (cut) check(reset()==1, "reset failed");
            auto out = step(false);
            destroy();
            return out;
        };
        auto fresh = run(false,false);
        auto cut = run(true,true);
        auto uncut = run(true,false);
        size_t changed = 0, control_changed = 0, nonfinite = 0;
        float max_error = 0;
        for (size_t p = 0; p < px; ++p) for (int c = 0; c < 3; ++c) {
            const size_t i = p*4+c;
            if (!std::isfinite(float(cut[i]))) ++nonfinite;
            if (memcmp(&fresh[i],&cut[i],sizeof(_Float16))) ++changed;
            if (memcmp(&fresh[i],&uncut[i],sizeof(_Float16))) ++control_changed;
            max_error = fmaxf(max_error, fabsf(float(cut[i])-float(fresh[i])));
        }
        printf("reset vs fresh: %zu changed RGB values, max error %.8g, nonfinite %zu: %s\n",
               changed,max_error,nonfinite,!changed&&!nonfinite ? "PASS" : "FAIL");
        printf("without reset: %zu changed RGB values: %s (history exercised)\n",
               control_changed,control_changed ? "PASS" : "FAIL");
        result = !changed && !nonfinite && control_changed ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr,"FAIL: %s\n",e.what());
    }
    destroy();
    if (d_input) (void)hipFree(d_input);
    if (d_output) (void)hipFree(d_output);
    if (d_motion) (void)hipFree(d_motion);
    if (d_depth) (void)hipFree(d_depth);
    FreeLibrary(dll);
    return result;
}
