// Dormant extra passes must not retain temporal images across skipped frames.
// Usage: resident_engine_pass_switch_test engine.dll config.txt
// Config: transfer=nvidia, motion=uv, external_controls=1, uncover=0.
// Two runs store different old images, then run identical lower-pass frames
// with history=0. Their resumed upper-pass output must be byte-identical.
// A continuous upper-pass control must differ, proving history is exercised.
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
        fputs("usage: resident_engine_pass_switch_test engine.dll config.txt\n", stderr);
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "cannot load engine (%lu)\n", GetLastError()); return 2; }
    auto create = (int (*)(const char*))GetProcAddress(dll, "re_create");
    auto destroy = (void (*)())GetProcAddress(dll, "re_destroy");
    auto passes = (int (*)(int))GetProcAddress(dll, "re_set_passes");
    auto history = (int (*)(float))GetProcAddress(dll, "re_set_history");
    auto controls = (int (*)(const float*, int))GetProcAddress(dll, "re_controls");
    auto motion = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_motion");
    auto submit = (int (*)(const void*, int, int, int, int, int))GetProcAddress(dll, "re_submit_full");
    auto completed = (unsigned long long (*)())GetProcAddress(dll, "re_completed");
    auto compose = (int (*)(void*, unsigned long long, int, int, int, float*))GetProcAddress(dll, "re_compose_device");
    auto error = (const char* (*)())GetProcAddress(dll, "re_last_error");
    if (!create || !destroy || !passes || !history || !controls || !motion ||
        !submit || !completed || !compose || !error) {
        fputs("missing engine export\n", stderr); return 2;
    }
    constexpr int w = 320, h = 320, pitch = w*8;
    constexpr size_t px = size_t(w)*h;
    void *d_input = nullptr, *d_output = nullptr, *d_motion = nullptr;
    int result = 1;
    try {
        hip_check(hipMalloc(&d_input, px*8));
        hip_check(hipMalloc(&d_output, px*8));
        hip_check(hipMalloc(&d_motion, px*sizeof(float2)));
        std::vector<_Float16> frame(px*4);
        std::vector<float2> mv(px, make_float2(0.0f, 0.0f));
        hip_check(hipMemcpy(d_motion, mv.data(), px*sizeof(float2), hipMemcpyHostToDevice));
        const float settings[6] = {1.0f, 0.0f, 1.0f, 1.0f, 0.73f, 1.0f};
        auto step = [&](float blue) {
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const size_t p = size_t(y)*w+x;
                const float tile = ((x/16+y/16)&1) ? 0.12f : 0.0f;
                frame[p*4] = (_Float16)(0.15f+0.45f*x/w+tile);
                frame[p*4+1] = (_Float16)(0.2f+0.4f*y/h);
                frame[p*4+2] = (_Float16)blue;
                frame[p*4+3] = (_Float16)1.0f;
            }
            hip_check(hipMemcpy(d_input, frame.data(), px*8, hipMemcpyHostToDevice));
            hip_check(hipMemcpy(d_output, frame.data(), px*8, hipMemcpyHostToDevice));
            check(motion(d_motion,w,h,w*8)==1, error());
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
        struct Output { std::vector<_Float16> lower, resumed, next; };
        auto run = [&](int upper, int lower, float old_blue, float amount, bool fresh) {
            check(create(argv[2])==1, error());
            check(controls(settings,6)==1, "config needs external_controls=1");
            check(passes(fresh ? lower : upper)==1, "set passes failed");
            check(history(0.0f)==1, "set history failed");
            (void)step(old_blue);
            check(passes(lower)==1, "set lower passes failed");
            (void)step(0.45f);
            Output out;
            out.lower = step(0.45f);
            check(passes(upper)==1, "set upper passes failed");
            check(history(amount)==1, "set resumed history failed");
            out.resumed = step(0.45f);
            out.next = step(0.45f);
            destroy();
            return out;
        };
        auto differences = [&](const auto& a, const auto& b, float& peak) {
            size_t changed = 0;
            peak = 0;
            for (size_t p = 0; p < px; ++p) for (int c = 0; c < 3; ++c) {
                const size_t i = p*4+c;
                check(std::isfinite(float(a[i])) && std::isfinite(float(b[i])), "nonfinite output");
                if (memcmp(&a[i],&b[i],sizeof(_Float16))) ++changed;
                peak = fmaxf(peak, fabsf(float(a[i])-float(b[i])));
            }
            return changed;
        };
        bool ok = true;
        for (int upper : {2,3}) for (int lower = 1; lower < upper; ++lower) {
            for (float amount : {0.0f,0.5f,1.0f}) {
                auto a = run(upper,lower,0.15f,amount,false);
                auto b = run(upper,lower,0.75f,amount,false);
                auto fresh = run(upper,lower,0.15f,amount,true);
                float peak = 0;
                const auto gap = differences(a.lower,b.lower,peak);
                check(!gap, "lower-pass inputs/history did not converge");
                check(!differences(a.lower,fresh.lower,peak), "first-use lower-pass state did not converge");
                const auto resumed = differences(a.resumed,b.resumed,peak);
                printf("%d->%d->%d history %.2f: dormant %zu changed RGB values, max %.8g: %s\n",
                       upper,lower,upper,amount,resumed,peak,resumed ? "FAIL" : "PASS");
                const auto first = differences(a.resumed,fresh.resumed,peak);
                printf("  resumed vs first use: %zu changed RGB values, max %.8g: %s\n",
                       first,peak,first ? "FAIL" : "PASS");
                const auto next = differences(a.next,b.next,peak);
                printf("  next frame: %zu changed RGB values, max %.8g: %s\n",
                       next,peak,next ? "FAIL" : "PASS");
                const auto fresh_next = differences(a.next,fresh.next,peak);
                printf("  next vs first use: %zu changed RGB values, max %.8g: %s\n",
                       fresh_next,peak,fresh_next ? "FAIL" : "PASS");
                ok = ok && !resumed && !first && !next && !fresh_next;
            }
        }
        auto a = run(2,2,0.15f,1.0f,false);
        auto b = run(2,2,0.15f,0.0f,false);
        float peak = 0;
        const auto control = differences(a.resumed,b.resumed,peak);
        printf("continuous history control: %zu changed RGB values: %s\n",control,control ? "PASS" : "FAIL");
        result = ok && control ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr,"FAIL: %s\n",e.what());
    }
    destroy();
    if (d_input) (void)hipFree(d_input);
    if (d_output) (void)hipFree(d_output);
    if (d_motion) (void)hipFree(d_motion);
    FreeLibrary(dll);
    return result;
}
