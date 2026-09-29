// Loads resident_engine.dll like the shim does and checks one staged frame.
// Usage: resident_engine_test engine.dll config.txt capture.bin expected_dir out_dir
// expected_dir holds color.f32 (candidate input), rgb_native.f32, rgb_public.f32
// and candidate_preview.bin (D5PREV01, native gain) from the validated sidecar.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef int (*CreateFn)(const char*);
typedef int (*SubmitFn)(const void*, int, int, int, int);
typedef int (*PollFn)(int, unsigned char*, unsigned char*, float*);
typedef int (*LatestFn)(int, float*, float*);
typedef const char* (*ErrorFn)();
typedef void (*DestroyFn)();

static std::vector<unsigned char> slurp(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    std::vector<unsigned char> data;
    unsigned char buffer[1 << 16];
    size_t got;
    while ((got = fread(buffer, 1, sizeof(buffer), f)) > 0) data.insert(data.end(), buffer, buffer+got);
    fclose(f);
    return data;
}

static bool same(const char* label, const void* a, const std::vector<unsigned char>& b, size_t bytes) {
    bool ok = b.size() == bytes && !memcmp(a, b.data(), bytes);
    printf("%s: %s\n", label, ok ? "PASS (byte-exact)" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 6) { fputs("usage: resident_engine_test engine.dll config.txt capture.bin expected_dir out_dir\n", stderr); return 2; }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "cannot load %s (%lu)\n", argv[1], GetLastError()); return 1; }
    auto create = (CreateFn)GetProcAddress(dll, "re_create");
    auto submit = (SubmitFn)GetProcAddress(dll, "re_submit");
    auto poll = (PollFn)GetProcAddress(dll, "re_poll");
    auto latest = (LatestFn)GetProcAddress(dll, "re_latest_rgb");
    auto error = (ErrorFn)GetProcAddress(dll, "re_last_error");
    auto destroy = (DestroyFn)GetProcAddress(dll, "re_destroy");
    if (!create || !submit || !poll || !latest || !error || !destroy) { fputs("missing engine export\n", stderr); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    if (!create(argv[2])) { fprintf(stderr, "re_create failed: %s\n", error()); return 1; }
    double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    // D5INP001 capture: 36-byte header (w, h, bpp, pitch, passthrough, proxy, white) then rows.
    auto capture = slurp(argv[3]);
    if (capture.size() < 36 || memcmp(capture.data(), "D5INP001", 8)) { fputs("bad capture\n", stderr); return 1; }
    unsigned int fields[4];
    memcpy(fields, capture.data()+8, sizeof(fields));
    int w = int(fields[0]), h = int(fields[1]), bpp = int(fields[2]), pitch = int(fields[3]);
    size_t bytes = size_t(pitch)*h;
    if (capture.size() != 36+bytes) { fputs("capture size mismatch\n", stderr); return 1; }
    void* staging = nullptr;
    if (hipMalloc(&staging, bytes) != hipSuccess ||
        hipMemcpy(staging, capture.data()+36, bytes, hipMemcpyHostToDevice) != hipSuccess) return 1;
    std::string expected = argv[4], out = argv[5];
    std::vector<unsigned char> rgba8(256*256*4), rgba16(256*256*8);
    std::vector<float> native(256*256*3), pub(256*256*3), color(256*256*3);
    bool ok = true;
    std::vector<double> ms;
    for (int frame = 0; frame < 3; ++frame) {
        int accepted = submit(staging, w, h, pitch, bpp);
        if (accepted != 1) { fprintf(stderr, "submit %d returned %d: %s\n", frame, accepted, error()); return 1; }
        // A second submit while busy must be refused, never queued over the snapshot.
        if (submit(staging, w, h, pitch, bpp) != 0) { fputs("busy engine accepted a frame\n", stderr); return 1; }
        float device_ms = 0;
        int state = 0;
        for (int wait = 0; wait < 6000 && state == 0; ++wait) {
            state = poll(0, rgba8.data(), rgba16.data(), &device_ms);
            if (!state) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (state != 1) { fprintf(stderr, "poll failed (%d): %s\n", state, error()); return 1; }
        ms.push_back(device_ms);
        if (latest(0, native.data(), color.data()) != 1 || latest(1, pub.data(), nullptr) != 1) return 1;
        ok = same("candidate input colour", color.data(), slurp(expected+"/color.f32"), color.size()*4) && ok;
        ok = same("native-gain RGB", native.data(), slurp(expected+"/rgb_native.f32"), native.size()*4) && ok;
        ok = same("public-gain RGB", pub.data(), slurp(expected+"/rgb_public.f32"), pub.size()*4) && ok;
        auto preview = slurp(expected+"/candidate_preview.bin");
        size_t off8 = 16, off16 = 16+rgba8.size();
        int worst = 0; size_t differing = 0;
        if (preview.size() == 16+rgba8.size()+rgba16.size()) {
            for (size_t i = 0; i < rgba8.size(); ++i) {
                int d = std::abs(int(rgba8[i]) - int(preview[off8+i]));
                if (d) ++differing;
                if (d > worst) worst = d;
            }
        } else worst = 256;
        bool rgba16_consistent = true;
        for (size_t i = 0; i < rgba8.size(); ++i) {
            _Float16 hval; memcpy(&hval, rgba16.data()+i*2, 2);
            _Float16 want = (_Float16)((i % 4) == 3 ? 1.0f : float(rgba8[i])/255.0f);
            if (memcmp(&hval, &want, 2)) rgba16_consistent = false;
        }
        printf("preview RGBA8 vs sidecar: %zu values differ, max %d level(s): %s\n", differing, worst,
               worst <= 1 ? "PASS" : "FAIL");
        printf("preview RGBA16F from RGBA8: %s\n", rgba16_consistent ? "PASS" : "FAIL");
        ok = ok && worst <= 1 && rgba16_consistent;
    }
    FILE* f = fopen((out+"/engine_preview_rgba8.bin").c_str(), "wb");
    if (f) { fwrite(rgba8.data(), 1, rgba8.size(), f); fclose(f); }
    destroy();
    hipFree(staging);
    printf("{\"setup_seconds\":%.3f,\"device_ms\":[%.3f,%.3f,%.3f]}\n", setup_s, ms[0], ms[1], ms[2]);
    return ok ? 0 : 1;
}
