// transfer=nvidia end to end (S337): runs a network_sequence frame sequence
// (tools/nvidia_sequence.py writes them) through the engine DLL the way the
// shim feeds it -- re_motion, re_depth, re_submit_full, one job per frame --
// composes each result with re_compose_device and compares it with
// network_sequence's output for the same frames (the offline pipeline that
// was validated against NVIDIA's own sequences): both round to 8 bits.
// Usage: resident_engine_nvidia_test engine.dll config.txt seq_dir ref_dir [count]
//   config.txt: the 24 engine lines (and any options); transfer=nvidia,
//   motion=1 and extent=compact are appended.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static bool slurp(const std::string& path, std::vector<unsigned char>& data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END); const long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    data.resize(size_t(n));
    const bool ok = fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fputs("usage: resident_engine_nvidia_test engine.dll config.txt seq_dir ref_dir [count]\n", stderr);
        return 2;
    }
    const int count = argc > 5 ? atoi(argv[5]) : 4;
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    auto create = (int (*)(const char*))GetProcAddress(dll, "re_create");
    auto motion = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_motion");
    auto depth = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_depth");
    auto submit = (int (*)(const void*, int, int, int, int, int))GetProcAddress(dll, "re_submit_full");
    auto completed = (unsigned long long (*)())GetProcAddress(dll, "re_completed");
    auto compose = (int (*)(void*, unsigned long long, int, int, int, float*))GetProcAddress(dll, "re_compose_device");
    auto error = (const char* (*)())GetProcAddress(dll, "re_last_error");
    auto destroy = (void (*)())GetProcAddress(dll, "re_destroy");
    if (!create || !motion || !depth || !submit || !completed || !compose || !error || !destroy) {
        fputs("engine lacks an export\n", stderr); return 1;
    }
    // The config with the faithful options appended.
    std::string cfg = std::string(argv[2]) + ".nvidia.txt";
    {
        std::ifstream in(argv[2]);
        std::ofstream out(cfg, std::ios::binary);
        std::string line;
        for (int n = 0; n < 24 && std::getline(in, line); ++n) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            out << line << "\n";
        }
        out << "transfer=nvidia\nmotion=1\nextent=compact\n";
    }
    if (create(cfg.c_str()) != 1) { fprintf(stderr, "re_create: %s\n", error()); return 1; }
    unsigned char *d_rgba = nullptr, *d_out = nullptr;
    float2* d_mv = nullptr;
    float* d_depth = nullptr;
    int fw = 0, fh = 0, failures = 0;
    std::vector<unsigned char> data, ref;
    for (int k = 0; k < count; ++k) {
        char name[64];
        snprintf(name, sizeof(name), "/%04d.seq", k);
        if (!slurp(std::string(argv[3]) + name, data) || data.size() < 8) { fprintf(stderr, "missing frame %d\n", k); return 1; }
        unsigned dims[2]; memcpy(dims, data.data(), 8);
        if (!d_rgba) {
            fw = int(dims[0]); fh = int(dims[1]);
            if (hipMalloc(&d_rgba, size_t(fw)*fh*4) || hipMalloc(&d_out, size_t(fw)*fh*4) ||
                hipMalloc(&d_mv, size_t(fw)*fh*8) || hipMalloc(&d_depth, size_t(fw)*fh*4)) {
                fputs("hipMalloc failed\n", stderr); return 1;
            }
        }
        const size_t px = size_t(fw)*fh;
        std::vector<unsigned char> rgba(px*4);
        for (size_t i = 0; i < px; ++i) {
            rgba[i*4] = data[8+i*3]; rgba[i*4+1] = data[8+i*3+1]; rgba[i*4+2] = data[8+i*3+2]; rgba[i*4+3] = 255;
        }
        (void)hipMemcpy(d_rgba, rgba.data(), rgba.size(), hipMemcpyHostToDevice);
        (void)hipMemcpy(d_mv, data.data() + 8 + px*3, px*8, hipMemcpyHostToDevice);
        (void)hipMemcpy(d_depth, data.data() + 8 + px*11, px*4, hipMemcpyHostToDevice);
        if (motion(d_mv, fw, fh, fw*8) != 1 || depth(d_depth, fw, fh, fw*4) != 1) {
            fprintf(stderr, "frame %d: motion/depth refused: %s\n", k, error()); return 1;
        }
        const unsigned long long before = completed();
        if (submit(d_rgba, fw, fh, fw*4, 4, 0) != 1) { fprintf(stderr, "frame %d: submit: %s\n", k, error()); return 1; }
        while (completed() == before) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        float ms = 0.0f;
        if (compose(d_out, size_t(fw)*4, fw, fh, 4, &ms) != 1) { fprintf(stderr, "frame %d: compose: %s\n", k, error()); return 1; }
        std::vector<unsigned char> out(px*4);
        (void)hipMemcpy(out.data(), d_out, out.size(), hipMemcpyDeviceToHost);
        snprintf(name, sizeof(name), "/%04d.f32", k);
        if (!slurp(std::string(argv[4]) + name, ref) || ref.size() != 8 + px*12) { fprintf(stderr, "missing reference %d\n", k); return 1; }
        size_t differ = 0, beyond = 0;
        for (size_t i = 0; i < px; ++i)
            for (int c = 0; c < 3; ++c) {
                float r; memcpy(&r, ref.data() + 8 + (i*3 + c)*4, 4);
                const int want = int(nearbyintf(fminf(fmaxf(r, 0.0f), 1.0f)*255.0f)), got = out[i*4+c];
                if (got != want) { ++differ; if (abs(got - want) > 1) ++beyond; }
            }
        printf("frame %d: %zu of %zu channels differ (%zu by more than 1), %.2f ms\n", k, differ, px*3, beyond, ms);
        if (beyond || differ > px*3/10000) ++failures;
    }
    destroy();
    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
