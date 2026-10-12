// Faithful mode across frame-size changes (S341): the engine walked through a
// list of sizes (N frames each, zero motion, flat depth) against a fresh
// engine run only at the last size. A size change must behave as a reset;
// it also covers a second engine instance in one process.
// Usage: resident_engine_size_test engine.dll config.txt out_dir frames WxH.rgb [WxH.rgb ...]
//   config.txt: the 24 engine lines and transfer=nvidia etc.; .rgb files are
//   two uint32 (width, height) then RGB8 rows.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
static bool slurp(const std::string& path, std::vector<unsigned char>& data) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
    _fseeki64(f, 0, SEEK_END); const long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    data.resize(size_t(n)); const bool ok = fread(data.data(), 1, data.size(), f) == data.size(); fclose(f); return ok;
}
int main(int argc, char** argv) {
    if (argc < 6) return 2;
    const int frames = atoi(argv[4]);
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (int (*)(const char*))GetProcAddress(dll, "re_create");
    auto motion = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_motion");
    auto depth = (int (*)(const void*, int, int, int))GetProcAddress(dll, "re_depth");
    auto submit = (int (*)(const void*, int, int, int, int, int))GetProcAddress(dll, "re_submit_full");
    auto completed = (unsigned long long (*)())GetProcAddress(dll, "re_completed");
    auto compose = (int (*)(void*, unsigned long long, int, int, int, float*))GetProcAddress(dll, "re_compose_device");
    auto error = (const char* (*)())GetProcAddress(dll, "re_last_error");
    auto destroy = (void (*)())GetProcAddress(dll, "re_destroy");
    std::vector<std::vector<unsigned char>> last[2];
    int lw = 0, lh = 0;
    for (int pass = 0; pass < 2; ++pass) {
        if (create(argv[2]) != 1) { fprintf(stderr, "re_create: %s\n", error()); return 1; }
        for (int a = pass ? argc - 1 : 5; a < argc; ++a) {
            std::vector<unsigned char> data;
            if (!slurp(argv[a], data)) { fprintf(stderr, "missing %s\n", argv[a]); return 1; }
            unsigned dims[2]; memcpy(dims, data.data(), 8);
            const int fw = int(dims[0]), fh = int(dims[1]); const size_t px = size_t(fw)*fh;
            std::vector<unsigned char> rgba(px*4);
            for (size_t i = 0; i < px; ++i) { rgba[i*4] = data[8+i*3]; rgba[i*4+1] = data[8+i*3+1]; rgba[i*4+2] = data[8+i*3+2]; rgba[i*4+3] = 255; }
            unsigned char *d_rgba, *d_out; float2* d_mv; float* d_depth;
            if (hipMalloc(&d_rgba, px*4) || hipMalloc(&d_out, px*4) || hipMalloc(&d_mv, px*8) || hipMalloc(&d_depth, px*4)) return 1;
            std::vector<float> dz(px, 0.5f);
            (void)hipMemcpy(d_rgba, rgba.data(), px*4, hipMemcpyHostToDevice);
            (void)hipMemset(d_mv, 0, px*8); (void)hipMemcpy(d_depth, dz.data(), px*4, hipMemcpyHostToDevice);
            for (int k = 0; k < frames; ++k) {
                if (motion(d_mv, fw, fh, fw*8) != 1 || depth(d_depth, fw, fh, fw*4) != 1) { fprintf(stderr, "motion/depth refused: %s\n", error()); return 1; }
                const unsigned long long before = completed();
                if (submit(d_rgba, fw, fh, fw*4, 4, 0) != 1) { fprintf(stderr, "submit: %s\n", error()); return 1; }
                while (completed() == before) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                float ms = 0.0f;
                if (compose(d_out, size_t(fw)*4, fw, fh, 4, &ms) != 1) { fprintf(stderr, "compose: %s\n", error()); return 1; }
                if (a == argc - 1) {
                    std::vector<unsigned char> out(px*4);
                    (void)hipMemcpy(out.data(), d_out, out.size(), hipMemcpyDeviceToHost);
                    last[pass].push_back(out); lw = fw; lh = fh;
                }
            }
            printf("pass %d: %dx%d x %d frames done\n", pass, fw, fh, frames);
            (void)hipFree(d_rgba); (void)hipFree(d_out); (void)hipFree(d_mv); (void)hipFree(d_depth);
        }
        destroy();
    }
    for (int k = 0; k < frames; ++k) {
        size_t differ = 0, big = 0;
        for (size_t i = 0; i < last[0][k].size(); ++i) { const int d = abs(int(last[0][k][i]) - int(last[1][k][i])); differ += d != 0; big += d > 2; }
        printf("last size %dx%d frame %d: after the size walk vs fresh engine: %zu of %zu bytes differ (%zu by more than 2)\n", lw, lh, k, differ, last[0][k].size(), big);
        for (int pass = 0; pass < 2; ++pass) {
            char name[512]; snprintf(name, sizeof(name), "%s/p%d_f%d.rgba", argv[3], pass, k);
            FILE* f = fopen(name, "wb"); if (f) { fwrite(last[pass][k].data(), 1, last[pass][k].size(), f); fclose(f); }
        }
    }
    return 0;
}
