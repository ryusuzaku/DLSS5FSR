// Full-frame engine checks through the DLL exports (as the shim uses them).
// Usage: resident_engine_full_test engine.dll config.txt capture.bin out_dir
// 1) a 256x256 cut of the capture must give byte-identical colour and RGB in
//    crop and full mode (the crop resample is the identity at that size);
// 2) the whole capture runs at its padded network extent and returns a
//    finite frame of the capture's size and format; timings are reported.
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
typedef int (*SubmitFullFn)(const void*, int, int, int, int, int);
typedef int (*PollFn)(int, unsigned char*, unsigned char*, float*);
typedef int (*PollFrameFn)(unsigned char*, unsigned long long, int, int, int, float*);
typedef int (*LatestFn)(int, float*, float*);
typedef int (*ExtentFn)(int*, int*);
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

int main(int argc, char** argv) {
    if (argc != 5) { fputs("usage: resident_engine_full_test engine.dll config.txt capture.bin out_dir\n", stderr); return 2; }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (CreateFn)GetProcAddress(dll, "re_create");
    auto submit = (SubmitFn)GetProcAddress(dll, "re_submit");
    auto submit_full = (SubmitFullFn)GetProcAddress(dll, "re_submit_full");
    auto poll = (PollFn)GetProcAddress(dll, "re_poll");
    auto poll_frame = (PollFrameFn)GetProcAddress(dll, "re_poll_frame");
    auto latest = (LatestFn)GetProcAddress(dll, "re_latest_rgb");
    auto extent = (ExtentFn)GetProcAddress(dll, "re_latest_extent");
    auto error = (ErrorFn)GetProcAddress(dll, "re_last_error");
    auto destroy = (DestroyFn)GetProcAddress(dll, "re_destroy");
    if (!create || !submit || !submit_full || !poll || !poll_frame || !latest || !extent || !error || !destroy) return 1;
    if (!create(argv[2])) { fprintf(stderr, "re_create failed: %s\n", error()); return 1; }
    auto capture = slurp(argv[3]);
    if (capture.size() < 36 || memcmp(capture.data(), "D5INP001", 8)) return 1;
    unsigned int f[4];
    memcpy(f, capture.data()+8, sizeof(f));
    int w = int(f[0]), h = int(f[1]), bpp = int(f[2]), pitch = int(f[3]);
    const unsigned char* pixels = capture.data()+36;
    std::string out = argv[4];
    bool ok = true;

    auto wait = [&](auto&& step) {
        for (int i = 0; i < 60000; ++i) {
            int r = step();
            if (r) return r;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    };

    // 1) 256x256 cut: crop and full mode must agree exactly.
    int cut = 256, left = (w-cut)/2, top = (h-cut)/2;
    std::vector<unsigned char> cutout(size_t(cut)*cut*bpp);
    for (int y = 0; y < cut; ++y)
        memcpy(cutout.data()+size_t(y)*cut*bpp, pixels+size_t(top+y)*pitch+size_t(left)*bpp, size_t(cut)*bpp);
    void* device = nullptr;
    if (hipMalloc(&device, size_t(pitch)*h) != hipSuccess) return 1;
    if (hipMemcpy(device, cutout.data(), cutout.size(), hipMemcpyHostToDevice) != hipSuccess) return 1;
    std::vector<float> crop_native(256*256*3), crop_public(256*256*3), crop_color(256*256*3);
    std::vector<float> full_native(256*256*3), full_public(256*256*3), full_color(256*256*3);
    std::vector<unsigned char> rgba8(256*256*4), rgba16(256*256*8), frame(size_t(cut)*cut*bpp);
    float ms = 0;
    if (submit(device, cut, cut, cut*bpp, bpp) != 1 ||
        wait([&] { return poll(0, rgba8.data(), rgba16.data(), &ms); }) != 1 ||
        wait([&] { return latest(0, crop_native.data(), crop_color.data()); }) != 1 ||
        wait([&] { return latest(1, crop_public.data(), nullptr); }) != 1) {
        fprintf(stderr, "crop run failed: %s\n", error()); return 1;
    }
    if (submit_full(device, cut, cut, cut*bpp, bpp, 0) != 1 ||
        wait([&] { return poll_frame(frame.data(), size_t(cut)*bpp, cut, cut, bpp, &ms); }) != 1 ||
        wait([&] { return latest(0, full_native.data(), full_color.data()); }) != 1 ||
        wait([&] { return latest(1, full_public.data(), nullptr); }) != 1) {
        fprintf(stderr, "full 256 run failed: %s\n", error()); return 1;
    }
    for (auto item : {std::make_pair("colour", std::make_pair(&crop_color, &full_color)),
                      std::make_pair("native RGB", std::make_pair(&crop_native, &full_native)),
                      std::make_pair("public RGB", std::make_pair(&crop_public, &full_public))}) {
        bool same = !memcmp(item.second.first->data(), item.second.second->data(), item.second.first->size()*4);
        printf("256 cut crop vs full %s: %s\n", item.first, same ? "PASS (byte-exact)" : "FAIL");
        ok = ok && same;
    }

    // 2) Whole frame at its padded network extent.
    if (hipMemcpy(device, pixels, size_t(pitch)*h, hipMemcpyHostToDevice) != hipSuccess) return 1;
    std::vector<unsigned char> whole(size_t(w)*h*bpp);
    std::vector<float> timings;
    for (int run = 0; run < 3; ++run) {
        auto t0 = std::chrono::steady_clock::now();
        if (submit_full(device, w, h, pitch, bpp, 0) != 1 ||
            wait([&] { return poll_frame(whole.data(), size_t(w)*bpp, w, h, bpp, &ms); }) != 1) {
            fprintf(stderr, "full frame run failed: %s\n", error()); return 1;
        }
        double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
        printf("full frame run %d: device %.1f ms, submit-to-frame %.1f ms\n", run, ms, wall);
        timings.push_back(ms);
    }
    int ew = 0, eh = 0;
    extent(&ew, &eh);
    std::vector<float> rgb(size_t(ew)*eh*3);
    wait([&] { return latest(0, rgb.data(), nullptr); });
    size_t nonfinite = 0;
    double mean = 0;
    for (float v : rgb) { if (!std::isfinite(v)) ++nonfinite; else mean += v; }
    mean /= rgb.size();
    printf("network extent %dx%d for %dx%d frame; nonfinite %zu; mean linear %.4f: %s\n",
           ew, eh, w, h, nonfinite, mean, nonfinite == 0 ? "PASS" : "FAIL");
    ok = ok && nonfinite == 0;
    FILE* o = fopen((out+"/full_frame.bin").c_str(), "wb");
    if (o) {
        int header[3] = {w, h, bpp};
        fwrite(header, sizeof(header), 1, o);
        fwrite(whole.data(), 1, whole.size(), o);
        fclose(o);
    }
    destroy();
    hipFree(device);
    return ok ? 0 : 1;
}
