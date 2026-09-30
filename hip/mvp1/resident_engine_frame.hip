// Runs one capture through the engine DLL as the shim does (submit every
// frame, poll every frame) and writes the returned frame, so any output
// mode (transfer=full or luma) can be inspected offline.
// Usage: resident_engine_frame engine.dll config.txt capture.bin out.bin
// out.bin: 12-byte header (width, height, bpp as uint32) + tightly packed rows.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef int (*CreateFn)(const char*);
typedef int (*SubmitFullFn)(const void*, int, int, int, int, int);
typedef int (*PollFrameFn)(unsigned char*, unsigned long long, int, int, int, float*);
typedef const char* (*ErrorFn)();
typedef void (*DestroyFn)();

int main(int argc, char** argv) {
    if (argc != 5) { fputs("usage: resident_engine_frame engine.dll config.txt capture.bin out.bin\n", stderr); return 2; }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (CreateFn)GetProcAddress(dll, "re_create");
    auto submit_full = (SubmitFullFn)GetProcAddress(dll, "re_submit_full");
    auto poll_frame = (PollFrameFn)GetProcAddress(dll, "re_poll_frame");
    auto error = (ErrorFn)GetProcAddress(dll, "re_last_error");
    auto destroy = (DestroyFn)GetProcAddress(dll, "re_destroy");
    if (!create || !submit_full || !poll_frame || !error || !destroy) return 1;
    if (!create(argv[2])) { fprintf(stderr, "re_create failed: %s\n", error()); return 1; }
    FILE* f = fopen(argv[3], "rb");
    if (!f) return 1;
    std::vector<unsigned char> capture;
    unsigned char buffer[1 << 16];
    size_t got;
    while ((got = fread(buffer, 1, sizeof(buffer), f)) > 0) capture.insert(capture.end(), buffer, buffer+got);
    fclose(f);
    if (capture.size() < 36 || memcmp(capture.data(), "D5INP001", 8)) return 1;
    unsigned int hdr[4];
    memcpy(hdr, capture.data()+8, sizeof(hdr));
    int w = int(hdr[0]), h = int(hdr[1]), bpp = int(hdr[2]), pitch = int(hdr[3]);
    void* staging = nullptr;
    if (hipMalloc(&staging, size_t(pitch)*h) != hipSuccess ||
        hipMemcpy(staging, capture.data()+36, size_t(pitch)*h, hipMemcpyHostToDevice) != hipSuccess) return 1;
    std::vector<unsigned char> frame(size_t(w)*h*bpp);
    // Submit and poll every "frame" until a result arrives (at most ~10 s).
    int r = 0;
    float ms = 0.0f;
    for (int i = 0; i < 1000 && r != 1; ++i) {
        if (submit_full(staging, w, h, pitch, bpp, 1) < 0) { fprintf(stderr, "submit failed: %s\n", error()); return 1; }
        r = poll_frame(frame.data(), size_t(w)*bpp, w, h, bpp, &ms);
        if (r < 0) { fprintf(stderr, "poll failed: %s\n", error()); return 1; }
        if (r != 1) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (r != 1) { fputs("no frame\n", stderr); return 1; }
    FILE* o = fopen(argv[4], "wb");
    if (!o) return 1;
    unsigned int out_hdr[3] = {unsigned(w), unsigned(h), unsigned(bpp)};
    fwrite(out_hdr, 4, 3, o);
    fwrite(frame.data(), 1, frame.size(), o);
    fclose(o);
    printf("frame %dx%d bpp %d written (%.1f ms device)\n", w, h, bpp, ms);
    destroy();
    return 0;
}
