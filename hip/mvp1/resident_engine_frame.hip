// Runs one capture through the engine DLL as the shim does (submit every
// frame, poll every frame) and writes the returned frame, so any output
// mode (transfer=full or luma) can be inspected offline.
// Usage: resident_engine_frame engine.dll config.txt capture.bin out.bin [pan frames]
//   pan/frames: simulate a horizontal camera pan of `pan` pixels per frame
//   for `frames` frames (frame f shows the capture shifted by f*pan, edge
//   clamped) with the matching motion vectors through re_motion; the
//   output is the last frame's result.
// out.bin: 12-byte header (width, height, bpp as uint32) + tightly packed rows.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef int (*CreateFn)(const char*);
typedef int (*SubmitFullFn)(const void*, int, int, int, int, int);
typedef int (*MotionFn)(const void*, int, int, int);
typedef int (*PollFrameFn)(unsigned char*, unsigned long long, int, int, int, float*);
typedef const char* (*ErrorFn)();
typedef void (*DestroyFn)();

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7) {
        fputs("usage: resident_engine_frame engine.dll config.txt capture.bin out.bin [pan frames]\n", stderr);
        return 2;
    }
    const int pan = argc == 7 ? atoi(argv[5]) : 0, frames = argc == 7 ? atoi(argv[6]) : 1;
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (CreateFn)GetProcAddress(dll, "re_create");
    auto submit_full = (SubmitFullFn)GetProcAddress(dll, "re_submit_full");
    auto motion = (MotionFn)GetProcAddress(dll, "re_motion");
    auto poll_frame = (PollFrameFn)GetProcAddress(dll, "re_poll_frame");
    auto error = (ErrorFn)GetProcAddress(dll, "re_last_error");
    auto destroy = (DestroyFn)GetProcAddress(dll, "re_destroy");
    if (!create || !submit_full || !motion || !poll_frame || !error || !destroy) return 1;
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
    const unsigned char* pixels = capture.data()+36;
    void *staging = nullptr, *mv = nullptr;
    if (hipMalloc(&staging, size_t(pitch)*h) != hipSuccess || hipMalloc(&mv, size_t(w)*h*8) != hipSuccess) return 1;
    std::vector<float> field(size_t(w)*h*2);
    for (size_t i = 0; i < field.size(); i += 2) { field[i] = float(pan); field[i+1] = 0.0f; }
    if (hipMemcpy(mv, field.data(), field.size()*4, hipMemcpyHostToDevice) != hipSuccess) return 1;
    std::vector<unsigned char> shifted(size_t(pitch)*h), frame(size_t(w)*h*bpp);
    auto stage = [&](int index) {
        int offset = index*pan;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int sx = x + offset; sx = sx < 0 ? 0 : (sx >= w ? w-1 : sx);
                memcpy(&shifted[size_t(y)*pitch + size_t(x)*bpp], pixels + size_t(y)*pitch + size_t(sx)*bpp, bpp);
            }
        return hipMemcpy(staging, shifted.data(), shifted.size(), hipMemcpyHostToDevice) == hipSuccess;
    };
    int r = 0;
    float ms = 0.0f;
    // With a pan, each frame waits for the network so the result is several
    // frames stale when it lands (as in game); the last frame's output is kept.
    for (int index = 0; index < frames; ++index) {
        if (!stage(index)) return 1;
        r = 0;
        for (int tries = 0; tries < 1000 && r != 1; ++tries) {
            if (motion(mv, w, h, w*8) < 0 || submit_full(staging, w, h, pitch, bpp, 1) < 0) {
                fprintf(stderr, "submit failed: %s\n", error()); return 1;
            }
            r = poll_frame(frame.data(), size_t(w)*bpp, w, h, bpp, &ms);
            if (r < 0) { fprintf(stderr, "poll failed: %s\n", error()); return 1; }
            if (r == 1 && index + 1 < frames) {  // a result exists: move the camera
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
                break;
            }
            if (r != 1) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (r != 1) { fputs("no frame\n", stderr); return 1; }
    }
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
