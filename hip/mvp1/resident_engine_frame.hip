// Runs one capture through the engine DLL as the shim does (submit every
// frame, poll every frame) and writes the returned frame, so any output
// mode (transfer=full or luma) can be inspected offline.
// Usage: resident_engine_frame engine.dll config.txt capture.bin out.bin [pan frames [px|uv sign [hold]]]
//   pan/frames: simulate a horizontal camera pan of `pan` pixels per frame
//   for `frames` frames (frame f shows the capture shifted by f*pan, edge
//   clamped) with the matching motion vectors through re_motion; the
//   output is the last frame's result. px|uv and sign (1 or -1) choose how
//   the vectors are reported (default px 1), to exercise motion=auto.
//   hold (game cadence): the camera moves on every 16 ms frame (RE_FRAME_MS) whether or not
//   a network result landed, as in game, then holds still for `hold` frames
//   (0: none). Comparing hold 0 with a long hold shows the motion carry's
//   error at the final position. RE_SHIM_LAG=L hands the engine frame n-L at
//   game frame n, as the shim does (its prepare runs before frame n's list),
//   and also writes out.bin.map: the engine's carried ratio map
//   (re_compose_map_device, RGBA16F) after the last frame, for checking what
//   the shim's map resolve makes of it on frame n. RE_DUMP_PREFIX=P writes
//   every game-cadence frame's output to P_NNN.bin (same format as out.bin),
//   for measuring frame-to-frame stability.
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
    if (argc != 5 && argc != 7 && argc != 9 && argc != 10) {
        fputs("usage: resident_engine_frame engine.dll config.txt capture.bin out.bin [pan frames [px|uv sign [hold]]]\n", stderr);
        return 2;
    }
    const int pan = argc >= 7 ? atoi(argv[5]) : 0, frames = argc >= 7 ? atoi(argv[6]) : 1;
    const bool uv = argc >= 9 && strcmp(argv[7], "uv") == 0;
    const float sign = argc >= 9 ? float(atoi(argv[8])) : 1.0f;
    const int hold = argc == 10 ? atoi(argv[9]) : -1;  // -1: wait for results (no game cadence)
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
    void *staging = nullptr, *mv = nullptr, *still = nullptr;
    if (hipMalloc(&staging, size_t(pitch)*h) != hipSuccess || hipMalloc(&mv, size_t(w)*h*8) != hipSuccess ||
        hipMalloc(&still, size_t(w)*h*8) != hipSuccess || hipMemset(still, 0, size_t(w)*h*8) != hipSuccess) return 1;
    std::vector<float> field(size_t(w)*h*2);
    for (size_t i = 0; i < field.size(); i += 2) { field[i] = sign*float(pan)/(uv ? float(w) : 1.0f); field[i+1] = 0.0f; }
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
    if (hold >= 0) {
        // Warm up on the first position until a result lands (decides motion=auto too).
        if (!stage(0)) return 1;
        for (int tries = 0; tries < 1000 && r != 1; ++tries) {
            if (motion(still, w, h, w*8) < 0 || submit_full(staging, w, h, pitch, bpp, 1) < 0) return 1;
            r = poll_frame(frame.data(), size_t(w)*bpp, w, h, bpp, &ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        // RE_FRAME_MS: the game frame period (default 16); shorter than the
        // network time leaves the map several frames stale, as under load.
        const char* env = getenv("RE_FRAME_MS");
        const int period = env ? atoi(env) : 16;
        const char* lag_env = getenv("RE_SHIM_LAG");
        const int lag = lag_env ? atoi(lag_env) : 0;
        int landed = 0, staged = 0;
        for (int index = 1; index < frames + hold; ++index) {
            int k = index - lag;
            k = k < 0 ? 0 : (k > frames-1 ? frames-1 : k);
            const bool moving = k != staged;
            if (moving && !stage(k)) return 1;
            staged = k;
            if (motion(moving ? mv : still, w, h, w*8) < 0 || submit_full(staging, w, h, pitch, bpp, 1) < 0) {
                fprintf(stderr, "submit failed: %s\n", error()); return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(period));
            r = poll_frame(frame.data(), size_t(w)*bpp, w, h, bpp, &ms);
            if (r < 0) { fprintf(stderr, "poll failed: %s\n", error()); return 1; }
            landed += r == 1;
            if (const char* prefix = getenv("RE_DUMP_PREFIX")) {
                char name[1024];
                snprintf(name, sizeof(name), "%s_%03d.bin", prefix, index);
                if (FILE* d = fopen(name, "wb")) {
                    unsigned int dump_hdr[3] = {unsigned(w), unsigned(h), unsigned(bpp)};
                    fwrite(dump_hdr, 4, 3, d);
                    fwrite(frame.data(), 1, frame.size(), d);
                    fclose(d);
                }
            }
        }
        printf("game cadence: %d moving frames, %d held, %d composed, shim lag %d\n", frames, hold, landed, lag);
        typedef int (*ComposeMapFn)(void*, unsigned long long, int, int, float*);
        auto compose_map = (ComposeMapFn)GetProcAddress(dll, "re_compose_map_device");
        if (lag > 0 && compose_map && bpp == 8) {
            void* device_map = nullptr;
            std::vector<unsigned char> map(size_t(w)*h*8);
            if (hipMalloc(&device_map, map.size()) != hipSuccess ||
                compose_map(device_map, (unsigned long long)w*8, w, h, nullptr) != 1 ||
                hipMemcpy(map.data(), device_map, map.size(), hipMemcpyDeviceToHost) != hipSuccess) {
                fprintf(stderr, "map compose failed: %s\n", error()); return 1;
            }
            FILE* m = fopen((std::string(argv[4]) + ".map").c_str(), "wb");
            if (!m) return 1;
            unsigned int map_hdr[3] = {unsigned(w), unsigned(h), 8u};
            fwrite(map_hdr, 4, 3, m);
            fwrite(map.data(), 1, map.size(), m);
            fclose(m);
            (void)hipFree(device_map);
        }
    }
    // With a pan, each frame waits for the network so the result is several
    // frames stale when it lands (as in game); the last frame's output is kept.
    for (int index = 0; hold < 0 && index < frames; ++index) {
        if (!stage(index)) return 1;
        r = 0;
        for (int tries = 0; tries < 1000 && r != 1; ++tries) {
            // The pan's motion on the frame that moved, none on repeats of it.
            if (motion(tries == 0 ? mv : still, w, h, w*8) < 0 || submit_full(staging, w, h, pitch, bpp, 1) < 0) {
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
    typedef int (*ModeFn)();
    auto mode = (ModeFn)GetProcAddress(dll, "re_motion_mode");
    printf("frame %dx%d bpp %d written (%.1f ms device), motion mode %d\n", w, h, bpp, ms, mode ? mode() : -2);
    destroy();
    return 0;
}
