// Runs an in-game sequence capture (the shim's "<capture>.seq" folder: one
// D5INP001 file per staged frame with the D5DEP001 depth and D5MOV001 motion
// trailers) through the engine DLL, one network run per frame and in order,
// and writes each frame's ratio map, for offline replays of the shim's map
// resolve at any lag (tools/seq_replay.py).
// Usage: resident_engine_seq engine.dll config.txt seq_dir out_dir [first count]
// out_dir/map_NNNN.bin per frame: 12-byte header (width, height, 4) and the
// map as RG16F rows (ratio, Y_in; Y_in < 0 = no edit), NNNN = the index in
// the sorted sequence. Every network run waits for the previous one, so the
// engine's own smoothing sees one map per frame (in game it sees one per run).
#include <hip/hip_runtime.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef int (*CreateFn)(const char*);
typedef int (*SubmitFullFn)(const void*, int, int, int, int, int);
typedef int (*PlaneFn)(const void*, int, int, int);
typedef int (*ComposeMapFn)(void*, unsigned long long, int, int, float*);
typedef unsigned long long (*CompletedFn)();
typedef const char* (*ErrorFn)();
typedef void (*DestroyFn)();

static bool read_file(const std::string& path, std::vector<unsigned char>& data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    const long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    data.resize(size_t(n));
    const bool ok = fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

// The trailer `tag` after offset `at`: returns the payload offset or 0.
static size_t find_trailer(const std::vector<unsigned char>& d, size_t at, const char* tag, int w, int h, int bytes) {
    while (at + 16 <= d.size()) {
        unsigned int dims[2];
        memcpy(dims, d.data() + at + 8, 8);
        const bool is_mv = !memcmp(d.data() + at, "D5MOV001", 8);
        if (!is_mv && memcmp(d.data() + at, "D5DEP001", 8)) return 0;  // unknown trailer
        const size_t end = at + 16 + size_t(dims[0])*dims[1]*(is_mv ? 8 : 4);
        if (!memcmp(d.data() + at, tag, 8))
            return int(dims[0]) == w && int(dims[1]) == h && end <= d.size() && (is_mv ? 8 : 4) == bytes ? at + 16 : 0;
        at = end;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7) {
        fputs("usage: resident_engine_seq engine.dll config.txt seq_dir out_dir [first count]\n", stderr);
        return 2;
    }
    std::vector<std::string> files;
    WIN32_FIND_DATAA fd;
    HANDLE find = FindFirstFileA((std::string(argv[3]) + "\\*.bin").c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) { fputs("no frames\n", stderr); return 1; }
    do files.push_back(fd.cFileName); while (FindNextFileA(find, &fd));
    FindClose(find);
    std::sort(files.begin(), files.end());
    const int first = argc == 7 ? atoi(argv[5]) : 0;
    const int count = argc == 7 ? atoi(argv[6]) : int(files.size());
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (CreateFn)GetProcAddress(dll, "re_create");
    auto submit_full = (SubmitFullFn)GetProcAddress(dll, "re_submit_full");
    auto motion = (PlaneFn)GetProcAddress(dll, "re_motion");
    auto depth = (PlaneFn)GetProcAddress(dll, "re_depth");
    auto compose_map = (ComposeMapFn)GetProcAddress(dll, "re_compose_map_device");
    auto completed = (CompletedFn)GetProcAddress(dll, "re_completed");
    auto error = (ErrorFn)GetProcAddress(dll, "re_last_error");
    auto destroy = (DestroyFn)GetProcAddress(dll, "re_destroy");
    if (!create || !submit_full || !motion || !depth || !compose_map || !completed || !error || !destroy) {
        fputs("engine DLL lacks an export (re_completed needs S332 or later)\n", stderr);
        return 1;
    }
    if (!create(argv[2])) { fprintf(stderr, "re_create failed: %s\n", error()); return 1; }
    void *staging = nullptr, *mv = nullptr, *dep = nullptr, *map = nullptr;
    size_t staging_bytes = 0;
    int w0 = 0, h0 = 0;
    std::vector<unsigned char> data, out;
    for (int i = first; i < first + count && i < int(files.size()); ++i) {
        const std::string path = std::string(argv[3]) + "\\" + files[i];
        if (!read_file(path, data) || data.size() < 36 || memcmp(data.data(), "D5INP001", 8)) {
            fprintf(stderr, "%s: not a capture\n", path.c_str()); return 1;
        }
        unsigned int hdr[4];
        memcpy(hdr, data.data() + 8, sizeof(hdr));
        const int w = int(hdr[0]), h = int(hdr[1]), bpp = int(hdr[2]), pitch = int(hdr[3]);
        const size_t trailers = 36 + size_t(pitch)*h;
        const size_t mv_at = find_trailer(data, trailers, "D5MOV001", w, h, 8);
        const size_t dep_at = find_trailer(data, trailers, "D5DEP001", w, h, 4);
        if (!mv_at || bpp != 8) { fprintf(stderr, "%s: no motion trailer or not RGBA16F\n", path.c_str()); return 1; }
        if (w != w0 || h != h0) {
            if (w0) printf("frame %d: size %dx%d (was %dx%d)\n", i, w, h, w0, h0);
            hipFree(staging); hipFree(mv); hipFree(dep); hipFree(map);
            staging_bytes = size_t(pitch)*h;
            if (hipMalloc(&staging, staging_bytes) != hipSuccess || hipMalloc(&mv, size_t(w)*h*8) != hipSuccess ||
                hipMalloc(&dep, size_t(w)*h*4) != hipSuccess || hipMalloc(&map, size_t(w)*h*8) != hipSuccess) return 1;
            w0 = w; h0 = h;
        }
        if (hipMemcpy(staging, data.data() + 36, size_t(pitch)*h, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(mv, data.data() + mv_at, size_t(w)*h*8, hipMemcpyHostToDevice) != hipSuccess ||
            (dep_at && hipMemcpy(dep, data.data() + dep_at, size_t(w)*h*4, hipMemcpyHostToDevice) != hipSuccess)) return 1;
        const unsigned long long before = completed();
        if (motion(mv, w, h, w*8) < 0 || (dep_at && depth(dep, w, h, w*4) < 0) ||
            submit_full(staging, w, h, pitch, bpp, 1) < 0) {
            fprintf(stderr, "frame %d: submit failed: %s\n", i, error()); return 1;
        }
        // One network run per frame: wait for this frame's result.
        for (int t = 0; completed() == before; ++t) {
            if (t > 20000) { fprintf(stderr, "frame %d: no result: %s\n", i, error()); return 1; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (compose_map(map, (unsigned long long)w*8, w, h, nullptr) != 1) {
            fprintf(stderr, "frame %d: map compose failed: %s\n", i, error()); return 1;
        }
        out.resize(size_t(w)*h*8);
        if (hipMemcpy(out.data(), map, out.size(), hipMemcpyDeviceToHost) != hipSuccess) return 1;
        char name[64];
        snprintf(name, sizeof(name), "\\map_%04d.bin", i);
        FILE* f = fopen((std::string(argv[4]) + name).c_str(), "wb");
        if (!f) return 1;
        const unsigned int ohdr[3] = {unsigned(w), unsigned(h), 4};
        fwrite(ohdr, 4, 3, f);
        for (size_t p = 0; p < size_t(w)*h; ++p) fwrite(out.data() + p*8, 1, 4, f);  // R, G halves
        fclose(f);
        if ((i - first) % 10 == 0) printf("frame %d (%s) done\n", i, files[i].c_str());
    }
    destroy();
    return 0;
}
