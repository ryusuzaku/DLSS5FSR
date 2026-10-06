// Checks the synchronous path's ordering (re_sync_submit, S332): runs a
// sequence capture (as resident_engine_seq) through re_sync_submit with
// shared D3D12 fences, the way the shim drives it in game -- four staging
// and output slots, frames submitted back to back, the "staged" fence
// signalled only after a frame's slot is written, the slot rewritten only
// after the job that read it signalled -- and compares each frame's map with
// resident_engine_seq's (the blocking path) byte for byte.
// Usage: resident_engine_sync_test engine.dll config.txt seq_dir ref_map_dir [first count]
#define NOMINMAX
#include <hip/hip_runtime.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

struct ReSyncJob {
    const void* staging; int w, h, pitch, bpp, gain;
    const void* motion; int motion_pitch;
    const void* depth; int depth_pitch;
    void* wait_sem; unsigned long long wait_value;
    void* out; unsigned long long out_pitch;
    void* signal_sem; unsigned long long signal_value;
};

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

static size_t trailer(const std::vector<unsigned char>& d, size_t at, const char* tag) {
    while (at + 16 <= d.size()) {
        unsigned int dims[2];
        memcpy(dims, d.data() + at + 8, 8);
        const bool mv = !memcmp(d.data() + at, "D5MOV001", 8);
        if (!mv && memcmp(d.data() + at, "D5DEP001", 8)) return 0;
        if (!memcmp(d.data() + at, tag, 8)) return at + 16;
        at += 16 + size_t(dims[0])*dims[1]*(mv ? 8 : 4);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 5 && argc != 7) {
        fputs("usage: resident_engine_sync_test engine.dll config.txt seq_dir ref_map_dir [first count]\n", stderr);
        return 2;
    }
    std::vector<std::string> files;
    WIN32_FIND_DATAA fd;
    HANDLE find = FindFirstFileA((std::string(argv[3]) + "\\*.bin").c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) return 1;
    do files.push_back(fd.cFileName); while (FindNextFileA(find, &fd));
    FindClose(find);
    std::sort(files.begin(), files.end());
    const int first = argc == 7 ? atoi(argv[5]) : 0;
    const int count = std::min(argc == 7 ? atoi(argv[6]) : int(files.size()), int(files.size()) - first);

    // D3D12 side: the device the fences live on (no queue work needed: the
    // CPU signals "staged" as the game's queue would).
    IDXGIFactory6* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device* device = nullptr;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return 1;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.VendorId == 0x1002 && SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) break;
    }
    if (!device) { puts("no AMD D3D12 device"); return 1; }
    ID3D12Fence *staged = nullptr, *result = nullptr;
    HANDLE hs = nullptr, hr = nullptr;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&staged))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result))) ||
        FAILED(device->CreateSharedHandle(staged, nullptr, GENERIC_ALL, nullptr, &hs)) ||
        FAILED(device->CreateSharedHandle(result, nullptr, GENERIC_ALL, nullptr, &hr))) return 1;
    hipExternalSemaphore_t sem_staged = nullptr, sem_result = nullptr;
    hipExternalSemaphoreHandleDesc sd{};
    sd.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = hs;
    if (hipImportExternalSemaphore(&sem_staged, &sd) != hipSuccess) return 1;
    sd.handle.win32.handle = hr;
    if (hipImportExternalSemaphore(&sem_result, &sd) != hipSuccess) return 1;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    auto wait_result = [&](unsigned long long v) {
        if (result->GetCompletedValue() >= v) return true;
        result->SetEventOnCompletion(v, ev);
        return WaitForSingleObject(ev, 20000) == WAIT_OBJECT_0;
    };

    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 1;
    auto create = (int (*)(const char*))GetProcAddress(dll, "re_create");
    auto sync_submit = (int (*)(const ReSyncJob*))GetProcAddress(dll, "re_sync_submit");
    auto error = (const char* (*)())GetProcAddress(dll, "re_last_error");
    auto destroy = (void (*)())GetProcAddress(dll, "re_destroy");
    if (!create || !sync_submit || !error || !destroy) { puts("engine lacks re_sync_submit"); return 1; }
    if (!create(argv[2])) { fprintf(stderr, "re_create failed: %s\n", error()); return 1; }

    std::vector<unsigned char> data;
    if (!read_file(std::string(argv[3]) + "\\" + files[first], data)) return 1;
    unsigned int hdr[4];
    memcpy(hdr, data.data() + 8, sizeof(hdr));
    const int w = int(hdr[0]), h = int(hdr[1]), bpp = int(hdr[2]), pitch = int(hdr[3]);
    const int slots = 4;
    void *staging[4], *mv[4], *dep[4], *out[4];
    for (int k = 0; k < slots; ++k)
        if (hipMalloc(&staging[k], size_t(pitch)*h) != hipSuccess || hipMalloc(&mv[k], size_t(w)*h*8) != hipSuccess ||
            hipMalloc(&dep[k], size_t(w)*h*4) != hipSuccess || hipMalloc(&out[k], size_t(w)*h*8) != hipSuccess) return 1;
    std::vector<unsigned char> map(size_t(w)*h*8), ref;
    int compared = 0, identical = 0, skipped = 0;
    double worst = 0.0;
    auto check = [&](int i) {
        // Frame i's job has signalled: compare its output slot with the reference.
        hipMemcpy(map.data(), out[i % slots], map.size(), hipMemcpyDeviceToHost);
        char name[64];
        snprintf(name, sizeof(name), "\\map_%04d.bin", i);
        if (!read_file(std::string(argv[4]) + name, ref) || ref.size() != 12 + size_t(w)*h*4) { ++skipped; return; }
        size_t diff = 0;
        double maxd = 0.0;
        for (size_t p = 0; p < size_t(w)*h; ++p) {
            if (memcmp(map.data() + p*8, ref.data() + 12 + p*4, 4)) {
                ++diff;
                _Float16 a, b;
                memcpy(&a, map.data() + p*8, 2); memcpy(&b, ref.data() + 12 + p*4, 2);
                maxd = std::max(maxd, (double)std::abs(float(a) - float(b)));
            }
        }
        ++compared;
        identical += diff == 0;
        worst = std::max(worst, maxd);
        if (diff) printf("frame %d: %zu pixels differ (max ratio diff %.4g)\n", i, diff, maxd);
    };
    int submitted = 0, rejected = 0;
    long long last_accepted = -1;
    std::vector<int> pending;
    for (int i = first; i < first + count; ++i) {
        const int slot = i % slots;
        // The slot's previous reader (frame i-4) must have signalled before it is rewritten.
        if (i - slots >= first && !wait_result((unsigned long long)(i - slots + 1))) { puts("timeout (slot reuse)"); return 1; }
        if (!read_file(std::string(argv[3]) + "\\" + files[i], data)) return 1;
        const size_t base = 36 + size_t(pitch)*h;
        const size_t mv_at = trailer(data, base, "D5MOV001"), dep_at = trailer(data, base, "D5DEP001");
        if (!mv_at || hipMemcpy(staging[slot], data.data() + 36, size_t(pitch)*h, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(mv[slot], data.data() + mv_at, size_t(w)*h*8, hipMemcpyHostToDevice) != hipSuccess ||
            (dep_at && hipMemcpy(dep[slot], data.data() + dep_at, size_t(w)*h*4, hipMemcpyHostToDevice) != hipSuccess)) return 1;
        ReSyncJob j{};
        j.staging = staging[slot]; j.w = w; j.h = h; j.pitch = pitch; j.bpp = bpp; j.gain = 1;
        j.motion = mv[slot]; j.motion_pitch = w*8;
        j.depth = dep_at ? dep[slot] : nullptr; j.depth_pitch = w*4;
        j.wait_sem = sem_staged; j.wait_value = (unsigned long long)i + 1;
        j.out = out[slot]; j.out_pitch = (unsigned long long)w*8;
        j.signal_sem = sem_result; j.signal_value = (unsigned long long)i + 1;
        // Like the shim: submit first, the "list" (here the CPU) signals staged after.
        int r;
        while ((r = sync_submit(&j)) == 0) {
            // A job is still queued: in game this frame would be skipped; here
            // every frame must run to compare, so wait for the queue to drain.
            ++rejected;
            if (!wait_result((unsigned long long)last_accepted + 1)) { puts("timeout (queue)"); return 1; }
        }
        if (r < 0) { fprintf(stderr, "frame %d: %s\n", i, error()); return 1; }
        staged->Signal((unsigned long long)i + 1);
        last_accepted = i;
        ++submitted;
        pending.push_back(i);
        // Results are checked lazily, keeping up to three jobs in flight.
        while (!pending.empty() && (pending.size() > 3 || result->GetCompletedValue() >= (unsigned long long)pending.front() + 1)) {
            if (!wait_result((unsigned long long)pending.front() + 1)) { puts("timeout (result)"); return 1; }
            check(pending.front());
            pending.erase(pending.begin());
        }
    }
    for (int i : pending) {
        if (!wait_result((unsigned long long)i + 1)) { puts("timeout (drain)"); return 1; }
        check(i);
    }
    printf("sync: %d frames submitted (%d busy retries), %d compared, %d byte-identical, %d without reference, "
           "max ratio diff %.4g\n", submitted, rejected, compared, identical, skipped, worst);
    destroy();
    return identical == compared && compared > 0 ? 0 : 3;
}
