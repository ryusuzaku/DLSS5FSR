// Does hipWaitExternalSemaphoresAsync on a D3D12 fence hold a stream back?
//
// The synchronous paths (re_sync_submit: the shim's SyncMode and the
// final-picture front end) queue "wait for the staged fence, then read the
// shared buffers" on a HIP stream and rely on the reads not running before
// the caller's D3D12 queue has written the frame and signalled. This checks
// that reliance on its own: a wait for a fence value nobody has signalled,
// then a copy (and a kernel) on the same stream; the source is rewritten
// while the stream should be waiting; the fence is signalled late. The copy
// must see the rewritten source, and the stream must not be idle before the
// signal.
// Usage: sem_wait_test [hold_ms] [gpu|cpu]
//   gpu: the fence is signalled by a D3D12 queue (as a game's queue does),
//   cpu (default): by ID3D12Fence::Signal.
#define NOMINMAX
#include <hip/hip_runtime.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

__global__ void k_copy(const unsigned int* src, unsigned int* dst, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

int main(int argc, char** argv) {
    const int hold = argc > 1 ? atoi(argv[1]) : 400;
    const bool gpu = argc > 2 && !strcmp(argv[2], "gpu");
    IDXGIFactory6* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device* device = nullptr;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return 1;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.VendorId == 0x1002 && SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) break;
    }
    if (!device) { puts("no AMD D3D12 device"); return 1; }
    ID3D12CommandQueue* queue = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return 1;
    ID3D12Fence* fence = nullptr;
    HANDLE shared = nullptr;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) ||
        FAILED(device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &shared))) return 1;
    hipExternalSemaphore_t sem = nullptr;
    hipExternalSemaphoreHandleDesc sd{};
    sd.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    sd.handle.win32.handle = shared;
    if (hipImportExternalSemaphore(&sem, &sd) != hipSuccess) { puts("import failed"); return 1; }

    const int n = 1 << 20;
    unsigned int *src = nullptr, *dst = nullptr, *dst2 = nullptr, *dst3 = nullptr;
    hipStream_t waiting = nullptr, other = nullptr, second = nullptr;
    hipEvent_t ready = nullptr;
    if (hipMalloc(&src, n*4) || hipMalloc(&dst, n*4) || hipMalloc(&dst2, n*4) || hipMalloc(&dst3, n*4) ||
        hipStreamCreate(&waiting) || hipStreamCreate(&other) || hipStreamCreate(&second) || hipEventCreate(&ready)) return 1;
    std::vector<unsigned int> host(n);
    auto fill = [&](unsigned int* d, unsigned int v) {
        for (auto& x : host) x = v;
        (void)hipMemcpyAsync(d, host.data(), n*4, hipMemcpyHostToDevice, other);
        (void)hipStreamSynchronize(other);
    };
    int failures = 0;
    for (int round = 1; round <= 3; ++round) {
        const unsigned int before = 0x1000u + round, after = 0x2000u + round;
        fill(src, before); fill(dst, 0); fill(dst2, 0); fill(dst3, 0);
        hipExternalSemaphoreWaitParams wp{};
        wp.params.fence.value = (unsigned long long)round;
        const hipError_t we = hipWaitExternalSemaphoresAsync(&sem, &wp, 1, waiting);
        (void)hipMemcpyAsync(dst, src, n*4, hipMemcpyDeviceToDevice, waiting);
        hipLaunchKernelGGL(k_copy, dim3((n+255)/256), dim3(256), 0, waiting, src, dst2, n);
        // The engine's second stream: behind an event recorded after the reads.
        (void)hipEventRecord(ready, waiting);
        (void)hipStreamWaitEvent(second, ready, 0);
        hipLaunchKernelGGL(k_copy, dim3((n+255)/256), dim3(256), 0, second, dst, dst3, n);
        Sleep(hold/2);
        fill(src, after);  // what the caller's queue writes before it signals
        Sleep(hold/2);
        const hipError_t idle = hipStreamQuery(waiting);  // hipSuccess: nothing left to run
        const hipError_t idle2 = hipStreamQuery(second);
        if (gpu) queue->Signal(fence, (unsigned long long)round);
        else fence->Signal((unsigned long long)round);
        (void)hipStreamSynchronize(waiting);
        (void)hipStreamSynchronize(second);
        unsigned int got[3] = {0, 0, 0};
        (void)hipMemcpyAsync(&got[0], dst, 4, hipMemcpyDeviceToHost, other);
        (void)hipMemcpyAsync(&got[1], dst2 + n/2, 4, hipMemcpyDeviceToHost, other);
        (void)hipMemcpyAsync(&got[2], dst3 + n/3, 4, hipMemcpyDeviceToHost, other);
        (void)hipStreamSynchronize(other);
        const bool ok = got[0] == after && got[1] == after && got[2] == after && idle != hipSuccess && idle2 != hipSuccess;
        failures += !ok;
        auto saw = [&](unsigned int v) {
            return v == after ? "the late data" : v == before ? "the EARLY data" : v == 0 ? "NOTHING (ran first)" : "garbage";
        };
        printf("round %d: wait call %s; before the signal the streams were %s / %s; copy saw %s, kernel saw %s, "
               "second stream saw %s -> %s\n", round, we == hipSuccess ? "ok" : hipGetErrorString(we),
               idle == hipSuccess ? "IDLE (ran through)" : "busy", idle2 == hipSuccess ? "IDLE (ran through)" : "busy",
               saw(got[0]), saw(got[1]), saw(got[2]), ok ? "held" : "NOT HELD");
    }
    printf("%s (%s signal, %d ms)\n", failures ? "FAIL: the wait does not hold the stream" : "PASS: the wait holds the stream",
           gpu ? "queue" : "CPU", hold);
    return failures ? 3 : 0;
}
