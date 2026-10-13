// Final-picture front end: see final_picture.h.
#include "final_picture.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>  // interop types only; the functions are resolved at run time

#include "shaders/final_picture_vs.h"
#include "shaders/final_picture_down.h"
#include "shaders/final_picture_up.h"

using Microsoft::WRL::ComPtr;

namespace final_picture {
namespace {

// The engine's synchronous job (resident_engine.hip, ReSyncJob).
struct ReSyncJob {
    const void* staging; int w, h, pitch, bpp, gain;
    const void* motion; int motion_pitch;
    const void* depth; int depth_pitch;
    void* wait_sem; unsigned long long wait_value;
    void* out; unsigned long long out_pitch;
    void* signal_sem; unsigned long long signal_value;
};

constexpr int kRing = 4;       // command allocator sets in flight
constexpr int kBackRtvs = 8;   // back buffers of one swap chain

struct State {
    Config cfg;
    Stats stats;
    bool ready = false;
    std::mutex mutex;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;

    // Engine and HIP runtime.
    HMODULE engine = nullptr, hip = nullptr;
    int (*reCreate)(const char*) = nullptr;
    int (*reControls)(const float*, int) = nullptr;
    int (*reSyncSubmit)(const ReSyncJob*) = nullptr;
    const char* (*reLastError)() = nullptr;
    void (*reDestroy)() = nullptr;
    hipError_t (*importMemory)(hipExternalMemory_t*, const hipExternalMemoryHandleDesc*) = nullptr;
    hipError_t (*mappedBuffer)(void**, hipExternalMemory_t, const hipExternalMemoryBufferDesc*) = nullptr;
    hipError_t (*destroyMemory)(hipExternalMemory_t) = nullptr;
    hipError_t (*importSemaphore)(hipExternalSemaphore_t*, const hipExternalSemaphoreHandleDesc*) = nullptr;
    hipError_t (*destroySemaphore)(hipExternalSemaphore_t) = nullptr;
    hipError_t (*deviceSynchronize)() = nullptr;

    // Fences: staged / result are shared with HIP; done marks a frame's last list.
    ComPtr<ID3D12Fence> staged, result, done;
    hipExternalSemaphore_t semStaged = nullptr, semResult = nullptr;
    HANDLE doneEvent = nullptr;
    unsigned long long frame = 0;

    ComPtr<ID3D12CommandAllocator> alloc[kRing][2];
    ComPtr<ID3D12GraphicsCommandList> list[2];
    unsigned long long ringFrame[kRing] = {};

    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> psoDown, psoUp;
    DXGI_FORMAT psoUpFormat = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D12DescriptorHeap> srvHeap, rtvHeap;
    UINT srvStep = 0, rtvStep = 0;

    // Per frame size.
    UINT width = 0, height = 0, workW = 0, workH = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT64 pitch = 0, allocBytes = 0;
    ComPtr<ID3D12Resource> frameCopy, work, delta, bufIn, bufOut;
    hipExternalMemory_t extIn = nullptr, extOut = nullptr;
    void *ptrIn = nullptr, *ptrOut = nullptr;
    // The game's motion and depth for the frame about to be presented: the
    // caller records copies into these on the game's own list. A list that
    // names them may still be waiting to run when the layout changes, so old
    // ones are kept for a few frames instead of being released at once.
    ComPtr<ID3D12Resource> bufMotion, bufDepth;
    hipExternalMemory_t extMotion = nullptr, extDepth = nullptr;
    void *ptrMotion = nullptr, *ptrDepth = nullptr;
    UINT motionPitch = 0, depthPitch = 0;
    std::atomic<int> guides{0};  // 1 motion, 2 depth: staged for the next Process
    unsigned long long guideFrames = 0;
    std::vector<std::pair<unsigned long long, ComPtr<ID3D12Resource>>> retired;
    ID3D12Resource* backRtv[kBackRtvs] = {};
    int backRtvNext = 0;
    // A frame layout that could not be set up is not retried every frame.
    UINT failedW = 0, failedH = 0;
    DXGI_FORMAT failedFormat = DXGI_FORMAT_UNKNOWN;

    // Sequence capture (Config::capturePath): one readback buffer a frame.
    struct CaptureFrame { ComPtr<ID3D12Resource> buf; unsigned long long frame = 0; int guides = 0; };
    std::vector<CaptureFrame> capture;
    size_t captureAt = 0;          // frames recorded so far
    bool capturing = false;
    std::atomic<bool> captureWriting{false};
    long long captureChecked = 0;  // when the trigger file was last looked for
    std::atomic<int> captureRequest{0};  // RequestCapture: frames asked for
    std::wstring captureDir;

    // A job the queue is waiting for; the watchdog releases a stuck one.
    std::atomic<unsigned long long> waitingFor{0};
    std::atomic<long long> waitingSince{0};
    std::atomic<bool> stop{false}, stuck{false};
    std::thread watchdog;
};

// Never destroyed: a game ends without Shutdown, and by the time statics are
// torn down the watchdog thread and the device are gone.
State& S() { static State* s = new State; return *s; }

void (*g_log)(const char*) = nullptr;

void Log(const char* fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (g_log) g_log(line);
    else fprintf(stderr, "final_picture: %s\n", line);
}

bool Fail(const char* what) {
    State& s = S();
    s.stats.error = what;
    s.stats.active = false;
    Log("%s", what);
    return false;
}

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The view format that reads and writes a back buffer's stored (display-
// encoded) values as they are. Unknown: not handled (HDR, typed sRGB).
DXGI_FORMAT ViewFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return f;
        default:
            return DXGI_FORMAT_UNKNOWN;
    }
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}

bool CreatePso(DXGI_FORMAT target, const void* ps, size_t psBytes, ComPtr<ID3D12PipelineState>& pso) {
    State& s = S();
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = s.root.Get();
    d.VS = {g_fpVs, sizeof(g_fpVs)};
    d.PS = {ps, psBytes};
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.SampleMask = 0xFFFFFFFFu;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = 1;
    d.RTVFormats[0] = target;
    d.SampleDesc = {1, 0};
    return SUCCEEDED(s.device->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso)));
}

bool CreateFixed() {
    State& s = S();
    ID3D12Device* dev = s.device.Get();
    // Root: a table of two SRVs (t0, t1), four float constants (b0), a linear clamp sampler.
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 2;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {0, 0, 4};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rd{};
    rd.NumParameters = 2;
    rd.pParameters = params;
    rd.NumStaticSamplers = 1;
    rd.pStaticSamplers = &sampler;
    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&s.root))))
        return Fail("root signature");
    if (!CreatePso(DXGI_FORMAT_R16G16B16A16_FLOAT, g_fpDown, sizeof(g_fpDown), s.psoDown)) return Fail("down pipeline");

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 4;  // down: frame, frame; up: frame, delta
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s.srvHeap)))) return Fail("srv heap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 1 + kBackRtvs;  // work, back buffers
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s.rtvHeap)))) return Fail("rtv heap");
    s.srvStep = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    s.rtvStep = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    for (int r = 0; r < kRing; ++r)
        for (int k = 0; k < 2; ++k)
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.alloc[r][k]))))
                return Fail("command allocator");
    for (int k = 0; k < 2; ++k) {
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.alloc[0][k].Get(), nullptr, IID_PPV_ARGS(&s.list[k]))))
            return Fail("command list");
        s.list[k]->Close();
    }

    HANDLE hs = nullptr, hr = nullptr;
    bool ok = SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s.staged))) &&
              SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s.result))) &&
              SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.done))) &&
              SUCCEEDED(dev->CreateSharedHandle(s.staged.Get(), nullptr, GENERIC_ALL, nullptr, &hs)) &&
              SUCCEEDED(dev->CreateSharedHandle(s.result.Get(), nullptr, GENERIC_ALL, nullptr, &hr));
    if (ok) {
        hipExternalSemaphoreHandleDesc d{};
        d.type = hipExternalSemaphoreHandleTypeD3D12Fence;
        d.handle.win32.handle = hs;
        ok = s.importSemaphore(&s.semStaged, &d) == hipSuccess && s.semStaged;
        d.handle.win32.handle = hr;
        ok = ok && s.importSemaphore(&s.semResult, &d) == hipSuccess && s.semResult;
    }
    if (hs) CloseHandle(hs);
    if (hr) CloseHandle(hr);
    if (!ok) return Fail("shared fences");
    s.doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return s.doneEvent != nullptr;
}

void WaitDone(unsigned long long value) {
    State& s = S();
    if (!s.done || s.done->GetCompletedValue() >= value) return;
    if (SUCCEEDED(s.done->SetEventOnCompletion(value, s.doneEvent))) WaitForSingleObject(s.doneEvent, 10000);
}

// Sequence capture: the recorded frames go to a thread that waits for the
// last one's lists and writes the files.
struct CaptureJob {
    std::vector<State::CaptureFrame> frames;
    std::wstring dir, trigger;
    ComPtr<ID3D12Fence> done;
    unsigned long long last = 0;
    UINT w = 0, h = 0, pitch = 0, motionPitch = 0, depthPitch = 0;
};

void CaptureWrite(CaptureJob* job) {
    State& s = S();
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (ev && job->done->GetCompletedValue() < job->last && SUCCEEDED(job->done->SetEventOnCompletion(job->last, ev)))
        WaitForSingleObject(ev, 20000);
    if (ev) CloseHandle(ev);
    const size_t rows = size_t(job->pitch) * job->h, plane = size_t(job->w) * job->h;
    const size_t motionBytes = size_t(job->motionPitch) * job->h;
    unsigned int written = 0;
    std::vector<unsigned char> file;
    for (auto& f : job->frames) {
        unsigned char* p = nullptr;
        if (FAILED(f.buf->Map(0, nullptr, reinterpret_cast<void**>(&p))) || !p) continue;
        file.resize(36 + rows + ((f.guides & 2) ? 16 + plane * 4 : 0) + ((f.guides & 1) ? 16 + plane * 8 : 0) + 16 + plane * 8);
        unsigned char* o = file.data();
        memcpy(o, "D5INP001", 8);
        // passthrough 1: the picture is display-encoded as it is.
        const unsigned int fields[6] = {job->w, job->h, 8, job->pitch, 1, 0};
        memcpy(o + 8, fields, sizeof(fields));
        const float white = 1.0f;
        memcpy(o + 32, &white, sizeof(float));
        memcpy(o + 36, p, rows);
        size_t at = 36 + rows;
        const unsigned int dims[2] = {job->w, job->h};
        if (f.guides & 2) {
            memcpy(o + at, "D5DEP001", 8); memcpy(o + at + 8, dims, 8);
            for (UINT y = 0; y < job->h; ++y)
                memcpy(o + at + 16 + size_t(y) * job->w * 4, p + 2 * rows + motionBytes + size_t(y) * job->depthPitch, size_t(job->w) * 4);
            at += 16 + plane * 4;
        }
        if (f.guides & 1) {
            memcpy(o + at, "D5MOV001", 8); memcpy(o + at + 8, dims, 8);
            for (UINT y = 0; y < job->h; ++y)
                memcpy(o + at + 16 + size_t(y) * job->w * 8, p + 2 * rows + size_t(y) * job->motionPitch, size_t(job->w) * 8);
            at += 16 + plane * 8;
        }
        memcpy(o + at, "D5OUT001", 8); memcpy(o + at + 8, dims, 8);
        for (UINT y = 0; y < job->h; ++y)
            memcpy(o + at + 16 + size_t(y) * job->w * 8, p + rows + size_t(y) * job->pitch, size_t(job->w) * 8);
        f.buf->Unmap(0, nullptr);
        f.buf.Reset();
        wchar_t name[32];
        swprintf(name, 32, L"\\%08llu.bin", f.frame);
        FILE* out = _wfopen((job->dir + name).c_str(), L"wb");
        if (out && fwrite(file.data(), 1, file.size(), out) == file.size()) ++written;
        if (out) fclose(out);
    }
    DeleteFileW(job->trigger.c_str());
    Log("capture: %u of %zu frames written to %ls", written, job->frames.size(), job->dir.c_str());
    delete job;
    s.captureWriting = false;
}

// Ends the recording (complete, or cut short by a new frame layout).
void CaptureFinish() {
    State& s = S();
    if (!s.capturing) return;
    s.capturing = false;
    CaptureJob* job = new CaptureJob;
    job->frames.assign(s.capture.begin(), s.capture.begin() + ptrdiff_t(s.captureAt));
    s.capture.clear();
    job->dir = s.captureDir;
    job->trigger = s.cfg.capturePath + L".seq";
    job->done = s.done;
    job->last = s.frame;
    job->w = s.workW; job->h = s.workH; job->pitch = UINT(s.pitch);
    job->motionPitch = s.motionPitch; job->depthPitch = s.depthPitch;
    Log("capture: %zu frames recorded up to frame %llu; writing in the background", job->frames.size(), s.frame);
    std::thread(CaptureWrite, job).detach();
}

// Looks for the trigger file (a few times a second) and sets a recording up.
void CaptureCheck() {
    State& s = S();
    if (s.cfg.capturePath.empty() || s.capturing || s.captureWriting) return;
    const std::wstring trigger = s.cfg.capturePath + L".seq";
    int count = s.captureRequest.exchange(0);
    if (count <= 0) {
        const long long now = NowMs();
        if (now - s.captureChecked < 250) return;
        s.captureChecked = now;
        if (GetFileAttributesW(trigger.c_str()) == INVALID_FILE_ATTRIBUTES) return;
        count = 60;
        if (FILE* t = _wfopen(trigger.c_str(), L"rb")) {
            int v = 0;
            if (fscanf(t, "%d", &v) == 1 && v > 0) count = v;
            fclose(t);
        }
    }
    count = std::min(count, 600);
    s.captureWriting = true;  // until the files are written, or this attempt fails
    s.captureDir = s.cfg.capturePath + L".pseq." + std::to_wstring(GetTickCount64());
    const UINT64 bytes = s.pitch * s.workH * 2 + UINT64(s.motionPitch) * s.workH + UINT64(s.depthPitch) * s.workH;
    // The frames wait in memory until the last is recorded: 4 GB at most.
    count = std::max(1, std::min(count, int((4ull << 30) / bytes)));
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = bytes; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc = {1, 0}; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    s.capture.assign(size_t(count), State::CaptureFrame{});
    bool ok = CreateDirectoryW(s.captureDir.c_str(), nullptr) != 0;
    for (auto& f : s.capture)
        ok = ok && SUCCEEDED(s.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                               IID_PPV_ARGS(&f.buf)));
    if (!ok) {
        s.capture.clear();
        DeleteFileW(trigger.c_str());
        s.captureWriting = false;
        Log("capture: could not set up %d frames of %llu MB in %ls", count, (unsigned long long)(bytes >> 20), s.captureDir.c_str());
        return;
    }
    s.captureAt = 0;
    s.capturing = true;
    Log("capture: %d frames of %ux%u (%llu MB each) into %ls", count, s.workW, s.workH, (unsigned long long)(bytes >> 20),
        s.captureDir.c_str());
}

void DropSized() {
    State& s = S();
    CaptureFinish();  // its frames have this layout
    if (s.deviceSynchronize) s.deviceSynchronize();  // nothing of the engine's still reads the shared buffers
    if (s.extIn) s.destroyMemory(s.extIn);
    if (s.extOut) s.destroyMemory(s.extOut);
    if (s.extMotion) s.destroyMemory(s.extMotion);
    if (s.extDepth) s.destroyMemory(s.extDepth);
    s.extIn = s.extOut = s.extMotion = s.extDepth = nullptr;
    s.ptrIn = s.ptrOut = s.ptrMotion = s.ptrDepth = nullptr;
    if (s.bufMotion) s.retired.emplace_back(s.frame + 16, std::move(s.bufMotion));
    if (s.bufDepth) s.retired.emplace_back(s.frame + 16, std::move(s.bufDepth));
    s.guides = 0;
    s.frameCopy.Reset(); s.work.Reset(); s.delta.Reset(); s.bufIn.Reset(); s.bufOut.Reset();
    memset(s.backRtv, 0, sizeof(s.backRtv));
    s.backRtvNext = 0;
    s.width = s.height = 0;
}

bool CreateTexture(UINT w, UINT h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = f; d.SampleDesc = {1, 0}; d.Flags = flags;
    return SUCCEEDED(S().device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&out)));
}

bool CreateSized(const D3D12_RESOURCE_DESC& bb) {
    State& s = S();
    ID3D12Device* dev = s.device.Get();
    const DXGI_FORMAT view = ViewFormat(bb.Format);
    if (view == DXGI_FORMAT_UNKNOWN) return Fail("back buffer format not handled (HDR or typed sRGB)");
    const UINT w = (UINT)bb.Width, h = bb.Height;
    float scale = s.cfg.workingScale;
    if (!(scale > 0.0f)) {
        scale = sqrtf(s.cfg.maxWorkingPixels / (float(w) * float(h)));
        if (scale > 1.0f) scale = 1.0f;
    }
    if (scale > 1.0f) scale = 1.0f;
    const UINT ww = std::max(64u, (UINT)(w * scale + 0.5f)), wh = std::max(64u, (UINT)(h * scale + 0.5f));

    if (!CreateTexture(w, h, bb.Format, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, s.frameCopy) ||
        !CreateTexture(ww, wh, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, s.work) ||
        !CreateTexture(ww, wh, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, s.delta))
        return Fail("textures");

    // Buffers shared with HIP: RGBA16F rows at D3D12's copy pitch. Rounded up
    // as the NGX shim's staging is (small shared buffers failed in game).
    s.pitch = (UINT64(ww) * 8 + 255) & ~UINT64(255);
    const UINT64 bytes = s.pitch * wh, step = 1ull << 20;
    s.allocBytes = std::max<UINT64>((bytes + step - 1) / step * step, 4ull << 20);
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = s.allocBytes; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc = {1, 0}; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto shared = [&](ComPtr<ID3D12Resource>& buf, hipExternalMemory_t& ext, void*& ptr) {
        HANDLE handle = nullptr;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf))) ||
            FAILED(dev->CreateSharedHandle(buf.Get(), nullptr, GENERIC_ALL, nullptr, &handle)))
            return false;
        hipExternalMemoryHandleDesc md{};
        md.type = hipExternalMemoryHandleTypeD3D12Resource;
        md.handle.win32.handle = handle;
        md.size = s.allocBytes;
        md.flags = hipExternalMemoryDedicated;
        hipExternalMemoryBufferDesc gd{};
        gd.size = s.allocBytes;
        const bool ok = s.importMemory(&ext, &md) == hipSuccess && s.mappedBuffer(&ptr, ext, &gd) == hipSuccess && ptr;
        CloseHandle(handle);
        return ok;
    };
    if (!shared(s.bufIn, s.extIn, s.ptrIn) || !shared(s.bufOut, s.extOut, s.ptrOut)) return Fail("shared buffers");
    // Guides are optional: without them the engine reads its history in place.
    s.motionPitch = UINT((UINT64(ww) * 8 + 255) & ~UINT64(255));
    s.depthPitch = UINT((UINT64(ww) * 4 + 255) & ~UINT64(255));
    if (!shared(s.bufMotion, s.extMotion, s.ptrMotion) || !shared(s.bufDepth, s.extDepth, s.ptrDepth)) {
        Log("no shared buffers for motion and depth: running without them");
        s.bufMotion.Reset(); s.bufDepth.Reset();
        s.ptrMotion = s.ptrDepth = nullptr;
    }

    if (s.psoUpFormat != view) {
        s.psoUp.Reset();
        if (!CreatePso(view, g_fpUp, sizeof(g_fpUp), s.psoUp)) return Fail("up pipeline");
        s.psoUpFormat = view;
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE srv = s.srvHeap->GetCPUDescriptorHandleForHeapStart();
    sd.Format = view;
    for (int k = 0; k < 3; ++k) {  // down: frame, frame; up: frame
        dev->CreateShaderResourceView(s.frameCopy.Get(), &sd, srv);
        srv.ptr += s.srvStep;
    }
    sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    dev->CreateShaderResourceView(s.delta.Get(), &sd, srv);
    dev->CreateRenderTargetView(s.work.Get(), nullptr, s.rtvHeap->GetCPUDescriptorHandleForHeapStart());

    s.width = w; s.height = h; s.format = bb.Format; s.workW = ww; s.workH = wh;
    s.stats.workWidth = ww; s.stats.workHeight = wh;
    Log("frame %ux%u, working %ux%u (scale %.3f), pitch %llu", w, h, ww, wh, scale, (unsigned long long)s.pitch);
    return true;
}

D3D12_CPU_DESCRIPTOR_HANDLE BackRtv(ID3D12Resource* bb) {
    State& s = S();
    D3D12_CPU_DESCRIPTOR_HANDLE base = s.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (int k = 0; k < kBackRtvs; ++k)
        if (s.backRtv[k] == bb) return {base.ptr + SIZE_T(1 + k) * s.rtvStep};
    const int k = s.backRtvNext++ % kBackRtvs;
    s.backRtv[k] = bb;
    D3D12_CPU_DESCRIPTOR_HANDLE h{base.ptr + SIZE_T(1 + k) * s.rtvStep};
    D3D12_RENDER_TARGET_VIEW_DESC rd{};
    rd.Format = s.psoUpFormat;
    rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    s.device->CreateRenderTargetView(bb, &rd, h);
    return h;
}

void Draw(ID3D12GraphicsCommandList* cl, ID3D12PipelineState* pso, D3D12_CPU_DESCRIPTOR_HANDLE target, UINT tw, UINT th, int table) {
    State& s = S();
    ID3D12DescriptorHeap* heaps[] = {s.srvHeap.Get()};
    cl->SetGraphicsRootSignature(s.root.Get());
    cl->SetPipelineState(pso);
    cl->SetDescriptorHeaps(1, heaps);
    D3D12_GPU_DESCRIPTOR_HANDLE srv = s.srvHeap->GetGPUDescriptorHandleForHeapStart();
    srv.ptr += UINT64(table) * 2 * s.srvStep;
    cl->SetGraphicsRootDescriptorTable(0, srv);
    const float constants[4] = {float(s.width), float(s.height), float(tw), float(th)};
    cl->SetGraphicsRoot32BitConstants(1, 4, constants, 0);
    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(tw), float(th), 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, LONG(tw), LONG(th)};
    cl->RSSetViewports(1, &vp);
    cl->RSSetScissorRects(1, &sc);
    cl->OMSetRenderTargets(1, &target, FALSE, nullptr);
    cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cl->DrawInstanced(3, 1, 0, 0);
}

// A job the engine never finishes would hold the caller's queue for ever:
// after 5 s the fence is signalled from here and the path turns itself off.
void Watchdog() {
    State& s = S();
    while (!s.stop) {
        Sleep(100);
        const unsigned long long want = s.waitingFor.load();
        if (!want || s.stuck) continue;
        if (s.result->GetCompletedValue() >= want) {
            unsigned long long expected = want;
            s.waitingFor.compare_exchange_strong(expected, 0);
            continue;
        }
        if (NowMs() - s.waitingSince.load() > 5000) {
            s.stuck = true;
            s.result->Signal(want);
            Log("the engine did not finish frame %llu within 5 s; path off", want);
        }
    }
}

template<class F> bool Resolve(HMODULE m, const char* name, F& f) {
    f = reinterpret_cast<F>(GetProcAddress(m, name));
    if (!f) Log("%s missing", name);
    return f != nullptr;
}

}  // namespace

bool Init(ID3D12Device* device, ID3D12CommandQueue* queue, const Config& config) {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.ready) return true;
    s.cfg = config;
    s.device = device;
    s.queue = queue;
    // Its directory first for what it depends on (the ROCm runtime beside it).
    s.engine = LoadLibraryExW(config.engineDll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!s.engine) return Fail("cannot load the engine DLL");
    s.hip = GetModuleHandleW(L"amdhip64_7.dll");
    if (!s.hip) s.hip = LoadLibraryW(L"amdhip64_7.dll");
    if (!s.hip) return Fail("amdhip64_7.dll not loaded");
    if (!Resolve(s.engine, "re_create", s.reCreate) || !Resolve(s.engine, "re_controls", s.reControls) ||
        !Resolve(s.engine, "re_sync_submit", s.reSyncSubmit) || !Resolve(s.engine, "re_last_error", s.reLastError) ||
        !Resolve(s.engine, "re_destroy", s.reDestroy) ||
        !Resolve(s.hip, "hipImportExternalMemory", s.importMemory) ||
        !Resolve(s.hip, "hipExternalMemoryGetMappedBuffer", s.mappedBuffer) ||
        !Resolve(s.hip, "hipDestroyExternalMemory", s.destroyMemory) ||
        !Resolve(s.hip, "hipImportExternalSemaphore", s.importSemaphore) ||
        !Resolve(s.hip, "hipDestroyExternalSemaphore", s.destroySemaphore) ||
        !Resolve(s.hip, "hipDeviceSynchronize", s.deviceSynchronize))
        return Fail("engine or HIP export missing");
    if (s.reCreate(config.engineConfig.c_str()) != 1) {
        Log("re_create: %s", s.reLastError());
        return Fail("the engine did not start");
    }
    if (s.reControls(config.controls, 6) < 0) Log("re_controls: %s", s.reLastError());
    if (!CreateFixed()) return false;
    s.stop = false;
    s.stuck = false;
    s.watchdog = std::thread(Watchdog);
    s.ready = true;
    s.stats.active = true;
    s.stats.error = "";
    return true;
}

void SetControls(const float controls[6]) {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    memcpy(s.cfg.controls, controls, sizeof(s.cfg.controls));
    if (s.ready) s.reControls(controls, 6);
}

bool Process(ID3D12Resource* bb) {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ready || s.stuck || !bb) return false;
    const D3D12_RESOURCE_DESC desc = bb->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return false;
    if (desc.Width != s.width || desc.Height != s.height || desc.Format != s.format) {
        if (desc.Width == s.failedW && desc.Height == s.failedH && desc.Format == s.failedFormat) { ++s.stats.skipped; return false; }
        WaitDone(s.frame);
        DropSized();
        if (!CreateSized(desc)) {
            DropSized();
            s.failedW = (UINT)desc.Width; s.failedH = desc.Height; s.failedFormat = desc.Format;
            ++s.stats.skipped;
            return false;
        }
        s.failedW = s.failedH = 0;
        s.stats.active = true;
    }
    CaptureCheck();
    const unsigned long long n = ++s.frame;
    const int ring = int(n % kRing);
    WaitDone(s.ringFrame[ring]);  // this allocator set's last use has run
    s.ringFrame[ring] = n;
    while (!s.retired.empty() && s.retired.front().first <= n) s.retired.erase(s.retired.begin());

    // List A: the frame -> copy -> working size -> the shared buffer.
    ID3D12GraphicsCommandList* a = s.list[0].Get();
    s.alloc[ring][0]->Reset();
    a->Reset(s.alloc[ring][0].Get(), nullptr);
    {
        D3D12_RESOURCE_BARRIER b[2] = {Transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE),
                                       Transition(s.frameCopy.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)};
        a->ResourceBarrier(2, b);
        a->CopyResource(s.frameCopy.Get(), bb);
        b[0] = Transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        b[1] = Transition(s.frameCopy.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        a->ResourceBarrier(2, b);
        Draw(a, s.psoDown.Get(), s.rtvHeap->GetCPUDescriptorHandleForHeapStart(), s.workW, s.workH, 0);
        D3D12_RESOURCE_BARRIER w = Transition(s.work.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        a->ResourceBarrier(1, &w);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = s.bufIn.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, s.workW, s.workH, 1, (UINT)s.pitch};
        src.pResource = s.work.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        a->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        w = Transition(s.work.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        a->ResourceBarrier(1, &w);
    }
    a->Close();
    ID3D12CommandList* la[] = {a};
    s.queue->ExecuteCommandLists(1, la);
    s.queue->Signal(s.staged.Get(), n);

    // The engine's job: queued on the device behind `staged`.
    ReSyncJob j{};
    j.staging = s.ptrIn;
    j.w = int(s.workW); j.h = int(s.workH); j.pitch = int(s.pitch); j.bpp = 8; j.gain = 0;
    const int guides = s.guides.exchange(0);
    if ((guides & 1) && s.ptrMotion) {
        j.motion = s.ptrMotion; j.motion_pitch = int(s.motionPitch);
        if ((guides & 2) && s.ptrDepth) { j.depth = s.ptrDepth; j.depth_pitch = int(s.depthPitch); }
        ++s.guideFrames;
    }
    s.stats.guideFrames = s.guideFrames;
    j.wait_sem = s.semStaged; j.wait_value = n;
    j.out = s.ptrOut; j.out_pitch = s.pitch;
    j.signal_sem = s.semResult; j.signal_value = n;
    int accepted = 0;
    const long long t0 = NowMs();
    for (;;) {
        accepted = s.reSyncSubmit(&j);
        if (accepted != 0 || NowMs() - t0 > 50) break;  // 0: the worker still holds the previous job
        Sleep(0);
    }
    if (accepted != 1) {
        if (accepted < 0) { Log("re_sync_submit: %s", s.reLastError()); s.stats.error = "the engine rejected a job"; s.stuck = true; }
        ++s.stats.skipped;
        s.queue->Signal(s.done.Get(), n);
        return false;
    }
    s.waitingSince = NowMs();
    s.waitingFor = n;
    s.queue->Wait(s.result.Get(), n);

    // List B: the change map -> texture; frame copy + map -> the back buffer.
    ID3D12GraphicsCommandList* b = s.list[1].Get();
    s.alloc[ring][1]->Reset();
    b->Reset(s.alloc[ring][1].Get(), nullptr);
    {
        D3D12_RESOURCE_BARRIER t[2] = {Transition(s.delta.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                                       Transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        b->ResourceBarrier(2, t);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = s.delta.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = s.bufOut.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, s.workW, s.workH, 1, (UINT)s.pitch};
        b->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        t[0] = Transition(s.delta.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        b->ResourceBarrier(1, t);
        if (s.capturing) {
            // What the engine read and wrote for this frame, side by side.
            State::CaptureFrame& f = s.capture[s.captureAt];
            const UINT64 rows = s.pitch * s.workH, motionBytes = UINT64(s.motionPitch) * s.workH;
            f.frame = n;
            f.guides = j.motion ? (j.depth ? 3 : 1) : 0;
            b->CopyBufferRegion(f.buf.Get(), 0, s.bufIn.Get(), 0, rows);
            b->CopyBufferRegion(f.buf.Get(), rows, s.bufOut.Get(), 0, rows);
            if (f.guides & 1) b->CopyBufferRegion(f.buf.Get(), 2 * rows, s.bufMotion.Get(), 0, motionBytes);
            if (f.guides & 2) b->CopyBufferRegion(f.buf.Get(), 2 * rows + motionBytes, s.bufDepth.Get(), 0, UINT64(s.depthPitch) * s.workH);
        }
        Draw(b, s.psoUp.Get(), BackRtv(bb), s.width, s.height, 1);
        t[0] = Transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        b->ResourceBarrier(1, t);
    }
    b->Close();
    ID3D12CommandList* lb[] = {b};
    s.queue->ExecuteCommandLists(1, lb);
    s.queue->Signal(s.done.Get(), n);
    ++s.stats.frames;
    if (s.capturing && ++s.captureAt == s.capture.size()) CaptureFinish();
    return true;
}

bool GetGuideTargets(GuideTargets* out) {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!out || !s.ready || s.stuck || !s.width || !s.bufMotion || !s.ptrMotion) return false;
    out->motion = s.bufMotion.Get();
    out->depth = s.ptrDepth ? s.bufDepth.Get() : nullptr;
    out->width = s.workW; out->height = s.workH;
    out->motionPitch = s.motionPitch; out->depthPitch = s.depthPitch;
    return true;
}

void GuidesStaged(bool motion, bool depth) {
    S().guides = (motion ? 1 : 0) | (motion && depth ? 2 : 0);
}

void RequestCapture(int frames) { S().captureRequest = frames; }

void Flush() {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.ready) WaitDone(s.frame);
}

void ForgetBackBuffers() {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ready) return;
    WaitDone(s.frame);
    memset(s.backRtv, 0, sizeof(s.backRtv));
    s.backRtvNext = 0;
}

void SetLog(void (*log)(const char*)) { g_log = log; }

void Shutdown() {
    State& s = S();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.ready) return;
    WaitDone(s.frame);
    s.stop = true;
    if (s.watchdog.joinable()) s.watchdog.join();
    DropSized();
    if (s.semStaged) s.destroySemaphore(s.semStaged);
    if (s.semResult) s.destroySemaphore(s.semResult);
    s.semStaged = s.semResult = nullptr;
    s.reDestroy();
    s.staged.Reset(); s.result.Reset(); s.done.Reset();
    if (s.doneEvent) CloseHandle(s.doneEvent);
    s.doneEvent = nullptr;
    for (auto& r : s.alloc) for (auto& a : r) a.Reset();
    for (auto& l : s.list) l.Reset();
    s.psoDown.Reset(); s.psoUp.Reset(); s.root.Reset(); s.srvHeap.Reset(); s.rtvHeap.Reset();
    s.psoUpFormat = DXGI_FORMAT_UNKNOWN;
    s.queue.Reset(); s.device.Reset();
    s.ready = false;
    s.stats.active = false;
}

const Stats& GetStats() { return S().stats; }

}  // namespace final_picture
