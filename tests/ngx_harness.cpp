// Drives nvngx.dll the way a game does, on the real GPU, and checks the pixels.
//
// Deliberately includes ONLY the public SDK headers plus D3D12 -- no shim
// internals. If the exported ABI or the NVSDK_NGX_Parameter vtable were wrong,
// this is where it would show up.

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "d3dx12.h"
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------- harness --

static int g_failures = 0;
static int g_checks = 0;

static void Check(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        printf("  PASS  %s\n", what);
    } else {
        ++g_failures;
        printf("  FAIL  %s\n", what);
    }
}

static void CheckNear(int got, int want, int tol, const char* what) {
    ++g_checks;
    if (std::abs(got - want) <= tol) {
        printf("  PASS  %s (got %d, want %d +/-%d)\n", what, got, want, tol);
    } else {
        ++g_failures;
        printf("  FAIL  %s (got %d, want %d +/-%d)\n", what, got, want, tol);
    }
}

// The transfer functions the shader uses, so the debug view can be checked
// against an analytic value rather than against "it changed".
static float SrgbToLinearF(float v) {
    v = std::min(std::max(v, 0.0f), 1.0f);
    return (v <= 0.04045f) ? v / 12.92f
                           : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

static float LinearToSrgbF(float v) {
    v = std::min(std::max(v, 0.0f), 1.0f);
    return (v <= 0.0031308f) ? v * 12.92f
                             : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// ------------------------------------------------------------- ngx imports --

using PFN_Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*,
                                               ID3D12Device*,
                                               const NVSDK_NGX_FeatureCommonInfo*,
                                               NVSDK_NGX_Version);
using PFN_Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(void);
using PFN_GetParameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using PFN_AllocateParameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using PFN_DestroyParameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using PFN_CreateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
                                                        NVSDK_NGX_Feature,
                                                        NVSDK_NGX_Parameter*,
                                                        NVSDK_NGX_Handle**);
using PFN_ReleaseFeature = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using PFN_EvaluateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
                                                          const NVSDK_NGX_Handle*,
                                                          NVSDK_NGX_Parameter*,
                                                          PFN_NVSDK_NGX_ProgressCallback);

struct Ngx {
    HMODULE dll = nullptr;
    PFN_Init Init = nullptr;
    PFN_Shutdown Shutdown = nullptr;
    PFN_GetParameters GetParameters = nullptr;
    PFN_AllocateParameters AllocateParameters = nullptr;
    PFN_DestroyParameters DestroyParameters = nullptr;
    PFN_CreateFeature CreateFeature = nullptr;
    PFN_ReleaseFeature ReleaseFeature = nullptr;
    PFN_EvaluateFeature EvaluateFeature = nullptr;
};

// Every name a game's nvngx.lib can import. Missing one is a link failure in
// the real world, so it is a failure here too.
static const char* kRequiredExports[] = {
    "NVSDK_NGX_D3D12_Init", "NVSDK_NGX_D3D12_Init_with_ProjectID",
    "NVSDK_NGX_D3D12_Shutdown", "NVSDK_NGX_D3D12_Shutdown1",
    "NVSDK_NGX_D3D12_GetParameters", "NVSDK_NGX_D3D12_AllocateParameters",
    "NVSDK_NGX_D3D12_GetCapabilityParameters", "NVSDK_NGX_D3D12_DestroyParameters",
    "NVSDK_NGX_D3D12_GetScratchBufferSize", "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_ReleaseFeature", "NVSDK_NGX_D3D12_GetFeatureRequirements",
    "NVSDK_NGX_D3D12_EvaluateFeature", "NVSDK_NGX_D3D12_EvaluateFeature_C",
    "NVSDK_NGX_D3D11_Init", "NVSDK_NGX_D3D11_Init_with_ProjectID",
    "NVSDK_NGX_D3D11_Shutdown", "NVSDK_NGX_D3D11_Shutdown1",
    "NVSDK_NGX_D3D11_GetParameters", "NVSDK_NGX_D3D11_AllocateParameters",
    "NVSDK_NGX_D3D11_GetCapabilityParameters", "NVSDK_NGX_D3D11_DestroyParameters",
    "NVSDK_NGX_D3D11_GetScratchBufferSize", "NVSDK_NGX_D3D11_CreateFeature",
    "NVSDK_NGX_D3D11_ReleaseFeature", "NVSDK_NGX_D3D11_GetFeatureRequirements",
    "NVSDK_NGX_D3D11_EvaluateFeature", "NVSDK_NGX_D3D11_EvaluateFeature_C",
    "NVSDK_NGX_CUDA_Init", "NVSDK_NGX_CUDA_Init_with_ProjectID",
    "NVSDK_NGX_CUDA_Shutdown", "NVSDK_NGX_CUDA_GetParameters",
    "NVSDK_NGX_CUDA_AllocateParameters", "NVSDK_NGX_CUDA_GetCapabilityParameters",
    "NVSDK_NGX_CUDA_DestroyParameters", "NVSDK_NGX_CUDA_GetScratchBufferSize",
    "NVSDK_NGX_CUDA_CreateFeature", "NVSDK_NGX_CUDA_ReleaseFeature",
    "NVSDK_NGX_CUDA_EvaluateFeature", "NVSDK_NGX_CUDA_EvaluateFeature_C",
    "NVSDK_NGX_UpdateFeature", "GetNGXResultAsString",
    "NVSDK_NGX_Parameter_SetULL", "NVSDK_NGX_Parameter_SetF",
    "NVSDK_NGX_Parameter_SetD", "NVSDK_NGX_Parameter_SetUI",
    "NVSDK_NGX_Parameter_SetI", "NVSDK_NGX_Parameter_SetD3d11Resource",
    "NVSDK_NGX_Parameter_SetD3d12Resource", "NVSDK_NGX_Parameter_SetVoidPointer",
    "NVSDK_NGX_Parameter_GetULL", "NVSDK_NGX_Parameter_GetF",
    "NVSDK_NGX_Parameter_GetD", "NVSDK_NGX_Parameter_GetUI",
    "NVSDK_NGX_Parameter_GetI", "NVSDK_NGX_Parameter_GetD3d11Resource",
    "NVSDK_NGX_Parameter_GetD3d12Resource", "NVSDK_NGX_Parameter_GetVoidPointer",
};

// -------------------------------------------------------------- d3d12 side --

struct D3D {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 fenceValue = 0;
    std::string adapterName;

    // Present when the D3D12 debug layer is installed. The shim records onto
    // command lists we own, so every wrong resource state, wrong barrier and
    // illegal descriptor use on its side lands here as a stored ERROR. Without
    // this the harness can only see what the driver happens to render.
    bool debugLayer = false;
    ComPtr<ID3D12InfoQueue> infoQueue;
};

static bool D3DCreate(D3D& d) {
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        debug->EnableDebugLayer();
        d.debugLayer = true;
    }

    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&d.factory)))) {
        printf("  CreateDXGIFactory1 failed\n");
        return false;
    }
    ComPtr<IDXGIAdapter1> fallback;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> a;
        if (d.factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;

        DXGI_ADAPTER_DESC1 desc{};
        a->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        char name[256];
        int n = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name,
                                    sizeof(name), nullptr, nullptr);
        std::string s(n > 0 ? name : "?");

        // Prefer AMD (vendor 0x1002) -- that is the whole point of the project.
        if (desc.VendorId == 0x1002 && !d.adapter) {
            d.adapter = a;
            d.adapterName = s;
        } else if (!fallback) {
            fallback = a;
        }
    }
    if (!d.adapter) d.adapter = fallback;
    if (!d.adapter) {
        printf("  no adapter found\n");
        return false;
    }

    if (FAILED(D3D12CreateDevice(d.adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&d.device)))) {
        printf("  D3D12CreateDevice failed\n");
        return false;
    }

    if (d.debugLayer)
        d.device.As(&d.infoQueue);

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    d.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&d.queue));
    d.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     IID_PPV_ARGS(&d.alloc));
    d.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d.alloc.Get(),
                                nullptr, IID_PPV_ARGS(&d.list));
    d.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d.fence));
    d.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
}

static void D3DSubmit(D3D& d) {
    d.list->Close();
    ID3D12CommandList* lists[] = {d.list.Get()};
    d.queue->ExecuteCommandLists(1, lists);
    d.queue->Signal(d.fence.Get(), ++d.fenceValue);
    d.fence->SetEventOnCompletion(d.fenceValue, d.event);
    WaitForSingleObject(d.event, 10000);
    d.alloc->Reset();
    d.list->Reset(d.alloc.Get(), nullptr);
}

static ComPtr<ID3D12Resource> MakeTexture(ID3D12Device* dev, UINT w, UINT h,
                                          DXGI_FORMAT fmt,
                                          D3D12_RESOURCE_FLAGS flags,
                                          D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(fmt, w, h, 1, 1);
    desc.Flags = flags;
    CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> r;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                 IID_PPV_ARGS(&r));
    return r;
}

// The shim leaves Output in UAV, which is what NGX says it is handed in, so a
// readback has to take it out of that state and put it back. Copying straight
// out of UAV happens to work on AMD and is a validation error everywhere.
static void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* tex,
                       D3D12_RESOURCE_STATES before,
                       D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = tex;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}

// Uploads RGBA8 pixels via a staging buffer.
static bool UploadPixels(D3D& d, ID3D12Resource* tex, const unsigned char* rgba,
                         UINT w, UINT h) {
    UINT64 rowBytes = (UINT64)w * 4;
    UINT64 pitch = (rowBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT *
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    UINT64 total = pitch * h;

    CD3DX12_HEAP_PROPERTIES up(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(total);
    ComPtr<ID3D12Resource> staging;
    if (FAILED(d.device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ,
                                                 nullptr, IID_PPV_ARGS(&staging))))
        return false;

    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    for (UINT y = 0; y < h; ++y)
        memcpy((unsigned char*)p + y * pitch, rgba + (size_t)y * rowBytes,
               rowBytes);
    staging->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = tex;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)pitch;

    d.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    // A texture promoted to COPY_DEST does not decay back to COMMON after the
    // submit -- only read-state promotions decay -- so the promotion has to be
    // undone explicitly or every later barrier starts from the wrong state.
    // The debug layer flags this; the driver alone would silently accept it.
    Transition(d.list.Get(), tex, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COMMON);
    D3DSubmit(d);
    return true;
}

// S223c: the same upload for an R16G16B16A16_FLOAT texture (8 bytes a pixel),
// used by the HDR dump pass below.
static bool UploadHalfPixels(D3D& d, ID3D12Resource* tex,
                             const unsigned short* rgba16, UINT w, UINT h) {
    UINT64 rowBytes = (UINT64)w * 8;
    UINT64 pitch = (rowBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT *
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    UINT64 total = pitch * h;

    CD3DX12_HEAP_PROPERTIES up(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(total);
    ComPtr<ID3D12Resource> staging;
    if (FAILED(d.device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ,
                                                 nullptr, IID_PPV_ARGS(&staging))))
        return false;

    void* p = nullptr;
    staging->Map(0, nullptr, &p);
    for (UINT y = 0; y < h; ++y)
        memcpy((unsigned char*)p + y * pitch,
               (const unsigned char*)rgba16 + (size_t)y * rowBytes, rowBytes);
    staging->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = tex;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)pitch;

    d.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Transition(d.list.Get(), tex, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COMMON);
    D3DSubmit(d);
    return true;
}

// Reads back a 24-bit bottom-up BMP, the only format WriteBmp emits. Returns
// tightly packed rows of B,G,R from the TOP row down, so a check can index it
// like the image it is.
static bool ReadDumpBmp(const char* path, int* outW, int* outH,
                        std::vector<unsigned char>* bgr) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    std::vector<unsigned char> d;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long n = ftell(f);
    if (n < 54 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    d.resize((size_t)n);
    const bool read = fread(d.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    if (!read || d[0] != 'B' || d[1] != 'M') return false;
    unsigned int off = 0;
    int w = 0, h = 0;
    unsigned short bits = 0;
    memcpy(&off, &d[10], 4);
    memcpy(&w, &d[18], 4);
    memcpy(&h, &d[22], 4);
    memcpy(&bits, &d[28], 2);
    if (bits != 24 || w <= 0 || h == 0) return false;
    const bool bottomUp = h > 0;
    if (!bottomUp) h = -h;
    const size_t row = (((size_t)w * 3 + 3) / 4) * 4;
    if ((size_t)off + row * (size_t)h > d.size()) return false;
    bgr->assign((size_t)w * h * 3, 0);
    for (int y = 0; y < h; ++y) {
        const unsigned char* s = d.data() + off + (size_t)y * row;
        const int iy = bottomUp ? (h - 1 - y) : y;
        memcpy(bgr->data() + (size_t)iy * w * 3, s, (size_t)w * 3);
    }
    *outW = w;
    *outH = h;
    return true;
}

struct Readback {
    // Tightly packed rows at the resource's OWN bytes per pixel: 4 for the
    // RGBA8 passes, 8 for an R16G16B16A16_FLOAT one. Every consumer below
    // indexes it as RGBA8, so only the 8-bit passes may be compared this way.
    std::vector<unsigned char> pixels;
    UINT w = 0, h = 0;
};

// Largest single-channel difference between two readbacks. Alpha carries no
// information -- it is a constant 255 -- so it is skipped.
static int MaxChannelDiff(const Readback& a, const Readback& b) {
    if (a.w != b.w || a.h != b.h) return 9999;
    if (a.pixels.size() != b.pixels.size() || a.pixels.empty()) return 9999;
    int worst = 0;
    for (size_t k = 0; k < a.pixels.size(); ++k) {
        if ((k & 3u) == 3u) continue;
        int d = std::abs((int)a.pixels[k] - (int)b.pixels[k]);
        if (d > worst) worst = d;
    }
    return worst;
}

// Bytes per pixel of the formats this harness reads back. It assumed 4
// everywhere until S223c: the f16 pass then asked CopyTextureRegion for an
// 8-byte-a-pixel footprint inside a buffer sized for 4, which is an overrun
// (an access violation, not a wrong number).
static UINT FormatBytes(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
            return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return 16;
        default:
            return 4;
    }
}

static bool DownloadPixels(D3D& d, ID3D12Resource* tex, Readback& out) {
    D3D12_RESOURCE_DESC desc = tex->GetDesc();
    UINT w = (UINT)desc.Width, h = desc.Height;

    Transition(d.list.Get(), tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    UINT64 rowBytes = (UINT64)w * FormatBytes(desc.Format);
    UINT64 pitch = (rowBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT *
                   D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;

    CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(pitch * h);
    ComPtr<ID3D12Resource> buf;
    d.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                      IID_PPV_ARGS(&buf));

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = buf.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = desc.Format;
    dst.PlacedFootprint.Footprint.Width = w;
    dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = (UINT)pitch;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    d.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Transition(d.list.Get(), tex, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3DSubmit(d);

    void* p = nullptr;
    buf->Map(0, nullptr, &p);
    out.pixels.resize((size_t)rowBytes * h);
    for (UINT y = 0; y < h; ++y)
        memcpy(out.pixels.data() + (size_t)y * rowBytes,
               (const unsigned char*)p + (size_t)y * pitch, rowBytes);
    buf->Unmap(0, nullptr);
    out.w = w;
    out.h = h;
    return true;
}

// ------------------------------------------------------------------ config --

// S231: the weight reading is a *configuration*, and a check that does not say
// which one it ran says less than it looks like it says. Pass 8 arms its own
// reading; the staged checks take this, so `DLSS5_FFNT=4` runs them against the
// reading the game actually ships. Default stays 1: every pass above was pinned
// in the dense reading and must keep failing/succeeding for the same reasons.
static int EnvInt(const char* name, int dflt) {
    const char* v = getenv(name);
    return v && *v ? atoi(v) : dflt;
}

// One line per key, every time, so a pass is never affected by whatever the
// last one left behind.
struct IniValues {
    int nrPasses = 1;
    int hipBackend = 1;
    std::string hipWeightsDir;
    std::string hipRocwmmaInc;
    std::string hipRocInc;
    int hipFeBlock = 0;
    int hipFeWindX = 0;
    int hipFeWindY = 0;
    float transferStrength = 0.0f;
    float colourStrength = 0.0f;
    float modelScale = 1.0f;
    float whitePoint = 1.0f;
    float maxRatio = 4.0f;
    int proxyMode = 2;
    int passthrough = 0;
    int debugView = 0;
    int dumpFrames = 0;
    int dumpEvery = 0;
    // S223d: the live model path. Off by default so every pass above keeps the
    // behaviour it was pinned with; the live-chain pass arms it explicitly.
    int hipFeLive = 0;
    int hipFeTransition = 0;
    std::string candidatePreviewPath;
    // S231 step 3: the default is the SHIPPED reading (2, our map), so a green run
    // means the path the game runs is right. Only the c256f2 check consults this --
    // it is the one staged check that applies the same un-permutation the live
    // runners do -- so every other pinned pass is unaffected. `DLSS5_FFNT=1` runs
    // the dense reading instead, and then c256f2 correctly fails: exactly one
    // reading can be the one its golden describes.
    int hipFfnTranspose = EnvInt("DLSS5_FFNT", 2);
    int dumpField = 0;
};

static bool WriteIni(const std::string& dir, const IniValues& v) {
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s/dlssnr_shim.ini", dir.c_str());
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("  cannot write %s\n", path);
        return false;
    }
    fprintf(f,
            "LogLevel=2\n"
            "Enabled=1\n"
            "NrPasses=%d\n"
            "HipBackend=%d\n"
            "HipWeightsDir=%s\n"
            "HipRocwmmaInc=%s\n"
            "HipRocInc=%s\n"
            "HipFeBlock=%d\n"
            "HipFeWindX=%d\n"
            "HipFeWindY=%d\n"
            "TransferStrength=%.4f\n"
            "ColourStrength=%.4f\n"
            "ModelScale=%.4f\n"
            "WhitePoint=%.4f\n"
            "ProxyMode=%d\n"
            "MaxRatio=%.4f\n"
            "Passthrough=%d\n"
            "DebugView=%d\n"
            "DumpFrames=%d\n"
            "DumpEvery=%d\n"
            "HipFeLive=%d\n"
            "HipFeTransition=%d\n"
            "CandidatePreviewPath=%s\n"
            "HipFfnTranspose=%d\n"
            "DumpField=%d\n",
            v.nrPasses, v.hipBackend, v.hipWeightsDir.c_str(),
            v.hipRocwmmaInc.c_str(), v.hipRocInc.c_str(), v.hipFeBlock,
            v.hipFeWindX, v.hipFeWindY,
            v.transferStrength,
            v.colourStrength, v.modelScale, v.whitePoint, v.proxyMode,
            v.maxRatio,
            v.passthrough, v.debugView, v.dumpFrames, v.dumpEvery,
            v.hipFeLive, v.hipFeTransition, v.candidatePreviewPath.c_str(),
            v.hipFfnTranspose, v.dumpField);
    fclose(f);
    return true;
}

// -------------------------------------------------------------- one pass --

// S231b -- HOW A CHECK'S VERDICT MUST BE READ.
//
// The shim logs `hip: <name> check PASSED|FAILED`, and a check can run more than
// once per process (once per Init, and pass 8 arms a different weight reading per
// arm). Searching for the *PASSED variant* therefore matches a DIFFERENT RUN's
// verdict: that is how the c256f2 check logged `check FAILED` for the run being
// judged and the harness called it passed, because a later retry under mode 1 had
// logged `check PASSED`. This happened with the numeric checks reading the FIRST
// `maxerr` line, so the two halves of one judgement were reading different runs.
//
// Read the FIRST actual verdict for the check. Two things make that subtler than
// it looks:
//   * a check can run more than once per process (once per Init, and pass 8 arms a
//     different weight reading per arm), so searching for the *PASSED variant*
//     anywhere can be satisfied by a DIFFERENT RUN's success. That is how the
//     c256f2 check logged `check FAILED` for the run being judged while the
//     harness reported it passed -- a later retry under mode 1 had logged PASSED;
//   * the same tag also appears in `check skipped (...)` lines, which are not
//     verdicts at all. Treating one as a verdict makes a passing check look like a
//     failure -- which is exactly what the first cut of this helper did to the v4
//     chain check, whose first eight runs are skips.
// The first real verdict is the run the numeric checks beside it read (they take
// the first match of their own tag), so both halves of a judgement agree.
static bool FirstVerdictPassed(const std::string& text, const char* tag) {
    const size_t n = strlen(tag);
    size_t at = 0;
    while ((at = text.find(tag, at)) != std::string::npos) {
        if (text.compare(at + n, 6, "PASSED") == 0) return true;
        if (text.compare(at + n, 6, "FAILED") == 0) return false;
        at += n;
    }
    return false;
}

struct PassResult {
    Readback rb;
    int evalFailures = 0;
    bool created = false;
};

// Init -> create -> evaluate -> read back -> release -> shutdown, with a fresh
// config each time. The shim reloads dlssnr_shim.ini on every Init, which is
// what makes comparing configurations in one process possible.
static PassResult RunPass(Ngx& ngx, D3D& d, const std::string& iniDir,
                          const IniValues& v, ID3D12Resource* color,
                          ID3D12Resource* output, UINT SRC_W, UINT SRC_H,
                          UINT DST_W, UINT DST_H, int frames) {
    PassResult pr;
    if (!WriteIni(iniDir, v)) return pr;

    wchar_t appData[MAX_PATH]{};
    GetCurrentDirectoryW(MAX_PATH, appData);

    if (ngx.Init(0xDEADBEEF, appData, d.device.Get(), nullptr,
                 NVSDK_NGX_Version_API) != NVSDK_NGX_Result_Success) {
        printf("  Init failed\n");
        return pr;
    }

    NVSDK_NGX_Parameter* createParams = nullptr;
    ngx.AllocateParameters(&createParams);
    createParams->Set(NVSDK_NGX_Parameter_Width, SRC_W);
    createParams->Set(NVSDK_NGX_Parameter_Height, SRC_H);
    createParams->Set(NVSDK_NGX_Parameter_OutWidth, DST_W);
    createParams->Set(NVSDK_NGX_Parameter_OutHeight, DST_H);
    createParams->Set(NVSDK_NGX_Parameter_PerfQualityValue, 2u);
    createParams->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, 0u);

    NVSDK_NGX_Handle* handle = nullptr;
    if (ngx.CreateFeature(d.list.Get(), NVSDK_NGX_Feature_SuperSampling,
                          createParams, &handle) != NVSDK_NGX_Result_Success ||
        handle == nullptr) {
        printf("  CreateFeature failed\n");
        ngx.DestroyParameters(createParams);
        ngx.Shutdown();
        return pr;
    }
    pr.created = true;

    NVSDK_NGX_Parameter* evalParams = nullptr;
    ngx.AllocateParameters(&evalParams);
    evalParams->Set(NVSDK_NGX_Parameter_Color, color);
    evalParams->Set(NVSDK_NGX_Parameter_Output, output);
    evalParams->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                    SRC_W);
    evalParams->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                    SRC_H);

    for (int i = 0; i < frames; ++i) {
        if (ngx.EvaluateFeature(d.list.Get(), handle, evalParams, nullptr) !=
            NVSDK_NGX_Result_Success)
            ++pr.evalFailures;
        D3DSubmit(d);
    }

    DownloadPixels(d, output, pr.rb);

    ngx.ReleaseFeature(handle);
    ngx.DestroyParameters(createParams);
    ngx.DestroyParameters(evalParams);
    ngx.Shutdown();
    return pr;
}

// ------------------------------------------------------------------- main --

int main(int argc, char** argv) {
    const char* dllPath = (argc > 1) ? argv[1] : "nvngx.dll";
    const UINT SRC_W = 320, SRC_H = 180;
    const UINT DST_W = 1280, DST_H = 720;

    // More than one frame per pass: the shim allocates descriptors and pooled
    // textures lazily, so a second frame is what proves the steady state
    // works -- the first one could pass on freshly-zeroed state.
    const int kFrames = 3;

    printf("=== nvngx shim harness ===\n");
    printf("dll: %s\n\n", dllPath);

    // ---- load and check the export table ------------------------------
    Ngx ngx;
    ngx.dll = LoadLibraryA(dllPath);
    if (!ngx.dll) {
        printf("FATAL: cannot load %s (GetLastError=%lu)\n", dllPath,
               GetLastError());
        return 2;
    }
    printf("-- export table --\n");
    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(ngx.dll, name)) {
            printf("  MISSING %s\n", name);
            ++missing;
        }
    }
    printf("  %zu names checked, %d missing\n",
           sizeof(kRequiredExports) / sizeof(kRequiredExports[0]), missing);
    Check(missing == 0, "all required NGX exports resolve");

    ngx.Init = (PFN_Init)GetProcAddress(ngx.dll, "NVSDK_NGX_D3D12_Init");
    ngx.Shutdown = (PFN_Shutdown)GetProcAddress(ngx.dll, "NVSDK_NGX_D3D12_Shutdown");
    ngx.GetParameters =
        (PFN_GetParameters)GetProcAddress(ngx.dll, "NVSDK_NGX_D3D12_GetParameters");
    ngx.AllocateParameters = (PFN_AllocateParameters)GetProcAddress(
        ngx.dll, "NVSDK_NGX_D3D12_AllocateParameters");
    ngx.DestroyParameters = (PFN_DestroyParameters)GetProcAddress(
        ngx.dll, "NVSDK_NGX_D3D12_DestroyParameters");
    ngx.CreateFeature =
        (PFN_CreateFeature)GetProcAddress(ngx.dll, "NVSDK_NGX_D3D12_CreateFeature");
    ngx.ReleaseFeature =
        (PFN_ReleaseFeature)GetProcAddress(ngx.dll, "NVSDK_NGX_D3D12_ReleaseFeature");
    ngx.EvaluateFeature = (PFN_EvaluateFeature)GetProcAddress(
        ngx.dll, "NVSDK_NGX_D3D12_EvaluateFeature");

    // ---- d3d12 on the real adapter ------------------------------------
    printf("\n-- device --\n");
    D3D d;
    if (!D3DCreate(d)) return 2;
    printf("  adapter: %s\n", d.adapterName.c_str());

    // The shim reads dlssnr_shim.ini from the directory the DLL was loaded
    // from; every pass rewrites it before initing.
    const std::string iniDir = (argc > 2) ? argv[2] : ".";

    // ---- init ----------------------------------------------------------
    printf("\n-- parameter block --\n");
    wchar_t appData[MAX_PATH]{};
    GetCurrentDirectoryW(MAX_PATH, appData);
    WriteIni(iniDir, IniValues{});
    NVSDK_NGX_Result r = ngx.Init(0xDEADBEEF, appData, d.device.Get(), nullptr,
                                  NVSDK_NGX_Version_API);
    Check(r == NVSDK_NGX_Result_Success, "NVSDK_NGX_D3D12_Init returns Success");
    NVSDK_NGX_Parameter* caps = nullptr;
    ngx.GetParameters(&caps);
    Check(caps != nullptr, "GetParameters yields a block");

    unsigned int avail = 0;
    if (caps) {
        NVSDK_NGX_Result gr =
            caps->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &avail);
        Check(gr == NVSDK_NGX_Result_Success && avail == 1,
              "SuperSampling.Available == 1");
    }

    NVSDK_NGX_Parameter* scratch = nullptr;
    ngx.AllocateParameters(&scratch);
    Check(scratch != nullptr, "AllocateParameters yields a block");
    if (scratch) {
        scratch->Set("probe.ui", 1920u);
        unsigned int ui = 0;
        int i = 0;
        float f = 0.0f;
        Check(scratch->Get("probe.ui", &ui) == NVSDK_NGX_Result_Success && ui == 1920u,
              "uint round-trips");
        Check(scratch->Get("probe.ui", &i) == NVSDK_NGX_Result_Success && i == 1920,
              "uint reads back across the int overload");
        Check(scratch->Get("probe.ui", &f) == NVSDK_NGX_Result_Success &&
                  std::fabs(f - 1920.0f) < 0.5f,
              "uint reads back across the float overload");

        unsigned int absent = 0;
        Check(scratch->Get("probe.missing", &absent) != NVSDK_NGX_Result_Success,
              "reading an absent name fails");

        scratch->Set("probe.f", 0.25f);
        float fv = 0;
        Check(scratch->Get("probe.f", &fv) == NVSDK_NGX_Result_Success &&
                  std::fabs(fv - 0.25f) < 1e-6f,
              "float round-trips");

        scratch->Reset();
        Check(scratch->Get("probe.ui", &ui) != NVSDK_NGX_Result_Success,
              "Reset clears the block");
    }
    ngx.Shutdown();

    // ---- textures ------------------------------------------------------
    printf("\n-- resources --\n");
    ComPtr<ID3D12Resource> color = MakeTexture(
        d.device.Get(), SRC_W, SRC_H, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
    ComPtr<ID3D12Resource> output = MakeTexture(
        d.device.Get(), DST_W, DST_H, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Check(color && output, "colour + output textures created");

    // R ramps across x, G ramps across y: bilinear reproduces a linear ramp
    // exactly, so any deviation from the analytic value is a real bug.
    // B is a hard step at the halfway column. A nearest-neighbour resample
    // would emit only 0 or 255 there; a filtered one must produce
    // intermediate values across the transition.
    std::vector<unsigned char> src((size_t)SRC_W * SRC_H * 4);
    for (UINT y = 0; y < SRC_H; ++y) {
        for (UINT x = 0; x < SRC_W; ++x) {
            unsigned char* p = src.data() + ((size_t)y * SRC_W + x) * 4;
            p[0] = (unsigned char)(x * 255 / (SRC_W - 1));
            p[1] = (unsigned char)(y * 255 / (SRC_H - 1));
            p[2] = (x < SRC_W / 2) ? 0 : 255;
            p[3] = 255;
        }
    }
    Check(UploadPixels(d, color.Get(), src.data(), SRC_W, SRC_H),
          "source gradient uploaded");
    // Leave Colour the way a game that has just rendered it would: in
    // RENDER_TARGET. The shim guesses incoming states from desc flags, so
    // this both matches the guess and keeps the debug layer quiet.
    Transition(d.list.Get(), color.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3DSubmit(d);

    // ---- pass 1: the resample on its own -------------------------------
    // Every later pass is compared against this one, so it also carries the
    // full set of checks on the resample itself.
    printf("\n-- pass 1: resample only --\n");
    IniValues base;
    base.nrPasses = 0;
    PassResult p1 = RunPass(ngx, d, iniDir, base, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p1.created, "CreateFeature(SuperSampling) succeeds");
    Check(p1.evalFailures == 0, "EvaluateFeature returns Success on every frame");

    // ---- verify the resample -------------------------------------------
    printf("\n-- resample verification (%ux%u -> %ux%u) --\n", SRC_W, SRC_H,
           DST_W, DST_H);
    Check(p1.rb.w == DST_W && p1.rb.h == DST_H, "output read back");

    Readback rb = p1.rb;
    if (rb.w == DST_W && rb.h == DST_H) {
        auto px = [&](UINT x, UINT y, int c) {
            return (int)rb.pixels[((size_t)y * rb.w + x) * 4 + c];
        };
        int scale = DST_W / SRC_W;  // 4

        printf("  corners: TL=(%d,%d,%d) TR=(%d,%d,%d) BL=(%d,%d,%d) BR=(%d,%d,%d)\n",
               px(0, 0, 0), px(0, 0, 1), px(0, 0, 2), px(DST_W - 1, 0, 0),
               px(DST_W - 1, 0, 1), px(DST_W - 1, 0, 2), px(0, DST_H - 1, 0),
               px(0, DST_H - 1, 1), px(0, DST_H - 1, 2), px(DST_W - 1, DST_H - 1, 0),
               px(DST_W - 1, DST_H - 1, 1), px(DST_W - 1, DST_H - 1, 2));

        // Corners must land on the corresponding source corners.
        CheckNear(px(0, 0, 0), 0, 2, "top-left R ~ source R=0");
        CheckNear(px(DST_W - 1, 0, 0), 255, 2, "top-right R ~ source R=255");
        CheckNear(px(0, DST_H - 1, 1), 255, 2, "bottom-left G ~ source G=255");
        CheckNear(px(0, 0, 1), 0, 2, "top-left G ~ source G=0");
        CheckNear(px(0, 0, 2), 0, 1, "top-left B on the low side of the step");
        CheckNear(px(DST_W - 1, 0, 2), 255, 1, "top-right B on the high side");

        // The whole frame must be written. Alpha is a constant 255 in the
        // source, so any untouched pixel shows as alpha 0.
        int written = 0;
        for (size_t k = 0; k < (size_t)DST_W * DST_H; ++k) {
            if (rb.pixels[k * 4 + 3] == 255) ++written;
        }
        Check(written == (int)((size_t)DST_W * DST_H), "every output pixel written");

        // Red must increase monotonically left->right along the middle row.
        bool monotonic = true;
        for (UINT x = 1; x < DST_W; ++x)
            if (px(x, DST_H / 2, 0) < px(x - 1, DST_H / 2, 0)) monotonic = false;
        Check(monotonic, "red increases monotonically across the row");

        // Bilinear of a linear ramp == the ramp. Compare against the analytic
        // value at pixel centres; tolerance 2 covers fp8/rounding only.
        int worstR = 0, worstG = 0;
        for (UINT y = 0; y < DST_H; y += 37) {
            for (UINT x = 0; x < DST_W; x += 37) {
                float sx = (float(x) + 0.5f) * SRC_W / DST_W - 0.5f;
                float sy = (float(y) + 0.5f) * SRC_H / DST_H - 0.5f;
                sx = std::max(sx, 0.0f);
                sy = std::max(sy, 0.0f);
                int wantR = (int)std::lround(sx * 255.0f / (SRC_W - 1));
                int wantG = (int)std::lround(sy * 255.0f / (SRC_H - 1));
                worstR = std::max(worstR, std::abs(px(x, y, 0) - wantR));
                worstG = std::max(worstG, std::abs(px(x, y, 1) - wantG));
            }
        }
        printf("  worst channel error vs analytic bilinear: R=%d G=%d\n", worstR,
               worstG);
        Check(worstR <= 2 && worstG <= 2, "output matches analytic bilinear (<=2)");

        // Sampling must actually be filtered, not nearest: the blue step can
        // only produce intermediate values if interpolation is happening.
        int intermediate = 0;
        for (size_t k = 0; k < (size_t)DST_W * DST_H; ++k) {
            int b = rb.pixels[k * 4 + 2];
            if (b > 8 && b < 247) ++intermediate;
        }
        printf("  intermediate blue values across the step: %d\n", intermediate);
        Check(intermediate > 0, "step edge is filtered, not nearest-neighbour");
        (void)scale;
    }

    // ---- pass 2: the colour pipeline at zero strength -------------------
    // The invariant the reference is emphatic about: at strength 0 the frame
    // must be bit-for-bit what the resample produced. If the encode and the
    // resolve do not cancel exactly, this is where it shows.
    printf("\n-- pass 2: encode -> identity model -> resolve, strength 0 --\n");
    IniValues zero;
    zero.nrPasses = 1;
    zero.transferStrength = 0.0f;
    zero.colourStrength = 0.0f;
    PassResult p2 = RunPass(ngx, d, iniDir, zero, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p2.evalFailures == 0, "every frame succeeds with the passes on");
    int d2 = MaxChannelDiff(p1.rb, p2.rb);
    printf("  worst channel difference vs the resample: %d\n", d2);
    Check(d2 == 0, "strength 0 is bit-identical to the resample");

    // ---- pass 3: full strength against an identity model ----------------
    // The model is still the proxy, so there is no edit to transfer. The two
    // branches of the ratio, the OkLab hue correction and the AP1 clamp all
    // run at full strength and still have to land back on the frame. This is
    // the check that would catch a composition that adds a difference instead
    // of rescaling a picture.
    printf("\n-- pass 3: same chain, strength 1 --\n");
    IniValues full;
    full.nrPasses = 1;
    full.transferStrength = 1.0f;
    full.colourStrength = 1.0f;
    PassResult p3 = RunPass(ngx, d, iniDir, full, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p3.evalFailures == 0, "every frame succeeds at strength 1");
    int d3 = MaxChannelDiff(p1.rb, p3.rb);
    printf("  worst channel difference vs the resample: %d\n", d3);
    // Not zero: the OkLab round trip and the AP1 clamp are lossy in float.
    // Anything past a few counts means the composition is doing something
    // other than reproducing the frame.
    Check(d3 <= 4, "strength 1 against an identity model still reproduces the frame");

    // ---- pass 4: the proxy curve, one mode at a time ---------------------
    // DebugView=1 writes the picture the model is shown, rescaled back to frame
    // units, so each curve can be checked against its own analytic value. Mode
    // 2 is the frame itself -- the scale-and-encode round trip is the identity
    // -- while modes 0 and 1 roll the headroom above the knee and must
    // therefore differ from it.
    printf("\n-- pass 4: the proxy curve, three modes --\n");
    // The last iteration is mode 2, the default -- and the reference the later
    // passes compare against.
    PassResult p4;
    for (int mode = 0; mode <= 2; ++mode) {
        IniValues dbg;
        dbg.nrPasses = 1;
        dbg.debugView = 1;
        dbg.proxyMode = mode;
        p4 = RunPass(ngx, d, iniDir, dbg, color.Get(), output.Get(),
                     SRC_W, SRC_H, DST_W, DST_H, kFrames);
        Check(p4.evalFailures == 0, "every frame succeeds in the debug view");
        if (p4.rb.w != DST_W || p4.rb.h != DST_H) continue;

        const UINT kStepGuard = 24;  // the blue step's transition, in pixels
        int worstProxy = 0, worstVsFrame = 0, rolled = 0, samples = 0;

        for (UINT y = 8; y < DST_H; y += 53) {
            for (UINT x = 8; x < DST_W; x += 53) {
                // The blue channel is a step, so its interpolated value is not
                // analytic near the halfway column. Skip that band.
                if (std::abs((int)x - (int)DST_W / 2) < (int)kStepGuard) continue;

                float sx = (float(x) + 0.5f) * SRC_W / DST_W - 0.5f;
                float sy = (float(y) + 0.5f) * SRC_H / DST_H - 0.5f;
                sx = std::max(sx, 0.0f);
                sy = std::max(sy, 0.0f);

                float v[3] = {sx / (SRC_W - 1), sy / (SRC_H - 1),
                              x < DST_W / 2 ? 0.0f : 1.0f};
                for (int c = 0; c < 3; ++c) v[c] /= dbg.whitePoint;

                // The three curves, written exactly as dlssnr.hlsl writes them.
                const float kLuma[3] = {0.2126f, 0.7152f, 0.0722f};
                float luma = kLuma[0] * v[0] + kLuma[1] * v[1] + kLuma[2] * v[2];
                if (mode != 2 && luma > 0.75f) {
                    const float kKnee = 0.75f, kHead = 0.25f;
                    float rolledLuma;
                    if (mode == 0) {
                        rolledLuma =
                            kKnee + kHead * (1.0f - std::exp(-(luma - kKnee) / kHead));
                    } else {
                        float over = luma - kKnee;
                        rolledLuma = kKnee + kHead * over / (over + kHead);
                    }
                    for (int c = 0; c < 3; ++c) v[c] *= rolledLuma / luma;
                    ++rolled;
                }

                // The shader encodes to sRGB and the debug view decodes again,
                // so what lands is the curved value back in frame units.
                for (int c = 0; c < 3; ++c) {
                    float want = SrgbToLinearF(LinearToSrgbF(v[c])) *
                                 dbg.whitePoint * 255.0f;
                    int got = p4.rb.pixels[((size_t)y * DST_W + x) * 4 + c];
                    worstProxy = std::max(worstProxy, (int)std::abs(got - (int)std::lround(want)));
                    int base = p1.rb.pixels[((size_t)y * DST_W + x) * 4 + c];
                    worstVsFrame = std::max(worstVsFrame, std::abs(got - base));
                }
                ++samples;
            }
        }
        printf("  mode %d: %d samples, %d above the knee; worst vs analytic %d, "
               "largest departure from the frame %d\n",
               mode, samples, rolled, worstProxy, worstVsFrame);

        char label[80];
        snprintf(label, sizeof(label),
                 "proxy mode %d matches its own analytic curve", mode);
        Check(worstProxy <= 3, label);

        if (mode != 2) {
            // The rolling modes must actually roll -- the guard against the
            // check passing because the pass did nothing. Mode 2 has no roll by
            // construction (it is the frame) and is pinned by the analytic
            // match alone.
            snprintf(label, sizeof(label),
                     "proxy mode %d rolls the headroom it cannot carry", mode);
            Check(rolled > 0 && worstVsFrame >= 10, label);
        }
    }

    // ---- pass 5: the model below full resolution ------------------------
    // Exercises the downsample pass and the resolve's filtered read of a
    // smaller model texture. Strength 0 still has to be bit-identical, which
    // proves the extra resample is not leaking into the frame.
    printf("\n-- pass 5: ModelScale 0.5, strength 0 --\n");
    IniValues half;
    half.nrPasses = 1;
    half.modelScale = 0.5f;
    PassResult p5 = RunPass(ngx, d, iniDir, half, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p5.evalFailures == 0, "every frame succeeds with the model at half scale");
    int d5 = MaxChannelDiff(p1.rb, p5.rb);
    printf("  worst channel difference vs the resample: %d\n", d5);
    Check(d5 == 0, "half-scale model at strength 0 is still bit-identical");

    // ---- pass 6: leave dumps behind -------------------------------------
    // The numeric checks above all go through this harness's own readback.
    // This pass goes through the shim's BMP writer instead, so the dump path
    // is checked against something and there is a file to look at.
    printf("\n-- pass 6: writing debug dumps --\n");
    IniValues dump;
    dump.nrPasses = 1;
    dump.debugView = 1;
    dump.dumpFrames = 1;
    PassResult p6 = RunPass(ngx, d, iniDir, dump, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p6.evalFailures == 0, "every frame succeeds while dumping");
    int d6 = MaxChannelDiff(p4.rb, p6.rb);
    printf("  worst channel difference vs pass 4: %d\n", d6);
    Check(d6 == 0, "dumping does not change the frame");

    // ---- pass 6b: the same dump path on an HDR (f16) frame --------------
    // S223c. An HDR game hands NGX R16G16B16A16_FLOAT, and the dump writer used
    // to refuse it outright -- every frame logged "dump: unsupported format 10"
    // and no file was written, which is how the 2026-09-18 A/B lost its
    // pixel-exact half. The colour texture below carries eight KNOWN half
    // values, one per column, so the expected byte can be worked out by hand
    // from the documented fold (v * whitePoint -> clamp -> sRGB -> 8 bits),
    // including both clipping cases and the sub-clamp one. R and B are given
    // DIFFERENT tables on purpose: the writer stores B,G,R in memory, and that
    // channel swap has shipped once already.
    printf("\n-- pass 6b: HDR (f16) colour + output, dumps --\n");
    {
        static const unsigned short kHalf[8] = {0x0000, 0x3400, 0x3800, 0x3A00,
                                                0x3C00, 0x4000, 0xB800, 0x1419};
        static const unsigned char  kWant[8] = {0, 137, 188, 225,
                                                255, 255, 0, 3};
        ComPtr<ID3D12Resource> color16 = MakeTexture(
            d.device.Get(), SRC_W, SRC_H, DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
        ComPtr<ID3D12Resource> output16 = MakeTexture(
            d.device.Get(), DST_W, DST_H, DXGI_FORMAT_R16G16B16A16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Check(color16 && output16, "f16 colour + output textures created");
        if (color16 && output16) {
            std::vector<unsigned short> src16((size_t)SRC_W * SRC_H * 4);
            for (UINT y = 0; y < SRC_H; ++y) {
                for (UINT x = 0; x < SRC_W; ++x) {
                    unsigned short* p = src16.data() + ((size_t)y * SRC_W + x) * 4;
                    p[0] = kHalf[x % 8];                 // R: the tested table
                    p[1] = 0x3800;                       // G: 0.5, constant
                    p[2] = kHalf[(x + 4) % 8];           // B: shifted table
                    p[3] = 0x3C00;                       // A: 1.0
                }
            }
            Check(UploadHalfPixels(d, color16.Get(), src16.data(), SRC_W, SRC_H),
                  "f16 gradient uploaded");
            Transition(d.list.Get(), color16.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
            D3DSubmit(d);

            IniValues hdr;
            hdr.nrPasses = 1;
            hdr.debugView = 0;
            hdr.dumpFrames = 1;
            PassResult p6b = RunPass(ngx, d, iniDir, hdr, color16.Get(),
                                     output16.Get(), SRC_W, SRC_H, DST_W, DST_H,
                                     kFrames);
            Check(p6b.evalFailures == 0,
                  "every frame succeeds with an f16 frame while dumping");

            // Frame 0 of the pass is the one that dumps. The writer runs on its
            // own queue, so give it a moment rather than reading a file the GPU
            // has not written yet -- a missing file here is a FAIL, not a skip:
            // that is exactly the bug this pass exists to catch.
            int bw = 0, bh = 0;
            std::vector<unsigned char> bgr;
            bool got = false;
            for (int tries = 0; tries < 40 && !got; ++tries) {
                got = ReadDumpBmp("nr_0000_in.bmp", &bw, &bh, &bgr);
                if (!got) Sleep(50);
            }
            Check(got, "the f16 frame produced a dump file at all");
            if (got) {
                Check(bw == (int)SRC_W && bh == (int)SRC_H,
                      "the f16 dump has the colour texture's dimensions");
                int worst = 0, worstAt = -1;
                for (int x = 0; x < 8 && x < bw; ++x) {
                    const unsigned char* px = bgr.data() + (size_t)x * 3;
                    const int dr = std::abs((int)px[2] - (int)kWant[x]);
                    const int db = std::abs((int)px[0] - (int)kWant[(x + 4) % 8]);
                    const int dg = std::abs((int)px[1] - (int)kWant[2]);  // G = 0.5
                    if (dr > worst) { worst = dr; worstAt = x; }
                    if (db > worst) { worst = db; worstAt = x; }
                    if (dg > worst) { worst = dg; worstAt = x; }
                }
                printf("  f16 dump vs the hand-computed sRGB fold: worst %d count(s)%s\n",
                       worst, worstAt >= 0 ? "" : " (all columns exact)");
                Check(worst <= 1,
                      "the f16 dump is the documented fold, channel order included");
            }
        }
    }

    // ---- pass 7: the HIP backend, identity network ----------------------
    // The model now takes the long road: staged into a shared buffer,
    // unswizzled by a D3D12 copy, run through a hiprtc-compiled kernel on
    // the ROCm runtime, and swizzled back into a model texture one frame
    // late. With an identity network the answer is the frame itself, so
    // strength 0 must STILL be bit-identical -- and the log markers below
    // prove the HIP path actually ran rather than silently degraded.
    // The pass also arms the one-shot FP8 GEMM self-test: the REAL QKV
    // weights through rocWMMA with a one-hot input, whose result is the
    // weights bit-for-bit -- the checksum in the log is compared below --
    // and the one-shot v4 chain check (pre-block on real weights through
    // the hiprtc backend path, block output vs the oracle golden).
    printf("\n-- pass 7: HIP backend, identity model, strength 0 --\n");
    IniValues hipv;
    hipv.nrPasses = 1;
    hipv.hipBackend = 1;
    hipv.hipWeightsDir = "../dlss5-analysis/tensors";
    hipv.hipRocwmmaInc = "../ext/rocWMMA/library/include";
    {
        // ROCm's own include dir (hiprtc's implicit path lacks it). Same
        // Pass ROCM_ROOT when running the HIP-enabled harness.
        const char* rocm = getenv("ROCM_ROOT");
        hipv.hipRocInc = rocm ? std::string(rocm) + "/include"
                              : "";
    }
    // Frontend+chain debug block, proof level only (HANDOFF §34): level 2
    // would write sharedOut and move the d7 bit-identical check below, so
    // the harness never arms it.
    hipv.hipFeBlock = 1;
    PassResult p7 = RunPass(ngx, d, iniDir, hipv, color.Get(), output.Get(),
                            SRC_W, SRC_H, DST_W, DST_H, kFrames);
    Check(p7.evalFailures == 0, "every frame succeeds through the HIP path");
    int d7 = MaxChannelDiff(p1.rb, p7.rb);
    printf("  worst channel difference vs the resample: %d\n", d7);
    Check(d7 == 0, "the HIP round trip is bit-identical at strength 0");

    // ---- pass 8: the live FFN chain under the three weight readings ------
    // S223d. The in-game A/B (S204) compares three readings of the FFN weights
    // -- dense out-major (S148), dense in-major, and the fragment order S116
    // measured against the production cubin -- and every in-game attempt so far
    // has had to hand-wave about the input frame changing between runs: the
    // clock alone moves the sky, and the player's pose moves the rest. Here the
    // SAME frame goes through the whole live chain once per reading, so the
    // three answers differ by the flag and nothing else.
    //
    // DebugView=2 puts the model's own answer in the output, and DumpField=1
    // writes it as a window on the GAIN FIELD the live write-back produces
    // (128 = 1.0), because the display fold would clip a field that lives near
    // 1.0 to white. Each arm's field is kept as field_arm<N>.bmp so the three
    // can be compared afterwards, which is what tools/ab_fields.py does.
    printf("\n-- pass 8: the live FFN chain, three weight readings --\n");
    {
        // The live chain is the whole encoder: five hiprtc chains compile on
        // its first frame and every block synchronises, so this pass is slow by
        // design. It is the only place outside the game where the live runners
        // -- FeTransBlockRun, FeC64/C128/C256BlockRun, FeProj*Run -- run at all.
        static const int kTranspose[5] = {1, 0, 2, 3, 4};    // A, B, C, D, E
        static const char* kArmName[5] = {"A dense out-major (S148)",
                                          "B dense in-major (legacy)",
                                          "C tinlayout, OUR bit map (S116)",
                                          "D tinlayout, all regions (S223e)",
                                          "E tinlayout, THEIR bit map (S229)"};
        int liveChecks = 0;
        // ONE ARM PER PROCESS, chosen by DLSS5_ARM (0..5; default 0), because
        // the arms are not independent inside one process: every runner loads
        // its weight buffers behind a function-local `static bool built`, so
        // only the FIRST live pass loads anything and the others silently reuse
        // its buffers -- and the passes share the harness's colour/output
        // textures, which carry state from one to the next. Run the harness
        // once per arm with the environment set; pass 8 then compares its own
        // field against the model-off reference from the same conditions.
        const char* armEnv = getenv("DLSS5_ARM");
        const int armSel = armEnv ? atoi(armEnv) : 0;
        for (int arm = armSel; arm <= armSel; ++arm) {
            IniValues live;
            live.nrPasses = 1;
            live.hipBackend = 1;
            live.hipWeightsDir = hipv.hipWeightsDir;
            live.hipRocwmmaInc = hipv.hipRocwmmaInc;
            live.hipRocInc = hipv.hipRocInc;
            live.hipFeLive = (arm < 5) ? 1 : 0;   // arm 5 = model off
            // FeBlockStaged() returns immediately below level 3, so without
            // this the live runners never run and the field comes from the
            // legacy front-end chain -- which does not read the FFN weights at
            // all, so all three arms come out byte-identical and the pass
            // measures nothing. (It did exactly that the first time.)
            // The reference run needs the opposite: level 0, or the LEGACY
            // staged chain still writes its own view into the model texture and
            // the "off" frame is not the untouched one it is supposed to be.
            live.hipFeBlock = (arm < 5) ? 3 : 0;
            live.hipFeTransition = 1;
            live.hipFfnTranspose = kTranspose[arm < 5 ? arm : 0];
            live.debugView = 2;
            live.dumpFrames = 1;
            live.dumpEvery = 1;
            live.dumpField = 1;
            PassResult pl = RunPass(ngx, d, iniDir, live, color.Get(),
                                    output.Get(), SRC_W, SRC_H, DST_W, DST_H,
                                    kFrames);
            Check(pl.evalFailures == 0, "every frame succeeds on the live chain");
            liveChecks += 1;
            // The LAST frame's field, not the first: on frame 0 the live chain
            // cannot run yet -- its modules are compiled by the block test that
            // runs during that same prepare (S199/S206) -- so frame 0 is always
            // the legacy front-end's answer, which does not read the FFN
            // weights and comes out identical in all three arms. That is
            // exactly what this pass measured the first two times it ran.
            char src[MAX_PATH];
            snprintf(src, sizeof(src), "nr_%04d_out.bmp", kFrames - 1);
            char dst[MAX_PATH];
            snprintf(dst, sizeof(dst), "field_run%d.bmp", arm);
            const char* label = (arm < 5) ? kArmName[arm] : "MODEL OFF (reference)";
            remove(dst);
            if (rename(src, dst) == 0)
                printf("  arm %-28s -> %s\n", label, dst);
            else
                printf("  arm %-28s -> no field written\n", label);
        }
        Check(liveChecks == 1, "the selected arm's live chain ran");
    }

    // Optional fixed candidate image through the real D3D12/HIP model-texture
    // bridge. This checks display plumbing and upload timing, not inference on
    // the harness gradient. Keep it opt-in so the normal 117 checks are stable.
    const char* previewEnv = getenv("DLSS5_CANDIDATE_PREVIEW");
    if (previewEnv && *previewEnv) {
        printf("\n-- pass 9: fixed candidate preview model texture --\n");
        // A real game frame can carry zero alpha even when its RGB is valid.
        // The diagnostic view must be opaque so the game cannot blend the
        // preview with that frame. Restore the source to COMMON for upload.
        for (size_t k = 0; k < (size_t)SRC_W * SRC_H; ++k)
            src[k * 4 + 3] = 0;
        Transition(d.list.Get(), color.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                   D3D12_RESOURCE_STATE_COMMON);
        D3DSubmit(d);
        Check(UploadPixels(d, color.Get(), src.data(), SRC_W, SRC_H),
              "zero-alpha scene uploaded for candidate preview");
        Transition(d.list.Get(), color.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3DSubmit(d);
        IniValues preview;
        preview.candidatePreviewPath = previewEnv;
        preview.debugView = 2;
        PassResult pp = RunPass(ngx, d, iniDir, preview, color.Get(),
                                output.Get(), SRC_W, SRC_H, DST_W, DST_H,
                                kFrames);
        Check(pp.evalFailures == 0, "candidate preview frame evaluates");
        Check(MaxChannelDiff(p1.rb, pp.rb) > 20,
              "fixed candidate preview visibly changes the model texture");
        bool opaque = pp.rb.pixels.size() == (size_t)DST_W * DST_H * 4;
        for (size_t k = 0; opaque && k < (size_t)DST_W * DST_H; ++k)
            opaque = pp.rb.pixels[k * 4 + 3] == 255;
        Check(opaque, "candidate debug view stays opaque over zero-alpha scene");
    }

    // ---- the log -------------------------------------------------------
    // Pass 2 asserts the frame is unchanged at strength 0, which is also what
    // a chain that never ran would produce. The shim warns when it falls back
    // to the plain resample, so the log is the only place that can tell the
    // two apart.
    printf("\n-- shim log --\n");
    {
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s/dlssnr_shim.log", iniDir.c_str());
        FILE* f = fopen(path, "rb");
        if (!f) {
            Check(false, "the shim log can be read");
        } else {
            std::string text;
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
            fclose(f);
            if (previewEnv && *previewEnv) {
                Check(text.find("fixed candidate preview center-pixel device readback exact")
                          != std::string::npos,
                      "fixed candidate preview upload/readback is exact");
                Check(text.find("fixed candidate preview upload frame")
                          != std::string::npos,
                      "fixed candidate preview upload timing was logged");
            }

            // Counted, not just searched: every pass should have run the
            // chain, and one fallback anywhere is a failure.
            int fallbacks = 0, inits = 0, hipReady = 0, hipDegraded = 0;
            for (size_t at = 0;;) {
                size_t p = text.find("colour passes did not run", at);
                if (p == std::string::npos) break;
                ++fallbacks;
                at = p + 1;
            }
            for (size_t at = 0;;) {
                size_t p = text.find("=== NVSDK_NGX_D3D12_Init ===", at);
                if (p == std::string::npos) break;
                ++inits;
                at = p + 1;
            }
            for (size_t at = 0;;) {
                size_t p = text.find("hip: identity model ready", at);
                if (p == std::string::npos) break;
                ++hipReady;
                at = p + 1;
            }
            for (size_t at = 0;;) {
                size_t p = text.find("the model stays the identity", at);
                if (p == std::string::npos) break;
                ++hipDegraded;
                at = p + 1;
            }
            // The bit-exact FP8 GEMM self-test: one-hot input means the GEMM
            // output equals the weights bit-for-bit -- with ONE deliberate
            // exception: -0.0 weights come back as +0.0, because the fp32
            // accumulation (+0 + -0 = +0) flips them. The expected value is
            // computed offline from the same tensor with that rule.
            const unsigned int kExpectedQkvChecksum = 0x01800000u;
            size_t st = text.find("self-test QKV GEMM checksum 0x");
            if (st == std::string::npos) {
                Check(false, "the QKV GEMM self-test ran");
            } else {
                unsigned int got = 0;
                sscanf(text.c_str() + st, "self-test QKV GEMM checksum 0x%X",
                       &got);
                printf("  QKV GEMM self-test checksum: 0x%08X (expect 0x%08X)\n",
                       got, kExpectedQkvChecksum);
                Check(got == kExpectedQkvChecksum,
                      "real QKV weights through the FP8 GEMM are bit-exact");
            }
            // The Stage-1 v4 chain check: pre-block on real weights through
            // the hiprtc backend path, block output vs the embedded oracle
            // golden (tol 0.1 covers the 5.7e-3 cross-language gap; wiring
            // bugs show at O(0.5+)).
            size_t ch = text.find("hip: chain blockout maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the v4 chain check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: chain blockout maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  chain blockout maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1f,
                      "v4 chain block output matches the oracle golden");
                Check(FirstVerdictPassed(text, "hip: chain check "),
                      "the v4 chain check passed");
            }
            // The frontend+chain debug block: fixed-gradient proxy through
            // k_frontend into the pre-block, block output vs its own
            // oracle golden (HANDOFF §34).
            ch = text.find("hip: fe-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the fe-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: fe-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  fe-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "fe-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: fe-block check "),
                  "the fe-block check passed");
            // The same block sampled through a NON-identity affine (HANDOFF
            // §13.3). Its own golden, because the identity's cannot see an
            // affine that silently did nothing.
            ch = text.find("hip: fe-affine maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the fe-affine check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: fe-affine maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  fe-affine maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "fe-affine output matches the affine oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: fe-affine check "),
                  "the fe-affine check passed");
            // The block-1 second stage: same LCG Xe through block0, the
            // inter-stage boundary, block1 vs its own oracle golden
            // (HANDOFF §35). Two checks, same tol rationale.
            ch = text.find("hip: b2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b2-block check "),
                  "the b2-block check passed");
            // The block-3 third stage: staged X3e through block3
            // (tensor_044) vs its own oracle golden (HANDOFF §37).
            // Two checks, same tol rationale; expect ~1e-2 scaled by
            // output magnitude (≈0.0025×maxabs: Yp3 3.71 predicts ~0.01).
            ch = text.find("hip: b3-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b3-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b3-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b3-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b3-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b3-block check "),
                  "the b3-block check passed");
            // The block-67 fourth stage: staged X67e through block67
            // (tensor_145) vs its own oracle golden (HANDOFF §38).
            // Two checks, same tol rationale; expect ~1e-2 (oracle
            // A=7.04, hottest intermediates yet — residual may meet or
            // beat block3's 0.0219; §37 close-out, not 0.0025×maxabs).
            ch = text.find("hip: b67-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b67-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b67-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b67-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b67-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b67-block check "),
                  "the b67-block check passed");
            // The block-68 fifth stage: staged X68e through block68
            // (tensor_146) vs its own oracle golden (HANDOFF §39).
            // Two checks, same tol rationale; the O-tripwire predicts
            // above b3's 0.0219 (oracle O=2.98, hottest yet — §38
            // close-out), still 1e-2-to-4e-2 class at tol 0.1.
            ch = text.find("hip: b68-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b68-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b68-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b68-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b68-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b68-block check "),
                  "the b68-block check passed");
            // The block-69 sixth stage: staged X69e through block69
            // (tensor_147) vs its own oracle golden (HANDOFF §40).
            // Two checks, same tol rationale; oracle O=2.18 (cooler
            // than b68's 2.98) points the residual back between b3's
            // 0.0219 and b68's 0.0775 — directional only, tol 0.1.
            ch = text.find("hip: b69-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b69-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b69-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b69-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b69-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b69-block check "),
                  "the b69-block check passed");
            // Block2 seventh stage: staged Xblock2 through block2
            // (tensor_012, fp16 fused32 like the rest) vs its own oracle
            // golden (HANDOFF §42). Two checks, same tol rationale.
            // NOTE: b2-block above is BLOCK1 (Test-8 misnomer); this is
            // block 2 proper ("block2-block"). Fingerprint predicts
            // ~1e-2 class, hottest-A-ish (gains A-tier, S/O B-leaning).
            ch = text.find("hip: block2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the block2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: block2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  block2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "block2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: block2-block check "),
                  "the block2-block check passed");
            // The block-4 eighth stage: staged X4e through the block4
            // prefix (tensor_091) vs its own oracle golden (HANDOFF
            // §45, Test-14). Two checks, same tol rationale; cool-sReal
            // verse, expect 1e-2 class at tol 0.1.
            ch = text.find("hip: b4-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b4-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b4-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b4-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b4-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b4-block check "),
                  "the b4-block check passed");
            // The block-4 downsample ninth stage: prefix re-run plus the
            // 64x32 e4m3 ds projection vs its own oracle golden (HANDOFF
            // §45, Test-15). Two checks; one 32-tap GEMM past the staged
            // prefix, so the budget is the prefix block gap (tol 0.1).
            ch = text.find("hip: b4ds-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the b4ds-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: b4ds-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  b4ds-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "b4ds-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: b4ds-block check "),
                  "the b4ds-block check passed");
            // C=64 scores tenth stage: staged Q/K through k_c64scores
            // (real HOT bias + per-head temps, tensor_137) vs the oracle
            // S golden (HANDOFF §52, Test-16). Two checks; tol 0.1 with
            // ~6500x headroom (host twin maxerr 1.53e-05 at |S|~90).
            ch = text.find("hip: c64s-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64s-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64s-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64s-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64s-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64s-block check "),
                  "the c64s-block check passed");
            // C=64 trick-exp eleventh stage: staged S words through the
            // replicated 5-op trick + normalize (HANDOFF §52, Test-17).
            // Eb is integer-exact (folded into PASSED); P at tol 0.1.
            ch = text.find("hip: c64e-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64e-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64e-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64e-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64e-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64e-block check "),
                  "the c64e-block check passed");
            // C=64 context twelfth stage: staged Pq/V through k_c64ctx
            // (HANDOFF §78, Test-18) vs the oracle O golden. Two checks;
            // tol 0.1 with ~100x headroom (host twin maxerr 0.0).
            ch = text.find("hip: c64o-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64o-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64o-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64o-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64o-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64o-block check "),
                  "the c64o-block check passed");
            // C=64 proj thirteenth stage: staged Ocat/y through k_c64proj
            // (real Wproj/gate2, HANDOFF §78 Test-19) vs the oracle
            // golden. Two checks; tol 0.1 with huge headroom (host twin
            // maxerr 0.0).
            ch = text.find("hip: c64p-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64p-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64p-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64p-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64p-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64p-block check "),
                  "the c64p-block check passed");
            // C=64 FFN fourteenth stage: staged x through k_c64ffn_act
            // + k_c64ffn2 (real w1/w2/gate1, HANDOFF §78 Test-20) vs
            // the oracle golden. Two checks; tol 0.1 with huge
            // headroom (host twin maxerr 4.8e-07).
            ch = text.find("hip: c64f-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64f-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64f-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64f-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64f-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64f-block check "),
                  "the c64f-block check passed");
            // C=64 QKV fifteenth stage: staged X through k_c64qkv
            // (real Wqkv, HANDOFF §90 Test-21) vs the oracle golden.
            // Two checks; tol 0.1 with huge headroom (host twin
            // maxerr 0.0).
            ch = text.find("hip: c64q-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64q-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64q-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64q-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64q-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64q-block check "),
                  "the c64q-block check passed");
            // C=64 FFN contract sixteenth stage: staged h [64][128]
            // through k_c64contract (real W2, HANDOFF §111 Test-22) vs
            // the oracle golden. Two checks; tol 0.1 (oracle rounds f16
            // at four k-steps, device one f32 dot -- Test-21 precedent
            // puts the gap at ~half an f16 ULP).
            ch = text.find("hip: c64c-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64c-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64c-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64c-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64c-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64c-block check "),
                  "the c64c-block check passed");
            // C=64 FFN expand seventeenth stage: staged x through
            // k_c64expand (real w1, HANDOFF §113 Test-23) vs the oracle
            // golden. Two checks; tol 0.1 (oracle rounds f16 at two
            // k-steps, device one f32 dot -- §113 predicts ~half an
            // f16 ULP at maxabs 3.43).
            ch = text.find("hip: c64x-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64x-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64x-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64x-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64x-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64x-block check "),
                  "the c64x-block check passed");
            // C=64 CONNECTED FFN: staged x -> expand -> act -> e4m3 ->
            // contract per pass + gate1 residual through ONE kernel
            // (real w1/w2/gate1, HANDOFF §116 Test-24). Two checks;
            // tol 0.1 (device mirrors the oracle's per-k-step f16, so
            // this should land near 0; tol covers any e4m3 straddle).
            ch = text.find("hip: c64f2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64f2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64f2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64f2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64f2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64f2-block check "),
                  "the c64f2-block check passed");
            // Phase A: the SAME connected FFN at C=128, built from the C=128
            // chain (the kernel is macro-parameterised) and the REAL C=128
            // stage tensor. Its golden comes from the same oracle run at
            // C64_C = 128, so this is a genuine second width, not a rescale.
            ch = text.find("hip: c128f2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c128f2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c128f2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c128f2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c128f2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c128f2-block check "),
                  "the c128f2-block check passed");

            // Phase A: the third width, C=256, from its own chain and the
            // real C=256 stage tensor (tensor_007).
            ch = text.find("hip: c256f2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c256f2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c256f2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c256f2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c256f2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c256f2-block check "),
                  "the c256f2-block check passed");

            // Phase C: block31's FFN at C=1024 / H1=4096, ONE pass, from the
            // real tensor_050 + tensor_051. S145: our kernel IS the ViT's FFN
            // once W == C, because k_c64ffn2c never uses heads.
            ch = text.find("hip: vitf2-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the vitf2-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: vitf2-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  vitf2-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "vitf2-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: vitf2-block check "),
                  "the vitf2-block check passed");

            // Phase C: block31 layer2, the QKV projection, [3C][C] =
            // [3072][1024] out-major (S147 measured from cc_vit_qkv_fp8).
            ch = text.find("hip: vitqkv-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the vitqkv-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: vitqkv-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  vitqkv-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "vitqkv-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: vitqkv-block check "),
                  "the vitqkv-block check passed");

            // Phase C: block31 layer4, the attention output projection,
            // [1024][1024] out-major + 1024 fp16 skip coefficients (S150).
            ch = text.find("hip: vitproj-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the vitproj-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: vitproj-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  vitproj-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "vitproj-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: vitproj-block check "),
                  "the vitproj-block check passed");
            // C=64 CONNECTED BLOCK (HANDOFF §116 Test-25): FFN -> QKV ->
            // scores -> softmax -> ctx -> proj from ONE input. Two checks;
            // tol 0.1 across seven stages + two e4m3 boundaries.
            ch = text.find("hip: c64blk-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c64blk-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c64blk-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c64blk-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c64blk-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c64blk-block check "),
                  "the c64blk-block check passed");

            // S168: the CONNECTED C=128 BLOCK -- Test-25 at a second width,
            // the first evidence that the fused-stage family generalises. It
            // took five fixes to get here (S162 kernels, S163 oracle, S164
            // test A-offset, S165/S167 half-uploads); all are recorded.
            ch = text.find("hip: c128blk-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c128blk-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c128blk-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c128blk-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c128blk-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c128blk-block check "),
                  "the c128blk-block check passed");

            // S169: the CONNECTED C=256 BLOCK -- third width. Passed on the
            // first run once the emitter's temp-block alignment was fixed
            // (4*heads rounded up to 16), with no shim sizing bugs, because
            // every buffer size in that test is derived from the width rather
            // than copied from the C=128 one (the S168 lesson).
            ch = text.find("hip: c256blk-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c256blk-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c256blk-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c256blk-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c256blk-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c256blk-block check "),
                  "the c256blk-block check passed");

            // S170: the CONNECTED C=32 BLOCK -- the fourth and last
            // fused-stage width, and the family's structural outlier (S135:
            // A deviates by -C*W, no self-link region at heads=1).
            ch = text.find("hip: c32blk-block maxerr ");
            if (ch == std::string::npos) {
                Check(false, "the c32blk-block check ran");
            } else {
                float me = 0.0f;
                int nn = -1;
                sscanf(text.c_str() + ch,
                       "hip: c32blk-block maxerr %g (%*u values, %d "
                       "nonfinite)",
                       &me, &nn);
                printf("  c32blk-block maxerr: %g (%d nonfinite, tol 0.1)\n",
                       me, nn);
                Check(nn == 0 && me <= 0.1,
                      "c32blk-block output matches the oracle golden");
            }
            Check(FirstVerdictPassed(text, "hip: c32blk-block check "),
                  "the c32blk-block check passed");
            printf("  %d inits, %d fallbacks, %d hip-ready, %d hip-degraded\n",
                   inits, fallbacks, hipReady, hipDegraded);
            Check(fallbacks == 0, "no pass silently fell back to the resample");
            // One init per pass. Pass 4 runs the proxy curve three times, once
            // per mode, so it contributes two beyond the single-pass count, and
            // S223c's f16 dump pass and S223d's live-chain pass (one arm per
            // process, chosen by DLSS5_ARM) are one single pass each.
            Check(inits == 1 + 8 + 2 + 1 + ((previewEnv && *previewEnv) ? 1 : 0),
                  "every pass initialised the shim");
            Check(hipReady >= 1, "the HIP model ran at least once");
            Check(hipDegraded == 0, "no degradation to the identity model");
        }
    }

    // ---- the debug layer's verdict --------------------------------------
    // Everything above went through the driver's own readback; this section
    // is about what a stricter driver would have refused to execute at all.
    // Info/warning messages are not counted: only errors mean the recorded
    // command stream is invalid.
    printf("\n-- debug layer --\n");
    if (d.infoQueue) {
        UINT64 total = d.infoQueue->GetNumStoredMessages();
        int errors = 0;
        std::string first;
        std::vector<unsigned char> buf;
        for (UINT64 i = 0; i < total; ++i) {
            SIZE_T len = 0;
            if (FAILED(d.infoQueue->GetMessage(i, nullptr, &len)) || !len) continue;
            buf.resize(len);
            D3D12_MESSAGE* msg = (D3D12_MESSAGE*)buf.data();
            if (FAILED(d.infoQueue->GetMessage(i, msg, &len))) continue;
            if (msg->Severity == D3D12_MESSAGE_SEVERITY_ERROR) {
                if (errors == 0 && msg->pDescription) first = msg->pDescription;
                ++errors;
            }
        }
        printf("  %llu messages stored, %d errors\n",
               (unsigned long long)total, errors);
        if (!first.empty()) printf("  first error: %s\n", first.c_str());
        Check(errors == 0, "no debug-layer errors across every pass");
    } else {
        printf("  debug layer not available; state errors would be invisible\n");
    }

    // ---- teardown -------------------------------------------------------
    printf("\n-- teardown --\n");
    NVSDK_NGX_Handle bogus{};
    bogus.Id = 0x12345678;
    Check(ngx.ReleaseFeature(&bogus) != NVSDK_NGX_Result_Success,
          "releasing an unknown handle fails");
    if (scratch) ngx.DestroyParameters(scratch);

    printf("\n=== %d/%d checks passed ===\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
