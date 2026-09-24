#include "ngx_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "shaders/blit_cs.h"     // g_main[]
#include "shaders/dlssnr_cs.h"   // g_dlssnr[]
#include "shaders/counter_cs.h"  // g_counter[]

namespace ngx {

namespace {

constexpr UINT kDescriptorCount = 2048;  // 1024 evaluates in flight before wrap

// The colour shader reads five inputs and writes two, and not every mode uses
// all of them. The spare slots are declared anyway: temporal accumulation of
// the edit is designed but not implemented, and leaving them in the table
// means adding it later does not change the root signature.
constexpr UINT kNrSrvCount = 5;
constexpr UINT kNrUavCount = 2;
constexpr UINT kNrDescriptors = kNrSrvCount + kNrUavCount;

// -------------------------------------------------------------------------

D3D12_RESOURCE_STATES AssumeIncomingState(const D3D12_RESOURCE_DESC& d) {
    // We are handed resources with no state information and cannot query it.
    // These are the states a game written against the NGX contract will have
    // left them in: NGX requires Output to be UAV, and inputs have just been
    // rendered or depth-written.
    if (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    if (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    return D3D12_RESOURCE_STATE_COMMON;
}

bool SupportsTypedUav(ID3D12Device* dev, DXGI_FORMAT fmt) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{};
    fs.Format = fmt;
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))))
        return false;
    return (fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
}

struct DescriptorSlot {
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
};

DescriptorSlot AllocDescriptors(GpuContext& g, UINT n) {
    DescriptorSlot s{};
    if (g.nextDescriptor + n > kDescriptorCount) {
        // Wrapped. Conservatively drain our own queue before reusing.
        if (g.fence && g.fenceEvent) {
            g.queue->Signal(g.fence.Get(), ++g.fenceValue);
            g.fence->SetEventOnCompletion(g.fenceValue, g.fenceEvent);
            WaitForSingleObject(g.fenceEvent, INFINITE);
        }
        g.nextDescriptor = 0;
    }
    s.cpu = CD3DX12_CPU_DESCRIPTOR_HANDLE(
        g.heapCbvSrvUav->GetCPUDescriptorHandleForHeapStart(), g.nextDescriptor,
        g.cbvSrvUavStride);
    s.gpu = CD3DX12_GPU_DESCRIPTOR_HANDLE(
        g.heapCbvSrvUav->GetGPUDescriptorHandleForHeapStart(), g.nextDescriptor,
        g.cbvSrvUavStride);
    g.nextDescriptor += n;
    return s;
}

bool BuildBlitPipeline(GpuContext& g) {
    // Root: b0 = 16 root constants, t0 = SRV table, u0 = UAV table, s0 static.
    CD3DX12_ROOT_PARAMETER rp[3];
    rp[0].InitAsConstants(16, 0);  // b0
    CD3DX12_DESCRIPTOR_RANGE srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    CD3DX12_DESCRIPTOR_RANGE uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    rp[1].InitAsDescriptorTable(1, &srvRange);
    rp[2].InitAsDescriptorTable(1, &uavRange);

    CD3DX12_STATIC_SAMPLER_DESC samp(
        0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(3, rp, 1, &samp,
                                       D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &err))) {
        LOGE("blit: root signature serialize failed: %s",
             err ? (const char*)err->GetBufferPointer() : "?");
        return false;
    }
    if (FAILED(g.device->CreateRootSignature(0, sig->GetBufferPointer(),
                                             sig->GetBufferSize(),
                                             IID_PPV_ARGS(&g.blitRoot)))) {
        LOGE("blit: CreateRootSignature failed");
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};
    ps.pRootSignature = g.blitRoot.Get();
    ps.CS.pShaderBytecode = g_main;
    ps.CS.BytecodeLength = sizeof(g_main);
    if (FAILED(g.device->CreateComputePipelineState(&ps, IID_PPV_ARGS(&g.blitPso)))) {
        LOGE("blit: CreateComputePipelineState failed");
        return false;
    }
    return true;
}

bool BuildNrPipeline(GpuContext& g) {
    // b0 = root constants, then one table holding five SRVs followed by two
    // UAVs, then a static clamp-linear sampler.
    //
    // The constants are root constants rather than a CBV on purpose. Three
    // dispatches get recorded per frame and several frames can be in flight;
    // with a single upload buffer they all map and overwrite it before any of
    // them executes, so the encode silently runs with the resolve's
    // parameters. Root constants are copied into the command stream, so there
    // is nothing to overwrite.
    CD3DX12_ROOT_PARAMETER rp[2];
    rp[0].InitAsConstants(kNrConstantDwords, 0);

    CD3DX12_DESCRIPTOR_RANGE ranges[2];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kNrSrvCount, 0, 0, 0);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, kNrUavCount, 0, 0,
                   kNrSrvCount);
    rp[1].InitAsDescriptorTable(2, ranges);

    CD3DX12_STATIC_SAMPLER_DESC samp(
        0, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(2, rp, 1, &samp,
                                       D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &err))) {
        LOGE("nr: root signature serialize failed: %s",
             err ? (const char*)err->GetBufferPointer() : "?");
        return false;
    }
    if (FAILED(g.device->CreateRootSignature(0, sig->GetBufferPointer(),
                                             sig->GetBufferSize(),
                                             IID_PPV_ARGS(&g.nrRoot)))) {
        LOGE("nr: CreateRootSignature failed");
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};
    ps.pRootSignature = g.nrRoot.Get();
    ps.CS.pShaderBytecode = g_dlssnr;
    ps.CS.BytecodeLength = sizeof(g_dlssnr);
    if (FAILED(g.device->CreateComputePipelineState(&ps, IID_PPV_ARGS(&g.nrPso)))) {
        LOGE("nr: CreateComputePipelineState failed");
        return false;
    }
    return true;
}

bool BuildCounterPipeline(GpuContext& g) {
    // b0 = 4 root constants (the frame number), u0 = one raw-buffer UAV.
    CD3DX12_ROOT_PARAMETER rp[2];
    rp[0].InitAsConstants(4, 0);
    CD3DX12_DESCRIPTOR_RANGE uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    rp[1].InitAsDescriptorTable(1, &uavRange);

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(2, rp, 0, nullptr,
                                       D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> sig, err;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &sig, &err))) {
        LOGE("counter: root signature serialize failed: %s",
             err ? (const char*)err->GetBufferPointer() : "?");
        return false;
    }
    if (FAILED(g.device->CreateRootSignature(0, sig->GetBufferPointer(),
                                             sig->GetBufferSize(),
                                             IID_PPV_ARGS(&g.counterRoot)))) {
        LOGE("counter: CreateRootSignature failed");
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};
    ps.pRootSignature = g.counterRoot.Get();
    ps.CS.pShaderBytecode = g_counter;
    ps.CS.BytecodeLength = sizeof(g_counter);
    if (FAILED(g.device->CreateComputePipelineState(
            &ps, IID_PPV_ARGS(&g.counterPso)))) {
        LOGE("counter: CreateComputePipelineState failed");
        return false;
    }

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(16);
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(g.device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&g.counterBuf)))) {
        LOGE("counter: buffer creation failed");
        return false;
    }
    return true;
}

// Bytes per pixel of the formats the shim's views can be. The staging buffer
// is sized with this; anything unlisted falls back to 4, which is right for
// every 8-bit and 10-bit RGBA family member.
unsigned int BytesPerPixel(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
            return 8;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return 16;
        default:
            return 4;
    }
}

}  // namespace

// A typeless resource cannot be viewed, and the buffer the upscaler writes is
// occasionally declared that way, so the typed member of the same family is
// substituted. The sRGB variants map to their linear equivalents because an
// sRGB view cannot be bound as a typed UAV at all -- and the shader applies
// the transfer function itself, so nothing is lost by doing so.
DXGI_FORMAT TypedFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:  return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS:  return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:   return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:    return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:    return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:                                 return f;
    }
}

// The best guess available for a resource the game owns and we have not
// touched. See the header: wrong for anything our own passes transitioned.
D3D12_RESOURCE_STATES GpuGuessIncomingState(ID3D12Resource* res) {
    if (!res) return D3D12_RESOURCE_STATE_COMMON;
    return AssumeIncomingState(res->GetDesc());
}

// ------------------------------------------------------------------ setup --

GpuContext& Gpu() {
    static GpuContext g;
    return g;
}

bool GpuInit(ID3D12Device* dev) {
    GpuContext& g = Gpu();
    if (g.valid) return true;
    if (!dev) return false;

    g.device = dev;

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue)))) {
        LOGE("gpu: CreateCommandQueue failed");
        return false;
    }
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS(&g.alloc)))) {
        LOGE("gpu: CreateCommandAllocator failed");
        return false;
    }
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      g.alloc.Get(), nullptr,
                                      IID_PPV_ARGS(&g.scratchList)))) {
        LOGE("gpu: CreateCommandList failed");
        return false;
    }
    g.scratchList->Close();

    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence)))) {
        LOGE("gpu: CreateFence failed");
        return false;
    }
    g.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kDescriptorCount;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heapCbvSrvUav)))) {
        LOGE("gpu: CreateDescriptorHeap failed");
        return false;
    }
    g.cbvSrvUavStride = dev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    if (!BuildBlitPipeline(g)) return false;
    if (!BuildNrPipeline(g)) return false;
    if (!BuildCounterPipeline(g)) return false;

    // The counter check is an exact equality, and a fresh DEFAULT-heap
    // allocation is not guaranteed to read as zero -- so zero it now, on our
    // own queue, before the first evaluate can depend on it.
    {
        DescriptorSlot slot = AllocDescriptors(g, 1);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = 4;  // 16 bytes, raw
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        g.device->CreateUnorderedAccessView(g.counterBuf.Get(), nullptr, &uav,
                                            slot.cpu);

        if (SUCCEEDED(g.scratchList->Reset(g.alloc.Get(), nullptr))) {
            g.scratchList->SetComputeRootSignature(g.counterRoot.Get());
            g.scratchList->SetPipelineState(g.counterPso.Get());
            g.scratchList->SetDescriptorHeaps(1, g.heapCbvSrvUav.GetAddressOf());
            g.scratchList->SetComputeRootDescriptorTable(1, slot.gpu);
            const unsigned int zero[4] = {0, 0, 0, 0};
            g.scratchList->SetComputeRoot32BitConstants(0, 4, zero, 0);
            g.scratchList->Dispatch(1, 1, 1);
            g.scratchList->Close();
            ID3D12CommandList* lists[] = {g.scratchList.Get()};
            g.queue->ExecuteCommandLists(1, lists);
            g.queue->Signal(g.fence.Get(), ++g.fenceValue);
            g.fence->SetEventOnCompletion(g.fenceValue, g.fenceEvent);
            WaitForSingleObject(g.fenceEvent, 2000);
            g.alloc->Reset();
            // The list stays CLOSED: a Reset here would leave it in the
            // recording state, and the next user's Reset would fail with
            // "command list was not closed" -- which is exactly how the
            // first HIP frame quietly degraded.
        }
    }

    g.valid = true;
    LOGI("gpu: initialised (descriptor stride %u)", g.cbvSrvUavStride);
    return true;
}

void GpuShutdown() {
    GpuContext& g = Gpu();
    if (!g.valid) return;
    // HIP first: its imports and stream must be dropped while the D3D12
    // resources they reference are still alive.
    HipShutdown();
    // Flush before dropping anything the GPU may still reference.
    if (g.queue && g.fence && g.fenceEvent) {
        g.queue->Signal(g.fence.Get(), ++g.fenceValue);
        g.fence->SetEventOnCompletion(g.fenceValue, g.fenceEvent);
        WaitForSingleObject(g.fenceEvent, 2000);
    }
    if (g.fenceEvent) CloseHandle(g.fenceEvent);
    g = GpuContext{};
}

void GpuTransition(ID3D12GraphicsCommandList* cl, ID3D12Resource* res,
                   D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (!res || before == after) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    cl->ResourceBarrier(1, &b);
}

// ------------------------------------------------------------------- blit --

bool GpuBlit(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, SubRect srcRect,
             ID3D12Resource* dst, SubRect dstRect) {
    GpuContext& g = Gpu();
    if (!g.valid || !src || !dst || !cl) return false;

    D3D12_RESOURCE_DESC sd = src->GetDesc();
    D3D12_RESOURCE_DESC dd = dst->GetDesc();

    if (srcRect.w == 0 || srcRect.h == 0) {
        srcRect.x = 0; srcRect.y = 0;
        srcRect.w = (unsigned int)sd.Width; srcRect.h = sd.Height;
    }
    if (dstRect.w == 0 || dstRect.h == 0) {
        dstRect.x = 0; dstRect.y = 0;
        dstRect.w = (unsigned int)dd.Width; dstRect.h = dd.Height;
    }

    const DXGI_FORMAT srcFmt = TypedFormat(sd.Format);
    const DXGI_FORMAT dstFmt = TypedFormat(dd.Format);

    if (!SupportsTypedUav(g.device.Get(), dstFmt)) {
        LOGE("blit: output format %u (%u typed) does not support typed UAV",
             (unsigned)dd.Format, (unsigned)dstFmt);
        return false;
    }

    DescriptorSlot slot = AllocDescriptors(g, 2);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = srcFmt;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    g.device->CreateShaderResourceView(src, &srv, slot.cpu);

    D3D12_CPU_DESCRIPTOR_HANDLE uavCpu = slot.cpu;
    uavCpu.ptr += g.cbvSrvUavStride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = dstFmt;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Texture2D.MipSlice = 0;
    g.device->CreateUnorderedAccessView(dst, nullptr, &uav, uavCpu);

    D3D12_RESOURCE_STATES srcIn = AssumeIncomingState(sd);
    D3D12_RESOURCE_STATES dstIn = AssumeIncomingState(dd);
    if (dstIn != D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
        (dd.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        // NGX's documented contract: the app hands us Output already in UAV.
        dstIn = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }

    GpuTransition(cl, src, srcIn, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTransition(cl, dst, dstIn, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cl->SetComputeRootSignature(g.blitRoot.Get());
    cl->SetPipelineState(g.blitPso.Get());
    cl->SetDescriptorHeaps(1, g.heapCbvSrvUav.GetAddressOf());

    struct { unsigned int srcRect[4], dstRect[4], sizes[4], flags[4]; } cb{};
    cb.srcRect[0] = srcRect.x; cb.srcRect[1] = srcRect.y;
    cb.srcRect[2] = srcRect.w; cb.srcRect[3] = srcRect.h;
    cb.dstRect[0] = dstRect.x; cb.dstRect[1] = dstRect.y;
    cb.dstRect[2] = dstRect.w; cb.dstRect[3] = dstRect.h;
    cb.sizes[0] = (unsigned int)sd.Width;  cb.sizes[1] = sd.Height;
    cb.sizes[2] = (unsigned int)dd.Width;  cb.sizes[3] = dd.Height;
    cb.flags[0] = (srcRect.w == dstRect.w && srcRect.h == dstRect.h) ? 1u : 0u;

    cl->SetComputeRoot32BitConstants(0, 16, &cb, 0);
    cl->SetComputeRootDescriptorTable(1, slot.gpu);
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = slot.gpu;
    uavGpu.ptr += g.cbvSrvUavStride;
    cl->SetComputeRootDescriptorTable(2, uavGpu);

    UINT gx = (dstRect.w + 7) / 8;
    UINT gy = (dstRect.h + 7) / 8;
    cl->Dispatch(gx, gy, 1);

    // Leave the resources where we found them.
    GpuTransition(cl, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, srcIn);
    GpuTransition(cl, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, dstIn);

    return true;
}

// --------------------------------------------------------- colour passes --

// The model texture is ping-ponged. Frame n's resolve binds the slot prepare #n
// wrote as an SRV on the game's list; prepare #(n+1) rewrites on our own queue.
// With a single texture those two overlap -- the gate can only prove list n-1
// executed, so list n (the reader) may still be in flight when we rewrite. Two
// slots put the rewrite on the other one, and prepare #(n+2) reuses a slot only
// after the gate has proven the list that read it has executed.
static const wchar_t* HipModelTag(unsigned int parity) {
    return parity ? L"hipmodel1" : L"hipmodel0";
}

PooledTexture* GpuAcquireTexture(const wchar_t* tag, DXGI_FORMAT fmt,
                                 unsigned int w, unsigned int h) {
    GpuContext& g = Gpu();
    if (!g.valid || w == 0 || h == 0) return nullptr;

    for (auto& up : g.pool) {
        if (!up) continue;
        if (up->tag != tag) continue;
        if (up->fmt == fmt && up->w == w && up->h == h) return up.get();

        // Same name, different shape: a resize or a format change. The old
        // texture may still be referenced by a frame in flight, so it is
        // dropped here and the GPU is left to retire it on its own schedule.
        up.reset();
        break;
    }

    auto t = std::make_unique<PooledTexture>();
    t->tag = tag;
    t->fmt = fmt;
    t->w = w;
    t->h = h;

    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(fmt, w, h, 1, 1);
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);

    if (FAILED(g.device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&t->res)))) {
        LOGE("pool: CreateCommittedResource failed for %ls %ux%u fmt %u", tag, w,
             h, (unsigned)fmt);
        return nullptr;
    }
    t->state = D3D12_RESOURCE_STATE_COMMON;

    for (auto& up : g.pool) {
        if (!up) { up = std::move(t); return up.get(); }
    }
    g.pool.push_back(std::move(t));
    return g.pool.back().get();
}

void GpuSetState(ID3D12GraphicsCommandList* cl, PooledTexture& t,
                 D3D12_RESOURCE_STATES want) {
    if (!t.res) return;
    GpuTransition(cl, t.res.Get(), t.state, want);
    t.state = want;
}

bool GpuNrDispatch(ID3D12GraphicsCommandList* cl, const DlssNrConstants& c,
                   ID3D12Resource* source, ID3D12Resource* model,
                   ID3D12Resource* original, ID3D12Resource* motion,
                   ID3D12Resource* prevEdit, ID3D12Resource* target,
                   ID3D12Resource* keep) {
    GpuContext& g = Gpu();
    if (!g.valid || !cl || !source || !target) return false;

    // Every slot in the table gets a view, whether the mode reads it or not.
    // An unbound descriptor is not an empty read; it is a read from nothing,
    // and the source stands in wherever a mode has nothing of its own.
    ID3D12Resource* srvs[kNrSrvCount] = {
        source,
        model    ? model    : source,
        original ? original : source,
        motion   ? motion   : source,
        prevEdit ? prevEdit : source,
    };
    ID3D12Resource* uavs[kNrUavCount] = {
        target,
        keep ? keep : target,
    };

    DescriptorSlot slot = AllocDescriptors(g, kNrDescriptors);

    for (UINT i = 0; i < kNrSrvCount; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = TypedFormat(srvs[i]->GetDesc().Format);
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;

        D3D12_CPU_DESCRIPTOR_HANDLE h = slot.cpu;
        h.ptr += (SIZE_T)i * g.cbvSrvUavStride;
        g.device->CreateShaderResourceView(srvs[i], &srv, h);
    }

    for (UINT i = 0; i < kNrUavCount; ++i) {
        const DXGI_FORMAT f = TypedFormat(uavs[i]->GetDesc().Format);
        if (!SupportsTypedUav(g.device.Get(), f)) {
            LOGE("nr: %s format %u (%u typed) cannot be a typed UAV",
                 i == 0 ? "target" : "keep",
                 (unsigned)uavs[i]->GetDesc().Format, (unsigned)f);
            return false;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = f;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Texture2D.MipSlice = 0;

        D3D12_CPU_DESCRIPTOR_HANDLE h = slot.cpu;
        h.ptr += (SIZE_T)(kNrSrvCount + i) * g.cbvSrvUavStride;
        g.device->CreateUnorderedAccessView(uavs[i], nullptr, &uav, h);
    }

    cl->SetComputeRootSignature(g.nrRoot.Get());
    cl->SetPipelineState(g.nrPso.Get());
    cl->SetDescriptorHeaps(1, g.heapCbvSrvUav.GetAddressOf());
    cl->SetComputeRootDescriptorTable(1, slot.gpu);
    cl->SetComputeRoot32BitConstants(0, kNrConstantDwords, &c, 0);
    cl->Dispatch((c.Width + 7) / 8, (c.Height + 7) / 8, 1);
    return true;
}

bool GpuNeuralChain(ID3D12GraphicsCommandList* cl, ID3D12Resource* output,
                    SubRect outRect, const Config& cfg) {
    GpuContext& g = Gpu();
    if (!g.valid || !cl || !output) return false;

    if (outRect.w == 0 || outRect.h == 0) {
        D3D12_RESOURCE_DESC d = output->GetDesc();
        outRect.x = 0; outRect.y = 0;
        outRect.w = (unsigned int)d.Width; outRect.h = d.Height;
    }

    const DXGI_FORMAT fmt = TypedFormat(output->GetDesc().Format);
    if (!SupportsTypedUav(g.device.Get(), fmt)) {
        LOGW("nr: output format %u cannot be a typed UAV; skipping the passes",
             (unsigned)output->GetDesc().Format);
        return false;
    }

    PooledTexture* proxy = GpuAcquireTexture(L"proxy", fmt, outRect.w, outRect.h);
    PooledTexture* keep  = GpuAcquireTexture(L"keep", fmt, outRect.w, outRect.h);
    if (!proxy || !keep) {
        LOGE("nr: could not allocate the proxy/keep textures");
        return false;
    }

    DlssNrConstants c;
    c.Width = outRect.w;
    c.Height = outRect.h;
    c.WhitePoint = cfg.whitePoint;
    c.ProxyMode = (unsigned int)(cfg.proxyMode < 0 ? 0
                                 : (cfg.proxyMode > 2 ? 2 : cfg.proxyMode));
    c.TransferStrength = cfg.transferStrength;
    c.ColourStrength = cfg.colourStrength;
    c.DebugView = (unsigned int)cfg.debugView;
    c.MaxRatio = cfg.maxRatio;
    c.Passthrough = (unsigned int)(cfg.passthrough != 0);

    // ---- encode: the frame -> proxy, plus the untouched copy ------------
    // The resample just wrote `output` as a UAV, so it has to come back to
    // shader-readable before the encode can read it.
    GpuTransition(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuSetState(cl, *proxy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GpuSetState(cl, *keep, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    c.Mode = NR_MODE_ENCODE;
    if (!GpuNrDispatch(cl, c, output, nullptr, nullptr, nullptr, nullptr,
                       proxy->res.Get(), keep->res.Get())) {
        LOGE("nr: encode dispatch failed");
        return false;
    }

    // ---- stage the proxy for the next evaluate's HIP launch -------------
    // Path B from the interop probe: the model reads linear bytes from a
    // shared D3D12 buffer, and this copy is what unswizzles the texture into
    // it. The next evaluate's host-side counter check proves this copy (and
    // everything before it on this list) executed before HIP reads it.
    // HipStartup() is cached, so calling it here is what makes the staging
    // exist from the very first evaluate.
    bool staged = false;
    if (cfg.hipBackend && HipStartup()) {
        staged = HipEnsureStaging(outRect.w, outRect.h, BytesPerPixel(fmt));
        if (staged) {
            g.hipFmt = fmt;
            g.hipW = outRect.w;
            g.hipH = outRect.h;

            GpuSetState(cl, *proxy, D3D12_RESOURCE_STATE_COPY_SOURCE);

            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = HipStagingIn();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Offset = 0;
            dst.PlacedFootprint.Footprint.Format = fmt;
            dst.PlacedFootprint.Footprint.Width = outRect.w;
            dst.PlacedFootprint.Footprint.Height = outRect.h;
            dst.PlacedFootprint.Footprint.Depth = 1;
            dst.PlacedFootprint.Footprint.RowPitch = (UINT)HipStagingRowPitch();
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = proxy->res.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.SubresourceIndex = 0;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    }

    // ---- the model ------------------------------------------------------
    // Identity, for now: the proxy is both the model's input and its answer.
    // When the HIP backend produced an answer for the PREVIOUS frame, that
    // is bound instead -- the one-frame-late model. TensorBackend's real
    // network replaces the HIP launch, not this binding.
    ID3D12Resource* model = proxy->res.Get();

    PooledTexture* hipModel = nullptr;
    // S186 PROBE 1: the chain falls back to `proxy` SILENTLY when hipFrameReady
    // is false or the texture does not acquire -- and a proxy fallback renders
    // exactly like a broken output path, which is what we have been staring at.
    // Log the decision so we can tell "not bound" from "bound but empty".
    static unsigned long long bindN = 0;
    const bool bindLog = ((++bindN % 60) == 1);
    if (bindLog)
        LOGI("nr: chain model bind: hipFrameReady=%d parity=%u hipW=%u hipH=%u",
             (int)g.hipFrameReady, g.hipModelParity, g.hipW, g.hipH);
    if (g.hipFrameReady) {
        hipModel = GpuAcquireTexture(HipModelTag(g.hipModelParity), g.hipFmt,
                                     g.hipW, g.hipH);
        if (bindLog && !hipModel)
            LOGW("nr: chain model bind: hipFrameReady set but texture did NOT "
                 "acquire -- falling back to proxy");
        if (hipModel) {
            GpuSetState(cl, *hipModel,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            model = hipModel->res.Get();
        }
    }

    const float scale = cfg.modelScale;
    if (!hipModel && scale > 0.0f && scale < 0.999f) {
        unsigned int mw = (unsigned int)(outRect.w * scale);
        unsigned int mh = (unsigned int)(outRect.h * scale);
        mw = mw ? mw : 1;
        mh = mh ? mh : 1;

        // Not "small": the RPC headers #define small as char.
        PooledTexture* modelTex = GpuAcquireTexture(L"model", fmt, mw, mh);
        if (!modelTex) {
            LOGE("nr: could not allocate the %ux%u model texture", mw, mh);
            return false;
        }

        GpuSetState(cl, *proxy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GpuSetState(cl, *modelTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        DlssNrConstants dc = c;
        dc.Mode = NR_MODE_DOWNSAMPLE;
        dc.Width = mw;
        dc.Height = mh;
        if (!GpuNrDispatch(cl, dc, proxy->res.Get(), nullptr, nullptr, nullptr,
                           nullptr, modelTex->res.Get(), nullptr)) {
            LOGE("nr: downsample dispatch failed");
            return false;
        }

        GpuSetState(cl, *modelTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        model = modelTex->res.Get();
    }

    // ---- resolve: proxy + model + the untouched copy -> the frame -------
    GpuSetState(cl, *proxy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuSetState(cl, *keep, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GpuTransition(cl, output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    c.Mode = NR_MODE_RESOLVE;
    // `keep` as the second UAV would make it both an input and an output of
    // the same dispatch, so the stand-in is left to fall back to the target.
    if (!GpuNrDispatch(cl, c, proxy->res.Get(), model, keep->res.Get(), nullptr,
                       nullptr, output, nullptr)) {
        LOGE("nr: resolve dispatch failed");
        return false;
    }

    return true;
}

// ----------------------------------------------------- HIP choreography ----

void GpuWriteFrameCounter(ID3D12GraphicsCommandList* cl, uint64_t n) {
    GpuContext& g = Gpu();
    if (!g.valid || !cl || !g.counterPso) return;

    DescriptorSlot slot = AllocDescriptors(g, 1);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = 4;  // 16 bytes, raw
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    g.device->CreateUnorderedAccessView(g.counterBuf.Get(), nullptr, &uav,
                                        slot.cpu);

    cl->SetComputeRootSignature(g.counterRoot.Get());
    cl->SetPipelineState(g.counterPso.Get());
    cl->SetDescriptorHeaps(1, g.heapCbvSrvUav.GetAddressOf());
    cl->SetComputeRootDescriptorTable(1, slot.gpu);
    const unsigned int c[4] = {(unsigned int)n, 0, 0, 0};
    cl->SetComputeRoot32BitConstants(0, 4, c, 0);
    cl->Dispatch(1, 1, 1);
}

namespace {

// Reads the counter through our own queue. The read may race with a game
// list that has not been submitted yet -- that is the point: a stale value is
// the signal to skip the HIP work this evaluate. A 32-bit read cannot tear.
bool ReadCounter(GpuContext& g, unsigned int& out) {
    out = 0xFFFFFFFFu;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(16);
    ComPtr<ID3D12Resource> rb;
    if (FAILED(g.device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&rb))))
        return false;

    if (FAILED(g.scratchList->Reset(g.alloc.Get(), nullptr))) return false;
    g.scratchList->CopyBufferRegion(rb.Get(), 0, g.counterBuf.Get(), 0, 16);
    g.scratchList->Close();
    ID3D12CommandList* lists[] = {g.scratchList.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    g.queue->Signal(g.fence.Get(), ++g.fenceValue);
    g.fence->SetEventOnCompletion(g.fenceValue, g.fenceEvent);
    WaitForSingleObject(g.fenceEvent, 2000);
    g.alloc->Reset();  // the list stays closed for the next user

    void* mapped = nullptr;
    if (FAILED(rb->Map(0, nullptr, &mapped))) return false;
    unsigned int v = 0;
    memcpy(&v, mapped, 4);
    rb->Unmap(0, nullptr);
    out = v;
    return true;
}

}  // namespace

bool GpuHipFrameReady() { return Gpu().hipFrameReady; }

bool GpuPrepareHipModel(uint64_t n, const Config& cfg) {
    GpuContext& g = Gpu();

    // S187: DO NOT CLEAR THIS HERE. It was `g.hipFrameReady = false;` and that
    // single line is why the model texture has never been bound: the prepare
    // clears the latch at its start, the chain then reads it LATER in the same
    // frame, and finds 0 -- so `model` silently falls back to the proxy every
    // frame. Measured, not theorised: the chain logged
    // `hipFrameReady=0 parity=0 hipW=1680 hipH=1050` on every sample, while the
    // very same prepare run logged a successful texture copy.
    //
    // The flag is a LATCH across frames ("the one-frame-late model"): a good
    // prepare sets it, the next chain consumes the texture it names. Clearing it
    // on entry destroys it before any reader sees it. Leaving it set is also the
    // safer behaviour when a prepare is skipped by the counter gate: the last
    // good model stays bound rather than the frame silently reverting to the
    // proxy, which is exactly the failure that hid this for so long.

    if (!g.valid || !cfg.nrPasses || !cfg.hipBackend) return false;
    if (n == 0) return false;  // nothing staged yet
    if (!HipStartup()) return false;

    // Proof that a recent list's shim ops executed, from the counter the last
    // evaluate wrote as its FINAL op: reading k proves list k ran to the end
    // (the counter write is recorded after the resolve). A game that records
    // one frame ahead of submission -- Cyberpunk does, permanently -- leaves
    // exactly one list in flight, so the freshest value it can expose is n-2;
    // an exact n-1 test therefore never fires and the model stops after frame
    // 0. Accept n-2 (and n-1, so a submit-per-frame game is unaffected);
    // anything further behind still skips -- never poll: blocking here could
    // deadlock a game that submits from this very thread. n-2 is only safe
    // because the model texture ping-pongs (see HipModelTag): the one list
    // still in flight reads the other slot, not the one rewritten below.
    unsigned int counter = 0xFFFFFFFFu;
    const uint64_t need = (n >= 2) ? (n - 2) : 0;
    if (!ReadCounter(g, counter) || (uint64_t)counter < need) {
        static uint64_t throttled = 0;
        ++throttled;
        if (throttled <= 3 || (throttled % 300) == 0)
            LOGW("hip: previous frame not confirmed yet (counter %u, want >= %llu);"
                 " keeping the identity model",
                 counter, (unsigned long long)need);
        return false;
    }

    // The staged proxy bytes are the model's input; the launch synchronises,
    // so sharedOut holds the answer by the time it returns.
    if (!HipRunModel()) return false;

    // S176: the wiring. RunModel above is still a copy; this replaces the
    // patch it produces with real network output computed from the staged
    // proxy pixels. Off unless HipFeLive is set.
    if (Cfg().hipFeLive) {
        // S177: throttled. This is now a PER-FRAME call, so an unthrottled
        // warning would be 30 lines/s while, say, tensor_000.bin is missing --
        // the same class of hazard as the hex dump it sits next to.
        if (!HipFeBlockStaged()) {
            static unsigned long long liveFail = 0;
            if ((++liveFail % 300) == 1)
                LOGW("nr: live model path did not run (%llu)", liveFail);
        } else {
            static bool liveSeen = false;
            if (!liveSeen) {
                liveSeen = true;
                LOGI("nr: LIVE MODEL PATH ACTIVE -- real network output into the"
                     " frame (C=64 block; 1 of 71 blocks)");
            }
        }
    }

    // First successful model run: also prove the real-weights path (FP8
    // rocWMMA GEMM, bit-exact one-hot self-test) once. Its checksum goes to
    // the log; the harness compares it.
    HipSelfTest();
    // Same for the Stage-1 v4 chain through the hiprtc backend path (block
    // output vs the embedded oracle golden, tol 0.1). Log-only proof; the
    // harness compares the blockout line (HANDOFF §27).
    HipChainTest();

    // Frontend+chain debug block (HANDOFF §34): fixed-gradient proxy through
    // k_frontend into the pre-block, vs the embedded oracle golden.
    // HipFeBlock=1 is proof-only (never touches staging, like the chain
    // check); =2 additionally delivers the cached block into sharedOut on
    // every model run (debug view -- moves the strength-0 bit-identical
    // check while set). Identity path untouched at 0.
    HipFeBlockTest();
    // Block-1 second stage (HANDOFF §35): same LCG Xe through block0,
    // inter-stage boundary, block1 vs the embedded golden. Synthetic and
    // frame-safe like the chain check (never touches staging).
    HipB2BlockTest();
    // Block-3 third stage (HANDOFF §37): staged X3e through block3 vs
    // its own golden. Pure block3 proof (no block0/1 compute here).
    HipB3BlockTest();
    // Block-67 fourth stage (HANDOFF §38): staged X67e through block67
    // (tensor_145) vs its own golden. Same slim shape as Test-9.
    HipB67BlockTest();
    // Block-68 fifth stage (HANDOFF §39): staged X68e through block68
    // (tensor_146) vs its own golden. Same slim shape.
    HipB68BlockTest();
    // Block-69 sixth stage (HANDOFF §40): staged X69e through block69
    // (tensor_147) vs its own golden. Same slim shape.
    HipB69BlockTest();
    // Chained tail runner (HANDOFF §41): UNSCORED — runs 67->68->69 on
    // device from staged X67e, logs telescoping INFO only (no Checks).
    HipTailRun();
    // Block2 seventh stage (HANDOFF §42): staged Xblock2 through block2
    // (tensor_012) vs its own golden. NOTE: b2-block/B2* above mean
    // BLOCK1 (Test-8 misnomer); this is block 2 proper. Same slim shape.
    HipBlock2Test();
    // Block-4 eighth stage (HANDOFF §45, Test-14): staged X4e through
    // the block4 prefix (tensor_091) vs its own golden. Same slim
    // shape, synthetic and frame-safe like the earlier stages.
    HipB4BlockTest();
    // Block-4 downsample ninth stage (HANDOFF §45, Test-15): prefix
    // re-run plus the 64x32 e4m3 ds projection vs its own golden.
    HipB4DsBlockTest();
    // C=64 scores tenth stage (HANDOFF §52, Test-16): staged Q/K
    // through k_c64scores (real HOT bias + per-head temps) vs its own
    // golden. Same slim shape, synthetic and frame-safe.
    HipC64sBlockTest();
    // C=64 trick-exp eleventh stage (HANDOFF §52, Test-17): staged S
    // words through the replicated 5-op trick + normalize (Eb exact,
    // P tol 0.1). No new kernel.
    HipC64eBlockTest();
    // C=64 context twelfth stage (HANDOFF §78, Test-18): staged Pq/V
    // through k_c64ctx vs its own golden. Same slim shape, synthetic
    // and frame-safe.
    HipC64oBlockTest();
    // C=64 proj thirteenth stage (HANDOFF §78, Test-19): staged Ocat/y
    // through k_c64proj (real Wproj/gate2 from the dump) vs its own
    // golden. Same slim shape, synthetic and frame-safe.
    HipC64pBlockTest();
    // C=64 FFN fourteenth stage (HANDOFF §78, Test-20): staged x
    // through k_c64ffn_act + k_c64ffn2 (real w1/w2/gate1 from the
    // dump) vs its own golden. Same slim shape, synthetic and
    // frame-safe.
    HipC64fBlockTest();
    // C=64 QKV fifteenth stage (HANDOFF §90, Test-21): staged X
    // through k_c64qkv (real Wqkv from the dump) vs its own golden.
    // Same slim shape, synthetic and frame-safe.
    HipC64qBlockTest();
    // C=64 FFN contract sixteenth stage (HANDOFF §111, Test-22): staged
    // h through k_c64contract (real W2 from the dump) vs its own golden.
    // Same slim shape, synthetic and frame-safe.
    HipC64cBlockTest();
    // C=64 FFN expand seventeenth stage (HANDOFF §113, Test-23): staged
    // x through k_c64expand (real w1 from the dump, IN-MAJOR [64][128])
    // vs its own golden. Same slim shape, synthetic and frame-safe.
    HipC64xBlockTest();
    // CONNECTED C=64 FFN (real w1/w2/gate1, IN-MAJOR): staged x -> expand
    // -> act -> e4m3 -> contract per pass + gate1 residual (HANDOFF §116
    // Test-24). One kernel; the connection Test-20/22/23 lacked.
    HipC64f2BlockTest();
    // Phase A: the same connected FFN at C=128, from the C=128 chain.
    HipC128f2BlockTest();
    // Phase A: and at C=256, third width, own chain.
    HipC256f2BlockTest();
    // Phase C: block31's FFN at C=1024 (S145).
    HipVitFfn2BlockTest();
    HipVitQkvBlockTest();
    HipVitProjBlockTest();
    // CONNECTED C=64 BLOCK (HANDOFF §116 Test-25): FFN feeds QKV feeds
    // scores/softmax/ctx feeds proj, from ONE staged input -- the
    // inter-stage connections Test-16..24 tested only in isolation.
    HipC64blkBlockTest();
    HipC128blkBlockTest();
    HipC256blkBlockTest();
    HipC32blkBlockTest();
    HipFeBlockView();
    // Level 3 (HipFeBlock=3, manual ini only): top-left staged window
    // through the frontend+chain, block answer into sharedOut, window
    // bytes + block checksum logged for the offline oracle recompute.
    HipFeBlockStaged();

    // Refresh the model texture on our own queue. Ping-pong makes the rewrite
    // safe even though the gate above accepts a one-frame-late counter:
    // prepare #n writes slot n&1, frame n's resolve binds that slot, and this
    // code only reuses slot n&1 again at #(n+2) -- by then the gate has proven
    // list n, that slot's only reader, executed.
    const unsigned int parity = (unsigned int)(n & 1);
    PooledTexture* m =
        GpuAcquireTexture(HipModelTag(parity), g.hipFmt, g.hipW, g.hipH);
    if (!m) return false;

    if (FAILED(g.scratchList->Reset(g.alloc.Get(), nullptr))) return false;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = m->res.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = HipStagingOut();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Offset = 0;
    src.PlacedFootprint.Footprint.Format = g.hipFmt;
    src.PlacedFootprint.Footprint.Width = g.hipW;
    src.PlacedFootprint.Footprint.Height = g.hipH;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)HipStagingRowPitch();

    GpuSetState(g.scratchList.Get(), *m, D3D12_RESOURCE_STATE_COPY_DEST);
    g.scratchList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    GpuSetState(g.scratchList.Get(), *m,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g.scratchList->Close();
    ID3D12CommandList* lists[] = {g.scratchList.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    g.queue->Signal(g.fence.Get(), ++g.fenceValue);
    g.fence->SetEventOnCompletion(g.fenceValue, g.fenceEvent);
    WaitForSingleObject(g.fenceEvent, 2000);
    g.alloc->Reset();  // the list stays closed for the next user

    // S186 PROBE 2: CopyTextureRegion returns void, so a silent failure would
    // be invisible. A device-level error is the one thing it does leave behind.
    {
        static unsigned long long cr = 0;
        if ((++cr % 60) == 1) {
            HRESULT rr = g.device->GetDeviceRemovedReason();
            LOGI("nr: texture copy done; device reason=0x%08X (%s)", (unsigned)rr,
                 SUCCEEDED(rr) ? "ok" : "DEVICE ERROR");
        }
    }

    // The resolve for frame n binds this slot; publish it with the parity that
    // names it.
    g.hipModelParity = parity;
    g.hipFrameReady = true;
    return true;
}

// ------------------------------------------------------------------ dumps --

namespace {

#pragma pack(push, 2)
struct BmpHeader {
    unsigned short bfType = 0x4D42;
    unsigned int bfSize = 0;
    unsigned short bfReserved1 = 0;
    unsigned short bfReserved2 = 0;
    unsigned int bfOffBits = 54;
    unsigned int biSize = 40;
    int biWidth = 0;
    int biHeight = 0;
    unsigned short biPlanes = 1;
    unsigned short biBitCount = 24;
    unsigned int biCompression = 0;
    unsigned int biSizeImage = 0;
    int biXPelsPerMeter = 0;
    int biYPelsPerMeter = 0;
    unsigned int biClrUsed = 0;
    unsigned int biClrImportant = 0;
};
#pragma pack(pop)

// 24-bit BGR, bottom-up. rows are (w*3) padded to 4 bytes.
bool WriteBmp(const std::wstring& path, const unsigned char* rgb, int w, int h_) {
    const int rowBytes = ((w * 3 + 3) / 4) * 4;
    BmpHeader h{};
    h.biWidth = w;
    h.biHeight = h_;      // positive height = bottom-up row order
    h.biSizeImage = rowBytes * h_;
    h.bfSize = 54 + h.biSizeImage;

    std::vector<unsigned char> out;
    out.resize(54 + h.biSizeImage);
    memcpy(out.data(), &h, sizeof(h));
    unsigned char* p = out.data() + 54;
    for (int y = 0; y < h_; ++y) {
        const unsigned char* src = rgb + (size_t)(h_ - 1 - y) * w * 3;
        memcpy(p + (size_t)y * rowBytes, src, w * 3);
    }

    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return true;
}

}  // namespace

namespace {

struct PendingDump {
    ComPtr<ID3D12Resource> buf;
    unsigned int w = 0, h = 0;
    UINT64 pitch = 0;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    std::wstring path;
    uint64_t readyAt = 0;  // evaluate index at which the data is safe to read
};

std::mutex g_dumpMutex;
std::vector<PendingDump> g_pending;

bool FormatSupportedForDump(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        // S223c: an HDR game hands NGX R16G16B16A16_FLOAT, and without this the
        // dump instrument simply does not exist for it -- the whole run produces
        // "[E] dump: unsupported format 10" and no file at all, which is how the
        // 2026-09-18 A/B lost its pixel-exact half. See LinearToByte below for
        // what an 8-bit view of a linear f16 buffer means.
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return true;
        default:
            return false;
    }
}

// ---- S223c: the half path of the dump instrument -------------------------
//
// An HDR game (Cyberpunk with HDR on) hands NGX R16G16B16A16_FLOAT, so the
// dump writer needs to say what an 8-bit BMP of a LINEAR f16 buffer means. The
// answer is not invented here: it is the fold the pipeline itself applies on
// the way to the screen (dlssnr.hlsl's encode: scale by the white point, roll
// or clip the highlights, then LinearToSrgb), minus the proxy's knee, which
// exists for the MODEL's benefit rather than the eye's.
//
// So: v * whitePoint -> clamp to [0,1] -> sRGB transfer -> 8 bits. Values above
// white clip, exactly as they do on a display without tone mapping; a NaN is
// returned as black, which is also what clamp(NaN) does in the shader.
//
// The mapping is FIXED, which is what an A/B needs: every arm of a comparison
// is folded through the same curve, so a difference between two dumps is a
// difference in the buffer and never in the instrument.
unsigned int HalfToFloatBits(unsigned short h) {
    const unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) return sign;                       // +-0
        int shift = 0;                                   // subnormal: normalize
        while ((man & 0x400u) == 0) { man <<= 1; ++shift; }
        man &= 0x3FFu;
        return sign | ((unsigned int)(127 - 15 - shift) << 23) | (man << 13);
    }
    if (exp == 31) return sign | 0x7F800000u | (man << 13);   // inf / NaN
    return sign | ((exp + 112u) << 23) | (man << 13);
}

float HalfToFloat(unsigned short h) {
    unsigned int bits = HalfToFloatBits(h);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

unsigned char LinearToByte(float v, float whitePoint) {
    // S223d: the live write-back puts a GAIN FIELD in the model texture, not a
    // picture -- out = proxy * (1 + k*tanh(tok)), so every sane pixel sits near
    // 1.0 and the display fold above would clip the whole field to white. When
    // DumpField is set the dump is a window on that field instead: 0.0 maps to
    // grey and +-4.0 to the ends. The window is deliberately WIDE -- what the
    // staging actually holds is a measurement, not an assumption, and a window
    // that clips tells you nothing about which arm is which. In-game the raw
    // answer's maxabs runs 3.8..5.5, so the band that matters is +-2 or so.
    if (Cfg().dumpField) {
        const float b = (v + 4.0f) * 31.875f;         // [-4, 4] -> [0, 255]
        return (unsigned char)(!(b > 0.0f) ? 0.0f : (b >= 255.0f ? 255.0f : b));
    }
    v *= whitePoint;
    if (!(v > 0.0f)) return 0;                 // also catches NaN
    if (v > 1.0f) v = 1.0f;
    const float s = v <= 0.0031308f ? v * 12.92f
                                    : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
    const float b = s * 255.0f + 0.5f;
    return (unsigned char)(b <= 0.0f ? 0.0f : (b >= 255.0f ? 255.0f : b));
}

void WritePendingBmp(const PendingDump& pd) {
    void* mapped = nullptr;
    if (FAILED(pd.buf->Map(0, nullptr, &mapped))) {
        LOGE("dump: map failed for %ls", pd.path.c_str());
        return;
    }

    std::vector<unsigned char> rgb((size_t)pd.w * pd.h * 3);
    for (unsigned int y = 0; y < pd.h; ++y) {
        const unsigned char* srow =
            (const unsigned char*)mapped + (size_t)y * pd.pitch;
        unsigned char* drow = rgb.data() + (size_t)y * pd.w * 3;
        for (unsigned int x = 0; x < pd.w; ++x) {
            unsigned int r, gg, b;
            if (pd.fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                unsigned short h[4];
                memcpy(h, srow + (size_t)x * 8, 8);
                const float wp = Cfg().whitePoint;
                r = LinearToByte(HalfToFloat(h[0]), wp);
                gg = LinearToByte(HalfToFloat(h[1]), wp);
                b = LinearToByte(HalfToFloat(h[2]), wp);
            } else {
                unsigned int px;
                memcpy(&px, srow + (size_t)x * 4, 4);
                if (pd.fmt == DXGI_FORMAT_R10G10B10A2_UNORM) {
                    // 10-bit channels sit in the low bits of each lane.
                    r = (px >> 0) & 0x3FF;
                    gg = (px >> 10) & 0x3FF;
                    b = (px >> 20) & 0x3FF;
                    r >>= 2; gg >>= 2; b >>= 2;
                } else if (pd.fmt == DXGI_FORMAT_B8G8R8A8_UNORM ||
                           pd.fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) {
                    b = (px >> 0) & 0xFF;
                    gg = (px >> 8) & 0xFF;
                    r = (px >> 16) & 0xFF;
                } else {
                    r = (px >> 0) & 0xFF;
                    gg = (px >> 8) & 0xFF;
                    b = (px >> 16) & 0xFF;
                }
            }

            // A 24-bit BMP stores B, G, R in that order. Writing the channels
            // in memory order swaps red and blue -- which is invisible on a
            // grey ramp and invisible in the corners of a symmetric test
            // pattern, so it went unnoticed for a whole slice.
            drow[x * 3 + 0] = (unsigned char)b;
            drow[x * 3 + 1] = (unsigned char)gg;
            drow[x * 3 + 2] = (unsigned char)r;
        }
    }
    pd.buf->Unmap(0, nullptr);

    if (WriteBmp(pd.path, rgb.data(), (int)pd.w, (int)pd.h))
        LOGI("dump: wrote %ls (%ux%u)", pd.path.c_str(), pd.w, pd.h);
    else
        LOGE("dump: failed to write %ls", pd.path.c_str());
}

}  // namespace

bool GpuQueueDump(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, SubRect rect,
                  const std::wstring& path, uint64_t nowEvaluate,
                  D3D12_RESOURCE_STATES stateIn) {
    GpuContext& g = Gpu();
    if (!g.valid || !res || !cl) return false;

    D3D12_RESOURCE_DESC d = res->GetDesc();
    if (rect.w == 0 || rect.h == 0) {
        rect.x = 0; rect.y = 0;
        rect.w = (unsigned int)d.Width; rect.h = d.Height;
    }
    // Checked and read as the typed member of the family: a typeless resource
    // cannot be the format of a copy footprint either.
    const DXGI_FORMAT fmt = TypedFormat(d.Format);
    if (!FormatSupportedForDump(fmt)) {
        LOGE("dump: unsupported format %u (%u typed)", (unsigned)d.Format,
             (unsigned)fmt);
        return false;
    }

    UINT64 rowPitch = (UINT64)rect.w * BytesPerPixel(fmt);
    rowPitch = (rowPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) /
               D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    UINT64 total = rowPitch * rect.h;

    ComPtr<ID3D12Resource> readback;
    CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC bd = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(g.device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&readback)))) {
        LOGE("dump: readback buffer alloc failed");
        return false;
    }

    // `stateIn` comes from the caller because ours is the only code that
    // knows what the last pass left the resource in. Guessing from the desc
    // gets Output wrong: the chain leaves it in UAV, the guess says COMMON,
    // and a barrier with the wrong StateBefore is a validation error even
    // where the driver happens to render the right picture.
    GpuTransition(cl, res, stateIn, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = readback.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint.Offset = 0;
    dstLoc.PlacedFootprint.Footprint.Format = fmt;
    dstLoc.PlacedFootprint.Footprint.Width = rect.w;
    dstLoc.PlacedFootprint.Footprint.Height = rect.h;
    dstLoc.PlacedFootprint.Footprint.Depth = 1;
    dstLoc.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = res;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;

    D3D12_BOX box{};
    box.left = rect.x; box.top = rect.y;
    box.right = rect.x + rect.w; box.bottom = rect.y + rect.h;
    box.front = 0; box.back = 1;
    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box);

    GpuTransition(cl, res, D3D12_RESOURCE_STATE_COPY_SOURCE, stateIn);

    PendingDump pd;
    pd.buf = readback;
    pd.w = rect.w;
    pd.h = rect.h;
    pd.pitch = rowPitch;
    pd.fmt = fmt;
    pd.path = path;
    // Two evaluates of slack: the caller submits the list after we return, and
    // a frame of latency beyond that is normal.
    pd.readyAt = nowEvaluate + 2;

    std::lock_guard<std::mutex> lock(g_dumpMutex);
    g_pending.push_back(pd);
    return true;
}

void GpuDrainDumps(uint64_t nowEvaluate, bool drainAll) {
    std::vector<PendingDump> ready;
    {
        std::lock_guard<std::mutex> lock(g_dumpMutex);
        auto it = std::remove_if(g_pending.begin(), g_pending.end(),
                                 [&](const PendingDump& p) {
                                     if (drainAll || p.readyAt <= nowEvaluate) {
                                         ready.push_back(p);
                                         return true;
                                     }
                                     return false;
                                 });
        g_pending.erase(it, g_pending.end());
    }
    for (const auto& p : ready) WritePendingBmp(p);
}

}  // namespace ngx
