// A minimal D3D12 "game" for the present hook (src/present/hook.cpp): a
// window with a flip swap chain that shows one picture, panned a little
// every frame. Run from a directory that holds the hook as version.dll and
// its dlss5_present.ini: the hook loads with the process (this program
// imports the version API), patches the swap chain and processes each frame.
// After the last frame the last presented buffer is read back and compared
// with what was drawn: any change is the hook's.
//
// Usage: present_window image.rgb out_prefix [frames] [show] [shim.dll] [guides] [pan] [seq_dir] [linear] [linger]
//   image.rgb: two uint32 (width, height), then RGB8 rows.
//   show: 1 shows the window (default hidden).
//   shim.dll: the NGX shim (nvngx.dll) with a dlssnr_shim.ini beside it that
//     sets PresentMode=1, instead of the version.dll hook. The program then
//     plays the upscaler's part as OptiScaler's DLSS-NR pass does: every
//     frame it calls dlssnr_call_evaluate on its own list with a small
//     colour, the motion of the pan and a depth, before it presents.
//   guides: 0 hands the shim no motion (default 1). 2: the same motion in
//     pixels of the guides with scale 1, and a depth that is not inverted
//     (near small), stated so: the result must be what 1 gives. 3: pixels
//     only, the depth as it is.
//   pan: pixels the picture moves to the left a frame (default 4; 0 a still).
//   seq_dir: instead of the panned picture, the frames of a sequence capture
//     (D5INP001 + D5DEP001 + D5MOV001, as the shim and the front end record
//     them) of the image's size: frame k is drawn, and its own motion and
//     depth are what the NR evaluate is handed. linear 1: the pictures are
//     linear light and are sRGB-encoded first.
//   linger: seconds to stay alive after the last frame (a sequence capture
//     the front end recorded is written by a thread of this process).

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}

float HalfToFloat(unsigned short v) {
    const unsigned int sign = (v & 0x8000u) << 16, exp = (v >> 10) & 0x1Fu, man = v & 0x3FFu;
    unsigned int bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            int e = -1;
            unsigned int m = man;
            do { ++e; m <<= 1; } while (!(m & 0x400u));
            bits = sign | ((unsigned int)(112 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 3) { fputs("usage: present_window image.rgb out_prefix [frames] [show]\n", stderr); return 2; }
    const int frames = argc > 3 ? atoi(argv[3]) : 240;
    setvbuf(stdout, nullptr, _IONBF, 0);  // a crash under the hook must not lose what was printed
    const bool show = argc > 4 && atoi(argv[4]) != 0;
    const char* shimPath = argc > 5 && argv[5][0] ? argv[5] : nullptr;
    const int guideMode = argc > 6 ? atoi(argv[6]) : 1;
    const bool guides = guideMode != 0, pixelGuides = guideMode >= 2, flipDepth = guideMode == 2;
    const int pan = argc > 7 ? atoi(argv[7]) : 4;
    const char* seqDir = argc > 8 && argv[8][0] ? argv[8] : nullptr;
    const bool seqLinear = argc > 9 && atoi(argv[9]) != 0;
    std::vector<std::string> seqFiles;
    if (seqDir) {
        WIN32_FIND_DATAA fd;
        HANDLE find = FindFirstFileA((std::string(seqDir) + "\\*.bin").c_str(), &fd);
        if (find == INVALID_HANDLE_VALUE) { fputs("no sequence frames\n", stderr); return 1; }
        do seqFiles.push_back(std::string(seqDir) + "\\" + fd.cFileName); while (FindNextFileA(find, &fd));
        FindClose(find);
        std::sort(seqFiles.begin(), seqFiles.end());
    }
    // The import that brings version.dll (the hook) into the process.
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    DWORD unused = 0;
    printf("version info size of this program: %lu\n", GetFileVersionInfoSizeW(self, &unused));

    FILE* f = fopen(argv[1], "rb");
    unsigned dims[2];
    if (!f || fread(dims, 4, 2, f) != 2) { fputs("cannot read the image\n", stderr); return 1; }
    const UINT w = dims[0], h = dims[1];
    std::vector<unsigned char> rgb(size_t(w) * h * 3);
    if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) { fputs("short image\n", stderr); return 1; }
    fclose(f);

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"present_window";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"present_window", WS_POPUP, 0, 0, int(w), int(h), nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) { fputs("no window\n", stderr); return 1; }
    if (show) ShowWindow(hwnd, SW_SHOWNOACTIVATE);

    ComPtr<IDXGIFactory2> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12Fence> fence;
    ComPtr<IDXGISwapChain1> sc1;
    ComPtr<IDXGISwapChain3> sc;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = w; sd.Height = h;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc = {1, 0};
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;  // a presented buffer keeps its contents
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
        FAILED(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &sc1)) || FAILED(sc1.As(&sc))) {
        fputs("D3D12 setup failed\n", stderr); return 1;
    }
    cl->Close();
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    auto wait = [&] {
        queue->Signal(fence.Get(), ++fenceValue);
        fence->SetEventOnCompletion(fenceValue, event);
        WaitForSingleObject(event, 30000);
    };

    D3D12_HEAP_PROPERTIES def{}, up{}, rb{};
    def.Type = D3D12_HEAP_TYPE_DEFAULT; up.Type = D3D12_HEAP_TYPE_UPLOAD; rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w; td.Height = h; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc = {1, 0};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc = {1, 0}; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> source, upload, readback;
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&source))) ||
        FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) {
        fputs("resources failed\n", stderr); return 1;
    }
    {
        unsigned char* p = nullptr;
        upload->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x) {
                unsigned char* d = p + size_t(y) * fp.Footprint.RowPitch + x * 4;
                const unsigned char* s = &rgb[(size_t(y) * w + x) * 3];
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
            }
        upload->Unmap(0, nullptr);
    }
    D3D12_TEXTURE_COPY_LOCATION texSource{}, bufUp{}, bufRead{};
    texSource.pResource = source.Get(); texSource.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    bufUp.pResource = upload.Get(); bufUp.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; bufUp.PlacedFootprint = fp;
    bufRead.pResource = readback.Get(); bufRead.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; bufRead.PlacedFootprint = fp;
    alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
    cl->CopyTextureRegion(&texSource, 0, 0, 0, &bufUp, nullptr);
    D3D12_RESOURCE_BARRIER b = Transition(source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &b);
    cl->Close();
    ID3D12CommandList* lists[] = {cl.Get()};
    queue->ExecuteCommandLists(1, lists);
    wait();

    // The shim in present mode: a feature, and the textures an NR evaluate is
    // handed. The pan moves the picture `pan` pixels to the left a frame, so a
    // pixel was that far to the right the frame before: previous = current + (pan/w, 0)
    // in fractions of the frame.
    typedef void* (*CreateFn)(const wchar_t*, const wchar_t*, ID3D12Device*, ID3D12GraphicsCommandList*, void*, unsigned, unsigned,
                              int, float, int, float, float, float, int, int);
    typedef int (*EvaluateFn)(ID3D12GraphicsCommandList*, void*, void*, ID3D12Resource*, ID3D12Resource*, ID3D12Resource*,
                              ID3D12Resource*, unsigned, unsigned, unsigned, unsigned, int, int, float, int, float, float, float,
                              int, float, float);
    typedef int (*AllocFn)(void**);
    EvaluateFn shimEvaluate = nullptr;
    void *shimFeature = nullptr, *shimParams = nullptr;
    ComPtr<ID3D12Resource> nrColor, nrOutput, nrMotion, nrDepth;
    ComPtr<ID3D12DescriptorHeap> nrRtvs;
    const UINT nrW = 256, nrH = 144;
    // The guides: half the frame for the pan, the frame's size for a sequence.
    const UINT guideW = seqDir ? w : w / 2, guideH = seqDir ? h : h / 2;
    // The scale an upscaler host states (OptiScaler): vector units -> pixels
    // of the guides (their extent for fractions of the frame, 1 for pixels),
    // times the NR working size over the frame's.
    const float mvScaleX = (pixelGuides ? 1.0f : float(guideW)) * float(nrW) / float(w);
    const float mvScaleY = (pixelGuides ? 1.0f : float(guideH)) * float(nrH) / float(h);
    int shimFailures = 0;
    // A sequence's motion and depth go through upload buffers every frame.
    ComPtr<ID3D12Resource> mvUpload, depthUpload;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mvFp{}, depthFp{};
    if (seqDir) {
        D3D12_RESOURCE_DESC gd = td;
        UINT64 bytes = 0;
        gd.Format = DXGI_FORMAT_R32G32_FLOAT;
        device->GetCopyableFootprints(&gd, 0, 1, 0, &mvFp, nullptr, nullptr, &bytes);
        D3D12_RESOURCE_DESC ub = bd;
        ub.Width = bytes;
        bool ok = SUCCEEDED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mvUpload)));
        gd.Format = DXGI_FORMAT_R32_FLOAT;
        device->GetCopyableFootprints(&gd, 0, 1, 0, &depthFp, nullptr, nullptr, &bytes);
        ub.Width = bytes;
        ok = ok && SUCCEEDED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&depthUpload)));
        if (!ok) { fputs("guide upload buffers failed\n", stderr); return 1; }
    }
    if (shimPath) {
        HMODULE shim = LoadLibraryA(shimPath);
        const CreateFn create = shim ? reinterpret_cast<CreateFn>(GetProcAddress(shim, "dlssnr_call_create")) : nullptr;
        const AllocFn allocParams = shim ? reinterpret_cast<AllocFn>(GetProcAddress(shim, "NVSDK_NGX_D3D12_AllocateParameters")) : nullptr;
        shimEvaluate = shim ? reinterpret_cast<EvaluateFn>(GetProcAddress(shim, "dlssnr_call_evaluate")) : nullptr;
        if (!create || !allocParams || !shimEvaluate) { fputs("the shim or its exports are missing\n", stderr); return 1; }
        auto texture = [&](UINT tw, UINT th, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d.Width = tw; d.Height = th; d.DepthOrArraySize = 1; d.MipLevels = 1;
            d.Format = f; d.SampleDesc = {1, 0}; d.Flags = flags;
            return SUCCEEDED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&out)));
        };
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 3;
        if (!texture(nrW, nrH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, nrColor) ||
            !texture(nrW, nrH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nrOutput) ||
            !texture(guideW, guideH, seqDir ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, nrMotion) ||
            !texture(guideW, guideH, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, nrDepth) ||
            FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&nrRtvs)))) {
            fputs("NR textures failed\n", stderr); return 1;
        }
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = nrRtvs->GetCPUDescriptorHandleForHeapStart();
        ID3D12Resource* targets[3] = {nrColor.Get(), nrMotion.Get(), nrDepth.Get()};
        const float clears[3][4] = {{0.3f, 0.3f, 0.3f, 1.0f},
                                    {pixelGuides ? float(pan) * float(guideW) / float(w) : float(pan) / float(w), 0.0f, 0.0f, 0.0f},
                                    {0.5f, 0.0f, 0.0f, 0.0f}};
        alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
        for (int k = 0; k < 3; ++k) {
            device->CreateRenderTargetView(targets[k], nullptr, rtv);
            cl->ClearRenderTargetView(rtv, clears[k], 0, nullptr);
            rtv.ptr += step;
        }
        cl->Close();
        queue->ExecuteCommandLists(1, lists);
        wait();
        if (allocParams(&shimParams) != 1 || !shimParams) { fputs("no parameter block\n", stderr); return 1; }
        wchar_t dir[MAX_PATH];
        GetCurrentDirectoryW(MAX_PATH, dir);
        shimFeature = create(L"", dir, device.Get(), nullptr, shimParams, nrW, nrH, 1, 1.0f, 0, 1.0f, 1.0f, -1.0f, 1, 0);
        if (!shimFeature) { fputs("the shim did not create the NR feature\n", stderr); return 1; }
        printf("shim %s: NR feature created, %s\n", shimPath,
               !guides ? "without motion" : seqDir ? "with the sequence's motion and depth" : "with the pan's motion and a depth");
    }

    // A real game spends seconds loading before its first frames; the hook's
    // thread needs a moment to find the swap chain's vtable.
    Sleep(1500);
    // The frames: the picture panned by `shift` pixels (a wrap-around copy).
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    double slowest = 0.0, sum = 0.0;
    int counted = 0, shift = 0;
    UINT lastIndex = 0;
    std::vector<unsigned char> seqData;
    bool seqDepth = false;
    for (int k = 0; k < frames && (!seqDir || k < int(seqFiles.size())); ++k) {
        if (seqDir) {
            // Frame k of the sequence: its picture into the upload buffer, its
            // motion and depth into theirs.
            FILE* sf = fopen(seqFiles[k].c_str(), "rb");
            if (!sf) { fputs("cannot read a sequence frame\n", stderr); return 1; }
            _fseeki64(sf, 0, SEEK_END);
            seqData.resize(size_t(_ftelli64(sf)));
            _fseeki64(sf, 0, SEEK_SET);
            const bool read = fread(seqData.data(), 1, seqData.size(), sf) == seqData.size();
            fclose(sf);
            unsigned hdr[4] = {};
            if (read && seqData.size() > 36) memcpy(hdr, seqData.data() + 8, 16);
            if (!read || memcmp(seqData.data(), "D5INP001", 8) || hdr[0] != w || hdr[1] != h || hdr[2] != 8) {
                fputs("sequence frame is not an RGBA16F picture of the image's size\n", stderr); return 1;
            }
            const size_t pitch = hdr[3], plane = size_t(w) * h;
            unsigned char* p = nullptr;
            upload->Map(0, nullptr, reinterpret_cast<void**>(&p));
            for (UINT y = 0; y < h; ++y)
                for (UINT x = 0; x < w; ++x) {
                    unsigned char* d = p + size_t(y) * fp.Footprint.RowPitch + x * 4;
                    for (int c = 0; c < 3; ++c) {
                        unsigned short hv;
                        memcpy(&hv, seqData.data() + 36 + size_t(y) * pitch + x * 8 + c * 2, 2);
                        float v = HalfToFloat(hv);
                        v = !(v > 0.0f) ? 0.0f : v > 1.0f ? 1.0f : v;
                        if (seqLinear) v = v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
                        d[c] = (unsigned char)(v * 255.0f + 0.5f);
                    }
                    d[3] = 255;
                }
            upload->Unmap(0, nullptr);
            size_t at = 36 + pitch * h;
            const float *depthAt = nullptr, *mvAt = nullptr;
            while (at + 16 <= seqData.size()) {
                if (!memcmp(seqData.data() + at, "D5DEP001", 8)) { depthAt = reinterpret_cast<const float*>(seqData.data() + at + 16); at += 16 + plane * 4; }
                else if (!memcmp(seqData.data() + at, "D5MOV001", 8)) { mvAt = reinterpret_cast<const float*>(seqData.data() + at + 16); at += 16 + plane * 8; }
                else break;
            }
            if (!mvAt) { fputs("sequence frame without motion\n", stderr); return 1; }
            mvUpload->Map(0, nullptr, reinterpret_cast<void**>(&p));
            for (UINT y = 0; y < h; ++y) {
                float* row = reinterpret_cast<float*>(p + size_t(y) * mvFp.Footprint.RowPitch);
                memcpy(row, mvAt + size_t(y) * w * 2, size_t(w) * 8);
                if (pixelGuides)
                    for (UINT x = 0; x < w; ++x) { row[x * 2] *= float(w); row[x * 2 + 1] *= float(h); }
            }
            mvUpload->Unmap(0, nullptr);
            seqDepth = depthAt != nullptr;
            if (depthAt) {
                depthUpload->Map(0, nullptr, reinterpret_cast<void**>(&p));
                for (UINT y = 0; y < h; ++y) {
                    float* row = reinterpret_cast<float*>(p + size_t(y) * depthFp.Footprint.RowPitch);
                    memcpy(row, depthAt + size_t(y) * w, size_t(w) * 4);
                    if (flipDepth)
                        for (UINT x = 0; x < w; ++x) row[x] = 1.0f - row[x];
                }
                depthUpload->Unmap(0, nullptr);
            }
        }
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        const UINT index = sc->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> back;
        sc->GetBuffer(index, IID_PPV_ARGS(&back));
        shift = seqDir ? 0 : (k * pan) % int(w);
        alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = back.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        if (seqDir) {
            cl->CopyTextureRegion(&dst, 0, 0, 0, &bufUp, nullptr);
            if (shimEvaluate && guides) {
                D3D12_RESOURCE_BARRIER g[2] = {Transition(nrMotion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST),
                                               Transition(nrDepth.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST)};
                cl->ResourceBarrier(2, g);
                D3D12_TEXTURE_COPY_LOCATION gt{}, gb{};
                gt.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                gb.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                gt.pResource = nrMotion.Get(); gb.pResource = mvUpload.Get(); gb.PlacedFootprint = mvFp;
                cl->CopyTextureRegion(&gt, 0, 0, 0, &gb, nullptr);
                if (seqDepth) {
                    gt.pResource = nrDepth.Get(); gb.pResource = depthUpload.Get(); gb.PlacedFootprint = depthFp;
                    cl->CopyTextureRegion(&gt, 0, 0, 0, &gb, nullptr);
                }
                g[0] = Transition(nrMotion.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
                g[1] = Transition(nrDepth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
                cl->ResourceBarrier(2, g);
            }
        } else if (shift) {
            D3D12_BOX left{UINT(shift), 0, 0, w, h, 1}, right{0, 0, 0, UINT(shift), h, 1};
            cl->CopyTextureRegion(&dst, 0, 0, 0, &texSource, &left);
            cl->CopyTextureRegion(&dst, w - UINT(shift), 0, 0, &texSource, &right);
        } else {
            cl->CopyTextureRegion(&dst, 0, 0, 0, &texSource, nullptr);
        }
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &b);
        // The upscaler's NR call, on the frame's own list.
        if (shimEvaluate &&
            shimEvaluate(cl.Get(), shimFeature, shimParams, nrColor.Get(), guides ? nrDepth.Get() : nullptr,
                         guides ? nrMotion.Get() : nullptr, nrOutput.Get(), nrW, nrH, guideW, guideH, flipDepth ? 0 : 1, 0,
                         1.0f, 0, 1.0f, 1.0f, -1.0f, 1, mvScaleX, mvScaleY) != 1)
            ++shimFailures;
        cl->Close();
        queue->ExecuteCommandLists(1, lists);
        back.Reset();
        sc->Present(0, 0);
        lastIndex = index;
        wait();
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        if (k >= frames / 2) { sum += ms; ++counted; slowest = ms > slowest ? ms : slowest; }
    }
    printf("%d frames; second half: %.2f ms a frame (slowest %.2f)\n", frames, counted ? sum / counted : 0.0, slowest);
    if (shimEvaluate) printf("NR evaluates that failed: %d\n", shimFailures);

    // The last presented buffer against what was drawn into it.
    ComPtr<ID3D12Resource> back;
    sc->GetBuffer(lastIndex, IID_PPV_ARGS(&back));
    D3D12_TEXTURE_COPY_LOCATION texBack{};
    texBack.pResource = back.Get(); texBack.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
    b = Transition(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &b);
    cl->CopyTextureRegion(&bufRead, 0, 0, 0, &texBack, nullptr);
    b = Transition(back.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    cl->ResourceBarrier(1, &b);
    cl->Close();
    queue->ExecuteCommandLists(1, lists);
    wait();
    std::vector<unsigned char> out(rgb.size()), drawn(rgb.size());
    {
        unsigned char* p = nullptr;
        readback->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x) {
                memcpy(&out[(size_t(y) * w + x) * 3], p + size_t(y) * fp.Footprint.RowPitch + x * 4, 3);
                memcpy(&drawn[(size_t(y) * w + x) * 3], &rgb[(size_t(y) * w + (x + UINT(shift)) % w) * 3], 3);
            }
        readback->Unmap(0, nullptr);
    }
    double diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) diff += std::abs(int(out[i]) - int(drawn[i]));
    diff /= double(out.size());
    printf("last frame: mean |presented - drawn| %.2f of 255 (%s)\n", diff, diff > 0.5 ? "the hook edited it" : "untouched");
    for (int k = 0; k < 2; ++k) {
        const std::string path = std::string(argv[2]) + (k ? "_out.rgb" : "_in.rgb");
        if (FILE* o = fopen(path.c_str(), "wb")) {
            fwrite(dims, 4, 2, o);
            fwrite(k ? out.data() : drawn.data(), 1, out.size(), o);
            fclose(o);
        }
    }
    back.Reset();
    wait();
    if (argc > 10) Sleep(DWORD(atoi(argv[10])) * 1000);
    return diff > 0.5 ? 0 : 1;
}
