// A minimal D3D12 "game" for the present hook (src/present/hook.cpp): a
// window with a flip swap chain that shows one picture, panned a little
// every frame. Run from a directory that holds the hook as version.dll and
// its dlss5_present.ini: the hook loads with the process (this program
// imports the version API), patches the swap chain and processes each frame.
// After the last frame the last presented buffer is read back and compared
// with what was drawn: any change is the hook's.
//
// Usage: present_window image.rgb out_prefix [frames] [show]
//   image.rgb: two uint32 (width, height), then RGB8 rows.
//   show: 1 shows the window (default hidden).

#include <windows.h>

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

int main(int argc, char** argv) {
    if (argc < 3) { fputs("usage: present_window image.rgb out_prefix [frames] [show]\n", stderr); return 2; }
    const int frames = argc > 3 ? atoi(argv[3]) : 240;
    setvbuf(stdout, nullptr, _IONBF, 0);  // a crash under the hook must not lose what was printed
    const bool show = argc > 4 && atoi(argv[4]) != 0;
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

    // A real game spends seconds loading before its first frames; the hook's
    // thread needs a moment to find the swap chain's vtable.
    Sleep(1500);
    // The frames: the picture panned by `shift` pixels (a wrap-around copy).
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    double slowest = 0.0, sum = 0.0;
    int counted = 0, shift = 0;
    UINT lastIndex = 0;
    for (int k = 0; k < frames; ++k) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        const UINT index = sc->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> back;
        sc->GetBuffer(index, IID_PPV_ARGS(&back));
        shift = (k * 4) % int(w);
        alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = back.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        if (shift) {
            D3D12_BOX left{UINT(shift), 0, 0, w, h, 1}, right{0, 0, 0, UINT(shift), h, 1};
            cl->CopyTextureRegion(&dst, 0, 0, 0, &texSource, &left);
            cl->CopyTextureRegion(&dst, w - UINT(shift), 0, 0, &texSource, &right);
        } else {
            cl->CopyTextureRegion(&dst, 0, 0, 0, &texSource, nullptr);
        }
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &b);
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
    return diff > 0.5 ? 0 : 1;
}
