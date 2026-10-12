// Headless test of the final-picture front end (src/present): a D3D12 device
// and queue, an off-screen "back buffer" refreshed from an image every frame
// as a game would render it, final_picture::Process on it, the result read
// back. Reports the time of a whole frame (copy, network, compose) and
// writes the last frame.
//
// Usage: present_harness engine.dll engine_config.txt image.rgb out_prefix [frames] [scale]
//   image.rgb: two uint32 (width, height), then RGB8 rows (display-encoded).
//   out_prefix_in.rgb / _out.rgb are written in the same format.

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

#include "../src/present/final_picture.h"

using Microsoft::WRL::ComPtr;

static double Ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    if (argc < 5) {
        fputs("usage: present_harness engine.dll engine_config.txt image.rgb out_prefix [frames] [scale]\n", stderr);
        return 2;
    }
    const int frames = argc > 5 ? atoi(argv[5]) : 30;
    const float scale = argc > 6 ? float(atof(argv[6])) : 0.0f;

    FILE* f = fopen(argv[3], "rb");
    unsigned dims[2];
    if (!f || fread(dims, 4, 2, f) != 2) { fputs("cannot read the image\n", stderr); return 1; }
    const UINT w = dims[0], h = dims[1];
    std::vector<unsigned char> rgb(size_t(w) * h * 3);
    if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) { fputs("short image\n", stderr); return 1; }
    fclose(f);

    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) { fputs("no D3D12 device\n", stderr); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) { fputs("D3D12 setup failed\n", stderr); return 1; }
    cl->Close();
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    auto wait = [&] {
        queue->Signal(fence.Get(), ++fenceValue);
        fence->SetEventOnCompletion(fenceValue, event);
        WaitForSingleObject(event, 20000);
    };

    // The source picture, the "back buffer", an upload and a read-back buffer.
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
    ComPtr<ID3D12Resource> source, back, upload, readback;
    D3D12_RESOURCE_DESC bbd = td;
    bbd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;  // as a swap chain's buffers are
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&source))) ||
        FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bbd, D3D12_RESOURCE_STATE_PRESENT, nullptr, IID_PPV_ARGS(&back))) ||
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
    D3D12_TEXTURE_COPY_LOCATION texSource{}, texBack{}, bufUp{}, bufRead{};
    texSource.pResource = source.Get(); texSource.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    texBack.pResource = back.Get(); texBack.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
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

    final_picture::Config cfg;
    wchar_t wide[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, argv[1], -1, wide, MAX_PATH);
    cfg.engineDll = wide;
    cfg.engineConfig = argv[2];
    cfg.workingScale = scale;
    const double t0 = Ms();
    if (!final_picture::Init(device.Get(), queue.Get(), cfg)) { fprintf(stderr, "init failed: %s\n", final_picture::GetStats().error); return 1; }
    printf("engine ready in %.0f ms\n", Ms() - t0);

    double sum = 0.0, worst = 0.0;
    int counted = 0;
    const int warm = frames > 12 ? 8 : 0;
    for (int k = 0; k < frames; ++k) {
        const double start = Ms();
        // The game "renders" the frame into the back buffer.
        alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &b);
        cl->CopyResource(back.Get(), source.Get());
        b = Transition(back.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &b);
        cl->Close();
        queue->ExecuteCommandLists(1, lists);
        const bool ok = final_picture::Process(back.Get());
        wait();  // "present"
        const double ms = Ms() - start;
        if (!ok) printf("frame %d: not processed (%s)\n", k, final_picture::GetStats().error);
        if (k >= warm) { sum += ms; worst = ms > worst ? ms : worst; ++counted; }
    }
    const final_picture::Stats& st = final_picture::GetStats();
    printf("frame %ux%u, working %ux%u: %.2f ms a frame (worst %.2f) over %d frames; %llu processed, %llu skipped\n",
           w, h, st.workWidth, st.workHeight, counted ? sum / counted : 0.0, worst, counted, st.frames, st.skipped);

    alloc->Reset(); cl->Reset(alloc.Get(), nullptr);
    b = Transition(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &b);
    cl->CopyTextureRegion(&bufRead, 0, 0, 0, &texBack, nullptr);
    cl->Close();
    queue->ExecuteCommandLists(1, lists);
    wait();
    std::vector<unsigned char> out(rgb.size());
    {
        unsigned char* p = nullptr;
        readback->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x)
                memcpy(&out[(size_t(y) * w + x) * 3], p + size_t(y) * fp.Footprint.RowPitch + x * 4, 3);
        readback->Unmap(0, nullptr);
    }
    double diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) diff += std::abs(int(out[i]) - int(rgb[i]));
    printf("last frame: mean |out - in| %.2f of 255\n", diff / out.size());
    for (int k = 0; k < 2; ++k) {
        const std::string path = std::string(argv[4]) + (k ? "_out.rgb" : "_in.rgb");
        if (FILE* o = fopen(path.c_str(), "wb")) {
            fwrite(dims, 4, 2, o);
            fwrite(k ? out.data() : rgb.data(), 1, out.size(), o);
            fclose(o);
        }
    }
    final_picture::Shutdown();
    return st.frames ? 0 : 1;
}
