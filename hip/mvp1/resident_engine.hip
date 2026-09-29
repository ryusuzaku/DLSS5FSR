// In-process resident candidate engine for the NGX shim (built as a DLL).
// The shim hands over its HIP staging pointer; the engine copies the frame on
// its own non-blocking stream, then a worker thread runs the candidate input
// crop, the C32 front end, encoder5-30, ViT31-38, decoder39-69 and head70, and
// publishes a 256x256 RGBA8/RGBA16F preview. Arithmetic is the resident path
// validated by the harness and server checks. Failures throw inside the engine
// and are reported through re_last_error; nothing here may exit the process.
#include <stdexcept>
#include <string>
#define HIP_CHECK(expr) do { hipError_t e_ = (expr); if (e_ != hipSuccess) \
    throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(e_)); } while (0)
#define HEAD70_FATAL(message) throw std::runtime_error(message)
#define HEAD70_INVALID(message) throw std::runtime_error(message)
#include "split512_resident.h"
#include "vit_bridge_gather.hip"
#include "vit_resident.h"
#include "decoder39_resident.h"
#include "c256_resident.h"
#include "c128_resident.h"
#include "c64_resident.h"
#include "c32_resident.h"
#include "front_resident.h"
#include "candidate_input_256.hip"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

using namespace c512_resident;

namespace {
std::vector<std::pair<std::string,int>> blocks(const std::string& path, size_t count) {
    std::ifstream file(path);
    std::vector<std::pair<std::string,int>> out;
    std::string line;
    if (!file) throw std::runtime_error("cannot read block list " + path);
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3 || line[1] != '\t' || line[0] < '0' || line[0] > '3' || out.size() >= count)
            throw std::runtime_error("invalid block list " + path);
        out.emplace_back(line.substr(2), line[0]-'0');
    }
    if (out.size() != count) throw std::runtime_error("incomplete block list " + path);
    return out;
}

std::vector<std::string> folders(const std::string& path, size_t count) {
    std::ifstream file(path);
    std::vector<std::string> out;
    std::string line;
    if (!file) throw std::runtime_error("cannot read folder list " + path);
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || out.size() >= count) throw std::runtime_error("invalid folder list " + path);
        out.push_back(line);
    }
    if (out.size() != count) throw std::runtime_error("incomplete folder list " + path);
    return out;
}

// numpy-equivalent sRGB encoding of the head's linear RGB (clip, sRGB, rint).
unsigned char srgb8(float linear) {
    float v = linear < 0.0f ? 0.0f : (linear > 1.0f ? 1.0f : linear);
    float s = v <= 0.0031308f ? 12.92f*v : 1.055f*powf(v, 1.0f/2.4f) - 0.055f;
    return (unsigned char)nearbyintf(s*255.0f);
}

// Configuration: 24 lines, the resident_frame_server arguments after the
// executable (c512 blocks, width, height, ... downsample8, front fixture).
struct Engine {
    int width = 8, height = 8;
    size_t n = 0, hn = 0;
    hipStream_t stream = nullptr;
    std::unique_ptr<c64_resident::Chain> encoder8;
    std::unique_ptr<c64_resident::Downsample> down8;
    std::unique_ptr<c128_resident::Chain> encoder14;
    std::unique_ptr<c128_resident::Downsample> down14;
    std::unique_ptr<c256_resident::Chain> encoder22;
    std::unique_ptr<c256_resident::Downsample> down22;
    std::unique_ptr<Chain> chain, decoder;
    std::unique_ptr<IndexMap> map;
    std::unique_ptr<Buffer> bridge, color, raw_input, rgb_native, rgb_public;
    size_t raw_bytes = 0;
    std::unique_ptr<vit_resident::Chain> vit;
    std::unique_ptr<decoder39_resident::Entry> entry;
    std::unique_ptr<c256_resident::Prefix> prefix48;
    std::unique_ptr<c256_resident::Chain> c256;
    std::unique_ptr<c128_resident::Prefix> prefix56;
    std::unique_ptr<c128_resident::Chain> c128;
    std::unique_ptr<c64_resident::Prefix> prefix62;
    std::unique_ptr<c64_resident::Chain> c64;
    std::unique_ptr<c32_resident::Prefix> prefix66;
    std::unique_ptr<c32_resident::Chain> c32;
    std::unique_ptr<c32_resident::Head> head70;
    std::unique_ptr<front_resident::FrontEnd> front;
    hipEvent_t start = nullptr, stop = nullptr;

    // Worker state.
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool quit = false, pending = false, busy = false, fresh = false, failed = false;
    int frame_w = 0, frame_h = 0, frame_pitch = 0, frame_bpp = 0;
    std::vector<unsigned char> out8, out16;
    std::vector<float> host_native, host_public, host_color;
    float last_ms = 0.0f;
    unsigned long long completed = 0;
    std::string error;

    explicit Engine(const std::vector<std::string>& a) {
        if (a.size() != 24) throw std::runtime_error("engine config needs 24 lines");
        width = std::stoi(a[1]); height = std::stoi(a[2]);
        if (width != 8 || height != 8) throw std::runtime_error("engine supports the 256x256 extent only");
        n = size_t(width)*height*512; hn = size_t(width)*height/4*1024;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        c512_resident::stream = stream;
        encoder8.reset(new c64_resident::Chain(width*8,height*8,blocks(a[21],4),true));
        down8.reset(new c64_resident::Downsample(width*8,height*8,a[22]));
        encoder14.reset(new c128_resident::Chain(width*4,height*4,blocks(a[19],6),true));
        down14.reset(new c128_resident::Downsample(width*4,height*4,a[20]));
        encoder22.reset(new c256_resident::Chain(width*2,height*2,blocks(a[17],8),true));
        down22.reset(new c256_resident::Downsample(width*2,height*2,a[18]));
        chain.reset(new Chain(width,height,blocks(a[0],8),a[3]));
        map.reset(new IndexMap(a[4].c_str(),hn));
        bridge.reset(new Buffer(hn));
        vit.reset(new vit_resident::Chain(width*height/4,folders(a[5],8)));
        entry.reset(new decoder39_resident::Entry(width/2,height/2,a[6]));
        decoder.reset(new Chain(width,height,blocks(a[7],8)));
        prefix48.reset(new c256_resident::Prefix(width,height,a[8]));
        c256.reset(new c256_resident::Chain(width*2,height*2,blocks(a[9],8)));
        prefix56.reset(new c128_resident::Prefix(width*2,height*2,a[10]));
        c128.reset(new c128_resident::Chain(width*4,height*4,blocks(a[11],6)));
        prefix62.reset(new c64_resident::Prefix(width*4,height*4,a[12]));
        c64.reset(new c64_resident::Chain(width*8,height*8,blocks(a[13],4)));
        prefix66.reset(new c32_resident::Prefix(width*8,height*8,a[14]));
        c32.reset(new c32_resident::Chain(width*16,height*16,blocks(a[15],4)));
        head70.reset(new c32_resident::Head(width*32,height*32,a[16]));
        front.reset(new front_resident::FrontEnd(width*32,height*32,a[23]));
        color.reset(new Buffer(6*n)); rgb_native.reset(new Buffer(6*n)); rgb_public.reset(new Buffer(6*n));
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipDeviceSynchronize());
        worker = std::thread([this] { loop(); });
    }

    ~Engine() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            quit = true;
        }
        wake.notify_all();
        if (worker.joinable()) worker.join();
        if (stream) (void)hipStreamSynchronize(stream);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        c512_resident::stream = nullptr;
        // Members release their device memory in reverse order; then the stream.
        front.reset(); head70.reset(); c32.reset(); prefix66.reset(); c64.reset(); prefix62.reset();
        c128.reset(); prefix56.reset(); c256.reset(); prefix48.reset(); decoder.reset(); entry.reset();
        vit.reset(); bridge.reset(); map.reset(); chain.reset(); down22.reset(); encoder22.reset();
        down14.reset(); encoder14.reset(); down8.reset(); encoder8.reset();
        color.reset(); rgb_native.reset(); rgb_public.reset(); raw_input.reset();
        if (stream) (void)hipStreamDestroy(stream);
    }

    // Caller thread: snapshot the staged frame; the shim's buffer is reused next frame.
    int submit(const void* staging, int w, int h, int pitch, int bpp) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (busy || pending) return 0;
        if (!staging || w <= 0 || h <= 0 || w > 4096 || h > 4096 || (bpp != 4 && bpp != 8) ||
            pitch < w*bpp) { error = "unsupported staging format"; return -1; }
        size_t bytes = size_t(pitch)*h;
        try {
            if (bytes > raw_bytes) {
                raw_input.reset(new Buffer((bytes+3)/4));
                raw_bytes = raw_input->count*sizeof(float);
            }
            HIP_CHECK(hipMemcpyAsync(raw_input->data, staging, bytes, hipMemcpyDeviceToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        frame_w = w; frame_h = h; frame_pitch = pitch; frame_bpp = bpp;
        pending = true;
        wake.notify_all();
        return 1;
    }

    void loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this] { return quit || pending; });
                if (quit) return;
                pending = false; busy = true;
            }
            try {
                run();
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex);
                failed = true; error = e.what();
            }
            std::lock_guard<std::mutex> lock(mutex);
            busy = false;
        }
    }

    void run() {
        HIP_CHECK(hipEventRecord(start, stream));
        const unsigned char* src = reinterpret_cast<const unsigned char*>(raw_input->data);
        hipLaunchKernelGGL(k_candidate_input_256, dim3(256), dim3(256), 0, stream,
                           src, color->data, frame_w, frame_h, frame_pitch, frame_bpp);
        HIP_CHECK(hipGetLastError());
        bool ok = front->run_from_device({color->data,6*n}) &&
            encoder8->run_from_device(front->c64_view()) && down8->run_from_device(encoder8->raw_view()) &&
            encoder14->run_from_device(down8->final_view()) && down14->run_from_device(encoder14->raw_view()) &&
            encoder22->run_from_device(down14->final_view()) && down22->run_from_device(encoder22->raw_view()) &&
            chain->run_from_device(down22->final_view());
        if (ok) {
            auto head = chain->head_view();
            hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, stream,
                               head.data, map->data, bridge->data, int(hn));
            HIP_CHECK(hipGetLastError());
            ok = vit->run_from_device({bridge->data,hn}) &&
                entry->run_from_device(vit->final_view(), chain->final_view()) &&
                decoder->run_from_device(entry->final_view()) &&
                prefix48->run_from_device(decoder->final_view(), encoder22->final_view()) &&
                c256->run_from_device(prefix48->final_view()) &&
                prefix56->run_from_device(c256->final_view(), encoder14->final_view()) &&
                c128->run_from_device(prefix56->final_view()) &&
                prefix62->run_from_device(c128->final_view(), encoder8->final_view()) &&
                c64->run_from_device(prefix62->final_view()) &&
                prefix66->run_from_device(c64->final_view(), front->skip4_view()) &&
                c32->run_from_device(prefix66->final_view()) &&
                head70->run_from_device(c32->final_view(), front->skip0_view(), {color->data,6*n});
        }
        if (!ok) throw std::runtime_error("resident execution rejected a device view");
        HIP_CHECK(hipMemcpyAsync(rgb_native->data, head70->final_view().data, 6*n*sizeof(float),
                                 hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(rgb_public->data, head70->public_view().data, 6*n*sizeof(float),
                                 hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipEventRecord(stop, stream));
        std::vector<float> native(6*n), pub(6*n), rgb(6*n);
        HIP_CHECK(hipMemcpyAsync(native.data(), rgb_native->data, native.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(pub.data(), rgb_public->data, pub.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(rgb.data(), color->data, rgb.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        std::lock_guard<std::mutex> lock(mutex);
        host_native.swap(native); host_public.swap(pub); host_color.swap(rgb);
        last_ms = elapsed; fresh = true; ++completed;
    }

    // Copy the newest preview for the chosen gain (0 native .03125, 1 public 1.0).
    int poll(int gain, unsigned char* rgba8, unsigned char* rgba16, float* device_ms) {
        std::vector<float> rgb;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (failed) return -1;
            if (!fresh) return 0;
            fresh = false;
            rgb = gain ? host_public : host_native;
            if (device_ms) *device_ms = last_ms;
        }
        for (size_t p = 0; p < 256*256; ++p) {
            for (int c = 0; c < 4; ++c) {
                unsigned char v = c < 3 ? srgb8(rgb[p*3+c]) : 255;
                if (rgba8) rgba8[p*4+c] = v;
                if (rgba16) {
                    _Float16 h = (_Float16)(c < 3 ? float(v)/255.0f : 1.0f);
                    memcpy(rgba16 + (p*4+c)*2, &h, 2);
                }
            }
        }
        return 1;
    }
};

std::mutex g_mutex;
std::unique_ptr<Engine> g_engine;
std::string g_error;
} // namespace

extern "C" {
// Returns 1 on success, 0 on failure (see re_last_error).
__declspec(dllexport) int re_create(const char* config_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        g_engine.reset();
        std::ifstream file(config_path);
        if (!file) throw std::runtime_error(std::string("cannot read engine config ") + config_path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        g_engine.reset(new Engine(lines));
        g_error.clear();
        return 1;
    } catch (const std::exception& e) {
        g_engine.reset();
        g_error = e.what();
        return 0;
    }
}

// 1 accepted, 0 busy (try again later), -1 failed/disabled.
__declspec(dllexport) int re_submit(const void* staging, int width, int height, int pitch, int bpp) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->submit(staging, width, height, pitch, bpp);
}

// 1 new preview written, 0 nothing new, -1 failed/disabled.
__declspec(dllexport) int re_poll(int gain, unsigned char* rgba8, unsigned char* rgba16, float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->poll(gain, rgba8, rgba16, device_ms);
}

// Diagnostic copy of the latest linear RGB (256x256x3 floats) and its input.
__declspec(dllexport) int re_latest_rgb(int gain, float* rgb, float* color) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    if (!g_engine->completed) return 0;
    const auto& source = gain ? g_engine->host_public : g_engine->host_native;
    if (rgb) memcpy(rgb, source.data(), source.size()*sizeof(float));
    if (color) memcpy(color, g_engine->host_color.data(), g_engine->host_color.size()*sizeof(float));
    return 1;
}

__declspec(dllexport) const char* re_last_error() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_engine) {
        std::lock_guard<std::mutex> inner(g_engine->mutex);
        if (!g_engine->error.empty()) g_error = g_engine->error;
    }
    return g_error.c_str();
}

__declspec(dllexport) void re_destroy() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_engine.reset();
}
}
