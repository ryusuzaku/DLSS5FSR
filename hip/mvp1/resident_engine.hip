// In-process resident candidate engine for the NGX shim (built as a DLL).
// The shim hands over its HIP staging pointer; the engine copies the frame on
// its own non-blocking stream, then a worker thread runs the candidate input,
// the C32 front end, encoder5-30, ViT31-38, decoder39-69 and head70.
//   crop mode (re_submit/re_poll): the validated 256x256 centre-crop preview.
//   full mode (re_submit_full/re_poll_frame): the whole frame, edge-padded to
//   the smallest network extent (multiples of 64 whose ViT token count is 16
//   or a multiple of 64), written back at the frame's own size and format.
// Arithmetic is the resident path validated by the harness and server checks.
// Failures throw inside the engine and are reported through re_last_error;
// nothing here may exit the process.
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
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

using namespace c512_resident;

// Full-frame candidate input: the staged proxy at 1:1, edge-replicated into
// the padded network extent, then the same sRGB decode as the 256 crop.
__global__ void k_candidate_input_full(const unsigned char* src, float* output, int frame_w, int frame_h,
                                       int pitch, int bpp, int width, int height) {
    const int pixel = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (pixel >= width*height) return;
    const int x = min(pixel % width, frame_w-1), y = min(pixel / width, frame_h-1);
    for (int c = 0; c < 3; ++c) {
        float srgb = candidate_proxy_channel(src, pitch, bpp, x, y, c);
        srgb = fminf(fmaxf(srgb, 0.0f), 1.0f);
        output[pixel*3+c] = srgb <= 0.04045f ? srgb/12.92f : powf((srgb+0.055f)/1.055f, 2.4f);
    }
}

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

float srgb_encode(float linear) {
    float v = linear < 0.0f ? 0.0f : (linear > 1.0f ? 1.0f : linear);
    return v <= 0.0031308f ? 12.92f*v : 1.055f*powf(v, 1.0f/2.4f) - 0.055f;
}

// numpy-equivalent sRGB encoding of the head's linear RGB (clip, sRGB, rint).
unsigned char srgb8(float linear) { return (unsigned char)nearbyintf(srgb_encode(linear)*255.0f); }

// Full mode: the frame's sRGB bytes (RGBA8, or RGBA16F when bpp is 8) from
// the network's linear RGB (row stride W pixels), on the GPU. powf may
// differ from the host's by an ulp, which can move a rare 8-bit rounding.
__global__ void k_encode_frame(const float* linear, unsigned char* out, int frame_w, int frame_h, int W, int bpp) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= frame_w*frame_h) return;
    int x = i % frame_w, y = i / frame_w;
    const float* p = linear + (size_t(y)*W + x)*3;
    for (int c = 0; c < 4; ++c) {
        float s = 1.0f;
        if (c < 3) {
            float v = fminf(fmaxf(p[c], 0.0f), 1.0f);
            s = v <= 0.0031308f ? 12.92f*v : 1.055f*powf(v, 1.0f/2.4f) - 0.055f;
        }
        if (bpp == 4) {
            out[size_t(i)*4+c] = c < 3 ? (unsigned char)nearbyintf(s*255.0f) : 255;
        } else {
            _Float16 h = (_Float16)s;
            __builtin_memcpy(out + size_t(i)*8 + c*2, &h, 2);
        }
    }
}

// Head -> ViT logical token/channel map (tools/audit_native_vit_logical_map.py).
std::vector<int32_t> vit_gather_map(int tokens) {
    std::vector<int32_t> map(size_t(tokens)*1024);
    for (int t = 0; t < tokens; ++t) {
        int raster = (t & ~15) | ((t & 1) << 3) | ((t & 14) >> 1);
        for (int c = 0; c < 1024; ++c) {
            int channel = (c & ~31) | ((c & 1) << 1) | ((c & 2) >> 1) | ((c & 4) << 2) | ((c & 24) >> 1);
            map[size_t(t)*1024+c] = raster*1024 + channel;
        }
    }
    return map;
}

// Smallest padded extent the resident network supports for a frame.
void network_extent(int frame_w, int frame_h, int& width, int& height) {
    int w0 = (frame_w+63)/64*64, h0 = (frame_h+63)/64*64;
    long long best = -1;
    for (int w = w0; w <= w0+512; w += 64)
        for (int h = h0; h <= h0+512; h += 64) {
            int tokens = (w/64)*(h/64);
            if (!(tokens % 64 == 0 || (w == 256 && h == 256))) continue;
            long long area = (long long)w*h;
            if (best < 0 || area < best) { best = area; width = w; height = h; }
        }
    if (best < 0) throw std::runtime_error("no supported network extent for the frame");
}

// All resident components for one network extent (RGB width x height).
struct Network {
    int W, H, width, height, tokens;
    size_t n, hn;
    std::unique_ptr<front_resident::FrontEnd> front;
    std::unique_ptr<c64_resident::Chain> encoder8;
    std::unique_ptr<c64_resident::Downsample> down8;
    std::unique_ptr<c128_resident::Chain> encoder14;
    std::unique_ptr<c128_resident::Downsample> down14;
    std::unique_ptr<c256_resident::Chain> encoder22;
    std::unique_ptr<c256_resident::Downsample> down22;
    std::unique_ptr<Chain> chain, decoder;
    std::unique_ptr<IndexMap> map;
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
    std::unique_ptr<Buffer> bridge, color, rgb_native, rgb_public;

    // The WMMA half weight cache is keyed by these weights' device pointers;
    // a later network can reuse the addresses, so drop the copies with them.
    ~Network() {
        if (c512_resident::stream) (void)hipStreamSynchronize(c512_resident::stream);
        wmma_gemm::state().clear();
    }

    // Config: the resident_frame_server arguments after the executable; the
    // width/height lines are ignored, weights come from the listed fixtures.
    Network(const std::vector<std::string>& a, int rgb_w, int rgb_h) :
        W(rgb_w), H(rgb_h), width(rgb_w/32), height(rgb_h/32), tokens((rgb_w/64)*(rgb_h/64)),
        n(size_t(width)*height*512), hn(size_t(tokens)*1024) {
        const int cap = 65536;  // C32 body rows per chunk above this
        front.reset(new front_resident::FrontEnd(W,H,a[23],cap));
        encoder8.reset(new c64_resident::Chain(width*8,height*8,blocks(a[21],4),true));
        down8.reset(new c64_resident::Downsample(width*8,height*8,a[22]));
        encoder14.reset(new c128_resident::Chain(width*4,height*4,blocks(a[19],6),true));
        down14.reset(new c128_resident::Downsample(width*4,height*4,a[20]));
        encoder22.reset(new c256_resident::Chain(width*2,height*2,blocks(a[17],8),true));
        down22.reset(new c256_resident::Downsample(width*2,height*2,a[18]));
        chain.reset(new Chain(width,height,blocks(a[0],8),a[3]));
        auto gather = vit_gather_map(tokens);
        std::vector<int32_t> inverse(gather.size());
        for (size_t i = 0; i < gather.size(); ++i) inverse[size_t(gather[i])] = int32_t(i);
        map.reset(new IndexMap(gather));
        bridge.reset(new Buffer(hn));
        vit.reset(new vit_resident::Chain(tokens,folders(a[5],8)));
        entry.reset(new decoder39_resident::Entry(width/2,height/2,a[6],inverse));
        decoder.reset(new Chain(width,height,blocks(a[7],8)));
        prefix48.reset(new c256_resident::Prefix(width,height,a[8]));
        c256.reset(new c256_resident::Chain(width*2,height*2,blocks(a[9],8)));
        prefix56.reset(new c128_resident::Prefix(width*2,height*2,a[10]));
        c128.reset(new c128_resident::Chain(width*4,height*4,blocks(a[11],6)));
        prefix62.reset(new c64_resident::Prefix(width*4,height*4,a[12]));
        c64.reset(new c64_resident::Chain(width*8,height*8,blocks(a[13],4)));
        prefix66.reset(new c32_resident::Prefix(width*8,height*8,a[14]));
        c32.reset(new c32_resident::Chain(width*16,height*16,blocks(a[15],4)));
        head70.reset(new c32_resident::Head(W,H,a[16],cap));
        color.reset(new Buffer(size_t(W)*H*3));
        rgb_native.reset(new Buffer(size_t(W)*H*3));
        rgb_public.reset(new Buffer(size_t(W)*H*3));
    }

    void run(const unsigned char* src, int fw, int fh, int pitch, int bpp, bool crop, hipStream_t stream) {
        if (crop) {
            if (W != 256 || H != 256) throw std::runtime_error("crop mode needs the 256x256 network");
            hipLaunchKernelGGL(k_candidate_input_256, dim3(256), dim3(256), 0, stream,
                               src, color->data, fw, fh, pitch, bpp);
        } else {
            hipLaunchKernelGGL(k_candidate_input_full, dim3((W*H+255)/256), dim3(256), 0, stream,
                               src, color->data, fw, fh, pitch, bpp, W, H);
        }
        HIP_CHECK(hipGetLastError());
        bool ok = front->run_from_device({color->data,color->count}) &&
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
                head70->run_from_device(c32->final_view(), front->skip0_view(), {color->data,color->count});
        }
        if (!ok) throw std::runtime_error("resident execution rejected a device view");
        HIP_CHECK(hipMemcpyAsync(rgb_native->data, head70->final_view().data, rgb_native->count*sizeof(float),
                                 hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(rgb_public->data, head70->public_view().data, rgb_public->count*sizeof(float),
                                 hipMemcpyDeviceToDevice, stream));
    }
};

struct Engine {
    std::vector<std::string> config;
    hipStream_t stream = nullptr;
    std::unique_ptr<Network> net;
    std::unique_ptr<Buffer> raw_input;
    size_t raw_bytes = 0;
    hipEvent_t start = nullptr, stop = nullptr;

    // Worker state.
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool quit = false, pending = false, busy = false, failed = false;
    bool fresh_crop = false, fresh_frame = false;
    bool job_crop = true;
    int job_gain = 0;
    int frame_w = 0, frame_h = 0, frame_pitch = 0, frame_bpp = 0;
    std::vector<float> host_native, host_public, host_color;
    bool host_valid = false;                        // full mode fetches the floats on demand
    std::unique_ptr<Buffer> encoded;                // full mode: device frame bytes
    unsigned char* pinned = nullptr;                // full mode: host readback of the frame bytes
    size_t pinned_bytes = 0;
    int result_w = 0, result_h = 0;                 // network extent of the latest result
    std::vector<unsigned char> frame_out;           // full mode: frame_w x frame_h x bpp
    int out_w = 0, out_h = 0, out_bpp = 0;
    float last_ms = 0.0f;
    unsigned long long completed = 0;
    std::string error;

    explicit Engine(const std::vector<std::string>& lines) : config(lines) {
        if (config.size() != 24) throw std::runtime_error("engine config needs 24 lines");
        // The caller's thread may carry a stale HIP error from its own work;
        // setup checks hipGetLastError() after its first launches.
        (void)hipGetLastError();
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        c512_resident::stream = stream;
        // Load weights and validate the configuration up front at 256x256.
        net.reset(new Network(config, 256, 256));
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
        if (pinned) (void)hipHostFree(pinned);
        net.reset(); raw_input.reset(); encoded.reset();
        wmma_gemm::state().clear();  // half weight copies are keyed by the freed pointers
        c512_resident::stream = nullptr;
        if (stream) (void)hipStreamDestroy(stream);
    }

    // Caller thread: snapshot the staged frame; the shim's buffer is reused next frame.
    int submit(const void* staging, int w, int h, int pitch, int bpp, bool crop, int gain) {
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
        job_crop = crop; job_gain = gain ? 1 : 0;
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
        int W = 256, H = 256;
        if (!job_crop) network_extent(frame_w, frame_h, W, H);
        if (!net || net->W != W || net->H != H) {
            net.reset();
            net.reset(new Network(config, W, H));
        }
        HIP_CHECK(hipEventRecord(start, stream));
        net->run(reinterpret_cast<const unsigned char*>(raw_input->data), frame_w, frame_h, frame_pitch,
                 frame_bpp, job_crop, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        size_t count = size_t(W)*H*3;
        std::vector<float> native, pub, rgb;
        std::vector<unsigned char> frame;
        if (job_crop) {
            native.resize(count); pub.resize(count); rgb.resize(count);
            HIP_CHECK(hipMemcpyAsync(native.data(), net->rgb_native->data, count*sizeof(float), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(pub.data(), net->rgb_public->data, count*sizeof(float), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(rgb.data(), net->color->data, count*sizeof(float), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        } else {
            // Encode on the GPU and read back only the frame bytes (pinned).
            size_t bytes = size_t(frame_w)*frame_h*frame_bpp;
            if (!encoded || encoded->count*sizeof(float) < bytes) encoded.reset(new Buffer((bytes+3)/4));
            if (pinned_bytes < bytes) {
                if (pinned) HIP_CHECK(hipHostFree(pinned));
                pinned = nullptr; pinned_bytes = 0;
                HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes));
                pinned_bytes = bytes;
            }
            auto* out = reinterpret_cast<unsigned char*>(encoded->data);
            hipLaunchKernelGGL(k_encode_frame, dim3((frame_w*frame_h+255)/256), dim3(256), 0, stream,
                               job_gain ? net->rgb_public->data : net->rgb_native->data, out, frame_w, frame_h, W, frame_bpp);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpyAsync(pinned, out, bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            frame.assign(pinned, pinned + bytes);
        }
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        std::lock_guard<std::mutex> lock(mutex);
        host_native.swap(native); host_public.swap(pub); host_color.swap(rgb);
        host_valid = job_crop;
        result_w = W; result_h = H;
        if (job_crop) {
            fresh_crop = true;
        } else {
            frame_out.swap(frame); out_w = frame_w; out_h = frame_h; out_bpp = frame_bpp;
            fresh_frame = true;
        }
        last_ms = elapsed; ++completed;
    }

    // Crop mode: newest 256x256 preview for the chosen gain (0 native .03125, 1 public 1.0).
    int poll(int gain, unsigned char* rgba8, unsigned char* rgba16, float* device_ms) {
        std::vector<float> rgb;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (failed) return -1;
            if (!fresh_crop) return 0;
            fresh_crop = false;
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

    // Full mode: copy the newest frame if it matches the destination's size/format.
    int poll_frame(unsigned char* dst, size_t pitch, int w, int h, int bpp, float* device_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!fresh_frame) return 0;
        if (out_w != w || out_h != h || out_bpp != bpp || pitch < size_t(w)*bpp) {
            fresh_frame = false;  // stale extent after a resolution change
            return 0;
        }
        for (int y = 0; y < h; ++y)
            memcpy(dst + size_t(y)*pitch, frame_out.data() + size_t(y)*w*bpp, size_t(w)*bpp);
        fresh_frame = false;
        if (device_ms) *device_ms = last_ms;
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

// Crop mode. 1 accepted, 0 busy (try again later), -1 failed/disabled.
__declspec(dllexport) int re_submit(const void* staging, int width, int height, int pitch, int bpp) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->submit(staging, width, height, pitch, bpp, true, 0);
}

// Full mode with the output gain (0 native, 1 public). Same return codes.
__declspec(dllexport) int re_submit_full(const void* staging, int width, int height, int pitch, int bpp, int gain) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->submit(staging, width, height, pitch, bpp, false, gain);
}

// Crop mode: 1 new preview written, 0 nothing new, -1 failed/disabled.
__declspec(dllexport) int re_poll(int gain, unsigned char* rgba8, unsigned char* rgba16, float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->poll(gain, rgba8, rgba16, device_ms);
}

// Full mode: 1 new frame written into dst (rows of pitch bytes), 0 nothing new, -1 failed.
__declspec(dllexport) int re_poll_frame(unsigned char* dst, unsigned long long pitch, int width, int height,
                                        int bpp, float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->poll_frame(dst, size_t(pitch), width, height, bpp, device_ms);
}

// Diagnostic copy of the latest linear RGB and its input (network extent, 3 floats per pixel).
__declspec(dllexport) int re_latest_rgb(int gain, float* rgb, float* color) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    if (!g_engine->completed) return 0;
    if (!g_engine->host_valid) {
        // Full mode keeps the floats on the device; the idle worker cannot
        // start another run while the engine mutex is held.
        if (g_engine->busy || g_engine->pending) return 0;
        auto& e = *g_engine;
        size_t count = size_t(e.result_w)*e.result_h*3;
        e.host_native.resize(count); e.host_public.resize(count); e.host_color.resize(count);
        try {
            HIP_CHECK(hipMemcpyAsync(e.host_native.data(), e.net->rgb_native->data, count*sizeof(float), hipMemcpyDeviceToHost, e.stream));
            HIP_CHECK(hipMemcpyAsync(e.host_public.data(), e.net->rgb_public->data, count*sizeof(float), hipMemcpyDeviceToHost, e.stream));
            HIP_CHECK(hipMemcpyAsync(e.host_color.data(), e.net->color->data, count*sizeof(float), hipMemcpyDeviceToHost, e.stream));
            HIP_CHECK(hipStreamSynchronize(e.stream));
        } catch (const std::exception&) { return -1; }
        e.host_valid = true;
    }
    const auto& source = gain ? g_engine->host_public : g_engine->host_native;
    if (rgb) memcpy(rgb, source.data(), source.size()*sizeof(float));
    if (color) memcpy(color, g_engine->host_color.data(), g_engine->host_color.size()*sizeof(float));
    return 1;
}

__declspec(dllexport) int re_latest_extent(int* width, int* height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    if (width) *width = g_engine->result_w;
    if (height) *height = g_engine->result_h;
    return g_engine->completed ? 1 : 0;
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
