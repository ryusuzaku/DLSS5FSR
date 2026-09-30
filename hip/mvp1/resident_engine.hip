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

// Full-frame output options (optional "key=value" lines after the 24 engine
// config lines):
//   transfer=full  the network's RGB (default)
//   transfer=luma  the game's colour with only the network's luminance change
//                  applied (Y_out / Y_in on the linearized input); keeps the
//                  game's hue where its colour is not display-referred
//   limit=L        clamp that luminance ratio to [1-L, 1+L] (0 = no clamp)
//   smooth=A       blend the ratio with the previous frame's by A, gated by
//                  exp(-|Y_in - Y_in_prev| / sigma) so moving content follows
//   sigma=S        the gate's luminance scale (default 0.02)
//   gain=0|1       override the caller's gain (0 native .03125, 1 public 1.0)
//   strength=K     scale the luminance change (ratio = 1 + K*(ratio-1))
//   gate=G         luma mode applies the latest ratio map to every current
//                  frame; where the current input luminance differs from the
//                  one the map was computed on, the ratio fades by
//                  exp(-|dY| / G) (moved content keeps the game's pixels)
// In luma mode values at or above 1.0 (HDR, which the network only sees
// clipped) pass through, and the ratio fades out as a channel nears 1.0.
//   denoise=R      luma mode: the ratio of neighbourhood sums over a
//                  (2R+1)^2 window, each neighbour weighted by how close its
//                  input luminance is to the centre's (edge-aware; removes
//                  the per-pixel grain of the ratio, keeps edges)
//   motion=0|1     luma mode: carry the ratio map along the game's motion
//                  vectors (re_motion) to the newest frame (default 1)
//   motion_sign=-1 flips the vectors (games that store cur - prev)
struct OutputOptions {
    int luma = 0, denoise = 0, motion = 1, motion_sign = 1;
    float limit = 0.0f, smooth = 0.0f, sigma = 0.02f, strength = 1.0f, gate = 0.03f;
    int gain = -1;
};

__device__ inline float srgb_decode_device(float s) {
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    return s <= 0.04045f ? s/12.92f : powf((s+0.055f)/1.055f, 2.4f);
}

__device__ inline float frame_channel(const unsigned char* px, int bpp, int c) {
    if (bpp == 4) return px[c] / 255.0f;
    unsigned short bits = (unsigned short)(px[2*c] | (px[2*c+1] << 8));
    return __half2float(__ushort_as_half(bits));
}

__device__ inline float srgb_encode_device(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return v <= 0.0031308f ? 12.92f*v : 1.055f*powf(v, 1.0f/2.4f) - 0.055f;
}

// Luma mode, after a network run: per frame pixel (ratio, Y_in) from the
// network's linear output and its linearized input, optionally smoothed
// against the previous map (gated by the input luminance change).
__global__ void k_ratio_map(const float* linear, const float* color, const float2* previous, int previous_valid,
                            OutputOptions o, float2* map, int frame_w, int frame_h, int W) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= frame_w*frame_h) return;
    int x = i % frame_w, y = i / frame_w;
    const float* p = linear + (size_t(y)*W + x)*3;
    const float* q = color + (size_t(y)*W + x)*3;
    float yin = .2126f*q[0] + .7152f*q[1] + .0722f*q[2];
    float yout = .2126f*fmaxf(p[0], 0.0f) + .7152f*fmaxf(p[1], 0.0f) + .0722f*fmaxf(p[2], 0.0f);
    if (o.denoise > 0) {
        float sum_in = 0.0f, sum_out = 0.0f;
        for (int dy = -o.denoise; dy <= o.denoise; ++dy)
            for (int dx = -o.denoise; dx <= o.denoise; ++dx) {
                int nx = min(max(x+dx, 0), frame_w-1), ny = min(max(y+dy, 0), frame_h-1);
                const float* pn = linear + (size_t(ny)*W + nx)*3;
                const float* qn = color + (size_t(ny)*W + nx)*3;
                float yi = .2126f*qn[0] + .7152f*qn[1] + .0722f*qn[2];
                float yo = .2126f*fmaxf(pn[0], 0.0f) + .7152f*fmaxf(pn[1], 0.0f) + .0722f*fmaxf(pn[2], 0.0f);
                float d = (yi - yin) / (yin + 0.02f);
                float wgt = expf(-d*d*50.0f);  // relative luminance difference, sigma 0.1
                sum_in += wgt*yi; sum_out += wgt*yo;
            }
        yout = (sum_out + 1e-4f) / (sum_in + 1e-4f) * (yin + 1e-4f) - 1e-4f;  // local gain at the centre
    }
    float ratio = 1.0f + o.strength*((yout + 1e-4f) / (yin + 1e-4f) - 1.0f);
    if (o.limit > 0.0f) ratio = fminf(1.0f + o.limit, fmaxf(1.0f - o.limit, ratio));
    if (previous && previous_valid && o.smooth > 0.0f) {
        float2 h = previous[i];
        float wgt = o.smooth * expf(-fabsf(yin - h.y) / o.sigma);
        ratio = ratio + (h.x - ratio)*wgt;
    }
    // A non-finite network output anywhere (softmax overflow on an unusual
    // frame) must not reach the resolve, which would turn it black.
    if (!isfinite(ratio) || !isfinite(yin)) { ratio = 1.0f; yin = -1.0f; }
    map[i] = make_float2(ratio, yin);
}

// Luma mode with motion: the map at frame t from the map at t-1, sampled
// bilinearly where each pixel was a frame earlier (prev = cur + mv, in frame
// pixels). The reference luminance moves with the ratio, so k_compose's gate
// rejects disocclusions; samples outside the frame carry no edit.
__global__ void k_negate(float* v, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) v[i] = -v[i];
}

__global__ void k_warp_map(const float2* src, const float2* motion, float2* dst, int w, int h) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= w*h) return;
    int x = i % w, y = i / w;
    float2 mv = motion[i];
    float sx = x + mv.x, sy = y + mv.y;
    if (!(sx >= 0.0f && sy >= 0.0f && sx <= w-1 && sy <= h-1)) { dst[i] = make_float2(1.0f, -1.0f); return; }
    int x0 = min(int(sx), w-2 < 0 ? 0 : w-2), y0 = min(int(sy), h-2 < 0 ? 0 : h-2);
    float fx = sx - x0, fy = sy - y0;
    float2 a = src[size_t(y0)*w + x0], b = src[size_t(y0)*w + x0+1];
    float2 c = src[size_t(y0+1)*w + x0], d = src[size_t(y0+1)*w + x0+1];
    // A corner without an edit (yin < 0) makes the sample carry none.
    if (a.y < 0.0f || b.y < 0.0f || c.y < 0.0f || d.y < 0.0f) { dst[i] = make_float2(1.0f, -1.0f); return; }
    float2 top = make_float2(a.x + (b.x-a.x)*fx, a.y + (b.y-a.y)*fx);
    float2 bottom = make_float2(c.x + (d.x-c.x)*fx, c.y + (d.y-c.y)*fx);
    dst[i] = make_float2(top.x + (bottom.x-top.x)*fy, top.y + (bottom.y-top.y)*fy);
}

// Luma mode, every frame: the current frame with the latest ratio map.
__global__ void k_compose(const unsigned char* src, int pitch, int bpp, const float2* map, OutputOptions o,
                          unsigned char* out, size_t out_pitch, int frame_w, int frame_h) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= frame_w*frame_h) return;
    int x = i % frame_w, y = i / frame_w;
    const unsigned char* px = src + size_t(y)*pitch + size_t(x)*bpp;
    out += size_t(y)*out_pitch + size_t(x)*bpp;
    float raw[3], lin[3], peak = 0.0f;
    for (int c = 0; c < 3; ++c) {
        raw[c] = frame_channel(px, bpp, c);
        if (!(raw[c] == raw[c])) raw[c] = 0.0f;
        lin[c] = srgb_decode_device(raw[c]);
        peak = fmaxf(peak, raw[c]);
    }
    float yin = .2126f*lin[0] + .7152f*lin[1] + .0722f*lin[2];
    float2 m = map[i];
    float fade = fminf(1.0f, fmaxf(0.0f, (1.0f - peak) / 0.05f));
    float wgt = fade * expf(-fabsf(yin - m.y) / o.gate);
    float ratio = 1.0f + (m.x - 1.0f)*wgt;
    if (!isfinite(ratio)) { ratio = 1.0f; wgt = 0.0f; }
    for (int c = 0; c < 4; ++c) {
        if (c == 3 || raw[c] >= 1.0f || wgt == 0.0f) {  // alpha, HDR and untouched values pass through
            for (int b = 0; b < bpp/4; ++b) out[c*(bpp/4) + b] = px[c*(bpp/4) + b];
            continue;
        }
        float s = srgb_encode_device(lin[c]*ratio);
        if (bpp == 4) {
            out[c] = (unsigned char)nearbyintf(s*255.0f);
        } else {
            _Float16 h = (_Float16)s;
            __builtin_memcpy(out + c*2, &h, 2);
        }
    }
}

// Full mode (transfer=full): the frame (RGBA8, or RGBA16F when bpp is 8)
// from the network's linear RGB (row stride W pixels), on the GPU; alpha is
// the input's. powf may differ from the host's by an ulp, which can move a
// rare 8-bit rounding.
__global__ void k_encode_frame(const float* linear, const unsigned char* src, int pitch,
                               unsigned char* out, int frame_w, int frame_h, int W, int bpp) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= frame_w*frame_h) return;
    int x = i % frame_w, y = i / frame_w;
    const float* p = linear + (size_t(y)*W + x)*3;
    const float rgb[3] = {p[0], p[1], p[2]};
    const unsigned char* in = src + size_t(y)*pitch + size_t(x)*bpp;
    for (int c = 0; c < 4; ++c) {
        if (bpp == 4) {
            out[size_t(i)*4+c] = c < 3 ? (unsigned char)nearbyintf(srgb_encode_device(rgb[c])*255.0f) : in[3];
        } else if (c < 3) {
            _Float16 h = (_Float16)srgb_encode_device(rgb[c]);
            __builtin_memcpy(out + size_t(i)*8 + c*2, &h, 2);
        } else {
            __builtin_memcpy(out + size_t(i)*8 + 6, in + 6, 2);
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
    OutputOptions options;                          // full-frame output (config lines after 24)
    // Luma mode: ratio maps (ratio, Y_in per frame pixel), double-buffered;
    // maps[published] is the latest finished one, the worker writes the other.
    std::unique_ptr<Buffer> maps[2];
    int published = -1, map_w = 0, map_h = 0;
    // Luma mode: the newest submitted frame and the composed output.
    hipStream_t compose_stream = nullptr;
    std::unique_ptr<Buffer> current, composed;
    size_t current_bytes = 0;
    int cur_w = 0, cur_h = 0, cur_pitch = 0, cur_bpp = 0;
    // Luma mode with motion (re_motion before each submit): frame numbers
    // count luma submits; ring[f % kRing] holds frame f's motion (to f-1).
    static constexpr int kRing = 16;
    std::unique_ptr<Buffer> ring[kRing];
    long long ring_frame[kRing] = {};
    int mv_w = 0, mv_h = 0;
    long long frame_counter = 0;       // the next luma submit's frame number
    long long job_frame = -1;          // frame of the job being computed
    long long map_frame[2] = {-1, -1}; // frame each ratio map was computed on
    // The published map carried to the newest frame.
    std::unique_ptr<Buffer> warped[2];
    int warped_cur = 0;
    long long warped_frame = -1;
    int warped_source = -1;            // which published generation it came from
    int published_generation = 0;
    unsigned char* compose_pinned = nullptr;        // luma mode: host readback of the composed frame
    size_t compose_pinned_bytes = 0;
    unsigned char* pinned = nullptr;                // full mode: host readback of the frame bytes
    size_t pinned_bytes = 0;
    int result_w = 0, result_h = 0;                 // network extent of the latest result
    std::vector<unsigned char> frame_out;           // full mode: frame_w x frame_h x bpp
    int out_w = 0, out_h = 0, out_bpp = 0;
    float last_ms = 0.0f;
    unsigned long long completed = 0;
    std::string error;

    explicit Engine(const std::vector<std::string>& lines) : config(lines) {
        if (config.size() < 24) throw std::runtime_error("engine config needs 24 lines");
        for (size_t i = 24; i < config.size(); ++i) {  // optional output options
            const std::string& line = config[i];
            size_t eq = line.find('=');
            if (eq == std::string::npos) throw std::runtime_error("engine option needs key=value: " + line);
            std::string key = line.substr(0, eq), value = line.substr(eq+1);
            if (key == "transfer" && (value == "luma" || value == "full")) options.luma = value == "luma";
            else if (key == "limit") options.limit = std::stof(value);
            else if (key == "smooth") options.smooth = std::stof(value);
            else if (key == "sigma") options.sigma = std::stof(value);
            else if (key == "gain" && (value == "0" || value == "1")) options.gain = value == "1";
            else if (key == "strength") options.strength = std::stof(value);
            else if (key == "gate") options.gate = std::stof(value);
            else if (key == "denoise") options.denoise = std::stoi(value);
            else if (key == "motion" && (value == "0" || value == "1")) options.motion = value == "1";
            else if (key == "motion_sign" && (value == "1" || value == "-1")) options.motion_sign = std::stoi(value);
            else throw std::runtime_error("unknown engine option: " + line);
        }
        if (options.sigma <= 0.0f || options.gate <= 0.0f || options.smooth < 0.0f || options.smooth > 1.0f ||
            options.limit < 0.0f || options.strength < 0.0f || options.denoise < 0 || options.denoise > 4)
            throw std::runtime_error("engine options out of range");
        // The caller's thread may carry a stale HIP error from its own work;
        // setup checks hipGetLastError() after its first launches.
        (void)hipGetLastError();
        // The network runs at the lowest priority and the per-frame compose
        // at the highest, so the game's frame never queues behind a network run.
        int least = 0, greatest = 0;
        HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
        HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least));
        HIP_CHECK(hipStreamCreateWithPriority(&compose_stream, hipStreamNonBlocking, greatest));
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
        if (compose_pinned) (void)hipHostFree(compose_pinned);
        if (compose_stream) (void)hipStreamSynchronize(compose_stream);
        net.reset(); raw_input.reset(); encoded.reset(); maps[0].reset(); maps[1].reset();
        current.reset(); composed.reset(); warped[0].reset(); warped[1].reset();
        for (auto& r : ring) r.reset();
        wmma_gemm::state().clear();  // half weight copies are keyed by the freed pointers
        c512_resident::stream = nullptr;
        if (stream) (void)hipStreamDestroy(stream);
        if (compose_stream) (void)hipStreamDestroy(compose_stream);
    }

    // Caller thread: snapshot the staged frame; the shim's buffer is reused next frame.
    // The motion of the frame about to be submitted (float2 per frame pixel,
    // prev = cur + mv, frame pixels; pitch in bytes). Luma full mode only.
    int motion(const void* mv, int w, int h, int pitch) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!options.luma || !options.motion) return 0;
        if (!mv || w <= 0 || h <= 0 || w > 4096 || h > 4096 || pitch < w*8) { error = "unsupported motion format"; return -1; }
        try {
            if (mv_w != w || mv_h != h) {
                for (auto& r : ring) r.reset(new Buffer(size_t(w)*h*2));
                for (auto& f : ring_frame) f = -1;
                mv_w = w; mv_h = h;
            }
            int slot = int(frame_counter % kRing);
            HIP_CHECK(hipMemcpy2DAsync(ring[slot]->data, size_t(w)*8, mv, pitch, size_t(w)*8, h,
                                       hipMemcpyDeviceToDevice, compose_stream));
            ring_frame[slot] = frame_counter;
            if (options.motion_sign < 0) {
                // Games that store prev - cur: flip on the device.
                hipLaunchKernelGGL(k_negate, dim3((w*h*2+255)/256), dim3(256), 0, compose_stream, ring[slot]->data, w*h*2);
            }
            HIP_CHECK(hipGetLastError());
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        return 1;
    }

    int submit(const void* staging, int w, int h, int pitch, int bpp, bool crop, int gain) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!staging || w <= 0 || h <= 0 || w > 4096 || h > 4096 || (bpp != 4 && bpp != 8) ||
            pitch < w*bpp) { error = "unsupported staging format"; return -1; }
        size_t bytes = size_t(pitch)*h;
        if (!crop && options.luma) {
            // Every frame: the newest frame for k_compose (the network may be busy).
            try {
                if (bytes > current_bytes) {
                    current.reset(new Buffer((bytes+3)/4));
                    composed.reset(new Buffer((bytes+3)/4));
                    current_bytes = current->count*sizeof(float);
                }
                HIP_CHECK(hipMemcpyAsync(current->data, staging, bytes, hipMemcpyDeviceToDevice, compose_stream));
                HIP_CHECK(hipStreamSynchronize(compose_stream));
            } catch (const std::exception& e) {
                failed = true; error = e.what(); return -1;
            }
            cur_w = w; cur_h = h; cur_pitch = pitch; cur_bpp = bpp;
            ++frame_counter;
        }
        if (busy || pending) return 0;
        job_frame = frame_counter - 1;
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
            int gain = options.gain >= 0 ? options.gain : job_gain;
            const float* linear = gain ? net->rgb_public->data : net->rgb_native->data;
            if (options.luma) {
                int next = published == 0 ? 1 : 0;
                bool previous_valid = published >= 0 && map_w == frame_w && map_h == frame_h;
                for (auto& m : maps)
                    if (!m || m->count < size_t(frame_w)*frame_h*2) m.reset(new Buffer(size_t(frame_w)*frame_h*2));
                hipLaunchKernelGGL(k_ratio_map, dim3((frame_w*frame_h+255)/256), dim3(256), 0, stream,
                                   linear, net->color->data,
                                   previous_valid ? reinterpret_cast<const float2*>(maps[published]->data) : nullptr,
                                   int(previous_valid), options, reinterpret_cast<float2*>(maps[next]->data),
                                   frame_w, frame_h, W);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipStreamSynchronize(stream));
                float elapsed = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                std::lock_guard<std::mutex> lock(mutex);
                published = next; map_w = frame_w; map_h = frame_h;
                map_frame[next] = job_frame; ++published_generation;
                result_w = W; result_h = H; host_valid = false;
                last_ms = elapsed; ++completed;
                return;
            }
            hipLaunchKernelGGL(k_encode_frame, dim3((frame_w*frame_h+255)/256), dim3(256), 0, stream,
                               linear, reinterpret_cast<const unsigned char*>(raw_input->data), frame_pitch,
                               out, frame_w, frame_h, W, frame_bpp);
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

    // Luma mode: the newest frame with the latest (motion-carried) ratio map,
    // composed into device memory `out` (rows of out_pitch bytes) on the
    // high-priority compose stream; returns once it is written. The caller
    // holds the mutex. 1 composed, 0 nothing yet, -1 error.
    int compose_into(unsigned char* out, size_t out_pitch, int w, int h, int bpp, size_t dst_pitch) {
        if (!out || published < 0 || map_w != w || map_h != h || cur_w != w || cur_h != h || cur_bpp != bpp ||
            dst_pitch < size_t(w)*bpp || out_pitch < size_t(w)*bpp) return 0;
        try {
            const float2* map = reinterpret_cast<const float2*>(maps[published]->data);
            if (options.motion && mv_w == w && mv_h == h && map_frame[published] >= 0) {
                // Carry the map from its own frame (a new result) or from
                // the last carried frame to the newest submitted frame.
                const long long target = frame_counter - 1;
                long long from;
                const float2* src;
                if (warped_source != published_generation) {
                    from = map_frame[published]; src = map;
                    for (auto& b : warped)
                        if (!b || b->count < size_t(w)*h*2) b.reset(new Buffer(size_t(w)*h*2));
                } else {
                    from = warped_frame; src = reinterpret_cast<const float2*>(warped[warped_cur]->data);
                }
                bool carried = true;
                for (long long f = from + 1; f <= target; ++f) {
                    int slot = int(f % kRing);
                    if (ring_frame[slot] != f) { carried = false; break; }  // motion lost: stop here
                    int next = src == reinterpret_cast<const float2*>(warped[0]->data) ? 1 : 0;
                    hipLaunchKernelGGL(k_warp_map, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                                       src, reinterpret_cast<const float2*>(ring[slot]->data),
                                       reinterpret_cast<float2*>(warped[next]->data), w, h);
                    src = reinterpret_cast<const float2*>(warped[next]->data);
                    warped_cur = next;
                }
                if (src != map) {
                    warped_frame = carried ? target : warped_frame;
                    warped_source = published_generation;
                    map = src;
                } else if (carried) {
                    // Nothing to carry yet (result for the newest frame).
                    HIP_CHECK(hipMemcpyAsync(warped[0]->data, map, size_t(w)*h*8, hipMemcpyDeviceToDevice, compose_stream));
                    warped_cur = 0; warped_frame = target; warped_source = published_generation;
                }
            }
            hipLaunchKernelGGL(k_compose, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                               reinterpret_cast<const unsigned char*>(current->data), cur_pitch, bpp,
                               map, options, out, out_pitch, w, h);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(compose_stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        return 1;
    }

    int compose_device(void* dst, size_t pitch, int w, int h, int bpp, float* device_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!options.luma) return 0;
        int r = compose_into(reinterpret_cast<unsigned char*>(dst), pitch, w, h, bpp, pitch);
        if (r == 1 && device_ms) *device_ms = last_ms;
        return r;
    }

    // Full mode: copy the newest frame if it matches the destination's size/format.
    int poll_frame(unsigned char* dst, size_t pitch, int w, int h, int bpp, float* device_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (options.luma) {
            size_t bytes = size_t(w)*h*bpp;
            try {
                if (compose_pinned_bytes < bytes) {
                    if (compose_pinned) HIP_CHECK(hipHostFree(compose_pinned));
                    compose_pinned = nullptr; compose_pinned_bytes = 0;
                    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&compose_pinned), bytes));
                    compose_pinned_bytes = bytes;
                }
            } catch (const std::exception& e) {
                failed = true; error = e.what(); return -1;
            }
            int r = compose_into(composed ? reinterpret_cast<unsigned char*>(composed->data) : nullptr,
                                 size_t(w)*bpp, w, h, bpp, pitch);
            if (r != 1) return r;
            try {
                HIP_CHECK(hipMemcpyAsync(compose_pinned, composed->data, bytes, hipMemcpyDeviceToHost, compose_stream));
                HIP_CHECK(hipStreamSynchronize(compose_stream));
            } catch (const std::exception& e) {
                failed = true; error = e.what(); return -1;
            }
            for (int y = 0; y < h; ++y)
                memcpy(dst + size_t(y)*pitch, compose_pinned + size_t(y)*w*bpp, size_t(w)*bpp);
            if (device_ms) *device_ms = last_ms;
            return 1;
        }
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
// Luma full mode: the motion of the frame the next re_submit_full carries
// (float2 per frame pixel, prev = cur + mv in frame pixels, pitch in bytes).
// Returns 1 stored, 0 not used (other modes), -1 error.
__declspec(dllexport) int re_motion(const void* mv, int width, int height, int pitch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->motion(mv, width, height, pitch);
}

// Luma full mode: compose the newest frame straight into device memory
// (rows of `pitch` bytes, the submit's format), no host round trip.
// Returns 1 written, 0 nothing yet or another mode (use re_poll_frame), -1 error.
__declspec(dllexport) int re_compose_device(void* dst, unsigned long long pitch, int width, int height,
                                            int bpp, float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->compose_device(dst, size_t(pitch), width, height, bpp, device_ms);
}

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
