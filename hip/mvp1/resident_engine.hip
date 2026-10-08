// In-process resident candidate engine for the NGX shim (built as a DLL).
// The shim hands over its HIP staging pointer; the engine copies the frame on
// its own non-blocking stream, then a worker thread runs the candidate input,
// the C32 front end, encoder5-30, ViT31-38, decoder39-69 and head70.
//   crop mode (re_submit/re_poll): the validated 256x256 centre-crop preview.
//   full mode (re_submit_full/re_poll_frame): the whole frame, edge-padded to
//   the smallest network extent (multiples of 64 whose ViT token count is 16
//   or a multiple of 64; with extent=compact any multiple of 64, the ViT then
//   padded with masked tokens), written back at the frame's own size and format.
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

// Full-frame candidate input: the staged proxy resampled to sw x sh content
// (1:1 when that is the frame size; bilinear in sRGB otherwise, like the 256
// crop), edge-replicated into the padded network extent, then sRGB decoded.
// tone=K (luma mode): the network is shown the frame through auto exposure
// (K / log-average luminance, tone[2]) and the ACES fit, i.e. a display-like
// picture instead of the game's flat pre-tonemap one; k_ratio_map maps its
// answer back through the exact inverse. tone: sums (log Y, count), the
// applied exposure, valid, the smoothed log2 exposure. The applied exposure
// moves in 1/32-stop steps and only past a deadband: the network answers even
// last-bit input changes with visible grain changes, so a still scene must
// keep a bit-identical input.
__device__ inline float tone_aces(float x) {
    return fminf(fmaxf(x*(2.51f*x + 0.03f) / (x*(2.43f*x + 0.59f) + 0.14f), 0.0f), 1.0f);
}
__device__ inline float tone_aces_inverse(float y) {
    y = fminf(fmaxf(y, 0.0f), 0.98f);
    return (-0.59f*y + 0.03f - sqrtf(-1.0127f*y*y + 1.3702f*y + 0.0009f)) / (2.0f*(2.43f*y - 2.51f));
}
__global__ void k_tone_measure(const unsigned char* src, int pitch, int bpp, int w, int h, float* tone) {
    const int step = 4, gw = (w + step - 1)/step, gh = (h + step - 1)/step;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f, count = 0.0f;
    if (i < gw*gh) {
        int x = (i % gw)*step, y = (i / gw)*step;
        float l[3];
        for (int c = 0; c < 3; ++c) {
            float s = fminf(fmaxf(candidate_proxy_channel(src, pitch, bpp, x, y, c), 0.0f), 1.0f);
            l[c] = s <= 0.04045f ? s/12.92f : powf((s+0.055f)/1.055f, 2.4f);
        }
        sum = logf(.2126f*l[0] + .7152f*l[1] + .0722f*l[2] + 1e-4f); count = 1.0f;
    }
    for (int o = 16; o; o >>= 1) { sum += __shfl_xor(sum, o); count += __shfl_xor(count, o); }
    if ((threadIdx.x & 31) == 0 && count > 0.0f) { atomicAdd(&tone[0], sum); atomicAdd(&tone[1], count); }
}
__global__ void k_tone_update(float* tone, float key) {
    if (tone[1] > 0.0f) {
        const float target = fminf(fmaxf(log2f(key) - tone[0] / tone[1] * 1.44269504f, -6.0f), 6.0f);
        // Slow and sticky: a car driving into view moves the frame average,
        // and every exposure step changes the network's answer in bright
        // areas (a brightness pump), so only lasting changes past 1/4 stop move it.
        tone[4] = tone[3] > 0.0f ? tone[4] + (target - tone[4])*0.05f : target;  // smoothed over runs
        const float applied = tone[3] > 0.0f ? log2f(tone[2]) : 1e9f;
        if (fabsf(tone[4] - applied) > 0.25f) tone[2] = exp2f(rintf(tone[4]*32.0f) / 32.0f);
        tone[3] = 1.0f;
    }
    tone[0] = 0.0f; tone[1] = 0.0f;
}

__global__ void k_candidate_input_full(const unsigned char* src, float* output, int frame_w, int frame_h,
                                       int pitch, int bpp, int width, int height, int sw, int sh,
                                       const float* tone = nullptr, int encoded = 0) {
    const int pixel = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (pixel >= width*height) return;
    // Past the content the original mirrors it, without repeating the edge
    // (peer DLSSNR-AMD image_input.glsl: sx = 2*sw - x - 2).
    const int px = pixel % width, py = pixel / width;
    const int x = max(0, px < sw ? px : 2*sw - px - 2), y = max(0, py < sh ? py : 2*sh - py - 2);
    for (int c = 0; c < 3; ++c) {
        float srgb;
        if (sw == frame_w && sh == frame_h) {
            srgb = candidate_proxy_channel(src, pitch, bpp, x, y, c);
        } else {
            float fx = fminf(fmaxf((x+0.5f)*frame_w/sw - 0.5f, 0.0f), float(frame_w-1));
            float fy = fminf(fmaxf((y+0.5f)*frame_h/sh - 0.5f, 0.0f), float(frame_h-1));
            int x0 = int(fx), y0 = int(fy), x1 = min(x0+1, frame_w-1), y1 = min(y0+1, frame_h-1);
            float tx = fx - x0, ty = fy - y0;
            float a = candidate_proxy_channel(src, pitch, bpp, x0, y0, c), b = candidate_proxy_channel(src, pitch, bpp, x1, y0, c);
            float d = candidate_proxy_channel(src, pitch, bpp, x0, y1, c), e = candidate_proxy_channel(src, pitch, bpp, x1, y1, c);
            srgb = (a*(1.0f-tx) + b*tx)*(1.0f-ty) + (d*(1.0f-tx) + e*tx)*ty;
        }
        srgb = fminf(fmaxf(srgb, 0.0f), 1.0f);
        float linear = srgb <= 0.04045f ? srgb/12.92f : powf((srgb+0.055f)/1.055f, 2.4f);
        float v = tone ? tone_aces(linear*tone[2]) : linear;
        // encoded: the display-encoded value, what the original is fed.
        if (encoded) v = tone ? (v <= 0.0031308f ? 12.92f*v : 1.055f*powf(v, 1.0f/2.4f) - 0.055f) : srgb;
        output[pixel*3+c] = v;
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
//   motion=auto|1|uv|0 luma mode: carry the ratio map along the game's motion
//                  vectors (re_motion) to the newest frame. auto (default)
//                  detects the units (frame pixels or UV) and sign from how
//                  well each predicts the frame from the previous one (it
//                  needs real motion: nothing is carried until then); 1
//                  takes frame pixels with motion_sign, uv UV with
//                  motion_sign; 0 disables.
//   motion_sign=-1 flips the vectors (games that store cur - prev)
//   scale=S        luma mode: run the network on an S x downscale of the frame
//                  (0.25..1) and upsample its ratio map (about S^2 the work)
//   budget=N       luma mode: at most N network pixels; lowers the scale further
//                  for large frames (5120x1440 ultrawide) so the cost stays put
struct OutputOptions {
    int luma = 0, denoise = 0, motion = 2, motion_sign = 1;  // motion 2 = auto, 3 = uv
    int compact = 0;  // extent=compact: smallest multiple-of-64 extent (fast path only)
    float tone = 0.0f;  // tone=K: luma mode network input through auto exposure (key K) + ACES
    float controls[5] = {0, 0, 0, 0, 0};  // controls=a,b,c,d,e: stem features 10..14
    int encoded = 0;      // encoded=1: feed the network display-encoded colour, as the original
    int external = 0;     // external_controls=1: controls and strength follow re_controls
    float depth_sigma = 0.0f;  // depth_sigma=S: denoise also weights neighbours by relative depth difference
    int albedo_guide = 0;      // albedo_guide=1: denoise compares lighting (luminance / RR albedo), not texture
    int interval = 1;          // interval=K: start a network run at most every K submitted frames (luma);
                               // the motion-carried map covers the frames between, at 1/K of the GPU time
    float temporal = 0.0f;  // temporal=S: the original's history recurrence, S x its blend scale
    int tone_inverse = 1;  // tone: map the ratio back through the inverse curve (0: use it as it is)
    int history_clamp = 2;  // history guard: 0 none, 1 neighbourhood clamp, 2 reject where the input changed
    float reject = 0.06f;   // history_clamp=2: largest input channel change that keeps the history
    float scale = 1.0f;
    float budget = 0.0f;  // budget=N: cap on network pixels after scale (0 = none)
    float additive = 0.0f;  // additive=A: luma edit as the network's luminance change added to the
                            // frame (A=1) instead of a ratio (A=0): a ratio cannot light a black pixel
    float limit = 0.0f, smooth = 0.0f, sigma = 0.02f, strength = 1.0f, gate = 0.03f;
    // S331: relative forms of sigma and gate, in log2 luminance units (0 = the
    // absolute forms). An absolute linear tolerance barely separates a dark
    // object from a dim background (a weapon's old edit kept ~30% weight on
    // the road behind it, several swings stacked through the smoothing).
    float rel_sigma = 0.0f, rel_gate = 0.0f;
    // smooth_depth=1 (S332): the smoothing also rejects the history where this
    // pixel's depth differs from the previous frame's at the motion-carried
    // position (passes below 10%, gone at 25%). Luminance alone let a dark
    // weapon's edit trail over a dark street (tools/seq_replay.py, recorded
    // sequence). Only with a one-frame-old history and staged depth.
    int smooth_depth = 0;
    // input_temporal=A (S332): a TAA pass on the network's input only -- the
    // frame blended with the previous network input carried along the motion
    // by A, the history clamped to the frame's 3x3 range and dropped where the
    // depth changed. The network then sees a steadier picture (its answer
    // follows input noise several-fold); the edit is still applied to the
    // real frame. RGBA16F staging, one-frame-old history.
    float input_temporal = 0.0f;
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
// The network's luminance ratio at network pixel (x, y) of sw x sh content
// (row stride W); with denoise the ratio of edge-weighted neighbourhood sums.
// depth (frame pixels, fw x fh, optional): neighbours at a different depth
// (relative difference over dsig) count less, so the window does not mix
// surfaces across a silhouette even where their luminance matches.
__device__ inline float net_ratio(const float* linear, const float* color, int x, int y, int sw, int sh, int W,
                                  int denoise, int encoded = 0, const float* depth = nullptr, int fw = 0,
                                  int fh = 0, float dsig = 0.0f, const float* albedo = nullptr) {
    // albedo (frame pixels, optional): the edge test compares demodulated
    // luminance -- the light on the surface -- so texture detail is averaged
    // out of the ratio while lighting and geometry edges stay.
    auto alb = [&](int nx, int ny) {
        int fx = min(int((nx+0.5f)*fw/sw), fw-1), fy = min(int((ny+0.5f)*fh/sh), fh-1);
        return fmaxf(albedo[size_t(fy)*fw + fx], 0.02f);
    };
    auto dep = [&](int nx, int ny) {
        int fx = min(int((nx+0.5f)*fw/sw), fw-1), fy = min(int((ny+0.5f)*fh/sh), fh-1);
        return depth[size_t(fy)*fw + fx];
    };
    const bool use_depth = depth && dsig > 0.0f && denoise > 0;
    const float dc = use_depth ? dep(x, y) : 0.0f;
    auto lum = [&](const float* v, bool clamp) {
        if (encoded)  // display-encoded network colours: luminance of their light
            return .2126f*srgb_decode_device(v[0]) + .7152f*srgb_decode_device(v[1]) + .0722f*srgb_decode_device(v[2]);
        return clamp ? .2126f*fmaxf(v[0], 0.0f) + .7152f*fmaxf(v[1], 0.0f) + .0722f*fmaxf(v[2], 0.0f)
                     : .2126f*v[0] + .7152f*v[1] + .0722f*v[2];
    };
    const float yin = lum(color + (size_t(y)*W + x)*3, false);
    const bool use_albedo = albedo && fw > 0 && denoise > 0;
    const float lin = use_albedo ? yin / alb(x, y) : yin;
    float sum_in = 0.0f, sum_out = 0.0f;
    for (int dy = -denoise; dy <= denoise; ++dy)
        for (int dx = -denoise; dx <= denoise; ++dx) {
            int nx = min(max(x+dx, 0), sw-1), ny = min(max(y+dy, 0), sh-1);
            float yi = lum(color + (size_t(ny)*W + nx)*3, false);
            float yo = lum(linear + (size_t(ny)*W + nx)*3, true);
            float d = use_albedo ? (yi / alb(nx, ny) - lin) / (lin + 0.02f) : (yi - yin) / (yin + 0.02f);
            float wgt = denoise ? expf(-d*d*50.0f) : 1.0f;  // relative luminance difference, sigma 0.1
            if (use_depth) {
                float rd = fabsf(dep(nx, ny) - dc) / (fabsf(dc) + 1e-6f) / dsig;
                wgt *= expf(-rd*rd);
            }
            sum_in += wgt*yi; sum_out += wgt*yo;
        }
    return (sum_out + 1e-4f) / (sum_in + 1e-4f);
}

// The network's luminance change at one of its pixels (linear light).
__device__ inline float net_delta(const float* linear, const float* color, int x, int y, int W, int encoded) {
    const float* o = linear + (size_t(y)*W + x)*3;
    const float* i = color + (size_t(y)*W + x)*3;
    if (encoded)
        return .2126f*(srgb_decode_device(o[0]) - srgb_decode_device(i[0])) +
               .7152f*(srgb_decode_device(o[1]) - srgb_decode_device(i[1])) +
               .0722f*(srgb_decode_device(o[2]) - srgb_decode_device(i[2]));
    return .2126f*(fmaxf(o[0], 0.0f) - i[0]) + .7152f*(fmaxf(o[1], 0.0f) - i[1]) + .0722f*(fmaxf(o[2], 0.0f) - i[2]);
}

// Luma mode, after a network run: per frame pixel (ratio, Y_in). The ratio
// comes from the network at its own resolution (sw x sh content, bilinear
// up to the frame when scaled); Y_in is the frame's own linearized
// luminance, the reference the compose gate compares the current frame to.
// Optionally smoothed against the previous map (gated by Y_in's change).
__global__ void k_ratio_map(const float* linear, const float* color, const unsigned char* src, int pitch, int bpp,
                            const float2* previous, int previous_valid, OutputOptions o, float2* map,
                            int frame_w, int frame_h, int sw, int sh, int W, const float* tone = nullptr,
                            const float* depth = nullptr, const float* albedo = nullptr,
                            const float* prev_depth = nullptr, const float2* motion = nullptr,
                            float mx = 0.0f, float my = 0.0f) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= frame_w*frame_h) return;
    int x = i % frame_w, y = i / frame_w;
    float raw[3];
    const unsigned char* px = src + size_t(y)*pitch + size_t(x)*bpp;
    for (int c = 0; c < 3; ++c) raw[c] = srgb_decode_device(frame_channel(px, bpp, c));
    float yin = .2126f*raw[0] + .7152f*raw[1] + .0722f*raw[2];
    float r, delta = 0.0f;
    if (sw == frame_w && sh == frame_h) {
        r = net_ratio(linear, color, x, y, sw, sh, W, o.denoise, o.encoded, depth, frame_w, frame_h, o.depth_sigma,
                      albedo);
        if (o.additive > 0.0f) delta = net_delta(linear, color, x, y, W, o.encoded);
    } else {
        float fx = fminf(fmaxf((x+0.5f)*sw/frame_w - 0.5f, 0.0f), float(sw-1));
        float fy = fminf(fmaxf((y+0.5f)*sh/frame_h - 0.5f, 0.0f), float(sh-1));
        int x0 = int(fx), y0 = int(fy), x1 = min(x0+1, sw-1), y1 = min(y0+1, sh-1);
        float tx = fx - x0, ty = fy - y0;
        const float ds = o.depth_sigma;
        float a = net_ratio(linear, color, x0, y0, sw, sh, W, o.denoise, o.encoded, depth, frame_w, frame_h, ds, albedo);
        float b = net_ratio(linear, color, x1, y0, sw, sh, W, o.denoise, o.encoded, depth, frame_w, frame_h, ds, albedo);
        float c = net_ratio(linear, color, x0, y1, sw, sh, W, o.denoise, o.encoded, depth, frame_w, frame_h, ds, albedo);
        float d = net_ratio(linear, color, x1, y1, sw, sh, W, o.denoise, o.encoded, depth, frame_w, frame_h, ds, albedo);
        r = (a*(1.0f-tx) + b*tx)*(1.0f-ty) + (c*(1.0f-tx) + d*tx)*ty;
        if (o.additive > 0.0f) {
            float da = net_delta(linear, color, x0, y0, W, o.encoded), db = net_delta(linear, color, x1, y0, W, o.encoded);
            float dc = net_delta(linear, color, x0, y1, W, o.encoded), dd = net_delta(linear, color, x1, y1, W, o.encoded);
            delta = (da*(1.0f-tx) + db*tx)*(1.0f-ty) + (dc*(1.0f-tx) + dd*tx)*ty;
        }
    }
    if (o.additive > 0.0f) {
        // The same change as a ratio on this pixel; near black it may be large.
        const float ra = fmaxf(yin + delta, 0.0f) / fmaxf(yin, 1e-4f);
        r += o.additive*(fminf(ra, 64.0f) - r);
    }
    if (tone && yin > 1e-5f && o.tone_inverse) {
        // The network's ratio is on the toned picture: carry it back to the
        // frame's own light through the inverse curve (luminance). Near the
        // curve's shoulder the inverse amplifies any change in the answer
        // without bound (tone_inverse=0 applies the toned ratio as it is).
        const float e = tone[2], t = tone_aces(yin*e);
        r = tone_aces_inverse(t*r) / e / yin;
    }
    float ratio = 1.0f + o.strength*(r - 1.0f);
    if (o.limit > 0.0f) ratio = fminf(1.0f + o.limit, fmaxf(1.0f - o.limit, ratio));
    if (previous && previous_valid && o.smooth > 0.0f) {
        float2 h = previous[i];
        float wgt;
        if (o.rel_sigma > 0.0f) {
            // The history's luminance against this pixel's 3x3 neighbourhood
            // range (a carried map is off by a fraction of a pixel; a
            // per-pixel ratio test rejects it along every edge).
            float lo = yin, hi = yin;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = min(max(x+dx, 0), frame_w-1), ny = min(max(y+dy, 0), frame_h-1);
                    const unsigned char* q = src + size_t(ny)*pitch + size_t(nx)*bpp;
                    const float l = .2126f*srgb_decode_device(frame_channel(q, bpp, 0)) +
                                    .7152f*srgb_decode_device(frame_channel(q, bpp, 1)) +
                                    .0722f*srgb_decode_device(frame_channel(q, bpp, 2));
                    lo = fminf(lo, l); hi = fmaxf(hi, l);
                }
            const float nearest = fminf(hi, fmaxf(lo, h.y));
            wgt = h.y < 0.0f ? 0.0f : o.smooth * expf(-fabsf(log2f((nearest + 0.005f) / (h.y + 0.005f))) / o.rel_sigma);
        }
        else
            wgt = o.smooth * expf(-fabsf(yin - h.y) / o.sigma);
        if (depth && prev_depth && motion) {
            // smooth_depth: was this pixel the same surface a frame ago?
            const float2 mv = motion[i];
            const int px = min(max(int(x + 0.5f + mv.x*mx), 0), frame_w-1);
            const int py = min(max(int(y + 0.5f + mv.y*my), 0), frame_h-1);
            const float d = depth[i], dp = prev_depth[size_t(py)*frame_w + px];
            const float rel = fabsf(d - dp) / fmaxf(fmaxf(d, dp), 1e-7f);
            if (isfinite(rel)) wgt *= fminf(fmaxf((0.25f - rel) / 0.15f, 0.0f), 1.0f);
        }
        ratio = ratio + (h.x - ratio)*wgt;
    }
    // A non-finite network output anywhere (softmax overflow on an unusual
    // frame) must not reach the resolve, which would turn it black.
    if (!isfinite(ratio) || !isfinite(yin)) { ratio = 1.0f; yin = -1.0f; }
    map[i] = make_float2(ratio, yin);
}

// input_temporal: the network's input, stabilised (see OutputOptions). cur
// and hist are RGBA16F frames (rows of pitch bytes); hist is the previous
// stabilised input, sampled where this pixel was (prev = cur + mv*scale).
__global__ void k_input_stabilize(const unsigned char* cur, const unsigned char* hist, unsigned char* out,
                                  int pitch, int w, int h, const float2* motion, float mx, float my,
                                  const float* depth, const float* prev_depth, float a) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= w*h) return;
    int x = i % w, y = i / w;
    const _Float16* c = reinterpret_cast<const _Float16*>(cur + size_t(y)*pitch + size_t(x)*8);
    _Float16* o = reinterpret_cast<_Float16*>(out + size_t(y)*pitch + size_t(x)*8);
    float v[3], lo[3], hi[3];
    for (int k = 0; k < 3; ++k) { v[k] = float(c[k]); lo[k] = hi[k] = v[k]; }
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = min(max(x+dx, 0), w-1), ny = min(max(y+dy, 0), h-1);
            const _Float16* q = reinterpret_cast<const _Float16*>(cur + size_t(ny)*pitch + size_t(nx)*8);
            for (int k = 0; k < 3; ++k) { const float f = float(q[k]); lo[k] = fminf(lo[k], f); hi[k] = fmaxf(hi[k], f); }
        }
    const float2 mv = motion[i];
    const float sx = x + mv.x*mx, sy = y + mv.y*my;  // pixel coordinates of the same point a frame ago
    float wgt = a;
    if (!(sx >= 0.0f && sy >= 0.0f && sx <= w-1 && sy <= h-1) || !isfinite(sx) || !isfinite(sy)) wgt = 0.0f;
    if (wgt > 0.0f && depth && prev_depth) {
        const int px = min(max(int(sx + 0.5f), 0), w-1), py = min(max(int(sy + 0.5f), 0), h-1);
        const float d = depth[i], dp = prev_depth[size_t(py)*w + px];
        const float rel = fabsf(d - dp) / fmaxf(fmaxf(d, dp), 1e-7f);
        wgt *= isfinite(rel) ? fminf(fmaxf((0.25f - rel) / 0.15f, 0.0f), 1.0f) : 0.0f;
    }
    if (wgt > 0.0f) {
        const int x0 = min(int(sx), w-2 < 0 ? 0 : w-2), y0 = min(int(sy), h-2 < 0 ? 0 : h-2);
        const float fx = sx - x0, fy = sy - y0;
        const _Float16* p00 = reinterpret_cast<const _Float16*>(hist + size_t(y0)*pitch + size_t(x0)*8);
        const _Float16* p10 = reinterpret_cast<const _Float16*>(hist + size_t(y0)*pitch + size_t(x0+1)*8);
        const _Float16* p01 = reinterpret_cast<const _Float16*>(hist + size_t(y0+1)*pitch + size_t(x0)*8);
        const _Float16* p11 = reinterpret_cast<const _Float16*>(hist + size_t(y0+1)*pitch + size_t(x0+1)*8);
        for (int k = 0; k < 3; ++k) {
            float hk = (float(p00[k])*(1.0f-fx) + float(p10[k])*fx)*(1.0f-fy) + (float(p01[k])*(1.0f-fx) + float(p11[k])*fx)*fy;
            if (!isfinite(hk)) { hk = v[k]; }
            hk = fminf(fmaxf(hk, lo[k]), hi[k]);
            v[k] += (hk - v[k])*wgt;
        }
    }
    for (int k = 0; k < 3; ++k) o[k] = (_Float16)v[k];
    o[3] = c[3];
}

// Luma mode with motion: the map at frame t from the map at t-1, sampled
// bilinearly where each pixel was a frame earlier (prev = cur + mv, in frame
// pixels). The reference luminance moves with the ratio, so k_compose's gate
// rejects disocclusions; samples outside the frame carry no edit.
__global__ void k_negate(float* v, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) v[i] = -v[i];
}

__global__ void k_warp_map(const float2* src, const float2* motion, float2* dst, int w, int h,
                           float mx, float my) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= w*h) return;
    int x = i % w, y = i / w;
    float2 mv = motion[i];
    mv.x *= mx; mv.y *= my;
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

// Motion convention vote: on a sparse grid, how well the previous frame
// predicts the current one at x + m*mv for the hypotheses m = (+1,+1),
// (-1,-1) (vectors in frame pixels), (+w,+h), (-w,-h) (vectors in UV), and
// with no motion (index 4): luminance differences in sums[0..4], the samples
// each hypothesis could score (inside the frame) in sums[5..9].
__global__ void k_motion_vote(const unsigned char* cur, const unsigned char* prev, int pitch, int bpp,
                              const float2* motion, int w, int h, float* sums) {
    const int step = 6;
    int gx = blockIdx.x*blockDim.x + threadIdx.x, gw = (w + step - 1)/step, gh = (h + step - 1)/step;
    if (gx >= gw*gh) return;
    int x = (gx % gw)*step, y = (gx / gw)*step;
    auto lum = [&](const unsigned char* f, int px, int py) {
        const unsigned char* p = f + size_t(py)*pitch + size_t(px)*bpp;
        return .2126f*frame_channel(p, bpp, 0) + .7152f*frame_channel(p, bpp, 1) + .0722f*frame_channel(p, bpp, 2);
    };
    float2 mv = motion[size_t(y)*w + x];
    if (!(fabsf(mv.x) + fabsf(mv.y) > 0.0f) || !isfinite(mv.x) || !isfinite(mv.y)) return;
    const float yc = lum(cur, x, y);
    const float ms[4][2] = {{1.0f, 1.0f}, {-1.0f, -1.0f}, {float(w), float(h)}, {-float(w), -float(h)}};
    for (int k = 0; k < 4; ++k) {
        int sx = int(floorf(x + mv.x*ms[k][0] + 0.5f)), sy = int(floorf(y + mv.y*ms[k][1] + 0.5f));
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        atomicAdd(&sums[k], fabsf(yc - lum(prev, sx, sy)));
        atomicAdd(&sums[5+k], 1.0f);
    }
    atomicAdd(&sums[4], fabsf(yc - lum(prev, x, y)));
    atomicAdd(&sums[9], 1.0f);
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
    float wgt;
    if (o.rel_gate > 0.0f) {
        // Same neighbourhood gate as the shim's map resolve (dlssnr.hlsl).
        float lo = yin, hi = yin;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = min(max(x+dx, 0), frame_w-1), ny = min(max(y+dy, 0), frame_h-1);
                const unsigned char* q = src + size_t(ny)*pitch + size_t(nx)*bpp;
                const float l = .2126f*srgb_decode_device(frame_channel(q, bpp, 0)) +
                                .7152f*srgb_decode_device(frame_channel(q, bpp, 1)) +
                                .0722f*srgb_decode_device(frame_channel(q, bpp, 2));
                lo = fminf(lo, l); hi = fmaxf(hi, l);
            }
        const float nearest = fminf(hi, fmaxf(lo, m.y));
        wgt = m.y < 0.0f ? 0.0f : fade * expf(-fabsf(log2f((nearest + 0.005f) / (m.y + 0.005f))) / o.rel_gate);
    } else {
        wgt = fade * expf(-fabsf(yin - m.y) / o.gate);
    }
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

// Luma mode, map output (re_compose_map_device): the carried map as RGBA16F
// (ratio, Y_in, 0, 1) for a consumer that applies it to a later frame itself
// (the shim's resolve, which runs on frame n while the engine has seen n-1
// or n-2). Y_in < 0 marks pixels without an edit.
__global__ void k_map_out(const float2* map, unsigned char* out, size_t out_pitch, int w, int h) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= w*h) return;
    int x = i % w, y = i / w;
    float2 m = map[i];
    const _Float16 v[4] = {(_Float16)m.x, (_Float16)m.y, (_Float16)0.0f, (_Float16)1.0f};
    __builtin_memcpy(out + size_t(y)*out_pitch + size_t(x)*8, v, 8);
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
    // Tokens past the last whole group of 16 (compact extents) keep raster
    // order: the ViT has no positional terms, so only the channel map matters.
    for (int t = 0; t < tokens; ++t) {
        int raster = t < tokens/16*16 ? (t & ~15) | ((t & 1) << 3) | ((t & 14) >> 1) : t;
        for (int c = 0; c < 1024; ++c) {
            int channel = (c & ~31) | ((c & 1) << 1) | ((c & 2) >> 1) | ((c & 4) << 2) | ((c & 24) >> 1);
            map[size_t(t)*1024+c] = raster*1024 + channel;
        }
    }
    return map;
}

// Smallest padded extent the resident network supports for a frame. Compact
// (fast path only): the frame rounded up to multiples of 64.
void network_extent(int frame_w, int frame_h, int& width, int& height, bool compact = false) {
    int w0 = (frame_w+63)/64*64, h0 = (frame_h+63)/64*64;
    if (compact && !c512_resident::exact_math) { width = w0; height = h0; return; }
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
// temporal=S. The original's temporal pre/post blocks (peer DLSSNR-AMD
// fswin_t.comp, NR_TEMPORAL_HISTORY): the previous network image, carried to
// this frame along the motion, is stem features 7..9, and the head's fourth
// row (coefficient row 6) gives w = sigmoid(row . features) * 0.7397 (the
// model's blend scale) * S; the image is lerp(rgb, history, w) and becomes
// the next history.
__global__ void k_coord_identity(float2* coord, int w, int h) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < w*h) coord[i] = make_float2(float(i % w), float(i / w));
}
// Network pixel (x, y) of sw x sh content: the previous image (same layout)
// at this pixel's place in the history's frame (coord: frame pixel -> its
// frame position then, (.., < 0) none; null: same frame), Catmull-Rom. A
// pixel without one is marked by a negative first channel.
__global__ void k_history_sample(const float* previous, const float2* coord, float* out,
                                 int fw, int fh, int sw, int sh, int W, int H) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= W*H) return;
    int x = min(i % W, sw-1), y = min(i / W, sh-1);
    float px = float(x), py = float(y);
    if (coord) {
        int fx = min(max(int((x+0.5f)*fw/sw), 0), fw-1), fy = min(max(int((y+0.5f)*fh/sh), 0), fh-1);
        float2 c = coord[size_t(fy)*fw + fx];
        if (c.y < 0.0f) { out[size_t(i)*3] = -1.0f; out[size_t(i)*3+1] = 0.0f; out[size_t(i)*3+2] = 0.0f; return; }
        // c is where frame pixel (fx, fy) was; move this pixel by the same.
        px += (c.x - fx)*sw/fw; py += (c.y - fy)*sh/fh;
    }
    px = fminf(fmaxf(px, 0.0f), float(sw-1)); py = fminf(fmaxf(py, 0.0f), float(sh-1));
    int ix = int(floorf(px)), iy = int(floorf(py));
    float tx = px - ix, ty = py - iy;
    float wx[4], wy[4];
    auto cr = [](float t, float* wt) {
        float t2 = t*t, t3 = t2*t;
        wt[0] = -0.5f*t3 + t2 - 0.5f*t; wt[1] = 1.5f*t3 - 2.5f*t2 + 1.0f;
        wt[2] = -1.5f*t3 + 2.0f*t2 + 0.5f*t; wt[3] = 0.5f*t3 - 0.5f*t2;
    };
    cr(tx, wx); cr(ty, wy);
    for (int c = 0; c < 3; ++c) {
        float v = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int sy = min(max(iy-1+j, 0), sh-1);
            float row = 0.0f;
            for (int k = 0; k < 4; ++k) row += wx[k]*previous[(size_t(sy)*W + min(max(ix-1+k, 0), sw-1))*3 + c];
            v += wy[j]*row;
        }
        out[size_t(i)*3+c] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}
// Clamp the carried history (in place) to the current input's 3x3
// neighbourhood range per channel, as TAA does: a pixel the moving object
// has just uncovered (zero motion, history = the object) cannot bring the
// object's colour back as a see-through ghost. The original relies on its
// network to reject such history; ours lets part of it through.
__global__ void k_history_clamp(const float* color, float* history, int sw, int sh, int W, int H) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= W*H) return;
    float* q = history + size_t(i)*3;
    if (q[0] < 0.0f) return;
    int x = min(i % W, sw-1), y = min(i / W, sh-1);
    for (int c = 0; c < 3; ++c) {
        float lo = 1e9f, hi = -1e9f;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = min(max(x+dx, 0), sw-1), ny = min(max(y+dy, 0), sh-1);
                float v = color[(size_t(ny)*W + nx)*3 + c];
                lo = fminf(lo, v); hi = fmaxf(hi, v);
            }
        q[c] = fminf(fmaxf(q[c], lo), hi);
    }
}
// history_clamp=2 (default): reject instead of clamp. hin is the previous
// run's network input carried the same way as the history; where it differs
// from the current input (largest channel > thr, e.g. road the car just
// uncovered) the history is dropped for that pixel, so the network's own
// edit survives everywhere the content stayed the same.
__global__ void k_history_reject(const float* color, float* history, const float* hin, int sw, int sh,
                                 int W, int H, float thr) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= W*H) return;
    float* q = history + size_t(i)*3;
    const float* p = hin + size_t(i)*3;
    if (q[0] < 0.0f) return;
    if (p[0] < 0.0f) { q[0] = -1.0f; return; }
    int x = min(i % W, sw-1), y = min(i / W, sh-1);
    const float* cc = color + (size_t(y)*W + x)*3;
    float d = fmaxf(fabsf(p[0]-cc[0]), fmaxf(fabsf(p[1]-cc[1]), fabsf(p[2]-cc[2])));
    if (d > thr) q[0] = -1.0f;
}
// The head's history weight from its 32 features (tile layout of
// k_head70_finish) and the blend into both head images, in place.
__global__ void k_head_temporal(const float* tiles, const float* row, const float* history,
                                float* rgb_a, float* rgb_b, int width, int height, float scale) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= width*height) return;
    const float* h = history + size_t(p)*3;
    if (h[0] < 0.0f) return;
    int x = p % width, y = p / width;
    const float* t = tiles + size_t(((y/8)*(width/8) + x/8)*64 + (y%8)*8 + x%8)*32;
    float o = 0.0f;
    for (int k = 0; k < 32; ++k) o += t[k]*row[k];
    float w = fminf(fmaxf(scale / (1.0f + expf(-o)), 0.0f), 1.0f);
    for (int c = 0; c < 3; ++c) {
        rgb_a[size_t(p)*3+c] += (h[c] - rgb_a[size_t(p)*3+c])*w;
        rgb_b[size_t(p)*3+c] += (h[c] - rgb_b[size_t(p)*3+c])*w;
    }
}

// NVIDIA's level sizes (peer DLSSNR-AMD nr_native_plan.cpp, validated against
// the DLL): every reduction maps a side d to up(ceil(d/2), 4). For the network
// extents used here (multiples of 64) that changes nothing down to 1/16, but
// the C512 grid (1/32) and the ViT grid (1/64) are rounded up to multiples of
// 4, and the extra rows and columns are real tokens -- zero where the encoder
// hands over, then computed through every C512 block and attended by the ViT
// (1080p: C512 36x60 for 34x60 of image, ViT 32x20 = 640 tokens for 30x17).
// The decoder's upsample takes the encoder level's own shape again.
static inline int up4(int v) { return (v + 3)/4*4; }

__global__ void k_pad_hwc(const float* src, float* dst, int w, int h, int dw, int c) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= size_t(w)*h*c) return;
    const size_t pixel = i / c;
    const int x = int(pixel % w), y = int(pixel / w);
    dst[(size_t(y)*dw + x)*c + i % c] = src[i];
}

struct Network {
    // width x height: the image at 1/32; w32 x h32: NVIDIA's C512 grid (multiples
    // of 4); tokens: the C512 head's (w32/2 x h32/2); vit_valid: NVIDIA's ViT
    // sequence (up4(w32/2) x up4(h32/2)); vit_tokens: that padded to a multiple
    // of 64 for the fused attention, which masks only past vit_valid.
    int W, H, width, height, w32, h32, tokens, vit_valid, vit_tokens;
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
    std::unique_ptr<Buffer> padded;  // the C512 chain's input on NVIDIA's grid (null when it is the image's)

    // The encoder's 1/32 output into the zeroed, rounded-up C512 grid.
    bool pad_c512() {
        if (!padded) return true;
        const DeviceTensor src = down22->final_view();
        if (!src.data || src.count != size_t(width)*height*512) return false;
        hipLaunchKernelGGL(k_pad_hwc, dim3((src.count+255)/256), dim3(256), 0, c512_resident::stream,
                           src.data, padded->data, width, height, w32, 512);
        HIP_CHECK(hipGetLastError());
        return true;
    }

    // The WMMA half weight cache is keyed by these weights' device pointers;
    // a later network can reuse the addresses, so drop the copies with them.
    ~Network() {
        if (c512_resident::stream) (void)hipStreamSynchronize(c512_resident::stream);
        wmma_gemm::state().clear();
    }

    // Config: the resident_frame_server arguments after the executable; the
    // width/height lines are ignored, weights come from the listed fixtures.
    Network(const std::vector<std::string>& a, int rgb_w, int rgb_h) :
        W(rgb_w), H(rgb_h), width(rgb_w/32), height(rgb_h/32), w32(up4(width)), h32(up4(height)),
        tokens((w32/2)*(h32/2)), vit_valid(up4(w32/2)*up4(h32/2)),
        vit_tokens(vit_valid == 16 || vit_valid % 64 == 0 ? vit_valid : (vit_valid+63)/64*64),
        n(size_t(w32)*h32*512), hn(size_t(tokens)*1024) {
        if (vit_tokens != vit_valid && c512_resident::exact_math)
            throw std::runtime_error("exact mode needs a ViT token count of 16 or a multiple of 64");
        const int cap = 65536;  // C32 body rows per chunk above this
        front.reset(new front_resident::FrontEnd(W,H,a[23],cap));
        encoder8.reset(new c64_resident::Chain(width*8,height*8,blocks(a[21],4),true));
        down8.reset(new c64_resident::Downsample(width*8,height*8,a[22]));
        encoder14.reset(new c128_resident::Chain(width*4,height*4,blocks(a[19],6),true));
        down14.reset(new c128_resident::Downsample(width*4,height*4,a[20]));
        encoder22.reset(new c256_resident::Chain(width*2,height*2,blocks(a[17],8),true));
        down22.reset(new c256_resident::Downsample(width*2,height*2,a[18]));
        chain.reset(new Chain(w32,h32,blocks(a[0],8),a[3]));
        if (w32 != width || h32 != height) {
            padded.reset(new Buffer(n));
            HIP_CHECK(hipMemset(padded->data, 0, n*sizeof(float)));  // the extra rows/columns enter as zero
        }
        auto gather = vit_gather_map(tokens);
        std::vector<int32_t> inverse(gather.size());
        for (size_t i = 0; i < gather.size(); ++i) inverse[size_t(gather[i])] = int32_t(i);
        map.reset(new IndexMap(gather));
        // Padding tokens stay zero (the gather writes only the real ones).
        bridge.reset(new Buffer(size_t(vit_tokens)*1024));
        HIP_CHECK(hipMemset(bridge->data, 0, bridge->count*sizeof(float)));
        vit.reset(new vit_resident::Chain(vit_tokens,folders(a[5],8),vit_valid));
        entry.reset(new decoder39_resident::Entry(w32/2,h32/2,a[6],inverse));
        decoder.reset(new Chain(w32,h32,blocks(a[7],8)));
        prefix48.reset(new c256_resident::Prefix(w32,h32,a[8],width*2,height*2));
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

    // sw x sh: the network content (the frame, or its downscale); 0 = the frame.
    void run(const unsigned char* src, int fw, int fh, int pitch, int bpp, bool crop, hipStream_t stream,
             int sw = 0, int sh = 0, const float* tone = nullptr, int encoded = 0,
             float* history_clamp = nullptr, const float* history_input = nullptr, float reject = 0.0f) {
        if (sw <= 0 || sh <= 0) { sw = fw; sh = fh; }
        if (crop) {
            if (W != 256 || H != 256) throw std::runtime_error("crop mode needs the 256x256 network");
            hipLaunchKernelGGL(k_candidate_input_256, dim3(256), dim3(256), 0, stream,
                               src, color->data, fw, fh, pitch, bpp);
        } else {
            hipLaunchKernelGGL(k_candidate_input_full, dim3((W*H+255)/256), dim3(256), 0, stream,
                               src, color->data, fw, fh, pitch, bpp, W, H, sw, sh, tone, encoded);
            if (history_clamp && history_input)
                hipLaunchKernelGGL(k_history_reject, dim3((W*H+255)/256), dim3(256), 0, stream,
                                   color->data, history_clamp, history_input, sw, sh, W, H, reject);
            else if (history_clamp)
                hipLaunchKernelGGL(k_history_clamp, dim3((W*H+255)/256), dim3(256), 0, stream,
                                   color->data, history_clamp, sw, sh, W, H);
        }
        HIP_CHECK(hipGetLastError());
        bool ok = front->run_from_device({color->data,color->count}) &&
            encoder8->run_from_device(front->c64_view()) && down8->run_from_device(encoder8->raw_view()) &&
            encoder14->run_from_device(down8->final_view()) && down14->run_from_device(encoder14->raw_view()) &&
            encoder22->run_from_device(down14->final_view()) && down22->run_from_device(encoder22->raw_view()) &&
            pad_c512() && chain->run_from_device(padded ? DeviceTensor{padded->data, n} : down22->final_view());
        if (ok) {
            auto head = chain->head_view();
            hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, stream,
                               head.data, map->data, bridge->data, int(hn));
            HIP_CHECK(hipGetLastError());
            ok = vit->run_from_device({bridge->data,bridge->count}) &&
                entry->run_from_device({vit->final_view().data,hn}, chain->final_view()) &&
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

// re_sync_submit (S332): one frame for the synchronous path. The worker
// queues everything on the device in order: wait for wait_sem >= wait_value
// (the caller's queue signals it once the frame's staging is written), copy
// motion, depth and colour, run the network, write the map (k_map_out,
// RGBA16F ratio / Y_in) into out, then signal signal_sem = signal_value --
// always, also after a failure, because the caller's queue waits on it.
struct ReSyncJob {
    const void* staging; int w, h, pitch, bpp, gain;
    const void* motion; int motion_pitch;   // may be null
    const void* depth; int depth_pitch;     // may be null
    void* wait_sem; unsigned long long wait_value;      // hipExternalSemaphore_t
    void* out; unsigned long long out_pitch;
    void* signal_sem; unsigned long long signal_value;  // hipExternalSemaphore_t
};

struct Engine {
    std::vector<std::string> config;
    hipStream_t stream = nullptr;
    std::unique_ptr<Network> net;
    std::unique_ptr<Buffer> raw_input;
    size_t raw_bytes = 0;
    hipEvent_t start = nullptr, stop = nullptr;
    // Recorded on compose_stream after a job's inputs are copied; the worker's
    // stream waits on it on the device, so the caller never blocks on them.
    hipEvent_t inputs_ready = nullptr;
    // Recorded after an asynchronous compose (re_compose_map_async); the
    // caller polls it with re_compose_ready instead of blocking.
    hipEvent_t compose_done = nullptr;
    // Synchronous jobs: recorded on the network's stream after the ratio map;
    // compose_stream waits on it on the device (run(false) never blocks).
    hipEvent_t net_done = nullptr;
    // Recorded after a submit's copies of the caller's buffers (re_motion,
    // re_depth, re_albedo and re_submit_full all copy on compose_stream):
    // re_inputs_ready says when the caller may rewrite them.
    hipEvent_t inputs_done = nullptr;

    // Worker state.
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool quit = false, pending = false, busy = false, failed = false;
    // re_sync_submit: one queued synchronous job; sync_inline lets the worker
    // reuse submit() for it without the asynchronous bookkeeping.
    bool sync_pending = false, sync_inline = false;
    ReSyncJob sync_job{};
    unsigned long long sync_done = 0, sync_failed = 0;
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
    // smooth: the published map carried to the job's frame along the motion
    // (history[0]; history[1] is the ping-pong scratch), so the new map blends
    // with the same content rather than whatever was at its pixel before.
    std::unique_ptr<Buffer> history[2];
    int history_cur = -1;              // which holds the job's previous map, -1 none
    // The published map carried to the newest frame.
    std::unique_ptr<Buffer> warped[2];
    int warped_cur = 0;
    long long warped_frame = -1;
    int warped_source = -1;            // which published generation it came from
    int published_generation = 0;
    // motion=auto: the previous submitted frame, vote sums and the decision.
    std::unique_ptr<Buffer> previous_frame;
    bool previous_valid = false;
    float* vote = nullptr;               // device: 5 error sums, 5 sample counts
    int vote_frames = 0;
    int motion_choice = -1;              // 0 pixels, 1 pixels negated, 2 UV, 3 UV negated; -1 undecided
    unsigned char* compose_pinned = nullptr;        // luma mode: host readback of the composed frame
    size_t compose_pinned_bytes = 0;
    unsigned char* pinned = nullptr;                // full mode: host readback of the frame bytes
    size_t pinned_bytes = 0;
    int result_w = 0, result_h = 0;                 // network extent of the latest result
    std::vector<unsigned char> frame_out;           // full mode: frame_w x frame_h x bpp
    int out_w = 0, out_h = 0, out_bpp = 0;
    float last_ms = 0.0f;
    unsigned long long completed = 0;
    // temporal=S: the previous network image (network layout) and the frame
    // it is on; per job the frame-pixel positions of the job's pixels in that
    // frame (coord[coord_cur], built in prepare), the sampled history, the
    // head's row 6 and the noise seed counter.
    std::unique_ptr<Buffer> hist_prev, hist_cur, coord[2], hist_row, hin_prev, hin_cur;
    long long hist_frame = -1;
    int hist_W = 0, hist_H = 0, hist_sw = 0, hist_sh = 0;
    int coord_cur = -1;                // -1: no history for this job; 2: same frame (identity)
    unsigned temporal_seed = 0;
    // re_depth: the newest submitted frame's depth (frame pixels) and the
    // job's copy, taken when the job starts.
    std::unique_ptr<Buffer> depth_next, job_depth;
    // smooth_depth: the previous job's depth and its frame.
    std::unique_ptr<Buffer> prev_depth;
    long long prev_depth_frame = -1;
    // input_temporal: the stabilised network input, ping-pong, and its frame.
    std::unique_ptr<Buffer> in_stab[2];
    int in_stab_cur = 0, in_stab_w = 0, in_stab_h = 0, in_stab_pitch = 0;
    long long in_stab_frame = -1;
    int depth_w = 0, depth_h = 0;
    bool depth_fresh = false, job_depth_valid = false;
    // re_albedo: the same for the RR diffuse albedo's luminance.
    std::unique_ptr<Buffer> albedo_next, job_albedo;
    int albedo_w = 0, albedo_h = 0;
    bool albedo_fresh = false, job_albedo_valid = false;
    std::unique_ptr<Buffer> tone_state;  // tone=K: log-luminance sums, applied exposure, valid, smoothed log2
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
            else if (key == "motion" && (value == "0" || value == "1" || value == "auto" || value == "uv"))
                options.motion = value == "auto" ? 2 : value == "uv" ? 3 : value == "1";
            else if (key == "scale") options.scale = std::stof(value);
            else if (key == "budget") options.budget = std::stof(value);
            else if (key == "additive") options.additive = std::stof(value);
            else if (key == "rel_sigma") options.rel_sigma = std::stof(value);
            else if (key == "rel_gate") options.rel_gate = std::stof(value);
            else if (key == "smooth_depth") options.smooth_depth = std::stoi(value);
            else if (key == "input_temporal") options.input_temporal = std::stof(value);
            else if (key == "extent" && (value == "compact" || value == "legacy")) options.compact = value == "compact";
            else if (key == "tone") options.tone = std::stof(value);
            else if (key == "encoded") options.encoded = std::stoi(value);
            else if (key == "external_controls") options.external = std::stoi(value);
            else if (key == "depth_sigma") options.depth_sigma = std::stof(value);
            else if (key == "albedo_guide") options.albedo_guide = std::stoi(value);
            else if (key == "interval") options.interval = std::stoi(value);
            else if (key == "temporal") options.temporal = std::stof(value);
            else if (key == "history_clamp") options.history_clamp = std::stoi(value);
            else if (key == "reject") options.reject = std::stof(value);
            else if (key == "tone_inverse") options.tone_inverse = std::stoi(value);
            else if (key == "controls") {
                // style/128, local tone, local structure, skin, background (the
                // original's defaults without its automatic mask: 0,0,1,-1,-1).
                size_t at = 0;
                for (int k = 0; k < 5; ++k) {
                    size_t comma = value.find(',', at);
                    if ((comma == std::string::npos) != (k == 4)) throw std::runtime_error("controls needs five values: " + line);
                    options.controls[k] = std::stof(value.substr(at, comma - at));
                    at = comma + 1;
                }
            }
            else if (key == "motion_sign" && (value == "1" || value == "-1")) options.motion_sign = std::stoi(value);
            else throw std::runtime_error("unknown engine option: " + line);
        }
        if (options.sigma <= 0.0f || options.gate <= 0.0f || options.smooth < 0.0f || options.smooth > 1.0f ||
            options.limit < 0.0f || options.strength < 0.0f || options.denoise < 0 || options.denoise > 4 ||
            options.scale < 0.25f || options.scale > 1.0f || (options.scale < 1.0f && !options.luma) ||
            options.budget < 0.0f || (options.budget > 0.0f && (options.budget < 64.0f*64.0f || !options.luma)) ||
            options.additive < 0.0f || options.additive > 1.0f || (options.additive > 0.0f && !options.luma) ||
            options.rel_sigma < 0.0f || options.rel_gate < 0.0f ||
            options.input_temporal < 0.0f || options.input_temporal > 0.98f ||
            options.tone < 0.0f || options.tone > 1.0f || (options.tone > 0.0f && !options.luma) ||
            options.encoded < 0 || options.encoded > 1 || (options.encoded && !options.luma) ||
            options.temporal < 0.0f || options.temporal > 1.0f || (options.temporal > 0.0f && !options.luma))
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
        HIP_CHECK(hipEventCreateWithFlags(&inputs_ready, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&compose_done, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&net_done, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&inputs_done, hipEventDisableTiming));
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
        if (inputs_ready) (void)hipEventDestroy(inputs_ready);
        if (compose_done) (void)hipEventDestroy(compose_done);
        if (net_done) (void)hipEventDestroy(net_done);
        if (inputs_done) (void)hipEventDestroy(inputs_done);
        if (pinned) (void)hipHostFree(pinned);
        if (compose_pinned) (void)hipHostFree(compose_pinned);
        if (compose_stream) (void)hipStreamSynchronize(compose_stream);
        net.reset(); raw_input.reset(); encoded.reset(); maps[0].reset(); maps[1].reset();
        current.reset(); composed.reset(); warped[0].reset(); warped[1].reset(); previous_frame.reset();
        if (vote) (void)hipFree(vote);
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
            if (options.motion == 1 && options.motion_sign < 0) {
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
                // motion=auto votes until it has decided: a game's convention
                // does not change, and the vote costs a frame copy and a
                // kernel per frame plus a blocking readback every 30 frames
                // (S331: 5-10 ms spikes on the game's thread).
                const bool voting = options.motion == 2 && motion_choice < 0;
                if (voting) {
                    if (!previous_frame || previous_frame->count < current->count) {
                        previous_frame.reset(new Buffer(current->count));
                        previous_valid = false;
                    }
                    if (cur_w == w && cur_h == h && cur_bpp == bpp && cur_pitch == pitch) {
                        HIP_CHECK(hipMemcpyAsync(previous_frame->data, current->data, bytes, hipMemcpyDeviceToDevice, compose_stream));
                        previous_valid = true;
                    } else {
                        previous_valid = false;
                    }
                }
                HIP_CHECK(hipMemcpyAsync(current->data, staging, bytes, hipMemcpyDeviceToDevice, compose_stream));
                // This frame's motion (re_motion came first) against the previous frame.
                const int slot = int(frame_counter % kRing);
                if (voting && previous_valid && mv_w == w && mv_h == h && ring_frame[slot] == frame_counter) {
                    if (!vote) { HIP_CHECK(hipMalloc(&vote, 10*sizeof(float))); HIP_CHECK(hipMemsetAsync(vote, 0, 10*sizeof(float), compose_stream)); }
                    const int samples = ((w+5)/6)*((h+5)/6);
                    hipLaunchKernelGGL(k_motion_vote, dim3((samples+255)/256), dim3(256), 0, compose_stream,
                                       reinterpret_cast<const unsigned char*>(current->data),
                                       reinterpret_cast<const unsigned char*>(previous_frame->data), pitch, bpp,
                                       reinterpret_cast<const float2*>(ring[slot]->data), w, h, vote);
                    if (++vote_frames >= 30) {
                        float sums[10];
                        HIP_CHECK(hipMemcpyAsync(sums, vote, sizeof(sums), hipMemcpyDeviceToHost, compose_stream));
                        HIP_CHECK(hipStreamSynchronize(compose_stream));
                        HIP_CHECK(hipMemsetAsync(vote, 0, 10*sizeof(float), compose_stream));
                        vote_frames = 0;
                        // Mean error per hypothesis that scored at least half the samples.
                        int best = -1;
                        float mean[5];
                        for (int k = 0; k < 5; ++k) mean[k] = sums[5+k] > 0.0f ? sums[k]/sums[5+k] : 1e30f;
                        for (int k = 0; k < 4; ++k)
                            if (sums[5+k] >= 0.5f*sums[9] && (best < 0 || mean[k] < mean[best])) best = k;
                        // Only with real motion in the window and a clear win over no motion.
                        if (best >= 0 && sums[9] > 100.0f && sums[4] > 1.0f && mean[best] < 0.85f*mean[4]) {
                            motion_choice = best;
                        }
                    }
                }
            } catch (const std::exception& e) {
                failed = true; error = e.what(); return -1;
            }
            cur_w = w; cur_h = h; cur_pitch = pitch; cur_bpp = bpp;
            ++frame_counter;
        }
        if (!sync_inline && (busy || pending)) return 0;
        if (!sync_inline && !crop && options.luma && options.interval > 1 && job_frame >= 0 &&
            frame_counter - 1 - job_frame < options.interval) return 0;
        job_frame = frame_counter - 1;
        try {
            if (bytes > raw_bytes) {
                raw_input.reset(new Buffer((bytes+3)/4));
                raw_bytes = raw_input->count*sizeof(float);
            }
            // The job's inputs are copied on the high-priority compose stream
            // (the network's low-priority stream is starved under a path-traced
            // frame, S326) and nobody waits for them here: the worker's stream
            // waits on inputs_ready, and compose_device's one sync per frame
            // covers the caller's buffers (S328: each sync here cost the game's
            // thread a GPU round trip, 4-8 ms mean per frame in all).
            HIP_CHECK(hipMemcpyAsync(raw_input->data, staging, bytes, hipMemcpyDeviceToDevice, compose_stream));
            job_depth_valid = false;
            if (!crop && depth_fresh && depth_w == w && depth_h == h) {
                if (!job_depth || job_depth->count < size_t(w)*h) job_depth.reset(new Buffer(size_t(w)*h));
                HIP_CHECK(hipMemcpyAsync(job_depth->data, depth_next->data, size_t(w)*h*sizeof(float),
                                         hipMemcpyDeviceToDevice, compose_stream));
                job_depth_valid = true;
            }
            depth_fresh = false;
            job_albedo_valid = false;
            if (!crop && albedo_fresh && albedo_w == w && albedo_h == h) {
                if (!job_albedo || job_albedo->count < size_t(w)*h) job_albedo.reset(new Buffer(size_t(w)*h));
                HIP_CHECK(hipMemcpyAsync(job_albedo->data, albedo_next->data, size_t(w)*h*sizeof(float),
                                         hipMemcpyDeviceToDevice, compose_stream));
                job_albedo_valid = true;
            }
            albedo_fresh = false;
            history_cur = -1;
            if (!crop && options.luma && options.smooth > 0.0f) carry_history(w, h);
            coord_cur = -1;
            if (!crop && options.luma && options.temporal > 0.0f) history_coords(w, h);
            HIP_CHECK(hipEventRecord(inputs_ready, compose_stream));
            // Only the luma path is followed by a compose that syncs; the
            // others hand the staging back to the caller on return.
            if (crop || !options.luma) HIP_CHECK(hipStreamSynchronize(compose_stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        frame_w = w; frame_h = h; frame_pitch = pitch; frame_bpp = bpp;
        job_crop = crop; job_gain = gain ? 1 : 0;
        if (sync_inline) return 1;  // the worker runs it right away
        pending = true;
        wake.notify_all();
        return 1;
    }

    void loop() {
        for (;;) {
            bool sync = false;
            ReSyncJob job{};
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this] { return quit || pending || sync_pending; });
                if (quit) return;
                if (sync_pending) {
                    sync = true; job = sync_job; sync_pending = false;
                } else {
                    pending = false;
                }
                busy = true;
            }
            if (sync) {
                run_sync(job);
            } else {
                try {
                    run();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(mutex);
                    failed = true; error = e.what();
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            busy = false;
        }
    }

    // Worker thread: queues one re_sync_submit job, start to finish, on the
    // device; no host wait for the GPU (the worker returns long before the
    // job runs). The caller's queue waits for the signal at the end, so it is
    // queued whatever happens. An allocation or network rebuild after a resize
    // may synchronize the device: it then waits behind the previous job's
    // signal and this job's own wait, both of which the caller's queue
    // satisfies, so it stalls the worker, never the game.
    void run_sync(const ReSyncJob& j) {
        bool ok = false;
        try {
            hipExternalSemaphore_t wait = reinterpret_cast<hipExternalSemaphore_t>(j.wait_sem);
            hipExternalSemaphoreWaitParams wp{};
            wp.params.fence.value = j.wait_value;
            HIP_CHECK(hipWaitExternalSemaphoresAsync(&wait, &wp, 1, compose_stream));
            if (j.motion && motion(j.motion, j.w, j.h, j.motion_pitch) < 0) throw std::runtime_error(error);
            if (j.depth && depth(j.depth, j.w, j.h, j.depth_pitch) < 0) throw std::runtime_error(error);
            sync_inline = true;
            const int r = submit(j.staging, j.w, j.h, j.pitch, j.bpp, false, j.gain);
            sync_inline = false;
            if (r != 1) throw std::runtime_error(r < 0 ? error : std::string("sync job not accepted"));
            // Queued, never waited for: the map write and the signal follow
            // the network on the device (net_done), and the next job's inputs
            // are copied on compose_stream behind them, so it cannot overwrite
            // what this one still reads.
            run(false);
            HIP_CHECK(hipStreamWaitEvent(compose_stream, net_done, 0));
            if (compose_map_async(j.out, size_t(j.out_pitch), j.w, j.h, nullptr) != 1)
                throw std::runtime_error("sync job: map compose failed");
            ok = true;
        } catch (const std::exception& e) {
            sync_inline = false;
            std::lock_guard<std::mutex> lock(mutex);
            error = e.what();
            ++sync_failed;
        }
        hipExternalSemaphore_t sig = reinterpret_cast<hipExternalSemaphore_t>(j.signal_sem);
        hipExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = j.signal_value;
        const hipError_t e = hipSignalExternalSemaphoresAsync(&sig, &sp, 1, compose_stream);
        std::lock_guard<std::mutex> lock(mutex);
        if (e != hipSuccess) { failed = true; error = "sync job: signal failed"; }
        if (ok) ++sync_done;
    }

    int sync_submit(const ReSyncJob& j) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!options.luma || !j.staging || !j.out || !j.wait_sem || !j.signal_sem || j.bpp != 8) {
            error = "sync job needs luma mode, RGBA16F staging and both semaphores";
            return -1;
        }
        if (sync_pending) return 0;  // one queued at a time: the caller keeps its last result
        sync_job = j;
        sync_pending = true;
        wake.notify_all();
        return 1;
    }

    // wait=false (synchronous jobs): a luma run is only queued -- nothing on
    // the host waits for the device; net_done marks the end of the ratio map.
    void run(bool wait = true) {
        HIP_CHECK(hipStreamWaitEvent(stream, inputs_ready, 0));
        int W = 256, H = 256;
        // Luma mode may run the network on a downscale of the frame (scale).
        int sw = frame_w, sh = frame_h;
        float scale = options.scale;
        if (options.budget > 0.0f)
            scale = std::min(scale, std::sqrt(options.budget/(float(frame_w)*float(frame_h))));
        if (!job_crop && options.luma && scale < 1.0f) {
            sw = std::max(64, int(frame_w*scale + 0.5f));
            sh = std::max(64, int(frame_h*scale + 0.5f));
        }
        if (!job_crop) network_extent(sw, sh, W, H, options.compact);
        if (!net || net->W != W || net->H != H) {
            net.reset();
            net.reset(new Network(config, W, H));
        }
        const float* tone = nullptr;
        if (!job_crop && options.luma && options.tone > 0.0f) {
            if (!tone_state) {
                tone_state.reset(new Buffer(5));
                HIP_CHECK(hipMemsetAsync(tone_state->data, 0, 5*sizeof(float), stream));
            }
            const int samples = ((frame_w+3)/4)*((frame_h+3)/4);
            hipLaunchKernelGGL(k_tone_measure, dim3((samples+255)/256), dim3(256), 0, stream,
                               reinterpret_cast<const unsigned char*>(raw_input->data), frame_pitch, frame_bpp,
                               frame_w, frame_h, tone_state->data);
            hipLaunchKernelGGL(k_tone_update, dim3(1), dim3(1), 0, stream, tone_state->data, options.tone);
            HIP_CHECK(hipGetLastError());
            tone = tone_state->data;
        }
        float job_strength;
        {
            std::lock_guard<std::mutex> lock(mutex);  // set_controls may change them
            for (int k = 0; k < 5; ++k) net->front->controls.v[k] = job_crop ? 0.0f : options.controls[k];
            job_strength = options.strength;
        }
        const bool temporal = !job_crop && options.luma && options.temporal > 0.0f;
        bool use_history = false, hin_ok = false;
        if (temporal) {
            if (!hist_row) {
                std::vector<float> row(32);
                std::string path = config[16] + "/coeff_history.f32";
                FILE* f = fopen(path.c_str(), "rb");
                if (!f || fread(row.data(), sizeof(float), 32, f) != 32) {
                    if (f) fclose(f);
                    throw std::runtime_error("temporal needs " + path);
                }
                fclose(f);
                hist_row.reset(new Buffer(32));
                HIP_CHECK(hipMemcpy(hist_row->data, row.data(), 32*sizeof(float), hipMemcpyHostToDevice));
            }
            size_t count = size_t(W)*H*3;
            if (!hist_cur || hist_cur->count < count) hist_cur.reset(new Buffer(count));
            use_history = coord_cur >= 0 && hist_prev && hist_W == W && hist_H == H && hist_sw == sw && hist_sh == sh;
            if (!hin_cur || hin_cur->count < count) hin_cur.reset(new Buffer(count));
            hin_ok = use_history && hin_prev;
            const float2* coords = coord_cur == 2 || !use_history ? nullptr : reinterpret_cast<const float2*>(coord[coord_cur]->data);
            if (use_history)
                hipLaunchKernelGGL(k_history_sample, dim3((W*H+255)/256), dim3(256), 0, stream,
                                   hist_prev->data, coords, hist_cur->data, frame_w, frame_h, sw, sh, W, H);
            if (hin_ok)
                hipLaunchKernelGGL(k_history_sample, dim3((W*H+255)/256), dim3(256), 0, stream,
                                   hin_prev->data, coords, hin_cur->data, frame_w, frame_h, sw, sh, W, H);
            // The original's seed: a counter per run, 0 on the first frame and
            // after a reset (here: whenever there is no history to blend with).
            temporal_seed = use_history ? temporal_seed + 1 : 0;
        }
        net->front->history = use_history ? hist_cur->data : nullptr;
        net->front->noise_seed = temporal ? temporal_seed : 0;
        net->head70->keep_native = temporal && use_history;  // k_head_temporal's input
        HIP_CHECK(hipEventRecord(start, stream));
        const unsigned char* net_src = reinterpret_cast<const unsigned char*>(raw_input->data);
        if (!job_crop && options.luma && options.input_temporal > 0.0f && frame_bpp == 8) {
            const size_t bytes = size_t(frame_pitch)*frame_h;
            for (auto& b : in_stab)
                if (!b || b->count*sizeof(float) < bytes) { b.reset(new Buffer((bytes+3)/4)); in_stab_frame = -1; }
            const int dst = in_stab_cur ^ 1;
            const int mslot = int(job_frame % kRing);
            const float2 ms = motion_scale(frame_w, frame_h);
            const bool hist_ok = in_stab_frame == job_frame - 1 && in_stab_w == frame_w && in_stab_h == frame_h &&
                                 in_stab_pitch == frame_pitch && mv_w == frame_w && mv_h == frame_h &&
                                 ring_frame[mslot] == job_frame && (ms.x != 0.0f || ms.y != 0.0f);
            const bool depth_ok = job_depth_valid && prev_depth && prev_depth_frame == job_frame - 1;
            auto* out = reinterpret_cast<unsigned char*>(in_stab[dst]->data);
            if (hist_ok) {
                hipLaunchKernelGGL(k_input_stabilize, dim3((frame_w*frame_h+255)/256), dim3(256), 0, stream,
                                   net_src, reinterpret_cast<const unsigned char*>(in_stab[in_stab_cur]->data), out,
                                   frame_pitch, frame_w, frame_h, reinterpret_cast<const float2*>(ring[mslot]->data),
                                   ms.x, ms.y, depth_ok ? job_depth->data : nullptr,
                                   depth_ok ? prev_depth->data : nullptr, options.input_temporal);
                HIP_CHECK(hipGetLastError());
            } else {
                HIP_CHECK(hipMemcpyAsync(out, net_src, bytes, hipMemcpyDeviceToDevice, stream));
            }
            in_stab_cur = dst; in_stab_frame = job_frame;
            in_stab_w = frame_w; in_stab_h = frame_h; in_stab_pitch = frame_pitch;
            net_src = out;
        }
        net->run(net_src, frame_w, frame_h, frame_pitch,
                 frame_bpp, job_crop, stream, sw, sh, tone, !job_crop && options.encoded,
                 use_history && options.history_clamp ? hist_cur->data : nullptr,
                 use_history && options.history_clamp == 2 && hin_ok ? hin_cur->data : nullptr, options.reject);
        if (temporal) {
            if (use_history) {
                hipLaunchKernelGGL(k_head_temporal, dim3((W*H+255)/256), dim3(256), 0, stream,
                                   net->head70->native_view().data, hist_row->data, hist_cur->data,
                                   net->rgb_native->data, net->rgb_public->data, W, H, 0.7397f*options.temporal);
                HIP_CHECK(hipGetLastError());
            }
            size_t count = size_t(W)*H*3;
            if (!hist_prev || hist_prev->count < count) hist_prev.reset(new Buffer(count));
            HIP_CHECK(hipMemcpyAsync(hist_prev->data, net->rgb_native->data, count*sizeof(float),
                                     hipMemcpyDeviceToDevice, stream));
            if (!hin_prev || hin_prev->count < count) hin_prev.reset(new Buffer(count));
            HIP_CHECK(hipMemcpyAsync(hin_prev->data, net->color->data, count*sizeof(float),
                                     hipMemcpyDeviceToDevice, stream));
            hist_W = W; hist_H = H; hist_sw = sw; hist_sh = sh;
            hist_frame = job_frame;
        }
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
                // smooth_depth: the history is the previous frame's map, both
                // frames have depth, and this frame's motion is in the ring.
                const float* prev_d = nullptr;
                const float2* job_mv = nullptr;
                float2 ms = make_float2(0.0f, 0.0f);
                const int mslot = int(job_frame % kRing);
                if (options.smooth_depth && previous_valid && job_depth_valid && prev_depth &&
                    prev_depth_frame == job_frame - 1 && map_frame[published] == job_frame - 1 &&
                    mv_w == frame_w && mv_h == frame_h && ring_frame[mslot] == job_frame) {
                    ms = motion_scale(frame_w, frame_h);
                    if (ms.x != 0.0f || ms.y != 0.0f) {
                        prev_d = prev_depth->data;
                        job_mv = reinterpret_cast<const float2*>(ring[mslot]->data);
                    }
                }
                hipLaunchKernelGGL(k_ratio_map, dim3((frame_w*frame_h+255)/256), dim3(256), 0, stream,
                                   linear, net->color->data,
                                   reinterpret_cast<const unsigned char*>(raw_input->data), frame_pitch, frame_bpp,
                                   !previous_valid ? nullptr : history_cur >= 0
                                       ? reinterpret_cast<const float2*>(history[history_cur]->data)
                                       : reinterpret_cast<const float2*>(maps[published]->data),
                                   int(previous_valid), job_options(job_strength), reinterpret_cast<float2*>(maps[next]->data),
                                   frame_w, frame_h, sw, sh, W, tone,
                                   job_depth_valid ? job_depth->data : nullptr,
                                   job_albedo_valid ? job_albedo->data : nullptr,
                                   job_mv ? prev_d : nullptr, job_mv, ms.x, ms.y);
                HIP_CHECK(hipGetLastError());
                if (options.smooth_depth || options.input_temporal > 0.0f) {
                    // Keep this job's depth for the next one (same stream, in order).
                    if (job_depth_valid) {
                        if (!prev_depth || prev_depth->count < size_t(frame_w)*frame_h)
                            prev_depth.reset(new Buffer(size_t(frame_w)*frame_h));
                        HIP_CHECK(hipMemcpyAsync(prev_depth->data, job_depth->data, size_t(frame_w)*frame_h*sizeof(float),
                                                 hipMemcpyDeviceToDevice, stream));
                        prev_depth_frame = job_frame;
                    } else {
                        prev_depth_frame = -1;
                    }
                }
                float elapsed = last_ms;
                if (wait) {
                    HIP_CHECK(hipStreamSynchronize(stream));
                    HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
                } else {
                    // Published now: everything that reads it (the next job's
                    // carry, the map write) is queued behind net_done.
                    HIP_CHECK(hipEventRecord(net_done, stream));
                }
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
    // smooth: the published map carried from its frame to job_frame into
    // history[history_cur] (caller holds the mutex, the worker is idle). Without
    // motion (off, undecided or lost on the way) it stays the plain previous map.
    void carry_history(int w, int h) {
        if (published < 0 || map_w != w || map_h != h || map_frame[published] < 0) return;
        const float2 ms = motion_scale(w, h);
        const float mx = ms.x, my = ms.y;
        if (!options.motion || (mx == 0.0f && my == 0.0f) || mv_w != w || mv_h != h) return;
        const long long from = map_frame[published];
        if (from >= job_frame) return;  // already on the job's frame: the plain map is right
        for (auto& b : history)
            if (!b || b->count < size_t(w)*h*2) b.reset(new Buffer(size_t(w)*h*2));
        const float2* src = reinterpret_cast<const float2*>(maps[published]->data);
        int cur = -1;
        for (long long f = from + 1; f <= job_frame; ++f) {
            int slot = int(f % kRing);
            if (ring_frame[slot] != f) return;  // motion lost: keep the plain previous map
            int next = cur == 0 ? 1 : 0;
            hipLaunchKernelGGL(k_warp_map, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                               src, reinterpret_cast<const float2*>(ring[slot]->data),
                               reinterpret_cast<float2*>(history[next]->data), w, h, mx, my);
            src = reinterpret_cast<const float2*>(history[next]->data);
            cur = next;
        }
        HIP_CHECK(hipGetLastError());
        history_cur = cur;
    }

    // external_controls: the original's mapping of its controls onto the stem
    // features (peer DLSSNR-AMD patch_push) without the automatic mask, which
    // we cannot compute: style/128, tone, structure, skin -1, background -1.
    // Intensity becomes the luma strength.
    // re_depth: copy the frame's depth (float per pixel, rows of pitch bytes)
    // for the next job. Only kept when depth_sigma is on.
    int depth(const void* src, int w, int h, int pitch) {
        if (options.depth_sigma <= 0.0f || !options.luma) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        try {
            if (!depth_next || depth_next->count < size_t(w)*h) depth_next.reset(new Buffer(size_t(w)*h));
            HIP_CHECK(hipMemcpy2DAsync(depth_next->data, size_t(w)*sizeof(float), src, size_t(pitch),
                                       size_t(w)*sizeof(float), h, hipMemcpyDeviceToDevice, compose_stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        depth_w = w; depth_h = h; depth_fresh = true;
        return 1;
    }

    // re_albedo: the RR diffuse albedo luminance for the next job (albedo_guide).
    int albedo(const void* src, int w, int h, int pitch) {
        if (!options.albedo_guide || !options.luma) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        try {
            if (!albedo_next || albedo_next->count < size_t(w)*h) albedo_next.reset(new Buffer(size_t(w)*h));
            HIP_CHECK(hipMemcpy2DAsync(albedo_next->data, size_t(w)*sizeof(float), src, size_t(pitch),
                                       size_t(w)*sizeof(float), h, hipMemcpyDeviceToDevice, compose_stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        albedo_w = w; albedo_h = h; albedo_fresh = true;
        return 1;
    }

    // The staged motion vectors' scale to frame pixels at w x h (prev = cur +
    // mv*scale), 0 while motion=auto is undecided. UV scales with the frame:
    // a scale fixed when the vote decided went stale on every resize (S332:
    // decided at 1510x850, still applied at 1587x893).
    float2 motion_scale(int w, int h) const {
        if (!options.motion) return make_float2(0.0f, 0.0f);
        if (options.motion == 1) return make_float2(1.0f, 1.0f);  // the ring is already flipped
        const int c = options.motion == 3 ? (options.motion_sign > 0 ? 2 : 3) : motion_choice;
        if (c < 0) return make_float2(0.0f, 0.0f);
        const float sign = (c & 1) ? -1.0f : 1.0f;
        return c >= 2 ? make_float2(sign*w, sign*h) : make_float2(sign, sign);
    }

    OutputOptions job_options(float strength) const {
        OutputOptions o = options;
        o.strength = strength;
        return o;
    }
    int set_controls(const float* v) {
        if (!options.external) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        const float style = fminf(fmaxf(v[1], 0.0f), 2.0f);
        options.controls[0] = style / 128.0f;
        options.controls[1] = fminf(fmaxf(v[3], 0.0f), 2.0f);
        options.controls[2] = fminf(fmaxf(v[2], 0.0f), 2.0f);
        options.controls[3] = -1.0f;
        options.controls[4] = -1.0f;
        options.strength = fminf(fmaxf(v[0], 0.0f), 2.0f);
        return 1;
    }

    // temporal: where each pixel of job_frame was on hist_frame, chaining the
    // per-frame motion (caller holds the mutex, the worker is idle).
    void history_coords(int w, int h) {
        if (hist_frame < 0 || !hist_prev) return;
        if (hist_frame == job_frame) { coord_cur = 2; return; }
        const float2 ms = motion_scale(w, h);
        const float mx = ms.x, my = ms.y;
        if (!options.motion || (mx == 0.0f && my == 0.0f) || mv_w != w || mv_h != h || hist_frame > job_frame) return;
        if (job_frame - hist_frame >= kRing) return;
        for (auto& b : coord)
            if (!b || b->count < size_t(w)*h*2) b.reset(new Buffer(size_t(w)*h*2));
        hipLaunchKernelGGL(k_coord_identity, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                           reinterpret_cast<float2*>(coord[0]->data), w, h);
        int cur = 0;
        for (long long f = hist_frame + 1; f <= job_frame; ++f) {
            int slot = int(f % kRing);
            if (ring_frame[slot] != f) return;  // motion lost: no history
            hipLaunchKernelGGL(k_warp_map, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                               reinterpret_cast<const float2*>(coord[cur]->data),
                               reinterpret_cast<const float2*>(ring[slot]->data),
                               reinterpret_cast<float2*>(coord[1-cur]->data), w, h, mx, my);
            cur = 1 - cur;
        }
        HIP_CHECK(hipGetLastError());
        coord_cur = cur;
    }

    // map_only: write the carried map (k_map_out, RGBA16F) instead of composing.
    int compose_into(unsigned char* out, size_t out_pitch, int w, int h, int bpp, size_t dst_pitch,
                     bool map_only = false, bool sync = true) {
        if (!out || published < 0 || map_w != w || map_h != h || cur_w != w || cur_h != h || cur_bpp != bpp ||
            dst_pitch < size_t(w)*bpp || out_pitch < size_t(w)*bpp) return 0;
        try {
            const float2* map = reinterpret_cast<const float2*>(maps[published]->data);
            const float2 ms = motion_scale(w, h);
            const float mx = ms.x, my = ms.y;
            if (options.motion && (mx != 0.0f || my != 0.0f) && mv_w == w && mv_h == h && map_frame[published] >= 0) {
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
                                       reinterpret_cast<float2*>(warped[next]->data), w, h, mx, my);
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
            if (map_only)
                hipLaunchKernelGGL(k_map_out, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                                   map, out, out_pitch, w, h);
            else
                hipLaunchKernelGGL(k_compose, dim3((w*h+255)/256), dim3(256), 0, compose_stream,
                                   reinterpret_cast<const unsigned char*>(current->data), cur_pitch, bpp,
                                   map, options, out, out_pitch, w, h);
            HIP_CHECK(hipGetLastError());
            if (sync) HIP_CHECK(hipStreamSynchronize(compose_stream));
            else HIP_CHECK(hipEventRecord(compose_done, compose_stream));
        } catch (const std::exception& e) {
            failed = true; error = e.what(); return -1;
        }
        return 1;
    }

    // Asynchronous map compose: queued on compose_stream, compose_done marks
    // the end. 1 queued, 0 nothing to compose (stream left idle), -1 error.
    int compose_map_async(void* dst, size_t pitch, int w, int h, float* device_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!options.luma) return 0;
        int r = compose_into(reinterpret_cast<unsigned char*>(dst), pitch, w, h, 8, pitch, true, false);
        if (r == 0 && hipStreamSynchronize(compose_stream) != hipSuccess) {
            failed = true; error = "compose stream sync failed"; return -1;
        }
        if (r == 1 && device_ms) *device_ms = last_ms;
        return r;
    }
    // 1 the last asynchronous compose (and everything queued before it on
    // compose_stream) finished, 0 still running, -1 error.
    int mark_inputs() {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (hipEventRecord(inputs_done, compose_stream) != hipSuccess) {
            failed = true; error = "inputs event record failed"; return -1;
        }
        return 1;
    }
    int inputs_copied() {
        hipError_t e = hipEventQuery(inputs_done);
        if (e == hipSuccess) return 1;
        if (e == hipErrorNotReady) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        failed = true; error = "inputs event query failed"; return -1;
    }

    int compose_ready() {
        hipError_t e = hipEventQuery(compose_done);
        if (e == hipSuccess) return 1;
        if (e == hipErrorNotReady) return 0;
        std::lock_guard<std::mutex> lock(mutex);
        failed = true; error = "compose event query failed"; return -1;
    }

    int compose_device(void* dst, size_t pitch, int w, int h, int bpp, float* device_ms, bool map_only = false) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) return -1;
        if (!options.luma || (map_only && bpp != 8)) return 0;
        int r = compose_into(reinterpret_cast<unsigned char*>(dst), pitch, w, h, bpp, pitch, map_only);
        if (r == 0 && hipStreamSynchronize(compose_stream) != hipSuccess) {
            failed = true; error = "compose stream sync failed"; return -1;
        }
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
            if (r == 0 && hipStreamSynchronize(compose_stream) != hipSuccess) {
                failed = true; error = "compose stream sync failed"; return -1;
            }
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
// Luma mode: the depth of the frame the next re_submit_full carries (float
// per frame pixel, rows of pitch bytes, device memory). 1 kept, 0 unused
// (depth_sigma off), -1 error.
__declspec(dllexport) int re_depth(const void* depth, int width, int height, int pitch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine || !depth) return -1;
    return g_engine->depth(depth, width, height, pitch);
}

// Luma mode: the RR diffuse albedo luminance of the frame the next
// re_submit_full carries (float per frame pixel). 1 kept, 0 unused, -1 error.
__declspec(dllexport) int re_albedo(const void* albedo, int width, int height, int pitch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine || !albedo) return -1;
    return g_engine->albedo(albedo, width, height, pitch);
}

// The caller's NR controls: intensity, style (0..2), local structure, local
// tone, skin structure, automatic mask (as OptiScaler's DLSS-NR passes them).
// Returns 1 applied, 0 ignored (external_controls=0), -1 no engine.
__declspec(dllexport) int re_controls(const float* v, int n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine || !v || n < 6) return -1;
    return g_engine->set_controls(v);
}

__declspec(dllexport) int re_submit(const void* staging, int width, int height, int pitch, int bpp) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->submit(staging, width, height, pitch, bpp, true, 0);
}

// Full mode with the output gain (0 native, 1 public). Same return codes.
__declspec(dllexport) int re_submit_full(const void* staging, int width, int height, int pitch, int bpp, int gain) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    const int r = g_engine->submit(staging, width, height, pitch, bpp, false, gain);
    return r < 0 ? r : g_engine->mark_inputs() < 0 ? -1 : r;
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

// Luma mode: the carried ratio map instead of a composed frame, RGBA16F
// (bpp must be 8) of (ratio, Y_in, 0, 1), Y_in < 0 = no edit. The map sits on
// the newest submitted frame; re_luma_map_params says how to carry it further
// and gate it. Returns 1 written, 0 nothing yet or not luma mode, -1 error.
__declspec(dllexport) int re_compose_map_device(void* dst, unsigned long long pitch, int width, int height,
                                                float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->compose_device(dst, size_t(pitch), width, height, 8, device_ms, true);
}

// Asynchronous re_compose_map_device: returns once the map is queued (1),
// 0 nothing yet (nothing queued), -1 error. dst may only be read, and the
// buffers handed to re_submit_full/re_motion/re_depth/re_albedo since the
// last ready poll may only be rewritten, once re_compose_ready says 1.
// Synchronous path (S332): queue one frame (ReSyncJob above). 1 accepted --
// signal_value will be signalled, also if the job fails; 0 a job is still
// queued (nothing will be signalled for this one); -1 error.
__declspec(dllexport) int re_sync_submit(const ReSyncJob* job) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine || !job) return -1;
    return g_engine->sync_submit(*job);
}

// Synchronous jobs finished and failed so far (failed ones still signalled).
__declspec(dllexport) int re_sync_stats(unsigned long long* done, unsigned long long* failed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    if (done) *done = g_engine->sync_done;
    if (failed) *failed = g_engine->sync_failed;
    return 1;
}

__declspec(dllexport) int re_compose_map_async(void* dst, unsigned long long pitch, int width, int height,
                                               float* device_ms) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->compose_map_async(dst, size_t(pitch), width, height, device_ms);
}

// Non-blocking: 1 the caller's buffers handed to the engine so far (motion,
// depth, albedo, staging) have been copied, so they may be rewritten; 0 not
// yet; -1 error. re_submit_full marks the point.
__declspec(dllexport) int re_inputs_ready() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->inputs_copied();
}

// Non-blocking: 1 the last re_compose_map_async finished, 0 not yet, -1 error.
__declspec(dllexport) int re_compose_ready() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    return g_engine->compose_ready();
}

// For a map consumer: params[0..1] scale the staged motion vectors to frame
// pixels (prev = cur + mv*scale; 0 when motion is off or not detected yet),
// params[2] = 1/gate, params[3] = 1 in luma mode. Returns 1, or -1.
__declspec(dllexport) int re_luma_map_params(float* params) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine || !params) return -1;
    auto& e = *g_engine;
    std::lock_guard<std::mutex> inner(e.mutex);
    // motion=1 flips the engine's own copy; the consumer reads the game's vectors.
    const float2 ms = e.options.motion == 1 ? make_float2(float(e.options.motion_sign), float(e.options.motion_sign))
                                            : e.motion_scale(e.cur_w, e.cur_h);
    params[0] = ms.x;
    params[1] = ms.y;
    // A negative value asks the resolve for the relative (log2) gate.
    params[2] = e.options.rel_gate > 0.0f ? -1.0f / e.options.rel_gate : 1.0f / e.options.gate;
    params[3] = e.options.luma ? 1.0f : 0.0f;
    return 1;
}

// motion=auto: the detected convention, -1 not yet, 0 frame pixels, 1 frame
// pixels negated, 2 UV, 3 UV negated.
__declspec(dllexport) int re_motion_mode() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return -1;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    const auto& e = *g_engine;
    return e.options.motion == 3 ? (e.options.motion_sign > 0 ? 2 : 3) : e.motion_choice;
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

// Network results finished so far (a replay waits for each run).
__declspec(dllexport) unsigned long long re_completed() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_engine) return 0;
    std::lock_guard<std::mutex> inner(g_engine->mutex);
    return g_engine->completed;
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
