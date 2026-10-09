#pragma once
// Resident candidate C32 front end: stem -> block0 -> pool -> encoder1-4 ->
// block4 downsample. Runs in the public (peer) channel basis and writes the
// resident chain's native-basis C64 input, block4 skip and preblock0 skip.
// C32 bodies reuse the existing head70 body kernels unchanged.
#include "c32_resident.h"

namespace front_resident {
#define FRONT_LAUNCH(kernel,count,...) do {     hipLaunchKernelGGL(kernel,dim3(((count)+255)/256),dim3(256),0,c512_resident::stream,__VA_ARGS__);     HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while(0)
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c32_resident::Body;
using c32_resident::Weights;
using c32_resident::check;

// Window order of an image with the resident shift convention (4-pixel pads).
__device__ inline bool front_window_pixel(int token, int width, int height, int shift, int& x, int& y) {
    int px = (shift&1) ? 4 : 0, py = (shift&2) ? 4 : 0;
    int wx = (width+px+7)/8;
    int window = token/64, local = token%64;
    x = (window%wx)*8 + c32_local_x(local) - px;
    y = (window/wx)*8 + c32_local_y(local) - py;
    return x >= 0 && x < width && y >= 0 && y < height;
}

// 15 input channels: noise(3), 1, x(3), x(3), controls(5); x=half((rgb-.5)*.125).
// The controls are the original's style/128, local tone, local structure, skin
// and background features (peer DLSSNR-AMD image_input.glsl); the fixtures
// were captured with all five zero, which stays the default.
struct FrontControls { float v[5]; };
// history (optional, rgb's layout): features 7..9 become the previous output
// carried to this frame, as in the original's temporal pre block; a negative
// first channel marks a pixel without history (it keeps the current colour).
__global__ void k_front_stem(const float* rgb, const float* noise, const float* stem,
                             float* tokens, int width, int height, FrontControls controls = {},
                             const float* history = nullptr) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id >= width*height*32) return;
    int token = id/32, c = id%32, x, y;
    front_window_pixel(token, width, height, 0, x, y);
    const float* p = rgb + (size_t(y)*width+x)*3;
    const float* n = noise + (size_t(y)*width+x)*3;
    float f[15];
    for (int k = 0; k < 3; ++k) f[k] = h70_h(n[k]);
    f[3] = 1.0f;
    for (int k = 0; k < 3; ++k) {
        float centered = p[k]-.5f;
        float v = h70_h(centered*.125f);
        f[4+k] = v; f[7+k] = v;
    }
    if (history) {
        const float* q = history + (size_t(y)*width+x)*3;
        if (q[0] >= 0.0f)
            for (int k = 0; k < 3; ++k) f[7+k] = h70_h(h70_h(h70_h(q[k])-.5f)*.125f);
    }
    for (int k = 10; k < 15; ++k) f[k] = h70_h(controls.v[k-10]);
    float part = 0.0f;
    for (int k = 0; k < 15; ++k) part += f[k]*stem[k*32+c];
    tokens[id] = h70_h(part);
}

// The original's pre-block noise (peer DLSSNR-AMD image_noise.glsl): PCG +
// Box-Muller of (x, y, seed). Seed 0 reproduces the captured 256x256 tile.
__device__ inline unsigned front_pcg(unsigned v) {
    v = (v >> ((v >> 28) + 4u)) ^ v;
    return v * 0x108EF2D9u;
}
__device__ inline float front_uniform(unsigned stream) {
    const unsigned t = front_pcg(stream);
    return float(((t >> 30) ^ (t >> 8)) + 1u) * 5.9604644775390625e-08f;
}
__global__ void k_front_noise(float* noise, int width, int height, unsigned seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= width*height) return;
    const unsigned x = unsigned(i % width), y = unsigned(i / width);
    const unsigned base = (x*0x8DA6B343u) ^ (seed*0x9E3779B9u) ^ (y*0xD8163841u) ^ 0x243F6A88u;
    const unsigned t = front_pcg(base), h = (t >> 22) ^ t;
    const float uA = front_uniform(h*0x2C9277B5u + 0xAC564B05u), uB = front_uniform(h*0xFA6DC5F9u + 0x4712A88Eu);
    const float uC = front_uniform(h*0xCAA5B80Du + 0x21DD796Bu), uD = front_uniform(h*0x83232C31u + 0x3463E0ACu);
    const float rA = sqrtf(-2.0f*logf(uA)), rC = sqrtf(-2.0f*logf(uC));
    noise[size_t(i)*3+0] = rA*cosf(uB*6.28318530718f);
    noise[size_t(i)*3+1] = rA*sinf(uB*6.28318530718f);
    noise[size_t(i)*3+2] = rC*cosf(uD*6.28318530718f);
}

__global__ void k_front_gather(const float* image, float* tokens, int width, int height, int shift) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int px = (shift&1) ? 4 : 0, py = (shift&2) ? 4 : 0;
    int count = ((width+px+7)/8)*((height+py+7)/8)*64*32;
    if (id >= count) return;
    int x, y;
    tokens[id] = front_window_pixel(id/32, width, height, shift, x, y) ?
                 image[(size_t(y)*width+x)*32+id%32] : 0.0f;
}

__global__ void k_front_scatter(const float* tokens, float* image, int width, int height, int shift) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int px = (shift&1) ? 4 : 0, py = (shift&2) ? 4 : 0;
    int count = ((width+px+7)/8)*((height+py+7)/8)*64*32;
    if (id >= count) return;
    int x, y;
    if (front_window_pixel(id/32, width, height, shift, x, y))
        image[(size_t(y)*width+x)*32+id%32] = tokens[id];
}

// Rounded 2x2 pool then FP8, as the other encoder pools.
__global__ void k_front_pool(const float* raw, float* pool, int width, int height, int channels) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int pw = width/2, ph = height/2;
    if (i >= pw*ph*channels) return;
    int c = i%channels, x = i/channels%pw, y = i/channels/pw;
    size_t top = (size_t(2*y)*width+2*x)*channels+c, bottom = (size_t(2*y+1)*width+2*x)*channels+c;
    float a = h70_h(raw[top]+raw[top+channels]);
    float b = h70_h(raw[bottom]+raw[bottom+channels]);
    pool[i] = h70_f(h70_h(h70_h(a+b)*.25f));
}

// C32 -> C64 projection with one 32-product half boundary, then FP8.
__global__ void k_front_down(const float* pool, const float* matrix, float* output, int pixels) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= pixels*64) return;
    const float* x = pool + (i/64)*32;
    const float* w = matrix + (i%64)*32;
    float part = 0.0f;
    for (int k = 0; k < 32; ++k) part += x[k]*w[k];
    output[i] = h70_f(h70_h(part));
}

__device__ inline int front_multihead64(int c) { return (c/16)*16+(c%8)*2+(c%16/8); }

__global__ void k_front_native64(const float* peer, float* native, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) native[(i&~63)+front_multihead64(i&63)] = peer[i];
}

__global__ void k_front_native32(const float* peer, float* native, int n, int clamp) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float v = peer[i];
    if (clamp) v = fminf(448.0f, fmaxf(-448.0f, v));
    native[(i&~31)+peer_to_native32(i&31)] = v;
}

class FrontEnd {
public:
    FrontControls controls{};  // stem features 10..14 (engine option controls=)
    const float* history = nullptr;  // stem features 7..9 (engine option temporal=), null: current colour
    unsigned noise_seed = 0;         // 0: the captured tile; otherwise the original's noise for this seed
    // noise_function: the original's noise also at seed 0 (computed over the
    // whole extent, as the original) instead of the fixtures' 256x256 tile.
    bool noise_function = false, loaded_function = false;
    // half_skip: the fused block0 writes skip0 as halves only (skip0_half), for
    // a head that reads them (Head::fusable); skip0_view is then empty.
    bool half_skip = false;
    // skip4_native: also write skip4 in native order (skip4_view); the
    // network reads skip4_peer_view instead and turns this off.
    bool skip4_native = true;
    // pool4: block4 pools its raw body itself (k_c32_t fz.pool), no raw4 buffer.
    bool pool4 = [] { const char* v = getenv("RESIDENT_POOL4"); return !(v && *v == '0'); }();
private:
    unsigned loaded_seed = 0;
    static int checked(int w, int h) {
        if (w <= 0 || h <= 0 || w % 64 || h % 64) throw std::invalid_argument("front end requires RGB extents that are multiples of 64");
        return w;
    }
    int width, height;
    size_t pixels;
    std::string dir;
    Buffer noise, stem, matrix, tokens, raw0, image, raw4, down_pool, down, c64, skip4, skip0, skip0h;
    Buffer matrix_native;  // matrix rows in the native order: the down GEMM writes c64 directly
    bool skip_is_half = false;  // this run wrote skip0 only as halves
    std::vector<std::unique_ptr<Weights>> weights;
    Body body;
    std::unique_ptr<Buffer> body_raw, body_quant;  // chunked body outputs above the cap
    bool ready = false;
public:
    size_t comparisons = 0;
    // max_rows caps the C32 body scratch; larger frames run in chunks without stage checks.
    FrontEnd(int w, int h, const std::string& fixture, int max_rows = 0) : width(checked(w,h)), height(h), pixels(size_t(w)*h),
        dir(fixture), noise(pixels*3), stem(dir,"stem_weights",15*32), matrix(dir,"matrix",64*32), matrix_native(64*32),
        tokens(size_t(w/2+8)*(h/2+8)*32 > pixels*32 ? size_t(w/2+8)*(h/2+8)*32 : pixels*32),
        raw0(pixels*32), image(pixels/4*32), raw4(pixels/4*32), down_pool(pixels/16*32), down(pixels/16*64),
        c64(pixels/16*64), skip4(pixels/4*32), skip0(pixels*32), skip0h(pixels*16),
        body(max_rows > 0 && size_t(max_rows) < pixels ? max_rows : int(pixels)) {
        for (int b = 0; b < 5; ++b) weights.emplace_back(new Weights(dir+"/block"+std::to_string(b)));
        {
            const auto m = read(dir, "matrix", 64*32);
            std::vector<float> nat(64*32);
            for (int r = 0; r < 64; ++r) {
                const int n = (r/16)*16 + (r%8)*2 + (r%16)/8;  // front_multihead64(r)
                for (int k = 0; k < 32; ++k) nat[size_t(n)*32 + k] = m[size_t(r)*32 + k];
            }
            HIP_CHECK(hipMemcpy(matrix_native.data.get(), nat.data(), nat.size()*sizeof(float), hipMemcpyHostToDevice));
        }
        load_noise();
        if (size_t(body.rows_capacity()) < pixels) {
            body_raw.reset(new Buffer(tokens.count));
            body_quant.reset(new Buffer(tokens.count));
        }
    }
    // The public 256x256 noise tile, repeated over this extent (the reference's
    // tiling). Fixtures store it tiled to their own 256-row extent.
    void load_noise() {
        std::string path = dir + "/noise.f32";
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) HEAD70_FATAL("cannot read " + path);
        std::vector<float> saved;
        float chunk[4096];
        size_t got;
        while ((got = fread(chunk, sizeof(float), 4096, f)) > 0) saved.insert(saved.end(), chunk, chunk+got);
        fclose(f);
        size_t saved_w = saved.size() / (256*3);
        if (saved.size() != saved_w*256*3 || saved_w < 256) HEAD70_FATAL("wrong size: " + path);
        std::vector<float> tiled(pixels*3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    tiled[(size_t(y)*width+x)*3+c] = saved[((size_t(y%256))*saved_w+(x%256))*3+c];
        HIP_CHECK(hipMemcpy(noise.data, tiled.data(), tiled.size()*sizeof(float), hipMemcpyHostToDevice));
        c512_resident::traffic.h2d_bytes += tiled.size()*sizeof(float);
    }
    // Block body over rows windows: whole-frame with stage checks, or chunked.
    bool block(int b, int rows, bool verify, const float*& raw, const float*& quant) {
        if (body_raw) {
            if (verify || !body.run_into(*weights[b], tokens.data, rows, body_raw->data, body_quant->data)) return false;
            raw = body_raw->data; quant = body_quant->data;
            return true;
        }
        if (!body.run(*weights[b], tokens.data, rows, verify, comparisons)) return false;
        raw = body.raw(); quant = body.quantized();
        return true;
    }
    bool run_from_device(DeviceTensor rgb, bool verify = false) {
        ready = false;
        if (!rgb.data || rgb.count != pixels*3) return false;
        if (!check(dir,"rgb",rgb.data,pixels*3,verify,comparisons)) return false;
        if (noise_seed != loaded_seed || noise_function != loaded_function) {
            if (noise_seed || noise_function) FRONT_LAUNCH(k_front_noise,pixels,noise.data,width,height,noise_seed);
            else load_noise();
            loaded_seed = noise_seed; loaded_function = noise_function;
        }
        const float *raw = nullptr, *quant = nullptr;
        const bool fused = !verify && !c512_resident::exact_math;  // bodies gather/scatter themselves
        // Stem, block0, pool and skip0 in one launch (k_c32_t MODE 1).
        const bool pre = fused && Weights::fusable();
        skip_is_half = pre && half_skip;
        const bool p4 = pool4 && Weights::fusable();  // the FP8 body (k_c32_t) pools; the f16 one does not
        if (!pre) FRONT_LAUNCH(k_front_stem,pixels*32,rgb.data,noise.data,stem.data,tokens.data,width,height,controls,history);
        auto io = [](int in_hwc, int w, int h, int shift) {
            C32Io v; v.in_hwc = in_hwc; v.out_hwc = 1; v.width = w; v.height = h;
            v.px = (shift&1) ? 4 : 0; v.py = (shift&2) ? 4 : 0; v.pw = ((w+v.px+7)/8)*8;
            return v;
        };
        if (pre) {
            C32Fuse fz; fz.width = width; fz.height = height; fz.rgb = rgb.data; fz.noise = noise.data;
            fz.stem = stem.data; fz.history = history; fz.pool = image.data;
            if (half_skip) fz.skip0h = reinterpret_cast<_Float16*>(skip0h.data.get()); else fz.skip0 = skip0.data;
            for (int k = 0; k < 5; ++k) fz.controls[k] = controls.v[k];
            weights[0]->fused<1>(int(pixels), fz);
        } else if (fused) {
            weights[0]->body(tokens.data, int(pixels), raw0.data, nullptr, io(0, width, height, 0));
        } else {
            if (!check(dir+"/block0","input",tokens.data,pixels*32,verify,comparisons) ||
                !block(0,int(pixels),verify,raw,quant)) return false;
            FRONT_LAUNCH(k_front_scatter,pixels*32,raw,raw0.data,width,height,0);
        }
        if (!pre) FRONT_LAUNCH(k_front_pool,pixels/4*32,raw0.data,image.data,width,height,32);
        if (!check(dir+"/block0","raw",raw0.data,pixels*32,verify,comparisons) ||
            !check(dir,"pre_down",image.data,pixels/4*32,verify,comparisons)) return false;
        const int shifts[5] = {0,0,3,0,3};
        int w = width/2, h = height/2;
        for (int b = 1; b <= 4; ++b) {
            int px = (shifts[b]&1) ? 4 : 0, py = (shifts[b]&2) ? 4 : 0;
            int rows = ((w+px+7)/8)*((h+py+7)/8)*64;
            std::string block = dir+"/block"+std::to_string(b);
            if (fused) {
                // block4: the 2x2 pool of its raw body in the epilogue (no raw4).
                C32Fuse pz; pz.pool = b == 4 && p4 ? down_pool.data.get() : nullptr;
                weights[b]->body(image.data, rows, b == 4 && !p4 ? raw4.data.get() : nullptr, image.data,
                                 io(1, w, h, shifts[b]), pz);
                continue;
            }
            FRONT_LAUNCH(k_front_gather,size_t(rows)*32,image.data,tokens.data,w,h,shifts[b]);
            if (!check(block,"input",tokens.data,size_t(rows)*32,verify,comparisons) ||
                !this->block(b,rows,verify,raw,quant)) return false;
            if (b == 4) FRONT_LAUNCH(k_front_scatter,size_t(rows)*32,raw,raw4.data,w,h,shifts[b]);
            FRONT_LAUNCH(k_front_scatter,size_t(rows)*32,quant,image.data,w,h,shifts[b]);
            if (!check(block,"image",image.data,size_t(w)*h*32,verify,comparisons) ||
                (b == 4 && !check(block,"raw",raw4.data,size_t(w)*h*32,verify,comparisons))) return false;
        }
        if (!(fused && p4)) FRONT_LAUNCH(k_front_pool,size_t(w/2)*(h/2)*32,raw4.data,down_pool.data,w,h,32);
        if (!verify) {
            // k_front_down + k_front_native64 in one GEMM: output column n of the
            // permuted matrix is peer column mh^-1(n), the same sums.
            tiled::gemm<c32_resident::H70Raw,tiled::FP8,false>(c512_resident::stream,down_pool.data,32,matrix_native.data,32,
                nullptr,0,nullptr,c64.data,nullptr,64,(w/2)*(h/2),64);
        } else {
            tiled::gemm<c32_resident::H70Raw,tiled::FP8,false>(c512_resident::stream,down_pool.data,32,matrix.data,32,nullptr,0,nullptr,
                down.data,nullptr,64,(w/2)*(h/2),64);  // k_front_down
            FRONT_LAUNCH(k_front_native64,down.count,down.data,c64.data,int(down.count));
        }
        if (skip4_native) FRONT_LAUNCH(k_front_native32,skip4.count,image.data,skip4.data,int(skip4.count),0);
        if (!pre) FRONT_LAUNCH(k_front_native32,skip0.count,raw0.data,skip0.data,int(skip0.count),1);
        if (!check(dir,"down_pool",down_pool.data,down_pool.count,verify,comparisons) ||
            !check(dir,"down",down.data,down.count,verify,comparisons) ||
            !check(dir,"c64_input",c64.data,c64.count,verify,comparisons) ||
            !check(dir,"skip4",skip4.data,skip4.count,verify,comparisons) ||
            !check(dir,"skip0",skip0.data,skip0.count,verify,comparisons)) return false;
        ready = true;
        return true;
    }
    DeviceTensor c64_view() const { return ready ? DeviceTensor{c64.data,c64.count} : DeviceTensor{}; }
    DeviceTensor skip4_view() const { return ready && skip4_native ? DeviceTensor{skip4.data,skip4.count} : DeviceTensor{}; }
    // The same values in the peer channel order (block4's image), for a reader
    // that maps the channels itself (k_upsample66_merge skip_peer).
    DeviceTensor skip4_peer_view() const { return ready ? DeviceTensor{image.data.get(),skip4.count} : DeviceTensor{}; }
    DeviceTensor skip0_view() const { return ready && !skip_is_half ? DeviceTensor{skip0.data,skip0.count} : DeviceTensor{}; }
    // skip0 as halves (the fused block0 with half_skip set), else null.
    const _Float16* skip0_half() const { return ready && skip_is_half ? reinterpret_cast<const _Float16*>(skip0h.data.get()) : nullptr; }
};
#undef FRONT_LAUNCH
} // namespace front_resident
