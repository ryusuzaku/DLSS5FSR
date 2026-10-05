#pragma once
// FP8 form of swin_block_wmma.hip (S330): the same C64/C128/C256 Swin block
// kernels, stage for stage, with every product on gfx12 FP8 WMMA (16x16x16,
// f32 accumulate) and the FP8 operands kept as E4M3 bytes (LDS, the window
// feature between the two launches, the split FFN's hidden and middle, and
// the weights). Every operand is a value the reference already quantizes to
// E4M3, so the products are exact and only the f32 order inside each
// 16-product step differs, as on the f16 path. Rounding boundaries,
// epilogues and the denominator order are unchanged.
// Operand layout (fp8_wmma_layout_test.hip): lane l holds A row / B column
// l%16 with K = (l/16)*8 .. +8 as 8 bytes; the accumulator holds column l%16,
// rows (l/16)*8 + e. RESIDENT_SWIN_F16=1 runs swin_block_wmma.hip instead.
// Needs swin_block_wmma.hip included first.

namespace swin_fp8 {
using swin_fused::Geometry;
using swin_fused::Tiles;
using h16 = rocwmma::float16_t;
typedef unsigned char u8;
typedef float acc8 __attribute__((ext_vector_type(8)));
typedef int op8 __attribute__((ext_vector_type(2)));

inline bool enabled = [] { const char* v = getenv("RESIDENT_SWIN_F16"); return !(v && *v == '1'); }();

// split512_fp8 as a byte (clamp to +-448, E4M3 nearest even).
__device__ inline u8 q8(float x) {
    x = fminf(448.0f, fmaxf(-448.0f, x));
    return (u8)(__builtin_amdgcn_cvt_pk_fp8_f32(x, 0.0f, 0, false) & 0xFF);
}
__device__ inline float f8(u8 b) { return __builtin_amdgcn_cvt_f32_fp8((int)b, 0); }

// A 32-deep slice from zero: A rows a[(l%16)*lda ..], B columns b[(l%16)*ldb ..].
__device__ inline acc8 slice(const u8* a, int lda, const u8* b, int ldb) {
    const int l = threadIdx.x % 32, off = (l / 16) * 8;
    const u8* ar = a + (l % 16) * lda + off;
    const u8* br = b + (l % 16) * ldb + off;
    acc8 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(*reinterpret_cast<const op8*>(ar),
                                                             *reinterpret_cast<const op8*>(br), c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(*reinterpret_cast<const op8*>(ar + 16),
                                                             *reinterpret_cast<const op8*>(br + 16), c);
    return c;
}

__global__ void k_to_fp8(const float* w, u8* out, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = q8(w[i]);
}

// Cached E4M3 copy of `count` weights at W (converted once on `stream`).
inline const u8* weights(hipStream_t stream, const float* W, size_t count) {
    static std::map<const float*, std::pair<u8*, size_t>> cache;
    auto found = cache.find(W);
    if (found == cache.end() || found->second.second != count) {
        u8* packed = nullptr;
        HIP_CHECK(hipMalloc(&packed, count));
        hipLaunchKernelGGL(k_to_fp8, dim3((count+255)/256), dim3(256), 0, stream, W, packed, count);
        if (found != cache.end()) (void)hipFree(found->second.first);
        cache[W] = {packed, count};
        return packed;
    }
    return found->second.first;
}

template<int C>
__global__ __launch_bounds__(256) void k_swin_ffn(const float* __restrict__ input, Geometry g,
        const u8* __restrict__ w1, const u8* __restrict__ w2, const u8* __restrict__ w3,
        const float* __restrict__ skip, u8* __restrict__ feature) {
    constexpr int LX = C+8, LH = 136, HID = 4*C, PER = Tiles<C>::PER;
    __shared__ __attribute__((aligned(16))) u8 xs[64*LX];  // gathered input, then FP8 middle
    __shared__ __attribute__((aligned(16))) u8 hs[64*LH];  // one 128-column hidden chunk
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x;
    for (int i = tid; i < 64*C; i += 256) {
        int t = i/C, c = i%C, x, y;
        xs[t*LX+c] = q8(split512_half(g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f));
    }
    __syncthreads();
    float mid[PER][8];
    for (int p = 0; p < PER; ++p) for (int e = 0; e < 8; ++e) mid[p][e] = 0.0f;
    for (int h0 = 0; h0 < HID; h0 += 128) {
        {   // Hidden chunk 64x128 (K=C): 32 tiles, wave -> row tile, 4 column tiles.
            const int rt = wave%4;
            for (int j = 0; j < 4; ++j) {
                const int ct = wave/4*4+j;
                float acc[8] = {};
                for (int k0 = 0; k0 < C; k0 += 32) {
                    acc8 part = slice(xs+rt*16*LX+k0, LX, w1+size_t(h0+ct*16)*C+k0, C);
                    for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part[e]);
                }
                for (int e = 0; e < 8; ++e) {
                    float a = acc[e];
                    float gate = fminf(4.0f, fmaxf(-4.0f, a));
                    float inner = split512_half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                    float poly = split512_half(gate*inner+0.89453125f);
                    hs[(rt*16+lr+e)*LH+ct*16+lc] = q8(split512_half(a*poly));
                }
            }
        }
        __syncthreads();
        for (int p = 0; p < PER; ++p) {  // w2: this chunk's four K slices, in order
            const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
            for (int s = 0; s < 4; ++s) {
                acc8 part = slice(hs+rt*16*LH+s*32, LH, w2+size_t(ct*16)*HID+h0+s*32, HID);
                for (int e = 0; e < 8; ++e) mid[p][e] = split512_half(mid[p][e]+part[e]);
            }
        }
        __syncthreads();
    }
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) xs[(rt*16+lr+e)*LX+ct*16+lc] = q8(mid[p][e]);
    }
    __syncthreads();
    for (int p = 0; p < PER; ++p) {  // w3 with the gathered skip
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        float acc[8];
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = ct*16+lc, x, y;
            float v = g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f;
            acc[e] = split512_half(v*skip[c]);
        }
        for (int k0 = 0; k0 < C; k0 += 32) {
            acc8 part = slice(xs+rt*16*LX+k0, LX, w3+size_t(ct*16)*C+k0, C);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e)
            feature[(size_t(win)*64+rt*16+lr+e)*C+ct*16+lc] = q8(acc[e]);
    }
}

template<int C, bool SPLIT>
__global__ __launch_bounds__(256) void k_swin_attention(const u8* __restrict__ feature, Geometry g,
        const u8* __restrict__ wqkv, const float* __restrict__ scales, const float* __restrict__ bias,
        const u8* __restrict__ wp, const float* __restrict__ attention_skip,
        float* output, float* raw_out, float* partial) {
    constexpr int LD = 40, LS = 72, PER = Tiles<C>::PER, HEADS = C/32;
    // q/k/v as half for the norm; their E4M3 forms for the products. After the
    // quantization qs (half) holds the probabilities and ks the context.
    __shared__ __attribute__((aligned(16))) h16 qs[64*LD], ks[64*LD], vs[64*LD];
    __shared__ __attribute__((aligned(16))) u8 q8s[64*32], k8s[64*32], vt8[32*64];
    __shared__ __attribute__((aligned(16))) h16 sc[64*LS];
    __shared__ float inv[128];
    u8* P = reinterpret_cast<u8*>(qs);    // 64x64 bytes (qs is 5 KB)
    u8* ctx = reinterpret_cast<u8*>(ks);  // 64x32 bytes
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x;
    const u8* src = feature + size_t(win)*64*C;
    float acc[PER][8];
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) {
            int c = ct*16+lc;
            acc[p][e] = split512_half(f8(src[(rt*16+lr+e)*C+c])*attention_skip[c]);
        }
    }
    const int head0 = SPLIT ? blockIdx.y : 0, head1 = SPLIT ? head0 + 1 : HEADS;
    for (int head = head0; head < head1; ++head) {
        for (int j = 0; j < 3; ++j) {  // qkv of this head: 24 tiles, 3 per wave
            const int k = wave*3+j, m = k/8, rt = k%8/2, ct = k%2;
            float q[8] = {};
            for (int k0 = 0; k0 < C; k0 += 32) {
                acc8 part = slice(src+rt*16*C+k0, C, wqkv+size_t(m*C+head*32+ct*16)*C+k0, C);
                for (int e = 0; e < 8; ++e) q[e] = split512_half(q[e]+part[e]);
            }
            h16* dst = m == 0 ? qs : m == 1 ? ks : vs;
            for (int e = 0; e < 8; ++e) dst[(rt*16+lr+e)*LD+ct*16+lc] = split512_to_half(q[e]);
        }
        __syncthreads();
        if (tid < 128) {
            const h16* row = (tid < 64 ? qs : ks) + (tid%64)*LD;
            float v[32];
            for (int c = 0; c < 32; ++c) v[c] = float(row[c]);
            inv[tid] = split512_norm_inverse_fast(v);
        }
        __syncthreads();
        const float scale = split512_half(scales[head]);
        for (int i = tid; i < 3*64*32; i += 256) {
            int m = i/2048, t = i/32%64, c = i%32;
            float value = float((m == 0 ? qs : m == 1 ? ks : vs)[t*LD + c]);
            if (m < 2) {
                value = split512_half(value*inv[m*64+t]);
                if (m == 0) value = split512_half(value*scale);
            }
            const u8 b = q8(value);
            if (m == 0) q8s[t*32+c] = b;
            else if (m == 1) k8s[t*32+c] = b;
            else vt8[c*64+t] = b;
        }
        __syncthreads();
        {   // Scores + bias + exponent transform: 16 tiles, 2 per wave.
            const int rt = wave%4;
            for (int j = 0; j < 2; ++j) {
                const int ct = wave/4*2+j;
                acc8 s = slice(q8s+rt*16*32, 32, k8s+ct*16*32, 32);
                for (int e = 0; e < 8; ++e) {
                    int q = rt*16+lr+e, k = ct*16+lc;
                    float score = split512_half(s[e]+bias[(head*64+q)*64+k]);
                    float b = score*.044921875f+1.30078125f;
                    float clipped = fminf(1.5693359375f, fmaxf(1.03125f, split512_half(b)));
                    _Float16 hc = split512_to_half(clipped);
                    unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
                    bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
                    __builtin_memcpy(&hc, &bits, 2);
                    sc[q*LS+k] = hc;
                }
            }
        }
        __syncthreads();
        if (tid < 64) {  // split512_den order on the half row
            const h16* e = sc + tid*LS;
            float parity[2]; const int lanes[4] = {0,2,8,10}; const int offs[3] = {4,32,36};
            for (int odd = 0; odd < 2; ++odd) {
                float partial4[4];
                for (int l = 0; l < 4; ++l) {
                    int b = odd+lanes[l];
                    float value = split512_half(float(e[b])+float(e[b+16]));
                    for (int t = 0; t < 3; ++t)
                        value = split512_half(value+split512_half(float(e[b+offs[t]])+float(e[b+offs[t]+16])));
                    partial4[l] = value;
                }
                float total = partial4[0];
                for (int l = 1; l < 4; ++l) total = split512_half(total+partial4[l]);
                parity[odd] = total;
            }
            inv[tid] = split512_half(1.0f/split512_half(parity[0]+parity[1]));
        }
        __syncthreads();
        for (int i = tid; i < 64*64; i += 256) {
            int q = i/64, k = i%64;
            P[q*64+k] = q8(split512_half(float(sc[q*LS+k])*inv[q]));
        }
        __syncthreads();
        {   // Context (two 32-key slices) -> FP8: 8 tiles, one per wave.
            const int rt = wave/2, ct = wave%2;
            float c8[8] = {};
            for (int k0 = 0; k0 < 64; k0 += 32) {
                acc8 part = slice(P+rt*16*64+k0, 64, vt8+(ct*16)*64+k0, 64);
                for (int e = 0; e < 8; ++e) c8[e] = split512_half(c8[e]+part[e]);
            }
            for (int e = 0; e < 8; ++e) ctx[(rt*16+lr+e)*32+ct*16+lc] = q8(c8[e]);
        }
        __syncthreads();
        for (int p = 0; p < PER; ++p) {  // projection K slice `head`
            const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
            acc8 part = slice(ctx+rt*16*32, 32, wp+size_t(ct*16)*C+head*32, C);
            if constexpr (SPLIT) {
                float* dst = partial + ((size_t(win)*HEADS + head)*64)*C;
                for (int e = 0; e < 8; ++e) dst[size_t(rt*16+lr+e)*C + ct*16+lc] = part[e];
            } else {
                for (int e = 0; e < 8; ++e) acc[p][e] = split512_half(acc[p][e]+part[e]);
            }
        }
        __syncthreads();
    }
    if constexpr (SPLIT) return;
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = ct*16+lc, x, y;
            if (!g.pixel(win, t, x, y)) continue;
            size_t at = (size_t(y)*g.width+x)*C+c;
            output[at] = split512_fp8(acc[p][e]);
            if (raw_out) raw_out[at] = acc[p][e];
        }
    }
}

// The few-window FFN in three stages (see swin_block_wmma.hip).
template<int C>
__global__ __launch_bounds__(256) void k_swin_hidden(const float* __restrict__ input, Geometry g,
        const u8* __restrict__ w1, u8* __restrict__ hidden) {
    constexpr int LX = C+8, HID = 4*C;
    __shared__ __attribute__((aligned(16))) u8 xs[64*LX];
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, h0 = blockIdx.y*128;
    for (int i = tid; i < 64*C; i += 256) {
        int t = i/C, c = i%C, x, y;
        xs[t*LX+c] = q8(split512_half(g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f));
    }
    __syncthreads();
    const int rt = wave%4;
    for (int j = 0; j < 4; ++j) {
        const int ct = wave/4*4+j;
        float acc[8] = {};
        for (int k0 = 0; k0 < C; k0 += 32) {
            acc8 part = slice(xs+rt*16*LX+k0, LX, w1+size_t(h0+ct*16)*C+k0, C);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e) {
            float a = acc[e];
            float gate = fminf(4.0f, fmaxf(-4.0f, a));
            float inner = split512_half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
            float poly = split512_half(gate*inner+0.89453125f);
            hidden[(size_t(win)*64+rt*16+lr+e)*HID + h0+ct*16+lc] = q8(split512_half(a*poly));
        }
    }
}

template<int C>
__global__ __launch_bounds__(256) void k_swin_w2(const u8* __restrict__ hidden, const u8* __restrict__ w2,
                                                 u8* __restrict__ middle) {
    constexpr int HID = 4*C;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, o0 = blockIdx.y*64;
    const int rt = wave%4;
    const u8* a = hidden + size_t(win)*64*HID;
    for (int j = 0; j < 2; ++j) {
        const int ct = wave/4*2+j;
        float acc[8] = {};
        for (int k0 = 0; k0 < HID; k0 += 32) {
            acc8 part = slice(a+rt*16*HID+k0, HID, w2+size_t(o0+ct*16)*HID+k0, HID);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e)
            middle[(size_t(win)*64+rt*16+lr+e)*C + o0+ct*16+lc] = q8(acc[e]);
    }
}

template<int C>
__global__ __launch_bounds__(256) void k_swin_w3(const float* __restrict__ input, Geometry g,
        const u8* __restrict__ middle, const u8* __restrict__ w3, const float* __restrict__ skip,
        u8* __restrict__ feature) {
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, o0 = blockIdx.y*64;
    const int rt = wave%4;
    const u8* a = middle + size_t(win)*64*C;
    for (int j = 0; j < 2; ++j) {
        const int ct = wave/4*2+j;
        float acc[8];
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = o0+ct*16+lc, x, y;
            float v = g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f;
            acc[e] = split512_half(v*skip[c]);
        }
        for (int k0 = 0; k0 < C; k0 += 32) {
            acc8 part = slice(a+rt*16*C+k0, C, w3+size_t(o0+ct*16)*C+k0, C);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e)
            feature[(size_t(win)*64+rt*16+lr+e)*C + o0+ct*16+lc] = q8(acc[e]);
    }
}

template<int C>
__global__ void k_swin_combine(const u8* __restrict__ feature, const float* __restrict__ partial,
                               const float* __restrict__ attention_skip, Geometry g, int windows,
                               float* output, float* raw_out) {
    constexpr int HEADS = C/32;
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i >= size_t(windows)*64*C) return;
    int c = int(i % C), t = int(i / C % 64), win = int(i / C / 64), x, y;
    if (!g.pixel(win, t, x, y)) return;
    float acc = split512_half(f8(feature[i])*attention_skip[c]);
    for (int h = 0; h < HEADS; ++h)
        acc = split512_half(acc + partial[((size_t(win)*HEADS + h)*64 + t)*C + c]);
    size_t at = (size_t(y)*g.width + x)*C + c;
    output[at] = split512_fp8(acc);
    if (raw_out) raw_out[at] = acc;
}

inline u8* ffn_scratch(size_t bytes) {
    static u8* data = nullptr;
    static size_t capacity = 0;
    if (capacity < bytes) {
        if (data) HIP_CHECK(hipFree(data));
        data = nullptr; capacity = 0;
        HIP_CHECK(hipMalloc(&data, bytes));
        capacity = bytes;
    }
    return data;
}

// Same contract as swin_fused::run (scratch holds 64*C halves per window; the
// E4M3 feature uses half of it).
template<int C>
inline bool run(const float* input, float* output, float* raw_out, int width, int height, int shift,
                const swin_fused::Block& w, void* scratch) {
    if (c512_resident::exact_math) return false;
    if (!enabled) return swin_fused::run<C>(input, output, raw_out, width, height, shift, w, scratch);
    hipStream_t s = c512_resident::stream;
    Geometry g{width, height, (shift&1) ? 4 : 0, (shift&2) ? 4 : 0, 0};
    g.pw = ((width+g.px+7)/8)*8;
    int ph = ((height+g.py+7)/8)*8, windows = (g.pw/8)*(ph/8);
    auto f8w = [&](const float* W, size_t n) { return weights(s, W, n); };
    auto* feature = reinterpret_cast<u8*>(scratch);
    if (windows < 128 && C >= 64) {
        u8* hidden = ffn_scratch(size_t(windows)*64*(4*C + C));
        u8* middle = hidden + size_t(windows)*64*4*C;
        hipLaunchKernelGGL(k_swin_hidden<C>, dim3(windows, 4*C/128), dim3(256), 0, s, input, g,
                           f8w(w.w1, size_t(4*C)*C), hidden);
        hipLaunchKernelGGL(k_swin_w2<C>, dim3(windows, C/64), dim3(256), 0, s, hidden,
                           f8w(w.w2, size_t(C)*4*C), middle);
        hipLaunchKernelGGL(k_swin_w3<C>, dim3(windows, C/64), dim3(256), 0, s, input, g, middle,
                           f8w(w.w3, size_t(C)*C), w.skip, feature);
    } else {
        hipLaunchKernelGGL(k_swin_ffn<C>, dim3(windows), dim3(256), 0, s, input, g,
                           f8w(w.w1, size_t(4*C)*C), f8w(w.w2, size_t(C)*4*C), f8w(w.w3, size_t(C)*C), w.skip, feature);
    }
    HIP_CHECK(hipGetLastError());
    if (c512_resident::launch_hook) c512_resident::launch_hook(C == 64 ? "k_swin_ffn8<64>" : C == 128 ? "k_swin_ffn8<128>" : "k_swin_ffn8<256>");
    const bool split = windows < 128 && C > 32;
    float* partial = split ? swin_fused::head_partials(size_t(windows)*64*C*(C/32)) : nullptr;
    if (split)
        hipLaunchKernelGGL((k_swin_attention<C, true>), dim3(windows, C/32), dim3(256), 0, s, feature, g,
                           f8w(w.qkv, size_t(3*C)*C), w.scales, w.bias, f8w(w.projection, size_t(C)*C),
                           w.attention_skip, output, raw_out, partial);
    else
        hipLaunchKernelGGL((k_swin_attention<C, false>), dim3(windows), dim3(256), 0, s, feature, g,
                           f8w(w.qkv, size_t(3*C)*C), w.scales, w.bias, f8w(w.projection, size_t(C)*C),
                           w.attention_skip, output, raw_out, nullptr);
    HIP_CHECK(hipGetLastError());
    if (split) {
        size_t n = size_t(windows)*64*C;
        hipLaunchKernelGGL(k_swin_combine<C>, dim3((n+255)/256), dim3(256), 0, s, feature, partial,
                           w.attention_skip, g, windows, output, raw_out);
        HIP_CHECK(hipGetLastError());
    }
    if (c512_resident::launch_hook) c512_resident::launch_hook(C == 64 ? "k_swin_attention8<64>" : C == 128 ? "k_swin_attention8<128>" : "k_swin_attention8<256>");
    return true;
}
} // namespace swin_fp8
