#pragma once
// Fused C32 body: one workgroup runs one 8x8 window (64 tokens) through the
// FFN and attention entirely in LDS and writes only the body output.
// Every stage keeps the per-output operation order of the h70 kernels /
// tiled forms it replaces (k_h70_expand+hidden, ffn, qkv, qknorm, scores, exp,
// den, prob, context, projection). Intermediates the reference rounds to half
// or FP8, and the FP8 weights, are stored as _Float16, which is lossless for
// those values. Each stage is a register-tiled product: a thread owns RM x CN
// outputs and walks K in index order, so loads are shared, sums are not.
// Needs head70.hip / head70_normalized.hip / split512_block.hip included first.

#ifdef C32F_PROF
__device__ unsigned long long c32f_prof[16];
#define C32F_MARK(i) do { if (threadIdx.x == 0) marks[i] = wall_clock64(); } while (0)
#else
#define C32F_MARK(i) do {} while (0)
#endif

__device__ inline float c32f_h(float x) { return split512_half(x); }  // == h70_h (exhaustively)

// out[r][c] for r < M, c < N (M/RM * N/CN == 256 threads). K walks slices of
// `slice`; combine(acc, part, first) folds each slice's sequential sum.
template<int RM, int CN, class A, class B, class Init, class Combine, class Epi>
__device__ inline void c32f_gemm(int N, int K, int slice, A a, B b, Init init, Combine combine, Epi epi) {
    const int groups = N / CN, rg = threadIdx.x / groups, cg = threadIdx.x % groups;
    float acc[RM][CN];
    for (int i = 0; i < RM; ++i)
        for (int j = 0; j < CN; ++j) acc[i][j] = init(rg*RM+i, cg*CN+j);
    for (int k0 = 0; k0 < K; k0 += slice) {
        float part[RM][CN];
        for (int i = 0; i < RM; ++i)
            for (int j = 0; j < CN; ++j) part[i][j] = 0.0f;
        for (int k = k0; k < k0+slice; ++k) {
            float xv[RM], wv[CN];
            for (int i = 0; i < RM; ++i) xv[i] = a(rg*RM+i, k);
            for (int j = 0; j < CN; ++j) wv[j] = b(k, cg*CN+j);
            for (int i = 0; i < RM; ++i)
                for (int j = 0; j < CN; ++j) part[i][j] += xv[i]*wv[j];
        }
        for (int i = 0; i < RM; ++i)
            for (int j = 0; j < CN; ++j) acc[i][j] = combine(acc[i][j], part[i][j], k0 == 0);
    }
    for (int i = 0; i < RM; ++i)
        for (int j = 0; j < CN; ++j) epi(rg*RM+i, cg*CN+j, acc[i][j]);
}

// Stage a [rows][cols] float weight block transposed ([cols][rows]) as half.
__device__ inline void c32f_stage(_Float16* dst, const float* src, int rows, int cols) {
    for (int i = threadIdx.x; i < rows*cols; i += 256) {
        int r = i / cols, c = i % cols;
        dst[c*rows+r] = split512_to_half(src[i]);
    }
}

// dw: packed weights (16449 floats): w1 @0 (128x32), w2 @4096 (32x128),
// qkv @8192 (96x32), projection @11264 (32x32), bias @12288 (64x64),
// ffn gate @16385 (32), attention gate @16417 (32).
__global__ __launch_bounds__(256) void k_c32_fused(const float* __restrict__ in, const float* __restrict__ dw,
                                                   float scale, float* raw_out, float* quant_out) {
    __shared__ float xf[64*32];          // F(input), F(ffn), F(context)
    __shared__ _Float16 hid[64*128];     // FP8 hidden
    __shared__ _Float16 fs[64*32];       // ffn output (half)
    __shared__ _Float16 qkv[64*96];      // raw qkv (half), then normalized (FP8)
    __shared__ _Float16 sc[64*64];       // scores -> exponents -> probabilities
    __shared__ _Float16 wt[4096];        // current stage's weights, transposed
    __shared__ float inv[128];
    __shared__ float den[64];
    const int tid = threadIdx.x;
#ifdef C32F_PROF
    __shared__ unsigned long long marks[16];
#endif
    C32F_MARK(0);
    const size_t base = size_t(blockIdx.x)*64*32;
    const float *g1 = dw+16385, *g2 = dw+16417, *bias = dw+12288;
    auto zero = [](int, int) { return 0.0f; };

    for (int i = tid; i < 64*32; i += 256) xf[i] = h70_f(in[base+i]);
    c32f_stage(wt, dw, 128, 32);                              // w1^T [32][128]
    __syncthreads();
    C32F_MARK(1);
    // Expand (K=32) and gate -> FP8 hidden.
    c32f_gemm<4,8>(128, 32, 32,
        [&](int r, int k) { return xf[r*32+k]; }, [&](int k, int c) { return float(wt[k*128+c]); },
        zero, [](float acc, float part, bool) { return c32f_h(acc+part); },
        [&](int r, int c, float a) {
            float gate = fminf(4.0f, fmaxf(-4.0f, a));
            float inner = c32f_h(fabsf(gate)*(-0.055908203125f)+0.447265625f);
            float poly = c32f_h(gate*inner+0.89453125f);
            hid[r*128+c] = split512_to_half(h70_f(c32f_h(a*poly)));
        });
    __syncthreads();
    C32F_MARK(2);
    c32f_stage(wt, dw+4096, 32, 128);                         // w2^T [128][32]
    __syncthreads();
    C32F_MARK(3);
    // FFN contract with residual (K=128, four 32-product slices).
    c32f_gemm<2,4>(32, 128, 32,
        [&](int r, int k) { return float(hid[r*128+k]); }, [&](int k, int c) { return float(wt[k*32+c]); },
        [&](int r, int c) { return c32f_h(in[base+r*32+c]*g1[c]); },
        [](float acc, float part, bool) { return c32f_h(acc+part); },
        [&](int r, int c, float a) { fs[r*32+c] = split512_to_half(a); });
    __syncthreads();
    C32F_MARK(4);
    for (int i = tid; i < 64*32; i += 256) xf[i] = h70_f(float(fs[i]));
    c32f_stage(wt, dw+8192, 96, 32);                          // wq^T [32][96]
    __syncthreads();
    C32F_MARK(5);
    // QKV (K=32).
    c32f_gemm<4,6>(96, 32, 32,
        [&](int r, int k) { return xf[r*32+k]; }, [&](int k, int c) { return float(wt[k*96+c]); },
        zero, [](float acc, float part, bool) { return c32f_h(acc+part); },
        [&](int r, int c, float a) { qkv[r*96+c] = split512_to_half(a); });
    __syncthreads();
    C32F_MARK(6);
    // Per-head inverse norms for Q and K, then in-place normalization.
    if (tid < 128) {
        int r = tid/2, m = tid%2;
        float v[32];
        for (int c = 0; c < 32; ++c) v[c] = float(qkv[r*96+m*32+c]);
        inv[tid] = h70_normalize_inverse_fast(v);
    }
    __syncthreads();
    C32F_MARK(7);
    const float head_scale = c32f_h(scale);
    for (int i = tid; i < 64*96; i += 256) {
        int r = i/96, m = i%96/32;
        float q = float(qkv[i]), value;
        if (m == 2) value = h70_f(q);
        else {
            value = c32f_h(q*inv[r*2+m]);
            if (m == 0) value = c32f_h(value*head_scale);
            value = h70_f(value);
        }
        qkv[i] = split512_to_half(value);
    }
    __syncthreads();
    C32F_MARK(8);
    // Scores (K=32, raw sum) + bias, exponent transform.
    c32f_gemm<4,4>(64, 32, 32,
        [&](int q, int c) { return float(qkv[q*96+c]); }, [&](int c, int k) { return float(qkv[k*96+32+c]); },
        zero, [](float, float part, bool) { return part; },
        [&](int q, int k, float sum) {
            float score = c32f_h(sum+bias[q*64+k]);
            float a = score*.044921875f;
            float b = a+1.30078125f;
            float clipped = fminf(1.5693359375f, fmaxf(1.03125f, c32f_h(b)));
            _Float16 hc = split512_to_half(clipped);
            unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
            bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
            __builtin_memcpy(&hc, &bits, 2);
            sc[q*64+k] = hc;
        });
    __syncthreads();
    C32F_MARK(9);
    // Softmax denominators (the reference's fixed 64-key reduction order).
    if (tid < 64) {
        const _Float16* e = sc + tid*64;
        float parity[2]; const int lanes[4] = {0,2,8,10}; const int offs[3] = {4,32,36};
        for (int odd = 0; odd < 2; ++odd) {
            float partial[4];
            for (int l = 0; l < 4; ++l) {
                int b = odd+lanes[l];
                float value = c32f_h(float(e[b])+float(e[b+16]));
                for (int t = 0; t < 3; ++t)
                    value = c32f_h(value+c32f_h(float(e[b+offs[t]])+float(e[b+offs[t]+16])));
                partial[l] = value;
            }
            float total = partial[0];
            for (int j = 1; j < 4; ++j) total = c32f_h(total+partial[j]);
            parity[odd] = total;
        }
        den[tid] = c32f_h(parity[0]+parity[1]);
    }
    __syncthreads();
    C32F_MARK(10);
    for (int i = tid; i < 64*64; i += 256) {
        float invd = 1.f/den[i/64];
        float p = float(sc[i])*c32f_h(invd);
        sc[i] = split512_to_half(h70_f(c32f_h(p)));
    }
    c32f_stage(wt, dw+11264, 32, 32);                         // wp^T [32][32]
    __syncthreads();
    C32F_MARK(11);
    // Context: two 32-key slices -> F(context) for the projection.
    c32f_gemm<2,4>(32, 64, 32,
        [&](int q, int k) { return float(sc[q*64+k]); }, [&](int k, int c) { return float(qkv[k*96+64+c]); },
        zero, [](float acc, float part, bool) { return c32f_h(acc+part); },
        [&](int q, int c, float a) { xf[q*32+c] = h70_f(a); });
    __syncthreads();
    C32F_MARK(12);
    // Projection: double-precision residual add, single half rounding.
    c32f_gemm<2,4>(32, 32, 32,
        [&](int r, int k) { return xf[r*32+k]; }, [&](int k, int c) { return float(wt[k*32+c]); },
        zero, [](float, float part, bool) { return part; },
        [&](int r, int c, float sum) {
            float residual = c32f_h(float(fs[r*32+c])*g2[c]);
            float body = h70_hd(double(sum)+double(residual));
            raw_out[base+r*32+c] = body;
            if (quant_out) quant_out[base+r*32+c] = h70_f(body);
        });
    __syncthreads();
#ifdef C32F_PROF
    if (tid == 0) {
        marks[15] = wall_clock64();
        unsigned long long previous = marks[0];
        for (int i = 1; i <= 12; ++i) { atomicAdd(&c32f_prof[i], marks[i]-previous); previous = marks[i]; }
        atomicAdd(&c32f_prof[15], marks[15]-previous);
    }
#endif
}

// Fast (non-exact) form of k_c32_fused: every product stage runs on f16 WMMA
// (16x16x16, fp32 accumulate). All operands are FP8 or half values, so the
// products are exact; only the fp32 summation order inside each 32-product
// slice differs from the sequential reference (rare one-ulp half ties). The
// rounding boundaries, epilogues and softmax reduction order are unchanged.
// hw: the first 12288 weights (w1, w2, qkv, projection) as half, [out][k].
// Accumulator layout (gfx12, verified): lane l, element e -> row (l/16)*8+e, col l%16.
#include <rocwmma/rocwmma.hpp>
namespace c32w {
using h16 = rocwmma::float16_t;
using FA = rocwmma::fragment<rocwmma::matrix_a,16,16,16,h16,rocwmma::row_major>;
using FBc = rocwmma::fragment<rocwmma::matrix_b,16,16,16,h16,rocwmma::col_major>;
using FBr = rocwmma::fragment<rocwmma::matrix_b,16,16,16,h16,rocwmma::row_major>;
using FC = rocwmma::fragment<rocwmma::accumulator,16,16,16,float>;
// One 32-wide K slice of a 16x16 tile: A rows at a (ld lda), B halves bstep apart.
template<class FB>
__device__ inline void slice(FC& part, const h16* a, int lda, const h16* b, int ldb, int bstep) {
    FA a0, a1; FB b0, b1;
    rocwmma::load_matrix_sync(a0, a, lda);
    rocwmma::load_matrix_sync(a1, a+16, lda);
    rocwmma::load_matrix_sync(b0, b, ldb);
    rocwmma::load_matrix_sync(b1, b+bstep, ldb);
    rocwmma::fill_fragment(part, 0.0f);
    rocwmma::mma_sync(part, a0, b0, part);
    rocwmma::mma_sync(part, a1, b1, part);
}
} // namespace c32w

__global__ void k_c32_half_weights(const float* dw, _Float16* hw, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) hw[i] = split512_to_half(dw[i]);
}

// Where a window's tokens live: token-major [window][64][32] (hwc = 0), or
// scattered in an HWC image through a shifted 8x8 partition (hwc = 1), with
// the peer channel order when permute is set (either layout). Invalid (padding) tokens read
// as zero and are not written; windows own disjoint pixels, so input and
// output may be the same image.
struct C32Io {
    int in_hwc = 0, out_hwc = 0, permute = 0;
    int width = 0, height = 0, px = 0, py = 0, pw = 0;
    __device__ long long at(bool hwc, int win, int t, int c) const {
        if (!hwc) return (long long)(size_t(win)*64 + t)*32 + (permute ? peer_to_native32(c) : c);
        int x = (win % (pw/8))*8 + t%8 - px, y = (win / (pw/8))*8 + t/8 - py;
        if (x < 0 || x >= width || y < 0 || y >= height) return -1;
        return (long long)(size_t(y)*width + x)*32 + (permute ? peer_to_native32(c) : c);
    }
    // A token's first element (channel 0 before the permutation), or -1.
    __device__ int base(bool hwc, int win, int t) const {
        if (!hwc) return (win*64 + t)*32;
        int x = (win % (pw/8))*8 + t%8 - px, y = (win / (pw/8))*8 + t/8 - py;
        if (x < 0 || x >= width || y < 0 || y >= height) return -1;
        return (y*width + x)*32;
    }
    __device__ int channel(int c) const { return permute ? peer_to_native32(c) : c; }
};

__global__ __launch_bounds__(256) void k_c32_wmma(const float* in, const float* __restrict__ dw,
                                                  const _Float16* __restrict__ hw, float scale,
                                                  float* raw_out, float* quant_out, C32Io io) {
    using namespace c32w;
    // H: FP8 hidden; after the FFN F(ffn) at [0,2048) then F(context) there,
    // probabilities at [4096,8192). Q: F(input) at [0,2048), then qkv.
    __shared__ _Float16 H[8192];
    __shared__ _Float16 Q[64*96];
    __shared__ float inv[128];
    __shared__ float den[64];
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    const int win = blockIdx.x;
    // Token addresses once per window (the mapping costs integer divisions).
    __shared__ int in_base[64], out_base[64];
    if (tid < 64) { in_base[tid] = io.base(io.in_hwc, win, tid); out_base[tid] = io.base(io.out_hwc, win, tid); }
    __syncthreads();
    auto input = [&](int t, int c) { int b = in_base[t]; return b < 0 ? 0.0f : in[b + io.channel(c)]; };
    const float *g1 = dw+16385, *g2 = dw+16417, *bias = dw+12288;
    _Float16* sc = H+4096;
#ifdef C32F_PROF
    __shared__ unsigned long long marks[16];
    if (tid < 16) marks[tid] = 0;
    __syncthreads();
#endif
    C32F_MARK(0);

    {   // 8 channels of one token per thread (two float4 loads in channel order).
        const int t = tid/4, c0 = tid%4*8, b = in_base[t];
        float v[8];
        if (b < 0) {
            for (int j = 0; j < 8; ++j) v[j] = 0.0f;
        } else if (!io.permute) {
            float4 lo = *reinterpret_cast<const float4*>(in + b + c0), hi = *reinterpret_cast<const float4*>(in + b + c0 + 4);
            v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w; v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
        } else {
            for (int j = 0; j < 8; ++j) v[j] = in[b + peer_to_native32(c0+j)];
        }
        for (int j = 0; j < 8; ++j) Q[t*32 + c0+j] = split512_to_half(h70_f(v[j]));
    }
    __syncthreads();
    C32F_MARK(1);
    {   // Expand (K=32) and gate: tiles 4x8, wave -> row tile, 4 column tiles.
        const int rt = wave%4;
        for (int j = 0; j < 4; ++j) {
            const int ct = wave/4*4+j;
            FC c; slice<FBc>(c, Q+rt*16*32, 32, hw+ct*16*32, 32, 16);
            for (int e = 0; e < 8; ++e) {
                float a = c32f_h(c.x[e]);
                float gate = fminf(4.0f, fmaxf(-4.0f, a));
                float inner = c32f_h(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                float poly = c32f_h(gate*inner+0.89453125f);
                H[(rt*16+lr+e)*128+ct*16+lc] = split512_to_half(h70_f(c32f_h(a*poly)));
            }
        }
    }
    __syncthreads();
    C32F_MARK(2);
    // FFN output (half), kept in registers: the projection tile of this wave
    // (rt = wave/2, ct = wave%2) covers the same outputs in the same lanes.
    float ffn[8];
    {   // FFN contract (K=128, four slices) with residual: tiles 4x2.
        const int rt = wave/2, ct = wave%2;
        float* acc = ffn;
        for (int e = 0; e < 8; ++e) {
            int r = rt*16+lr+e, c = ct*16+lc;
            acc[e] = c32f_h(input(r, c)*g1[c]);
        }
        for (int k0 = 0; k0 < 128; k0 += 32) {
            FC part; slice<FBc>(part, H+rt*16*128+k0, 128, hw+4096+ct*16*128+k0, 128, 16);
            for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part.x[e]);
        }
        __syncthreads();  // other waves read H (hidden) until here
        for (int e = 0; e < 8; ++e) {
            int at = (rt*16+lr+e)*32+ct*16+lc;
            H[at] = split512_to_half(h70_f(acc[e]));
        }
    }
    __syncthreads();
    C32F_MARK(4);
    {   // QKV (K=32) with the Q/K norm in registers. Wave w owns rows 16*(w%4)
        // of Q (w < 4) or K (w >= 4) as two 16-column tiles, so lane l holds
        // v[l] and v[l+16] of each of its 8 rows, plus one V tile.
        const int rt = wave%4, m = wave/4;
        FC c0, c1, cv;
        slice<FBc>(c0, H+rt*16*32, 32, hw+8192+(m*32)*32, 32, 16);
        slice<FBc>(c1, H+rt*16*32, 32, hw+8192+(m*32+16)*32, 32, 16);
        slice<FBc>(cv, H+rt*16*32, 32, hw+8192+(64+m*16)*32, 32, 16);
        const float head_scale = c32f_h(scale);
        for (int e = 0; e < 8; ++e) {
            const int r = rt*16+lr+e;
            float v0 = c32f_h(c0.x[e]), v1 = c32f_h(c1.x[e]);
            // h70_normalize_inverse_fast order: parts[j] = hsum(v[j]^2, h(v[j+16]^2)),
            // then pairs (2j, 2j+1) and widths 4, 2, 1 -> lane butterflies 1, 8, 4, 2.
            float s = split512_half_sum(v0*v0, c32f_h(v1*v1));
            s = c32f_h(s + __shfl_xor(s, 1));
            s = c32f_h(s + __shfl_xor(s, 8));
            s = c32f_h(s + __shfl_xor(s, 4));
            s = c32f_h(s + __shfl_xor(s, 2));
            const float inverse = c32f_h(1.f/sqrtf(fmaxf(s, .00006198883056640625f)));
            float q0 = c32f_h(v0*inverse), q1 = c32f_h(v1*inverse);
            if (m == 0) { q0 = c32f_h(q0*head_scale); q1 = c32f_h(q1*head_scale); }
            Q[r*96+m*32+lc] = split512_to_half(h70_f(q0));
            Q[r*96+m*32+16+lc] = split512_to_half(h70_f(q1));
            Q[r*96+64+m*16+lc] = split512_to_half(h70_f(c32f_h(cv.x[e])));
        }
    }
    __syncthreads();
    C32F_MARK(8);
    {   // Scores (K=32, raw sum) + bias, exponent transform: tiles 4x4.
        const int rt = wave%4;
        for (int j = 0; j < 2; ++j) {
            const int ct = wave/4*2+j;
            FC c; slice<FBc>(c, Q+rt*16*96, 96, Q+32+ct*16*96, 96, 16);
            for (int e = 0; e < 8; ++e) {
                int q = rt*16+lr+e, k = ct*16+lc;
                float score = c32f_h(c.x[e]+bias[q*64+k]);
                float b = score*.044921875f+1.30078125f;
                float clipped = fminf(1.5693359375f, fmaxf(1.03125f, c32f_h(b)));
                _Float16 hc = split512_to_half(clipped);
                unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
                bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
                __builtin_memcpy(&hc, &bits, 2);
                sc[q*64+k] = hc;
            }
        }
    }
    __syncthreads();
    C32F_MARK(9);
    if (tid < 64) {
        const _Float16* e = sc + tid*64;
        float parity[2]; const int lanes[4] = {0,2,8,10}; const int offs[3] = {4,32,36};
        for (int odd = 0; odd < 2; ++odd) {
            float partial[4];
            for (int l = 0; l < 4; ++l) {
                int b = odd+lanes[l];
                float value = c32f_h(float(e[b])+float(e[b+16]));
                for (int t = 0; t < 3; ++t)
                    value = c32f_h(value+c32f_h(float(e[b+offs[t]])+float(e[b+offs[t]+16])));
                partial[l] = value;
            }
            float total = partial[0];
            for (int j = 1; j < 4; ++j) total = c32f_h(total+partial[j]);
            parity[odd] = total;
        }
        den[tid] = c32f_h(1.f/c32f_h(parity[0]+parity[1]));  // half inverse, once per row
    }
    __syncthreads();
    C32F_MARK(10);
    for (int i = tid; i < 64*64; i += 256) {
        float p = float(sc[i])*den[i/64];
        sc[i] = split512_to_half(h70_f(c32f_h(p)));
    }
    __syncthreads();
    C32F_MARK(11);
    {   // Context (two 32-key slices) -> F(context) in H[0,2048): tiles 4x2.
        const int rt = wave/2, ct = wave%2;
        float acc[8] = {};
        for (int k0 = 0; k0 < 64; k0 += 32) {
            FC part; slice<FBr>(part, sc+rt*16*64+k0, 64, Q+64+k0*96+ct*16, 96, 16*96);
            for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e) H[(rt*16+lr+e)*32+ct*16+lc] = split512_to_half(h70_f(acc[e]));
    }
    __syncthreads();
    C32F_MARK(12);
    {   // Projection: double-precision residual add, single half rounding.
        const int rt = wave/2, ct = wave%2;
        FC c; slice<FBc>(c, H+rt*16*32, 32, hw+11264+ct*16*32, 32, 16);
        for (int e = 0; e < 8; ++e) {
            int r = rt*16+lr+e, col = ct*16+lc;
            float residual = c32f_h(ffn[e]*g2[col]);
            float body = split512_half_sum(c.x[e], residual);
            if (out_base[r] < 0) continue;
            int a = out_base[r] + io.channel(col);
            if (raw_out) raw_out[a] = body;
            if (quant_out) quant_out[a] = h70_f(body);
        }
    }
#ifdef C32F_PROF
    __syncthreads();
    if (tid == 0) {
        marks[15] = wall_clock64();
        unsigned long long previous = marks[0];
        for (int i = 1; i <= 12; ++i) if (marks[i]) { atomicAdd(&c32f_prof[i], marks[i]-previous); previous = marks[i]; }
        atomicAdd(&c32f_prof[15], marks[15]-previous);
    }
#endif
}
