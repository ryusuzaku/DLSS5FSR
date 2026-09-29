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

__global__ __launch_bounds__(256) void k_c32_wmma(const float* __restrict__ in, const float* __restrict__ dw,
                                                  const _Float16* __restrict__ hw, float scale,
                                                  float* raw_out, float* quant_out) {
    using namespace c32w;
    // H: FP8 hidden; after the FFN F(ffn) at [0,2048) then F(context) there,
    // probabilities at [4096,8192). Q: F(input) at [0,2048), then qkv.
    __shared__ _Float16 H[8192];
    __shared__ _Float16 Q[64*96];
    __shared__ _Float16 fs[64*32];
    __shared__ float inv[128];
    __shared__ float den[64];
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    const size_t base = size_t(blockIdx.x)*64*32;
    const float *g1 = dw+16385, *g2 = dw+16417, *bias = dw+12288;
    _Float16* sc = H+4096;
#ifdef C32F_PROF
    __shared__ unsigned long long marks[16];
    if (tid < 16) marks[tid] = 0;
    __syncthreads();
#endif
    C32F_MARK(0);

    for (int i = tid; i < 64*32; i += 256) Q[i] = split512_to_half(h70_f(in[base+i]));
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
    {   // FFN contract (K=128, four slices) with residual: tiles 4x2.
        const int rt = wave/2, ct = wave%2;
        float acc[8];
        for (int e = 0; e < 8; ++e) {
            int r = rt*16+lr+e, c = ct*16+lc;
            acc[e] = c32f_h(in[base+r*32+c]*g1[c]);
        }
        for (int k0 = 0; k0 < 128; k0 += 32) {
            FC part; slice<FBc>(part, H+rt*16*128+k0, 128, hw+4096+ct*16*128+k0, 128, 16);
            for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part.x[e]);
        }
        __syncthreads();  // other waves read H (hidden) until here
        for (int e = 0; e < 8; ++e) {
            int at = (rt*16+lr+e)*32+ct*16+lc;
            fs[at] = split512_to_half(acc[e]);
            H[at] = split512_to_half(h70_f(acc[e]));
        }
    }
    __syncthreads();
    C32F_MARK(4);
    {   // QKV (K=32): tiles 4x6, wave -> row tile, 3 column tiles.
        const int rt = wave%4;
        for (int j = 0; j < 3; ++j) {
            const int ct = wave/4*3+j;
            FC c; slice<FBc>(c, H+rt*16*32, 32, hw+8192+ct*16*32, 32, 16);
            for (int e = 0; e < 8; ++e) Q[(rt*16+lr+e)*96+ct*16+lc] = split512_to_half(c32f_h(c.x[e]));
        }
    }
    __syncthreads();
    C32F_MARK(6);
    if (tid < 128) {
        int r = tid/2, m = tid%2;
        float v[32];
        for (int c = 0; c < 32; ++c) v[c] = float(Q[r*96+m*32+c]);
        inv[tid] = h70_normalize_inverse_fast(v);
    }
    __syncthreads();
    C32F_MARK(7);
    const float head_scale = c32f_h(scale);
    for (int i = tid; i < 64*96; i += 256) {
        int r = i/96, m = i%96/32;
        float q = float(Q[i]), value;
        if (m == 2) value = h70_f(q);
        else {
            value = c32f_h(q*inv[r*2+m]);
            if (m == 0) value = c32f_h(value*head_scale);
            value = h70_f(value);
        }
        Q[i] = split512_to_half(value);
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
        den[tid] = c32f_h(parity[0]+parity[1]);
    }
    __syncthreads();
    C32F_MARK(10);
    for (int i = tid; i < 64*64; i += 256) {
        float p = float(sc[i])*c32f_h(1.f/den[i/64]);
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
            float residual = c32f_h(float(fs[r*32+col])*g2[col]);
            float body = h70_hd(double(c.x[e])+double(residual));
            raw_out[base+r*32+col] = body;
            if (quant_out) quant_out[base+r*32+col] = h70_f(body);
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
