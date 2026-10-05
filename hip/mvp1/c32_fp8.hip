#pragma once
// FP8 form of k_c32_wmma (fast path, S330). Same stages, epilogues, rounding
// boundaries and reduction orders; the differences:
// - every product runs on gfx12 FP8 WMMA (16x16x16, f32 accumulate). All
//   operands are values the reference already quantizes to E4M3 (F(x), the
//   FP8 weights, the hidden, Q/K/V, the probabilities, the context), so the
//   products are exact; only the f32 summation order inside each 16-product
//   step differs, as on the f16 path.
// - persistent workgroups: the FP8 weights and the bias are staged into LDS
//   once per workgroup and reused for every window it claims, instead of
//   every window re-reading 24 KB of half weights through the matrix loads.
// - activations live in LDS as E4M3 bytes (a quarter of the f32 / half size).
// Operand layout (gfx12 wave32, verified by fp8_wmma_layout_test.hip): lane l
// holds A row l%16 / B column l%16 with K = (l/16)*8 .. +8 as 8 bytes; the
// accumulator holds column l%16, rows (l/16)*8 + e.
// Needs c32_fused.hip included first.

namespace c32f8 {
typedef float acc8 __attribute__((ext_vector_type(8)));
typedef int op8 __attribute__((ext_vector_type(2)));

// E4M3 byte of h70_f(x) (half rounding, clamp to +-448, nearest even).
__device__ inline unsigned char q8(float x) {
    float h = __half2float(__float2half_rn(x));
    h = fminf(448.0f, fmaxf(-448.0f, h));
    return (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(h, 0.0f, 0, false) & 0xFF);
}
// The same for a value that is already a half (no second rounding needed).
__device__ inline unsigned char q8h(float h) {
    h = fminf(448.0f, fmaxf(-448.0f, h));
    return (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(h, 0.0f, 0, false) & 0xFF);
}
__device__ inline float f8(unsigned char b) { return __builtin_amdgcn_cvt_f32_fp8((int)b, 0); }

// One 16-deep step: A rows a[(l%16)*lda + k], B columns b[(l%16)*ldb + k].
__device__ inline acc8 step(acc8 c, const unsigned char* a, int lda, const unsigned char* b, int ldb, int k) {
    const int l = threadIdx.x % 32, off = (l / 16) * 8 + k;
    op8 av = *reinterpret_cast<const op8*>(a + (l % 16) * lda + off);
#if C32F8_EXP == 1
    op8 bv = {0x38383838, int((size_t)b & 0x3F)};
#else
    op8 bv = *reinterpret_cast<const op8*>(b + (l % 16) * ldb + off);
#endif
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av, bv, c);
}
// A 32-deep slice from zero (the reference's slice sum).
__device__ inline acc8 slice32(const unsigned char* a, int lda, const unsigned char* b, int ldb, int k0) {
    acc8 c = {};
    c = step(c, a, lda, b, ldb, k0);
    return step(c, a, lda, b, ldb, k0 + 16);
}
}  // namespace c32f8

// w8: the first 12288 weights as E4M3 bytes, [out][k] like hw.
__global__ void k_c32_fp8_weights(const float* dw, unsigned char* w8, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) w8[i] = c32f8::q8h(dw[i]);  // the weights are E4M3 values already
}

#ifndef C32F8_EXP
#define C32F8_EXP 0
#endif
#ifdef C32F8_PROF
__device__ unsigned long long c32f8_prof[16];
#define C32F8_MARK(i) do { if (threadIdx.x == 0) marks[i] = wall_clock64(); } while (0)
#else
#define C32F8_MARK(i) do {} while (0)
#endif
#ifndef C32F8_WAVES
#define C32F8_WAVES 1  // no occupancy target: forcing 16 waves (96 VGPRs) measured slower
#endif
// One window per workgroup. LDS is exactly 16 KB (four workgroups, 32 waves
// per CU) by sharing buffers whose lifetimes do not overlap:
//   X  (2 KB): F(input) -> F(ffn) -> row inverses (den) -> F(context)
//   H  (8 KB): hidden -> exponent scores (half)
//   QK (4 KB): token input addresses -> Q | K -> probabilities -> output addresses
//   Vt (2 KB): V transposed [channel][key]
// Weights and bias are read through the caches (staging them cost occupancy).
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(C32F8_WAVES)))
void k_c32_fp8(const float* in, const float* __restrict__ dw, const unsigned char* __restrict__ w8, float scale,
               float* raw_out, float* quant_out, C32Io io) {
    using namespace c32f8;
    __shared__ __attribute__((aligned(16))) unsigned char X[64*32];
    __shared__ __attribute__((aligned(16))) unsigned char H[64*128];
    __shared__ __attribute__((aligned(16))) unsigned char QK[64*64];
    __shared__ __attribute__((aligned(16))) unsigned char Vt[32*64];
    _Float16* sc = reinterpret_cast<_Float16*>(H);
    unsigned char *Qs = QK, *Ks = QK + 64*32, *P = QK;
    __shared__ int in_base[64], out_base[64];
    __shared__ float den[64];
    const unsigned char *W1 = w8, *W2 = w8+4096, *Wqkv = w8+8192, *Wp = w8+11264;
    const float *g1 = dw+16385, *g2 = dw+16417, *B = dw+12288;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    const int win = blockIdx.x;
    const float head_scale = c32f_h(scale);
#ifdef C32F8_PROF
    __shared__ unsigned long long marks[16];
    if (tid < 16) marks[tid] = 0;
    __syncthreads();
#endif
    C32F8_MARK(0);

    if (tid < 64) { in_base[tid] = io.base(io.in_hwc, win, tid); out_base[tid] = io.base(io.out_hwc, win, tid); }
    __syncthreads();
    C32F8_MARK(1);
    // One read of the input serves F(input) and the FFN residual: each thread
    // loads the 8 values its projection tile lanes own (rows wave/2*16+lr+e,
    // column wave%2*16+lc), which together cover the window once.
    float ffn[8];
    {
        const int c = wave%2*16+lc, ch = io.channel(c);
        const float gc = g1[c];
        for (int e = 0; e < 8; ++e) {
            const int r = wave/2*16+lr+e, b = in_base[r];
            const float v = b < 0 ? 0.0f : in[b + ch];
            X[r*32 + c] = q8(v);
            ffn[e] = c32f_h(v*gc);
        }
    }
    __syncthreads();
    C32F8_MARK(2);
    {   // Expand (K=32) and gate: 4x8 tiles, 4 per wave.
        const int rt = wave%4;
        for (int j = 0; j < 4; ++j) {
            const int ct = wave/4*4+j;
            acc8 c = slice32(X+rt*16*32, 32, W1+ct*16*32, 32, 0);
            for (int e = 0; e < 8; ++e) {
                float a = c32f_h(c[e]);
                float gate = fminf(4.0f, fmaxf(-4.0f, a));
                float inner = c32f_h(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                float poly = c32f_h(gate*inner+0.89453125f);
                H[(rt*16+lr+e)*128+ct*16+lc] = q8h(c32f_h(a*poly));
            }
        }
    }
    __syncthreads();
    C32F8_MARK(3);
    {   // Contract (K=128, four 32-slices in order) with the residual in ffn.
        const int rt = wave/2, ct = wave%2;
        for (int s = 0; s < 4; ++s) {
            acc8 part = slice32(H+rt*16*128, 128, W2+ct*16*128, 128, s*32);
            for (int e = 0; e < 8; ++e) ffn[e] = c32f_h(ffn[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e) X[(rt*16+lr+e)*32+ct*16+lc] = q8h(ffn[e]);
    }
    __syncthreads();
    C32F8_MARK(4);
    {   // QKV with the Q/K norm in registers (k_c32_wmma's lane mapping).
        const int rt = wave%4, m = wave/4;
        acc8 c0 = slice32(X+rt*16*32, 32, Wqkv+(m*32)*32, 32, 0);
        acc8 c1 = slice32(X+rt*16*32, 32, Wqkv+(m*32+16)*32, 32, 0);
        acc8 cv = slice32(X+rt*16*32, 32, Wqkv+(64+m*16)*32, 32, 0);
        unsigned char* dst = m == 0 ? Qs : Ks;
        for (int e = 0; e < 8; ++e) {
            const int r = rt*16+lr+e;
            float v0 = c32f_h(c0[e]), v1 = c32f_h(c1[e]);
            float s = split512_half_sum(v0*v0, c32f_h(v1*v1));
            s = c32f_h(s + __shfl_xor(s, 1));
            s = c32f_h(s + __shfl_xor(s, 8));
            s = c32f_h(s + __shfl_xor(s, 4));
            s = c32f_h(s + __shfl_xor(s, 2));
#if C32F8_EXP == 3
            const float inverse = 0.5f;
#else
            const float inverse = c32f_h(1.f/sqrtf(fmaxf(s, .00006198883056640625f)));
#endif
            float q0 = c32f_h(v0*inverse), q1 = c32f_h(v1*inverse);
            if (m == 0) { q0 = c32f_h(q0*head_scale); q1 = c32f_h(q1*head_scale); }
            dst[r*32+lc] = q8h(q0);
            dst[r*32+16+lc] = q8h(q1);
            Vt[(m*16+lc)*64+r] = q8h(c32f_h(cv[e]));
        }
    }
    __syncthreads();
    C32F8_MARK(5);
    {   // Scores (K=32) + bias, exponent transform: 4x4 tiles, 2 per wave.
        const int rt = wave%4;
        for (int j = 0; j < 2; ++j) {
            const int ct = wave/4*2+j;
            acc8 c = slice32(Qs+rt*16*32, 32, Ks+ct*16*32, 32, 0);
            for (int e = 0; e < 8; ++e) {
                int q = rt*16+lr+e, k = ct*16+lc;
#if C32F8_EXP == 2
                float score = c32f_h(c[e]+0.25f*k);
#else
                float score = c32f_h(c[e]+B[q*64+k]);
#endif
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
    C32F8_MARK(6);
#if C32F8_EXP == 4
    if (tid < 64) den[tid] = float(sc[tid*64]);
#else
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
        den[tid] = c32f_h(1.f/c32f_h(parity[0]+parity[1]));
    }
#endif
    __syncthreads();
    C32F8_MARK(7);
    for (int i = tid; i < 64*64; i += 256) P[i] = q8h(c32f_h(float(sc[i])*den[i/64]));
    __syncthreads();
    C32F8_MARK(8);
    {   // Context (two 32-key slices) -> F(context) in X: 4x2 tiles.
        const int rt = wave/2, ct = wave%2;
        float acc[8] = {};
        for (int k0 = 0; k0 < 64; k0 += 32) {
            acc8 part = slice32(P+rt*16*64, 64, Vt+ct*16*64, 64, k0);
            for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part[e]);
        }
        for (int e = 0; e < 8; ++e) X[(rt*16+lr+e)*32+ct*16+lc] = q8h(acc[e]);
    }
    __syncthreads();
    C32F8_MARK(9);
    {   // Projection: single half rounding of the residual sum.
        const int rt = wave/2, ct = wave%2;
        acc8 c = slice32(X+rt*16*32, 32, Wp+ct*16*32, 32, 0);
        for (int e = 0; e < 8; ++e) {
            int r = rt*16+lr+e, col = ct*16+lc;
            float residual = c32f_h(ffn[e]*g2[col]);
            float body = split512_half_sum(c[e], residual);
            if (out_base[r] < 0) continue;
            int a = out_base[r] + io.channel(col);
            if (raw_out) raw_out[a] = body;
            if (quant_out) quant_out[a] = h70_f(body);
        }
    }
#ifdef C32F8_PROF
    __syncthreads();
    if (tid == 0) {
        marks[15] = wall_clock64();
        unsigned long long previous = marks[0];
        for (int i = 1; i < 15; ++i) if (marks[i]) { atomicAdd(&c32f8_prof[i], marks[i]-previous); previous = marks[i]; }
        atomicAdd(&c32f8_prof[15], marks[15]-previous);
    }
#endif
}
