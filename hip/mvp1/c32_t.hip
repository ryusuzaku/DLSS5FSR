#pragma once
// Transposed C32 body (S336): k_c32_fp8's arithmetic carried as
// [feature][token] instead of [token][feature]. On gfx12 wave32 FP8 WMMA a
// lane holds A row l%16 / B column l%16 with K = (l/16)*8 .. +8, and the
// accumulator column l%16 with rows (l/16)*8 + e, so an accumulator tile is,
// component for component, the B operand of a product contracting over its
// rows. With the weights as A and the activations as B (the orientation the
// peer DLSSNR-AMD's fswin_t uses), every intermediate feeds the next product
// from registers: x -> expand -> gate -> contract -> QKV -> logits^T -> P^T ->
// context -> projection. Only K and V cross waves (LDS, 4 KB); the Q/K norm
// and the softmax denominator become per-lane column reductions with one
// exchange across the lane halves.
// Same stage values, rounding boundaries and reduction orders as k_c32_fp8
// (the products are the same exact E4M3 products in the same 16-deep steps),
// so the outputs are bit-identical (build/c32_fp8_bench.exe checks).
// One window per workgroup, TT 16-token tiles per wave (4/TT waves).
// Needs c32_fused.hip and c32_fp8.hip included first.

namespace c32t {
using c32f8::acc8;
using c32f8::op8;
__device__ inline float clamp448(float h) { return fminf(448.0f, fmaxf(-448.0f, h)); }
// Eight halves -> E4M3 bytes in K order (c32f8::q8h per element).
__device__ inline op8 pack(const float* v) {
    int lo = 0, hi = 0;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[0]), clamp448(v[1]), lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[2]), clamp448(v[3]), lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[4]), clamp448(v[5]), hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[6]), clamp448(v[7]), hi, true);
    return op8{lo, hi};
}
__device__ inline acc8 mma(op8 a, op8 b, acc8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c); }
// A operand: row l%16 of a [rows][ld] byte matrix, K step k.
__device__ inline op8 arow(const unsigned char* m, int ld, int k) {
    const int l = threadIdx.x % 32;
    return *reinterpret_cast<const op8*>(m + (l % 16) * ld + (l / 16) * 8 + k);
}
// The gate of the expand output (one hidden value, half in and out).
__device__ inline float gate(float c) {
    float a = c32f_h(c);
    float g = fminf(4.0f, fmaxf(-4.0f, a));
    float inner = c32f_h(fabsf(g)*(-0.055908203125f)+0.447265625f);
    float poly = c32f_h(g*inner+0.89453125f);
    return c32f_h(a*poly);
}
// The exponent transform of a logit plus its bias (a half).
__device__ inline float expo(float c, float bias) {
    float score = c32f_h(c+bias);
    float b = score*.044921875f+1.30078125f;
    float clipped = fminf(1.5693359375f, fmaxf(1.03125f, c32f_h(b)));
    _Float16 hc = split512_to_half(clipped);
    unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
    bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
    __builtin_memcpy(&hc, &bits, 2);
    return float(hc);
}
}  // namespace c32t

// Fused full-resolution blocks (MODE, one 16-token tile per wave). Windows
// are the 8x8 raster windows of a width x height image (shift 0).
// MODE 1, the front's block0: the input is k_front_stem's token computed in
// place (rgb, noise, optional history, controls); the outputs are
// k_front_native32's clamped native skip0 and k_front_pool's FP8 pool (the
// 2x2 block of a pool is in one wave: lanes n, n^1, n^8).
// MODE 2, head70: the input is k_head70_merge's value (main at half
// resolution and the skip, native order) read in the body's peer order; the
// outputs are k_head70_finish_pair's two RGB images (the aligned sums are
// integer sums of truncated terms, so the two lane halves' partial sums
// combine exactly) and, when native is set, the native body.
struct C32Fuse {
    int width = 0, height = 0;
    const float *rgb = nullptr, *noise = nullptr, *stem = nullptr, *history = nullptr;
    float controls[5] = {};
    float *skip0 = nullptr, *pool = nullptr;
    const float *main = nullptr, *skip = nullptr, *sm = nullptr, *ss = nullptr, *coeff = nullptr, *color = nullptr;
    float *rgb_native = nullptr, *rgb_public = nullptr, *native = nullptr;
    float native_scale = 1.0f;
};

namespace c32t {
// float(static_cast<_Float16>(d)) without the software f64 -> f16 lowering:
// round to float by odd (RNE, then one ulp toward d when inexact and even),
// then to half; a float carries more than two extra bits, so the second
// rounding is correct.
__device__ inline float half_of(double d) {
    float f = float(d);
    if (double(f) != d) {
        unsigned u = __float_as_uint(f);
        if (!(u & 1u)) f = __uint_as_float(fabs(d) > fabs(double(f)) ? u + 1u : u - 1u);
    }
    return split512_half(f);
}
// head70_aligned_half over the 16 products of a lane pair: this lane's 8
// (a[k], b[k]) and the other lane half's 8, with the running acc. ea / eb are
// frexp exponents of a / b (a value's own once, not per colour). Every
// truncated term is below 2^27 in magnitude, so 16 of them sum exactly in
// an int.
__device__ inline float aligned_pair(const float* a, const int* ea, const float* b, float acc) {
    int e = -1000;
    for (int k = 0; k < 8; ++k) {
        int be;
        frexpf(fabsf(b[k]), &be);
        if (a[k] != 0.0f && b[k] != 0.0f) e = max(e, ea[k] + be);
    }
    e = max(e, __shfl_xor(e, 16));
    if (acc != 0.0f) {
        int ae;
        frexpf(fabsf(acc), &ae);
        e = max(e, ae + 1);
    }
    if (e == -1000) e = 0;
    int sum = 0;
    for (int k = 0; k < 8; ++k) sum += int(truncf(ldexpf(a[k] * b[k], 27 - e)));
    sum += __shfl_xor(sum, 16);
    return half_of(double(sum) * ldexp(1.0, e - 27) + double(acc));
}
}  // namespace c32t

template<int TT, int MODE = 0>
__global__ __launch_bounds__(128/TT) void k_c32_t(const float* in, const float* __restrict__ dw,
                                                  const unsigned char* __restrict__ w8, float scale,
                                                  float* raw_out, float* quant_out, C32Io io,
                                                  C32Fuse fz = C32Fuse()) {
    static_assert(MODE == 0 || TT == 1, "fused blocks run one tile per wave");
    using namespace c32t;
    __shared__ __attribute__((aligned(16))) unsigned char Ks[64*32];  // [key][d]
    __shared__ __attribute__((aligned(16))) unsigned char Vt[32*64];  // [d][key]
    __shared__ int in_base[64], out_base[64];
    const unsigned char *W1 = w8, *W2 = w8+4096, *Wqkv = w8+8192, *Wp = w8+11264;
    const float *g1 = dw+16385, *g2 = dw+16417, *B = dw+12288;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32, n = lane%16, h = lane/16;
    const int win = blockIdx.x;
    int token[TT];
    for (int u = 0; u < TT; ++u) token[u] = (wave*TT + u)*16 + n;
    if constexpr (MODE == 0) {
        for (int t = tid; t < 64; t += 128/TT) { in_base[t] = io.base(io.in_hwc, win, t); out_base[t] = io.base(io.out_hwc, win, t); }
        __syncthreads();
    }
    // The fused modes' pixel (fx, fy) of this lane's token.
    const int wx = fz.width/8 > 0 ? fz.width/8 : 1;
    const int fx = (win % wx)*8 + token[0]%8, fy = (win / wx)*8 + token[0]/8;
    const size_t fp = size_t(fy)*fz.width + fx;

    // x^T as two B operands (channels s*16 + h*8 + e) and the FFN residual,
    // which sits in the contract output's layout (feature tile s, rows h*8+e).
    op8 xb[TT][2];
    float ffn[TT][2][8];
    for (int u = 0; u < TT; ++u) {
        const int b = MODE == 0 ? in_base[token[u]] : 0;
        float f[15];
        if constexpr (MODE == 1) {  // k_front_stem's features
            const float* p = fz.rgb + fp*3;
            const float* nz = fz.noise + fp*3;
            for (int k = 0; k < 3; ++k) f[k] = h70_h(nz[k]);
            f[3] = 1.0f;
            for (int k = 0; k < 3; ++k) {
                float v = h70_h((p[k]-.5f)*.125f);
                f[4+k] = v; f[7+k] = v;
            }
            if (fz.history) {
                const float* q = fz.history + fp*3;
                if (q[0] >= 0.0f)
                    for (int k = 0; k < 3; ++k) f[7+k] = h70_h(h70_h(h70_h(q[k])-.5f)*.125f);
            }
            for (int k = 10; k < 15; ++k) f[k] = h70_h(fz.controls[k-10]);
        }
        for (int s = 0; s < 2; ++s) {
            float v[8], x[8];
            if constexpr (MODE == 1) {
                for (int e = 0; e < 8; ++e) {
                    const int c = s*16 + h*8 + e;
                    float part = 0.0f;
                    for (int k = 0; k < 15; ++k) part += f[k]*fz.stem[k*32+c];
                    x[e] = h70_h(part);
                }
            } else if constexpr (MODE == 2) {  // k_head70_merge
                const size_t m = (size_t(fy/2)*(fz.width/2) + fx/2)*32;
                for (int e = 0; e < 8; ++e) {
                    const int nc = peer_to_native32(s*16 + h*8 + e);
                    float main_product = fz.main[m + nc]*fz.sm[nc];
                    float skip_product = fz.skip[fp*32 + nc]*fz.ss[nc];
                    x[e] = head70_half(head70_half(main_product) + skip_product);
                }
            } else if (b < 0) {
                for (int e = 0; e < 8; ++e) x[e] = 0.0f;
            } else if (!io.permute) {
                const float4 lo = *reinterpret_cast<const float4*>(in + b + s*16 + h*8);
                const float4 hi = *reinterpret_cast<const float4*>(in + b + s*16 + h*8 + 4);
                x[0] = lo.x; x[1] = lo.y; x[2] = lo.z; x[3] = lo.w; x[4] = hi.x; x[5] = hi.y; x[6] = hi.z; x[7] = hi.w;
            } else {
                // peer c = s*16 + h*8 + e is native s*16 + 4(e>>1) + 2h + (e&1): four pairs.
                for (int q = 0; q < 4; ++q) {
                    const float2 v2 = *reinterpret_cast<const float2*>(in + b + s*16 + 4*q + 2*h);
                    x[2*q] = v2.x; x[2*q+1] = v2.y;
                }
            }
            for (int e = 0; e < 8; ++e) {
                v[e] = c32f_h(x[e]);
                ffn[u][s][e] = c32f_h(x[e]*g1[s*16 + h*8 + e]);
            }
            xb[u][s] = pack(v);
        }
    }
    // Expand + gate: hidden^T tile t (rows t*16..) is contract K step t.
    op8 hb[TT][8];
    for (int t = 0; t < 8; ++t) {
        const op8 a0 = arow(W1 + t*16*32, 32, 0), a1 = arow(W1 + t*16*32, 32, 16);
        for (int u = 0; u < TT; ++u) {
            acc8 c = {};
            c = mma(a1, xb[u][1], mma(a0, xb[u][0], c));
            float v[8];
            for (int e = 0; e < 8; ++e) v[e] = gate(c[e]);
            hb[u][t] = pack(v);
        }
    }
    // Contract: four 32-deep slices in order onto the residual.
    op8 fb[TT][2];
    for (int f = 0; f < 2; ++f) {
        for (int s = 0; s < 4; ++s) {
            const op8 a0 = arow(W2 + f*16*128, 128, (2*s)*16), a1 = arow(W2 + f*16*128, 128, (2*s+1)*16);
            for (int u = 0; u < TT; ++u) {
                acc8 part = {};
                part = mma(a1, hb[u][2*s+1], mma(a0, hb[u][2*s], part));
                for (int e = 0; e < 8; ++e) ffn[u][f][e] = c32f_h(ffn[u][f][e]+part[e]);
            }
        }
        for (int u = 0; u < TT; ++u) fb[u][f] = pack(ffn[u][f]);
    }
    // QKV: Q^T, K^T, V^T tiles (rows 0-31 Q, 32-63 K, 64-95 V), then the
    // Q/K norm per token: parts[j] = hsum(v[j]^2, h(v[j+16]^2)) for j = h*8+e,
    // pairs (j, j^1) in the lane, then j^8 (the other half), j^4, j^2.
    const float head_scale = c32f_h(scale);
    op8 qb[TT][2];
    for (int m = 0; m < 3; ++m) {
        const op8 a00 = arow(Wqkv + (2*m)*16*32, 32, 0), a01 = arow(Wqkv + (2*m)*16*32, 32, 16);
        const op8 a10 = arow(Wqkv + (2*m+1)*16*32, 32, 0), a11 = arow(Wqkv + (2*m+1)*16*32, 32, 16);
        for (int u = 0; u < TT; ++u) {
            acc8 c0 = {}, c1 = {};
            c0 = mma(a01, fb[u][1], mma(a00, fb[u][0], c0));
            c1 = mma(a11, fb[u][1], mma(a10, fb[u][0], c1));
            if (m == 2) {
                for (int e = 0; e < 8; ++e) {
                    Vt[(h*8+e)*64 + token[u]] = c32f8::q8h(c32f_h(c0[e]));
                    Vt[(16+h*8+e)*64 + token[u]] = c32f8::q8h(c32f_h(c1[e]));
                }
                continue;
            }
            float v0[8], v1[8], p[4], s[8];
            for (int e = 0; e < 8; ++e) { v0[e] = c32f_h(c0[e]); v1[e] = c32f_h(c1[e]); }
            for (int e = 0; e < 8; ++e) s[e] = split512_half_sum(v0[e]*v0[e], c32f_h(v1[e]*v1[e]));
            for (int q = 0; q < 4; ++q) p[q] = c32f_h(s[2*q] + s[2*q+1]);
            for (int q = 0; q < 4; ++q) p[q] = c32f_h(p[q] + __shfl_xor(p[q], 16));
            const float total = c32f_h(c32f_h(p[0] + p[2]) + c32f_h(p[1] + p[3]));
            const float inverse = c32f_h(1.f/sqrtf(fmaxf(total, .00006198883056640625f)));
            for (int e = 0; e < 8; ++e) {
                float q0 = c32f_h(v0[e]*inverse), q1 = c32f_h(v1[e]*inverse);
                if (m == 0) { q0 = c32f_h(q0*head_scale); q1 = c32f_h(q1*head_scale); }
                v0[e] = q0; v1[e] = q1;
            }
            if (m == 0) {
                qb[u][0] = pack(v0); qb[u][1] = pack(v1);
            } else {
                *reinterpret_cast<op8*>(Ks + token[u]*32 + h*8) = pack(v0);
                *reinterpret_cast<op8*>(Ks + token[u]*32 + 16 + h*8) = pack(v1);
            }
        }
    }
    __syncthreads();
    // Per query tile: logits^T (key tile k), bias, exponent transform, the
    // denominator, the probabilities, the context and the projection.
    for (int u = 0; u < TT; ++u) {
        float ex[4][8];
        for (int k = 0; k < 4; ++k) {
            acc8 c = {};
            c = mma(arow(Ks + k*16*32, 32, 16), qb[u][1], mma(arow(Ks + k*16*32, 32, 0), qb[u][0], c));
            const float4* bias = reinterpret_cast<const float4*>(B + token[u]*64 + k*16 + h*8);
            const float4 b0 = bias[0], b1 = bias[1];
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            for (int e = 0; e < 8; ++e) ex[k][e] = expo(c[e], bv[e]);
        }
        // Denominator in k_c32_fp8's order. Key k sits in tile k/16, half
        // (k%16)/8, element k%8; partial b (b in 0-3 and 8-11, + {0,16,4,20,
        // 32,48,36,52}) stays inside one lane half: this half computes the
        // partials (odd, l) for l = h*8 + {0,2}; the other half's two come across.
        float den;
        {
            float mine[2][2];
            for (int odd = 0; odd < 2; ++odd)
                for (int j = 0; j < 2; ++j) {
                    const int o = odd + 2*j;
                    float value = c32f_h(ex[0][o]+ex[1][o]);
                    value = c32f_h(value+c32f_h(ex[0][o+4]+ex[1][o+4]));
                    value = c32f_h(value+c32f_h(ex[2][o]+ex[3][o]));
                    value = c32f_h(value+c32f_h(ex[2][o+4]+ex[3][o+4]));
                    mine[odd][j] = value;
                }
            float parity[2];
            for (int odd = 0; odd < 2; ++odd) {
                const float o0 = __shfl_xor(mine[odd][0], 16), o1 = __shfl_xor(mine[odd][1], 16);
                const float p0 = h ? o0 : mine[odd][0], p1 = h ? o1 : mine[odd][1];
                const float p2 = h ? mine[odd][0] : o0, p3 = h ? mine[odd][1] : o1;
                parity[odd] = c32f_h(c32f_h(c32f_h(p0+p1)+p2)+p3);
            }
            den = c32f_h(1.f/c32f_h(parity[0]+parity[1]));
        }
        op8 pb[4];
        for (int k = 0; k < 4; ++k) {
            float v[8];
            for (int e = 0; e < 8; ++e) v[e] = c32f_h(ex[k][e]*den);
            pb[k] = pack(v);
        }
        // Context^T in two 32-key slices, then the projection.
        op8 cb[2];
        for (int d = 0; d < 2; ++d) {
            float acc[8] = {};
            for (int s = 0; s < 2; ++s) {
                acc8 part = {};
                part = mma(arow(Vt + d*16*64, 64, (2*s+1)*16), pb[2*s+1], mma(arow(Vt + d*16*64, 64, (2*s)*16), pb[2*s], part));
                for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part[e]);
            }
            cb[d] = pack(acc);
        }
        if constexpr (MODE != 0) {
            float body[2][8];
            for (int o = 0; o < 2; ++o) {
                acc8 c = {};
                c = mma(arow(Wp + o*16*32, 32, 16), cb[1], mma(arow(Wp + o*16*32, 32, 0), cb[0], c));
                for (int e = 0; e < 8; ++e) body[o][e] = split512_half_sum(c[e], c32f_h(ffn[u][o][e]*g2[o*16 + h*8 + e]));
            }
            if constexpr (MODE == 1) {
                // skip0 (k_front_native32, clamped) and the pool (k_front_pool).
                const bool corner = (token[u] % 2) == 0 && (token[u] / 8) % 2 == 0;
                float* pool = fz.pool + (size_t(fy/2)*(fz.width/2) + fx/2)*32;
                for (int o = 0; o < 2; ++o)
                    for (int e = 0; e < 8; ++e) {
                        const int c = o*16 + h*8 + e;
                        const float v = body[o][e];
                        fz.skip0[fp*32 + peer_to_native32(c)] = fminf(448.0f, fmaxf(-448.0f, v));
                        const float a = h70_h(v + __shfl_xor(v, 1));
                        const float below = __shfl_xor(a, 8);
                        if (corner) pool[c] = h70_f(h70_h(h70_h(a + below)*.25f));
                    }
            } else {
                if (fz.native)
                    for (int o = 0; o < 2; ++o)
                        for (int e = 0; e < 8; ++e)
                            fz.native[(size_t(win)*64 + token[u])*32 + peer_to_native32(o*16 + h*8 + e)] = body[o][e];
                int ea[2][8];
                for (int o = 0; o < 2; ++o)
                    for (int e = 0; e < 8; ++e) frexpf(fabsf(body[o][e]), &ea[o][e]);
                for (int c = 0; c < 3; ++c) {
                    float coef[2][8];
                    for (int o = 0; o < 2; ++o)
                        for (int e = 0; e < 8; ++e) coef[o][e] = fz.coeff[c*32 + peer_to_native32(o*16 + h*8 + e)];
                    float value = aligned_pair(body[0], ea[0], coef[0], 0.0f);
                    value = aligned_pair(body[1], ea[1], coef[1], value);
                    if (h) continue;
                    const size_t i = fp*3 + c;
                    // k_head70_finish_pair's double forms in float: each is one
                    // rounding of an exact product/sum of floats (the scales are
                    // powers of two), and rounding a float sum through double
                    // first gives the same float.
                    float base = fz.color[i] * 0.125f - 0.0625f;
                    float nat = value * fz.native_scale + base;
                    float pub = value + base;
                    fz.rgb_native[i] = fminf(1.0f, fmaxf(0.0f, nat * 8.0f + 0.5f));
                    fz.rgb_public[i] = fminf(1.0f, fmaxf(0.0f, pub * 8.0f + 0.5f));
                }
            }
            continue;
        }
        const int ob = out_base[token[u]];
        for (int o = 0; o < 2; ++o) {
            acc8 c = {};
            c = mma(arow(Wp + o*16*32, 32, 16), cb[1], mma(arow(Wp + o*16*32, 32, 0), cb[0], c));
            if (ob < 0) continue;
            float body[8], quant[8];
            for (int e = 0; e < 8; ++e) {
                body[e] = split512_half_sum(c[e], c32f_h(ffn[u][o][e]*g2[o*16 + h*8 + e]));
                quant[e] = h70_f(body[e]);
            }
            if (!io.permute) {
                const int a = ob + o*16 + h*8;
                if (raw_out) {
                    *reinterpret_cast<float4*>(raw_out + a) = make_float4(body[0], body[1], body[2], body[3]);
                    *reinterpret_cast<float4*>(raw_out + a + 4) = make_float4(body[4], body[5], body[6], body[7]);
                }
                if (quant_out) {
                    *reinterpret_cast<float4*>(quant_out + a) = make_float4(quant[0], quant[1], quant[2], quant[3]);
                    *reinterpret_cast<float4*>(quant_out + a + 4) = make_float4(quant[4], quant[5], quant[6], quant[7]);
                }
            } else {
                for (int q = 0; q < 4; ++q) {
                    const int a = ob + o*16 + 4*q + 2*h;
                    if (raw_out) *reinterpret_cast<float2*>(raw_out + a) = make_float2(body[2*q], body[2*q+1]);
                    if (quant_out) *reinterpret_cast<float2*>(quant_out + a) = make_float2(quant[2*q], quant[2*q+1]);
                }
            }
        }
    }
}
