#pragma once
// Helpers of the transposed FP8 WMMA kernels (S336; c32_t.hip, swin_t.hip,
// split512_attention_wmma.hip): activations as [feature][token] B operands,
// weights as A. gfx12 wave32 FP8 WMMA layout: a lane holds A row l%16 / B
// column l%16 with K = (l/16)*8 .. +8, the accumulator column l%16 with rows
// (l/16)*8 + e, so an accumulator tile is the next product's B operand.
// Needs split512_block.hip first.

namespace wmma_t {
typedef float acc8 __attribute__((ext_vector_type(8)));
typedef int op8 __attribute__((ext_vector_type(2)));
__device__ inline float hf(float x) { return split512_half(x); }
__device__ inline float clamp448(float h) { return fminf(448.0f, fmaxf(-448.0f, h)); }
// Eight halves -> E4M3 bytes in K order (clamp to +-448, nearest even).
__device__ inline op8 pack(const float* v) {
    int lo = 0, hi = 0;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[0]), clamp448(v[1]), lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[2]), clamp448(v[3]), lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[4]), clamp448(v[5]), hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(clamp448(v[6]), clamp448(v[7]), hi, true);
    return op8{lo, hi};
}
// One half -> its E4M3 byte.
__device__ inline unsigned char byte8(float h) {
    return (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(clamp448(h), 0.0f, 0, false) & 0xFF);
}
__device__ inline acc8 mma(op8 a, op8 b, acc8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c); }
// A operand: row l%16 of a [rows][ld] byte matrix, K step k.
__device__ inline op8 arow(const unsigned char* m, int ld, int k) {
    const int l = threadIdx.x % 32;
    return *reinterpret_cast<const op8*>(m + (l % 16) * ld + (l / 16) * 8 + k);
}
// The gate of the expand output (one hidden value, half in and out).
__device__ inline float gate(float c) {
    float a = hf(c);
    float g = fminf(4.0f, fmaxf(-4.0f, a));
    float inner = hf(fabsf(g)*(-0.055908203125f)+0.447265625f);
    float poly = hf(g*inner+0.89453125f);
    return hf(a*poly);
}
// The exponent transform of a logit plus its bias (a half).
__device__ inline float expo(float c, float bias) {
    float score = hf(c+bias);
    float b = score*.044921875f+1.30078125f;
    float clipped = fminf(1.5693359375f, fmaxf(1.03125f, hf(b)));
    _Float16 hc = split512_to_half(clipped);
    unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
    bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
    __builtin_memcpy(&hc, &bits, 2);
    return float(hc);
}
// split512_norm_inverse_fast of one token's 32 values as a column reduction:
// this lane holds v[h*8+e] (c0) and v[16+h*8+e] (c1), the other lane half
// the rest. parts[j] = hsum(v[j]^2, h(v[j+16]^2)), pairs (j, j^1) in the
// lane, then j^8 (across the halves), j^4, j^2.
__device__ inline float norm_inverse(const float* c0, const float* c1) {
    float s[8], p[4];
    for (int e = 0; e < 8; ++e) s[e] = split512_half_sum(c0[e]*c0[e], hf(c1[e]*c1[e]));
    for (int q = 0; q < 4; ++q) p[q] = hf(s[2*q] + s[2*q+1]);
    for (int q = 0; q < 4; ++q) p[q] = hf(p[q] + __shfl_xor(p[q], 16));
    const float total = hf(hf(p[0] + p[2]) + hf(p[1] + p[3]));
    return hf(1.f/sqrtf(fmaxf(total, .00006198883056640625f)));
}
// The softmax denominator inverse of one query (split512_den order) from
// its 64 exponents: key k in ex[k/16][k%8] of lane half (k%16)/8. Partial b
// (b in 0-3 and 8-11, + {0,16,4,20,32,48,36,52}) stays in one lane half:
// this half computes partials (odd, l) for l = h*8 + {0,2}, the other
// half's come across.
__device__ inline float den_inverse(const float (*ex)[8]) {
    const int h = (threadIdx.x % 32) / 16;
    float mine[2][2];
    for (int odd = 0; odd < 2; ++odd)
        for (int j = 0; j < 2; ++j) {
            const int o = odd + 2*j;
            float value = hf(ex[0][o]+ex[1][o]);
            value = hf(value+hf(ex[0][o+4]+ex[1][o+4]));
            value = hf(value+hf(ex[2][o]+ex[3][o]));
            value = hf(value+hf(ex[2][o+4]+ex[3][o+4]));
            mine[odd][j] = value;
        }
    float parity[2];
    for (int odd = 0; odd < 2; ++odd) {
        const float o0 = __shfl_xor(mine[odd][0], 16), o1 = __shfl_xor(mine[odd][1], 16);
        const float p0 = h ? o0 : mine[odd][0], p1 = h ? o1 : mine[odd][1];
        const float p2 = h ? mine[odd][0] : o0, p3 = h ? mine[odd][1] : o1;
        parity[odd] = hf(hf(hf(p0+p1)+p2)+p3);
    }
    return hf(1.f/hf(parity[0]+parity[1]));
}
}  // namespace wmma_t
