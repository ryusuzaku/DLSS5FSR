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
// The gate of the expand output (one hidden value, half in and out):
//   a = h(c), g = clamp(a, +-4), inner = h(|g|*k1 + k2), poly = h(g*inner + k3), h(a*poly)
// in f16 instructions. Each float step rounds an exact product (of halves)
// plus a half, first to float, then to half; the f16 fma rounds it once.
// They agree for every half a (build/f16_equiv_test.exe, exhaustive).
__device__ inline float gate(float c) {
    const _Float16 a = (_Float16)c;
    const _Float16 g = __builtin_amdgcn_fmed3h(a, (_Float16)-4.0f, (_Float16)4.0f);
    const _Float16 ag = __builtin_fabsf16(g);
    const _Float16 inner = __builtin_fmaf16(ag, (_Float16)-0.055908203125f, (_Float16)0.447265625f);
    const _Float16 poly = __builtin_fmaf16(g, inner, (_Float16)0.89453125f);
    return float(a*poly);
}
// The exponent transform of a logit plus its bias (a half). The affine step
// h(score*k + k') in one f16 fma (exhaustively equal, as the gate's).
__device__ inline float expo(float c, float bias) {
    const _Float16 score = (_Float16)(c+bias);
    const _Float16 b = __builtin_fmaf16(score, (_Float16).044921875f, (_Float16)1.30078125f);
    const float clipped = float(__builtin_amdgcn_fmed3h(b, (_Float16)1.03125f, (_Float16)1.5693359375f));
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
    // f16 instructions: the fma rounds v^2 + h(w^2) once, as split512_half_sum;
    // sums and products of two halves round once either way.
    _Float16 s[8], p[4];
    for (int e = 0; e < 8; ++e) {
        const _Float16 a = (_Float16)c0[e], b = (_Float16)c1[e];
        s[e] = __builtin_fmaf16(a, a, b*b);
    }
    for (int q = 0; q < 4; ++q) p[q] = s[2*q] + s[2*q+1];
    for (int q = 0; q < 4; ++q) p[q] = p[q] + (_Float16)__shfl_xor(float(p[q]), 16);
    const float total = float((p[0] + p[2]) + (p[1] + p[3]));
    return hf(1.f/sqrtf(fmaxf(total, .00006198883056640625f)));
}
// The f16-accumulating MMA's step (S338): eight halves plus a K block's f32
// partial sums, each exact sum rounded once to half (nearest even), two
// v_fma_mix a register. NVIDIA's HMMA rounds its f16 accumulator once per
// block; the f32-add-then-half form (c32f_h(acc + part), the reference
// kernels') rounds twice and costs twice the instructions. Exact single
// rounding: build/mix_round_test.exe.
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
// RESIDENT_SR=0: the kernels keep the f32-add-then-half form. FP8 partial
// sums of a 32-deep block are exact in f32 at the network's magnitudes, so the
// two agree on real data (swin_t_bench, network_parity: byte-identical).
inline bool single_round = [] { const char* v = getenv("RESIDENT_SR"); return !(v && *v == '0'); }();
__device__ inline void acc_add(h8& acc, const acc8& part) {
    typedef unsigned u4 __attribute__((ext_vector_type(4)));
    u4 r = __builtin_bit_cast(u4, acc);
    for (int q = 0; q < 4; ++q) {
        unsigned v = r[q];
        asm("v_fma_mixlo_f16 %0, %1, 1.0, %0 op_sel_hi:[0,0,1]" : "+v"(v) : "v"(part[2*q]));
        asm("v_fma_mixhi_f16 %0, %1, 1.0, %0 op_sel:[0,0,1] op_sel_hi:[0,0,1]" : "+v"(v) : "v"(part[2*q+1]));
        r[q] = v;
    }
    acc = __builtin_bit_cast(h8, r);
}
// gate() of eight halves and their E4M3 bytes (pack's clamp to +-448), in
// packed f16 instructions: the same f16 operations, two elements at a time.
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
__device__ inline op8 gate_pack(h8 a) {
    float v[8];
    for (int q = 0; q < 4; ++q) {
        const h2 x = {a[2*q], a[2*q+1]};
        const h2 g = __builtin_elementwise_min(__builtin_elementwise_max(x, (h2)(_Float16)-4.0f), (h2)(_Float16)4.0f);
        const h2 inner = __builtin_elementwise_fma(__builtin_elementwise_abs(g), (h2)(_Float16)-0.055908203125f, (h2)(_Float16)0.447265625f);
        const h2 poly = __builtin_elementwise_fma(g, inner, (h2)(_Float16)0.89453125f);
        const h2 r = __builtin_elementwise_min(__builtin_elementwise_max(x*poly, (h2)(_Float16)-448.0f), (h2)(_Float16)448.0f);
        v[2*q] = float(r[0]); v[2*q+1] = float(r[1]);
    }
    int lo = 0, hi = 0;
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], lo, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], lo, true);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v[4], v[5], hi, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(v[6], v[7], hi, true);
    return op8{lo, hi};
}
// The f32-add-then-half form of acc_add (RESIDENT_SR=0).
__device__ inline void acc_add2(h8& acc, const acc8& part) {
    for (int e = 0; e < 8; ++e) acc[e] = (_Float16)(float(acc[e]) + part[e]);
}
// h(a + b) of two floats, the exact sum rounded once (== split512_half_sum,
// which takes it through TwoSum and rounding to odd): one v_fma_mix.
__device__ inline float mix_sum(float a, float b) {
    unsigned v;
    asm("v_fma_mixlo_f16 %0, %1, 1.0, %2" : "=v"(v) : "v"(a), "v"(b));
    return float(__builtin_bit_cast(_Float16, (unsigned short)(v & 0xFFFFu)));
}
// h(a * b) and h(a + b) of two halves in one f16 instruction.
__device__ inline float hmul(float a, float b) { return float((_Float16)a * (_Float16)b); }
__device__ inline float hadd(float a, float b) { return float((_Float16)a + (_Float16)b); }
// The softmax denominator inverse of one query (split512_den order) from
// its 64 exponents: key k in ex[k/16][k%8] of lane half (k%16)/8. Partial b
// (b in 0-3 and 8-11, + {0,16,4,20,32,48,36,52}) stays in one lane half:
// this half computes partials (odd, l) for l = h*8 + {0,2}, the other
// half's come across.
__device__ inline float den_inverse(const float (*ex)[8]) {
    const int h = (threadIdx.x % 32) / 16;
    _Float16 x[4][8];
    for (int k = 0; k < 4; ++k) for (int e = 0; e < 8; ++e) x[k][e] = (_Float16)ex[k][e];
    _Float16 mine[2][2];
    for (int odd = 0; odd < 2; ++odd)
        for (int j = 0; j < 2; ++j) {
            const int o = odd + 2*j;
            _Float16 value = x[0][o] + x[1][o];
            value = value + (x[0][o+4] + x[1][o+4]);
            value = value + (x[2][o] + x[3][o]);
            value = value + (x[2][o+4] + x[3][o+4]);
            mine[odd][j] = value;
        }
    _Float16 parity[2];
    for (int odd = 0; odd < 2; ++odd) {
        const _Float16 o0 = (_Float16)__shfl_xor(float(mine[odd][0]), 16), o1 = (_Float16)__shfl_xor(float(mine[odd][1]), 16);
        const _Float16 p0 = h ? o0 : mine[odd][0], p1 = h ? o1 : mine[odd][1];
        const _Float16 p2 = h ? mine[odd][0] : o0, p3 = h ? mine[odd][1] : o1;
        parity[odd] = ((p0 + p1) + p2) + p3;
    }
    return hf(1.f/float(parity[0] + parity[1]));
}
}  // namespace wmma_t
