// hip/mvp1/swin_1h.hip -- Stage-1 (pre-block) unfused Swin 1H port, gfx1201.
//
// WHAT: patch embedding 16->32 (f16 weights, f32 accum, e4m3 quant out) +
// unfused Swin 1H block, FFN-FIRST order (HANDOFF §21: QKV-mma-A =
// quant(contract-chain-end)): FFN expand (32->128) + MpCubicSilu, FFN
// contract (128->32) + gate_ffn residual, QKV proj (32->96), window
// self-attention (8x8 = 64 tokens, 1 head, head_dim 32, QK-norm +
// temperature, + 64x64 f16 bias on the raw dots, fast-exp softmax, NO
// 1/sqrt(32) -- HANDOFF §19), attn proj (32->32) + gate_attn residual.
// f32 compute on exactly-dequantised e4m3 inputs; f32 accumulation (a
// superset of NVIDIA's f16 accum -- same argument as HANDOFF p4: gfx1201
// FP8 accumulates in FP32 only).
//
// WEIGHT CARVE (tensor-relative; §21 re-carve decided 2026-09-03, see
// tools/swin1h_ref.py header + HANDOFF §21):
//   tensor_000.bin (pre-block, 21696 B):
//     ffn1   e4m3 (128,32) @0        (4096 B)
//     ffn2   e4m3 (32,128) @4096     (4096 B, cool contract)
//     pad16 @8192 (8 f16 zeros) | patch f16 (32,16) @8208 (1024 B;
//               §33.14 audit: was @8192 + A-residue16 @9216; residue
//               dissolved into patch tail, kernel window == patch)
//     gate_ffn f16x32 @9232 | pad16 @9296
//     qkv    e4m3 (96,32)  @9312     (3072 B)
//     bias   f16  (64,64)  @12384    (8192 B; hot region reinterpreted --
//                                     score-mma C trace, f16 rms ~5;
//                                     §27: was transposed with fused @11360)
//     P-scale16 @20576 = [f32 QK-temperature][12 zeros] (corr. HANDOFF §17)
//     proj   e4m3 (32,32)  @20592    (1024 B)
//     gate_attn f16x32 @21616 | trailer16 @21680
//   tensor_001.bin (fused C=32, 20672 B): same minus patch/A-residue:
//     ffn1 @0 | ffn2 @4096 | gate_ffn @8208 | pad16 @8272 | qkv @8288 |
//     bias @11360 | P-scale16 @19552 | proj @19568 | gate_attn @20592 |
//     trailer16 @20656.
// W is row-major (out,in): Y[m][o] = sum_k X[m][k]*W[o][k] (matches the ViT
// QKV "(out,in)" convention, HANDOFF p2). QKV rows split Q[0:32]/K[32:64]/
// V[64:96]. No attention mask (the 64x64 region is a score BIAS, added to
// the raw dots -- recipe below). No activation on patch embed (HANDOFF
// p13.3). Residuals:
//   ffn-half out Yf = ffn2(act(ffn1(x))) + gate_ffn[*]  * x    (H1)
//   attn-half out    = proj(attn(qkv(Yf))) + gate_attn[*] * Yf  (H1)
// Bias recipe: S[i][j] = dot(Qn[i],Kn[j]) + f16bias[i*64+j], NO scale,
// row-major (exact 64x64 lane map open -- HANDOFF §21).
// e4m3 NaN codes (0x7F/0xFF) flush to +0.0 on dequant (H4, presumed HW;
// no-op on NaN-free vectors; the bias bytes carry NaN codes -- the N1
// side-effect note moved with the reinterpretation).
//
// OPEN (fidelity vs NVIDIA, NOT operator correctness):
//   N2b: contract-path magnitude (the §19 softmax is attention-side and
//       explains nothing about the hot FFN contract) -- via dynamic probing.
//   (N2-rcp resolved §17 as the QK-temperature; N1 moved to the bias.)
// Synthetic verification below is unaffected.
//
// STRUCTURE: section 1 = pure-C++ host mirror (also compiled by g++ with
// -DSWIN1H_HOST_ONLY for CPU-side cross-checks vs tools/swin1h_ref.py).
// Section 2 = HIP kernels (direct ports, same k-ascending loop order).
// Section 3 = GPU self-test main (RX 9070 XT). Section 4 = host-only main.
//
// BUILD (Windows/ROCm box):
//   hipcc -std=c++17 --offload-arch=gfx1201 -O2 -ffp-contract=off
//     hip/mvp1/swin_1h.hip -o build/swin_1h.exe && ./build/swin_1h.exe
// HOST-ONLY (any g++):
//   g++ -std=c++17 -x c++ -DSWIN1H_HOST_ONLY -O2 -ffp-contract=off
//     hip/mvp1/swin_1h.hip -o /tmp/swin_1h_host && /tmp/swin_1h_host
//     [--mpcubic-fnv|--quant-fnv|--f32f16-fnv]
// (-ffp-contract=off on BOTH sides: forbids FMA contraction so device f32
// accumulation is bitwise identical to the mirror's for same-order code.
// expf remains the only allowed divergence, covered by the tol stages.)

// Silence MSVC's fopen deprecation (C4996) under hipcc-on-Windows (it uses
// MSVC headers); the POSIX name is what g++/clang use everywhere else.
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef SWIN1H_HOST_ONLY
#define HD
#else
#define HD __host__ __device__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif
// NOTE: sw_f16fma uses __int128 (emulated by clang on the Windows/hipcc box;
// MSVC proper never compiles this file -- it is built only by hipcc/g++).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#define SW_D_IN 16
#define SW_C 32
#define SW_TOK 64
#define SW_QKV 96
#define SW_H1 128  // §21: cool contract 32x128 @4096..8192

HD inline uint32_t sw_f32_as_u32(float f) {
    union {
        float ff;
        uint32_t u;
    } c;
    c.ff = f;
    return c.u;
}

// ---------------------------------------------------------------------------
// 1. Host mirror (pure C++; identical math to the kernels, f32)
// ---------------------------------------------------------------------------

static uint32_t sw_lcg_next(uint32_t s) {
    return 1664525u * s + 1013904223u;
}

// f32 -> f16 bits, round-to-nearest-even (matches __float2half_rn;
// NaN payloads preserved + quieted, like hardware converters).
// HD: the body is pure integer ops, so the device quant kernel shares it
// bitwise (no __float2half_rn divergence risk).
HD inline uint16_t sw_f32_to_f16_bits(float f) {
    uint32_t u = sw_f32_as_u32(f);
    uint32_t s = (u >> 16) & 0x8000u;
    uint32_t E = (u >> 23) & 0xFFu;
    uint32_t M = u & 0x7FFFFFu;
    if (E == 0xFFu) {
        if (M == 0) return (uint16_t)(s | 0x7C00u);  // inf
        uint32_t pay = (M >> 13) | 0x0200u;  // preserve payload, quiet
        return (uint16_t)(s | 0x7C00u | pay);
    }
    int e = (int)E - 112;  // f16-biased exponent
    if (e >= 31) return (uint16_t)(s | 0x7C00u);  // overflow -> inf
    if (e <= 0) {  // subnormal result (or zero)
        if (e < -10) return (uint16_t)s;
        uint32_t S = M | 0x800000u;  // 24-bit significand
        int shift = 14 - e;  // 14..24
        uint32_t half = 1u << (shift - 1);
        uint32_t mask = (shift >= 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u);
        uint32_t rem = S & mask;
        uint32_t res = S >> shift;  // candidate 10-bit mantissa
        if (rem > half || (rem == half && (res & 1u))) res++;
        if (res >= 1024u) return (uint16_t)(s | 0x0400u);  // -> min normal
        return (uint16_t)(s | res);
    }
    {  // normal result: drop 13 bits, RN-even
        uint32_t rem = M & 0x1FFFu;
        uint32_t res = M >> 13;
        if (rem > 0x1000u || (rem == 0x1000u && (res & 1u))) {
            res++;
            if (res >= 1024u) {
                res = 0;
                e++;
                if (e >= 31) return (uint16_t)(s | 0x7C00u);
            }
        }
        return (uint16_t)(s | ((uint32_t)e << 10) | res);
    }
}

// f16 bits -> f32 (exact).
static float sw_f16_to_f32(uint16_t u) {
    uint32_t s = (uint32_t)(u & 0x8000u) << 16;
    uint32_t e = ((uint32_t)u >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)u & 0x3FFu;
    uint32_t out;
    if (e == 0x1Fu) {
        out = s | 0x7F800000u | (m << 13);  // inf / nan (payload kept)
    } else if (e == 0u) {
        if (m == 0) {
            out = s;
        } else {  // subnormal -> normalize
            e = 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;
            out = s | ((uint32_t)(e + 112) << 23) | (m << 13);
        }
    } else {
        out = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// Single-rounding f16 FMA via exact __int128 arithmetic (mirrors __hfma2;
// NaN/Inf lanes fall back to double, like tools/swin1h_ref.py).
static uint16_t sw_f16fma(uint16_t ab, uint16_t bb, uint16_t cb) {
    uint32_t ea = (ab >> 10) & 0x1Fu, eb = (bb >> 10) & 0x1Fu,
             ec = (cb >> 10) & 0x1Fu;
    uint32_t ma = ab & 0x3FFu, mb = bb & 0x3FFu, mc = cb & 0x3FFu;
    if (ea == 0x1Fu || eb == 0x1Fu || ec == 0x1Fu) {
        bool an = (ea == 0x1Fu && ma != 0), bn = (eb == 0x1Fu && mb != 0),
             cn = (ec == 0x1Fu && mc != 0);
        if (an || bn || cn) return 0x7E00u;
        double v = (double)sw_f16_to_f32(ab) * (double)sw_f16_to_f32(bb) +
                   (double)sw_f16_to_f32(cb);
        return sw_f32_to_f16_bits((float)v);
    }
    // exact significands (11-bit incl. hidden 1) + binary exponents
    int xa, xb, xc;
    uint32_t Sa, Sb, Sc;
    if (ea == 0) {
        Sa = ma;
        xa = -24;
    } else {
        Sa = ma | 0x400u;
        xa = (int)ea - 15 - 10;
    }
    if (eb == 0) {
        Sb = mb;
        xb = -24;
    } else {
        Sb = mb | 0x400u;
        xb = (int)eb - 15 - 10;
    }
    if (ec == 0) {
        Sc = mc;
        xc = -24;
    } else {
        Sc = mc | 0x400u;
        xc = (int)ec - 15 - 10;
    }
    __int128 P = (__int128)Sa * (__int128)Sb;  // <= 22 significant bits
    int xp = xa + xb;
    __int128 Scs = ((cb & 0x8000u) ? -(__int128)Sc : (__int128)Sc);
    __int128 Ps = ((ab ^ bb) & 0x8000u) ? -P : P;
    // align to the smaller exponent
    int xmin = xp < xc ? xp : xc;
    __int128 acc = (Ps << (xp - xmin)) + (Scs << (xc - xmin));
    if (acc == 0) return 0x0000u;  // fma zero -> +0 (matches __hfma2 for
                                   // non-opposite-signed zero sums; the
                                   // opposite-signed case is data-dependent
                                   // and absent from our vectors)
    bool neg = acc < 0;
    unsigned __int128 mag = neg ? (unsigned __int128)(-acc) : (unsigned __int128)acc;
    // normalize: top bit position
    int top = 127;
    while (top > 0 && ((mag >> top) & 1u) == 0) top--;
    // want 11-bit significand: shift = top - 10; round RN-even on drop
    int shift = top - 10;
    int xres = xmin + shift;  // binary exponent of LSB of kept significand
    unsigned __int128 kept, rem = 0;
    if (shift >= 0) {
        kept = mag >> shift;
        unsigned __int128 mask =
            (shift >= 128) ? (unsigned __int128)(~0) : (((unsigned __int128)1 << shift) - 1);
        rem = mag & mask;
    } else {
        kept = mag << (-shift);
    }
    // rounding decision: rem vs half = 2^(shift-1)
    if (shift > 0) {
        unsigned __int128 half = (unsigned __int128)1 << (shift - 1);
        if (rem > half || (rem == half && (kept & 1u))) {
            kept++;
            if (kept >= (unsigned __int128)0x800u) {  // carry out
                kept >>= 1;
                xres++;
            }
        }
    }
    // kept is now the 11-bit significand (hidden 1 at bit 10) or subnormal
    // Pack: value = kept * 2^xres with kept in [0x400, 0x7FF] (normal).
    int e2 = xres + 10 + 15;  // f16-biased exponent (LSB weight xres, 11 bits)
    if (e2 >= 31) return (uint16_t)((neg ? 0x8000u : 0u) | 0x7C00u);
    if (e2 <= 0) {  // subnormal (or underflow to zero)
        int d = 1 - e2;
        if (d >= 12) return (uint16_t)(neg ? 0x8000u : 0u);
        unsigned __int128 sub = (unsigned __int128)kept;
        // kept has hidden bit; drop d bits with RN-even
        unsigned __int128 r = 0;
        if (d > 0) {
            unsigned __int128 m2 =
                (((unsigned __int128)1 << d) - 1);
            r = sub & m2;
            sub >>= d;
        }
        unsigned __int128 h2 = d > 0 ? ((unsigned __int128)1 << (d - 1)) : 0;
        if (d > 0 && (r > h2 || (r == h2 && (sub & 1u)))) {
            sub++;
            if (sub >= 1024u) return (uint16_t)((neg ? 0x8000u : 0u) | 0x0400u);
        }
        return (uint16_t)((neg ? 0x8000u : 0u) | (uint32_t)sub);
    }
    uint32_t mant = (uint32_t)kept & 0x3FFu;
    return (uint16_t)((neg ? 0x8000u : 0u) | ((uint32_t)e2 << 10) | mant);
}

static uint16_t sw_mpcubic_silu(uint16_t x) {
    uint32_t e = (x >> 10) & 0x1Fu;
    uint32_t m = x & 0x3FFu;
    if (e == 0x1Fu && m != 0) return 0x7E00u;  // NaN -> canonical
    float xv = sw_f16_to_f32(x);
    uint16_t y = x;
    if (xv > 4.0f)
        y = 0x4400u;
    else if (xv < -4.0f)
        y = 0xC400u;
    uint16_t t = sw_f16fma(0xAB28u, (uint16_t)(y & 0x7FFFu), 0x3728u);
    uint16_t u = sw_f16fma(y, t, 0x3B28u);
    return sw_f32_to_f16_bits(xv * sw_f16_to_f32(u));
}

// Manual f16 -> e4m3fn RN-satfinite (copied from stage1_prep.hip, GPU-verified
// 65536/65536; HD so the host mirror shares it bitwise).
HD inline uint8_t sw_h2e4m3(uint16_t u16) {
    uint8_t sign = (uint8_t)((u16 >> 15) & 1);
    uint16_t abs_u16 = u16 & 0x7FFFu;
    if (abs_u16 > 0x7C00u) return (uint8_t)((sign << 7) | 0x7F);
    if (abs_u16 >= 0x5EC0u) return (uint8_t)((sign << 7) | 0x7E);
    if (abs_u16 <= 0x1400u) return (uint8_t)((sign << 7) | 0x00);
    uint16_t exp = abs_u16 >> 10;
    uint16_t mant = abs_u16 & 0x3FFu;
    if (exp < 9) {
        uint32_t sig = 1024u + mant;
        int shift = 16 - (int)exp;
        uint32_t half_lsb = 1u << (shift - 1);
        uint32_t mm = (shift >= 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u);
        uint32_t rem = sig & mm;
        uint32_t res = sig >> shift;
        if (rem > half_lsb || (rem == half_lsb && (res & 1u))) res++;
        return (uint8_t)((sign << 7) | (uint8_t)res);
    }
    uint8_t e_8 = (uint8_t)(exp - 8);
    uint8_t m_8 = (uint8_t)(mant >> 7);
    uint16_t rem = mant & 0x7Fu;
    if (rem > 64 || (rem == 64 && (m_8 & 1))) {
        m_8++;
        if (m_8 == 8) {
            m_8 = 0;
            e_8++;
        }
    }
    return (uint8_t)((sign << 7) | (e_8 << 3) | m_8);
}

// e4m3 -> f32 dequant with H4 NaN-flush (matches tools/swin1h_ref.e4m3_decode).
HD inline float sw_e4m3_decode(uint8_t b) {
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    uint32_t s = (b >> 7) & 1u;
    uint32_t e = (b >> 3) & 0xFu;
    uint32_t m = b & 7u;
    float v;
    if (e == 0)
        v = (float)m / 512.0f;
    else {
        // (1 + m/8) * 2^(e-7); exact in f32 for all codes
        int e2 = (int)e - 7;
        v = (float)(8u + m) / 8.0f;
        v = (e2 >= 0) ? v * (float)(1u << e2) : v / (float)(1u << (-e2));
    }
    return s ? -v : v;
}

// Y[m][o] = sum_k X[m][k]*W[o][k], k ascending (W row-major (out,in)).
static void sw_gemm(const float* X, const float* W, float* Y, int M, int K,
                    int N) {
    for (int m = 0; m < M; m++)
        for (int o = 0; o < N; o++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += X[m * K + k] * W[o * K + k];
            Y[m * N + o] = acc;
        }
}

static uint32_t sw_fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x01000193u;
    }
    return h;
}

// NaN-normalising u32 compare: any NaN == any NaN (payloads differ across
// HW converters; the value class is what matters). Used by the GPU main;
// unused in HOST_ONLY builds.
__attribute__((unused)) static bool sw_f32eq(float a, float b) {
    if (a != a && b != b) return true;
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    return ua == ub;
}

// Synthetic vectors: MUST match tools/swin1h_ref.synth_inputs exactly
// (LCG(1664525,1013904223), (state>>8)%span + lo).
static void sw_synth(std::vector<float>& X16, std::vector<float>& Wq,
                     std::vector<float>& Wp, std::vector<float>& W1,
                     std::vector<float>& W2, std::vector<float>& Wpt) {
    uint32_t st = 0x12345678u;
    X16.resize(SW_TOK * SW_D_IN);
    for (size_t i = 0; i < X16.size(); i++) {
        st = sw_lcg_next(st);
        X16[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
    st = 0x9E3779B9u;
    Wq.resize(SW_QKV * SW_C);
    for (size_t i = 0; i < Wq.size(); i++) {
        st = sw_lcg_next(st);
        Wq[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
    Wp.resize(SW_C * SW_C);
    for (size_t i = 0; i < Wp.size(); i++) {
        st = sw_lcg_next(st);
        Wp[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
    W1.resize(SW_H1 * SW_C);
    for (size_t i = 0; i < W1.size(); i++) {
        st = sw_lcg_next(st);
        W1[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
    W2.resize(SW_C * SW_H1);
    for (size_t i = 0; i < W2.size(); i++) {
        st = sw_lcg_next(st);
        W2[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
    st = 0xC0FFEE11u;
    Wpt.resize(SW_C * SW_D_IN);
    for (size_t i = 0; i < Wpt.size(); i++) {
        st = sw_lcg_next(st);
        Wpt[i] = (float)(-1 + (int)((st >> 8) % 3u));
    }
}

// ---------------------------------------------------------------------------
// 1b. Entry-kernel front-end mirror (HANDOFF 13.3/path-1; gate F pins PTX).
// Host mirror (static): --fe-fnv checks share bitwise with the oracle on
// int-exact/f16-RN parts (sw_f16fma is exactly-rounded, like __hfma);
// Box-Muller/affine are tol (libm/approx). Device kernels live in
// stage1_prep.hip (__half ops, RN-equal on normal values).
// ---------------------------------------------------------------------------

static uint32_t fe_pcg_round(uint32_t st) {
    uint32_t xs = (st >> ((st >> 28) + 4u)) ^ st;
    return xs * 277803737u;
}
static uint32_t fe_pcg_out(uint32_t v) {
    return (((v >> 30) ^ (v >> 8)) + 1u);
}
// PTX immediates as unsigned (mul.lo/mad.lo wrap mod 2^32, like uint32_t).
static void fe_pcg_streams(uint32_t seed, uint32_t x, uint32_t y,
                              uint32_t o[4]) {
    uint32_t hseed = seed * 0x9E3779B9u;
    uint32_t s = (x * (0u - 1918454973u)) ^ hseed;
    uint32_t s1 = (y * (0u - 669632447u)) ^ s;
    uint32_t s2 = s1 ^ 608135816u;
    uint32_t f1 = fe_pcg_round(s2);
    f1 = ((f1 >> 22) ^ f1);
    const uint32_t C0[4] = {747796405u, (0u - 93469191u),
                            (0u - 895109107u), (0u - 2094846927u)};
    const uint32_t C1[4] = {(0u - 1403630843u), 1192405134u, 568162667u,
                            878960812u};
    for (int i = 0; i < 4; i++) {
        uint32_t v = f1 * C0[i] + C1[i];
        o[i] = fe_pcg_out(fe_pcg_round(v));
    }
}
// f16 normalize via single-rounding FMA (host: sw_f16fma; device kernels
// use __hsub/__hmul, bitwise-equal by correct rounding on normal values;
// subnormal inputs are an FTZ caveat, synthetic values are normal).
static uint16_t fe_normalize_bits(float tex, float half, float mult) {
    uint16_t t = sw_f32_to_f16_bits(tex);
    uint16_t h = sw_f32_to_f16_bits(half);
    uint16_t m = sw_f32_to_f16_bits(mult);
    uint16_t d = sw_f16fma(t, 0x3C00u, (uint16_t)(h ^ 0x8000u));
    return sw_f16fma(d, m, 0x0000u);
}
// Exposure packs. max.ftz NaN rule read from intent: BB1_7's ltu fallbacks
// require NaN exposure -> lo176 fallback, i.e. max returns the non-NaN
// operand (fmaxf semantics); both-NaN -> NaN -> p2 false -> -1.0 packs.
static void fe_exposure(float lo184, float hi184, float lo176,
                           float* s_sel, float* s_a, float* s_b) {
    float mx = fmaxf(lo184, hi184);
    bool p2 = mx >= 0.0f;
    *s_sel = p2 ? 1.0f : lo176;
    bool p38 = std::isnan(lo184) || (lo184 < 0.0f);
    bool p37 = std::isnan(hi184) || (hi184 < 0.0f);
    float r391 = p38 ? lo176 : lo184;
    float r392 = p37 ? lo176 : hi184;
    *s_a = p2 ? r391 : -1.0f;
    *s_b = p2 ? r392 : -1.0f;
}
// Box-Muller in float libm (tol vs oracle f64 / device approx).
static void fe_boxmuller_f32(float u1, float u2, float u3, float u4,
                                float* g0, float* g1, float* g2) {
    const float ln2 = 0.6931471824645996f;    // 0f3F317218
    const float two_pi = 6.283185307179586f;  // 0f40C90FDB
    float r1 = sqrtf(-2.0f * log2f(u1) * ln2);
    float r2 = sqrtf(-2.0f * log2(u3) * ln2);
    float t1 = u2 * two_pi;
    float t2 = u4 * two_pi;
    *g0 = r1 * cosf(t1);
    *g1 = r1 * sinf(t1);
    *g2 = r2 * cosf(t2);
}
// X[q][d] halves (gate-F F3 layout); tex/gauss float, params float.
static void fe_assemble_row(const float tex[3], const float gs[3],
                               float hi176, float s_sel, float s_a,
                               float s_b, uint16_t out[16]) {
    uint16_t n0 = fe_normalize_bits(tex[0], 0.5f, 1.0f);
    uint16_t n1 = fe_normalize_bits(tex[1], 0.5f, 1.0f);
    uint16_t n2 = fe_normalize_bits(tex[2], 0.5f, 1.0f);
    uint16_t h[8][2] = {
        {sw_f32_to_f16_bits(gs[0]), sw_f32_to_f16_bits(gs[1])},
        {sw_f32_to_f16_bits(gs[2]), 0x3C00u},
        {n0, n1},
        {n0, n2},
        {n1, n2},
        {sw_f32_to_f16_bits(hi176), sw_f32_to_f16_bits(s_sel)},
        {sw_f32_to_f16_bits(s_sel), sw_f32_to_f16_bits(s_a)},
        {sw_f32_to_f16_bits(s_b), 0x0000u},
    };
    const uint8_t slot[16] = {0, 0, 1, 1, 2, 2, 3, 3,
                                4, 4, 5, 5, 6, 6, 7, 7};
    for (int d = 0; d < 16; d++) out[d] = h[slot[d]][d & 1];
}
// Patch Y[q][o] = sum_d X[q][d]*W[o][d], k-ascending, f32.
static void fe_patch_f32(const float* X, const float* W, float* Y) {
    for (int q = 0; q < 64; q++)
        for (int o = 0; o < 32; o++) {
            float s = 0.0f;
            for (int d = 0; d < 16; d++) s += X[q * 16 + d] * W[o * 16 + d];
            Y[q * 32 + o] = s;
        }
}

#ifdef SWIN1H_HOST_ONLY
// Host-only verification entry points (g++ build; cross-checked against
// tools/swin1h_ref.py outputs).
static uint32_t host_mpcubic_fnv() {
    uint32_t h = 0x811C9DC5u;
    for (uint32_t u = 0; u < 65536u; u++) {
        uint16_t r = sw_mpcubic_silu((uint16_t)u);
        if ((r & 0x7C00u) == 0x7C00u && (r & 0x03FFu) != 0) r = 0x7E00u;
        h ^= (r & 0xFFu);
        h *= 0x01000193u;
        h ^= (r >> 8);
        h *= 0x01000193u;
    }
    return h;
}

static uint32_t host_quant_fnv() {
    uint32_t h = 0x811C9DC5u;
    for (uint32_t u = 0; u < 65536u; u++) {
        uint8_t r = sw_h2e4m3((uint16_t)u);
        h ^= r;
        h *= 0x01000193u;
    }
    return h;
}

static uint32_t host_fe_pcg_fnv() {
    uint32_t h = 0x811C9DC5u;
    for (int pas = 0; pas < 2; pas++)
        for (uint32_t wl = 0; wl < 32; wl++) {
            uint32_t o[4];
            fe_pcg_streams(0x12345678u, wl & 7u, wl >> 3, o);
            for (int i = 0; i < 4; i++)
                for (int b = 0; b < 4; b++) {
                    h ^= (uint8_t)(o[i] >> (8 * b));
                    h *= 0x01000193u;
                }
        }
    return h;
}

static uint32_t host_fe_xbits_fnv() {
    uint32_t h = 0x811C9DC5u;
    for (int pas = 0; pas < 2; pas++)
        for (uint32_t wl = 0; wl < 32; wl++) {
            uint32_t x = wl & 7u, y = (wl >> 3) + 4u * (uint32_t)pas;
            float tex[3] = {(float)x, (float)y, (float)(x + y)};
            float gs[3] = {1.5f, -2.25f, 0.75f};
            uint16_t row[16];
            fe_assemble_row(tex, gs, 3.0f, 1.0f, 0.0f, 0.0f, row);
            for (int d = 0; d < 16; d++) {
                uint32_t w = row[d];  // halves as u32 words (gate conv.)
                for (int b = 0; b < 4; b++) {
                    h ^= (uint8_t)(w >> (8 * b));
                    h *= 0x01000193u;
                }
            }
        }
    return h;
}

static uint32_t host_fe_patch_fnv() {
    float X[64 * 16], W[32 * 16], Y[64 * 32];
    for (int pas = 0; pas < 2; pas++)
        for (uint32_t wl = 0; wl < 32; wl++) {
            uint32_t x = wl & 7u, y = (wl >> 3) + 4u * (uint32_t)pas;
            float tex[3] = {(float)x, (float)y, (float)(x + y)};
            float gs[3] = {1.5f, -2.25f, 0.75f};
            uint16_t row[16];
            fe_assemble_row(tex, gs, 3.0f, 1.0f, 0.0f, 0.0f, row);
            int q = pas * 32 + (int)wl;
            for (int d = 0; d < 16; d++)
                X[q * 16 + d] = sw_f16_to_f32(row[d]);
        }
    for (int o = 0; o < 32; o++)
        for (int d = 0; d < 16; d++)
            W[o * 16 + d] = (float)((o * 16 + d) % 7 - 3);
    fe_patch_f32(X, W, Y);
    uint32_t h = 0x811C9DC5u;
    for (int i = 0; i < 64 * 32; i++) {
        uint32_t w;
        memcpy(&w, &Y[i], 4);
        for (int b = 0; b < 4; b++) {
            h ^= (uint8_t)(w >> (8 * b));
            h *= 0x01000193u;
        }
    }
    return h;
}

static void host_fe_bm_anchor() {
    uint32_t o[4];
    fe_pcg_streams(0x12345678u, 3u, 5u, o);
    float u[4];
    for (int i = 0; i < 4; i++)
        u[i] = (float)o[i] * 5.9604644775390625e-08f;  // 2^-24 exact
    float g0, g1, g2;
    fe_boxmuller_f32(u[0], u[1], u[2], u[3], &g0, &g1, &g2);
    std::printf("FE-BM-ANCHOR %a %a %a\n", g0, g1, g2);
    float ss, sa, sb;
    float nan = nanf("");
    fe_exposure(nan, 5.0f, 7.0f, &ss, &sa, &sb);
    uint32_t b0, b1, b2;
    memcpy(&b0, &ss, 4);
    memcpy(&b1, &sa, 4);
    memcpy(&b2, &sb, 4);
    std::printf("FE-EXPOSURE-NAN %08X %08X %08X\n", b0, b1, b2);
}

static uint32_t host_f32f16_fnv() {
    // deterministic bit-pattern sweep incl. subnormals/inf/nan
    uint32_t h = 0x811C9DC5u;
    uint32_t st = 0x5EEDu;
    for (int i = 0; i < 2000000; i++) {
        st = sw_lcg_next(st);
        float f;
        uint32_t u = st ^ (st >> 3);
        memcpy(&f, &u, 4);
        uint16_t r = sw_f32_to_f16_bits(f);
        if ((r & 0x7C00u) == 0x7C00u && (r & 0x03FFu) != 0) r = 0x7E00u;
        h ^= (r & 0xFFu);
        h *= 0x01000193u;
        h ^= (r >> 8);
        h *= 0x01000193u;
    }
    return h;
}

#endif  // host fnv helpers (host_checksums + main follow the chain below)

// Full mirror chain on synthetic vectors (production-faithful: inter-stage
// e4m3 quantisation included). Mirrors tools/swin1h_ref.linked_chain.
// Compiled in BOTH modes (GPU main compares against it).
struct SwChain {
    std::vector<uint8_t> Xe, Ye, Wq_e4, Wp_e4, W1_e4, W2_e4, Yf_e4;
    std::vector<uint16_t> Wpt_bits, Abits, Eb;
    std::vector<float> Ypatch, Xb, Yq, Qn, Kn, ssQ, ssK, S, P, O, Yp, Xh, H,
        Yf, Esum, B;
};

// Fast-exp softmax constants (HANDOFF §19): f16(cvt.rn.f32(PTX imm)).
#define SW_EXP_C1 0x29C0u  // 0.044921875 = 1.4375/32
#define SW_EXP_C2 0x3D34u  // 1.30078125
#define SW_EXP_LO 0x3C20u  // 1.03125 (E floor = 2^-14)
#define SW_EXP_HI 0x3E47u  // 1.5693359375 (E ceil = 9.75)
#define SW_EXP_MAGIC 0x7FF88000u

// Test temperature (matches tools/swin1h_ref.S_TEST bitwise as f32).
#define SW_S_TEST 0.5f
// Norm eps as max-clamp (HANDOFF §17), hex-float exact.
#define SW_EPS_NORM 0x1p-14f
// C=64 geometry (HANDOFF §49, Test-16): 2 heads x 64 tokens x 32 dims.
// (Rides the extractor expdefs block into the chain, like SW_TOK.)
#define SW_C64_HEADS 2
#define SW_C64_TOK 64
#define SW_C64_DIM 32
#define SW_C64_H1 224  // staged FFN width for the FALSIFIED Test-20 shape
                       // (§111/§113: real H1 is 128). Do NOT use in production.

// Phase A1 (HANDOFF §132): the stage's dimensions live here, once, instead of as
// literals scattered through the kernels. A different width means changing these
// and regenerating the chain -- the same route SW_C64_TOK/SW_DIM already take.
// SW_C64_W is the PASS width. It is now MEASURED, not guessed: the B region's
// size is B = 4C^2 + 8*C*W + pad(16, or 32 at C=256), which fits all four widths
// at W = 32 and fails at W = C/2 for C = 32, 128 and 256 (C=64 is the one case
// where the two coincide, which is why this went unnoticed). So W is a CONSTANT
// equal to headDim = C/heads = 32, not a function of C. See HANDOFF §135.
// SW_C64_H1P below is the production FFN width, per §111/§113.
#define SW_C64_C 64      // channels: the width of the template
#define SW_C64_W 32      // pass width = headDim, constant across widths (see above)
#define SW_C64_K 32      // the mma K-step. Equal to W at every fused width, but
                         // the ViT FFN is ONE dense pass with K still 32, so the
                         // two must not be the same symbol (S140).
#define SW_C64_H1P 128   // PRODUCTION FFN width (H1). MEASURED constant: the A
                         // region is heads*(C*H1 + H1*W + C*W), which fits
                         // C = 64/128/256 exactly at H1 = 128 (C=32 is the one
                         // structural outlier -- it omits the C*W term). §135.

// QK-norm rows (HANDOFF §17): per-token sumsq (f32, k-ascending), rsqrt
// factor, Q scaled by f16-rounded temperature s. Shared by the synthetic
// mirror and the real-weights chain.
static void sw_qknorm_rows(const float* Yq, float s_f16, float* Qn, float* Kn,
                           float* ssQ, float* ssK) {
    for (int m = 0; m < SW_TOK; m++) {
        float sq = 0.0f, sk = 0.0f;
        for (int d = 0; d < 32; d++) {
            float q = Yq[m * SW_QKV + d], k = Yq[m * SW_QKV + 32 + d];
            sq += q * q;
            sk += k * k;
        }
        ssQ[m] = sq;
        ssK[m] = sk;
        float fq = 1.0f / sqrtf(sq > SW_EPS_NORM ? sq : SW_EPS_NORM);
        float fk = 1.0f / sqrtf(sk > SW_EPS_NORM ? sk : SW_EPS_NORM);
        for (int d = 0; d < 32; d++) {
            Qn[m * SW_C + d] = s_f16 * Yq[m * SW_QKV + d] * fq;
            Kn[m * SW_C + d] = Yq[m * SW_QKV + 32 + d] * fk;
        }
    }
}

// Fast-exp softmax episode (§19): per-element f16 u + bit-trick E,
// k-ascending f16 row sums, f32 RN rcp rounded to f16, f16 normalize.
// The rcp here is RN; the device uses the hw approximation (probs stay
// TOL). Shared by the synthetic mirror and the real-weights chain.
static void sw_smexp_episode(const float* S, uint16_t* Eb, float* Esum,
                             float* P) {
    const float exp_lo = sw_f16_to_f32(SW_EXP_LO);
    const float exp_hi = sw_f16_to_f32(SW_EXP_HI);
    for (int i = 0; i < SW_TOK; i++) {
        for (int j = 0; j < SW_TOK; j++) {
            uint16_t x16 = sw_f32_to_f16_bits(S[i * SW_TOK + j]);
            // Single-rounding f16 fma (exact int128 path), like __hfma.
            uint16_t u16 = sw_f16fma(x16, SW_EXP_C1, SW_EXP_C2);
            float v = sw_f16_to_f32(u16);
            if (v < exp_lo) v = exp_lo;
            if (v > exp_hi) v = exp_hi;
            uint16_t w = sw_f32_to_f16_bits(v);
            uint32_t r = (((uint32_t)w << 16) | w);
            r = (r << 5) + SW_EXP_MAGIC;
            Eb[i * SW_TOK + j] = (uint16_t)(r & 0xFFFFu);
        }
        // K-ascending f16 sum via exact fma(x, 1.0) — single rounding, and
        // the device kernel uses the identical instruction sequence.
        uint16_t acc16 = 0x0000;
        for (int j = 0; j < SW_TOK; j++)
            acc16 = sw_f16fma(Eb[i * SW_TOK + j], 0x3C00u, acc16);
        float tot = sw_f16_to_f32(acc16);
        float sm = tot > SW_EPS_NORM ? tot : SW_EPS_NORM;
        uint16_t g16 = sw_f32_to_f16_bits(1.0f / sm);
        float gf = sw_f16_to_f32(g16);
        Esum[i] = tot;
        for (int j = 0; j < SW_TOK; j++)
            P[i * SW_TOK + j] = sw_f16_to_f32(
                sw_f32_to_f16_bits(sw_f16_to_f32(Eb[i * SW_TOK + j]) * gf));
    }
}

// C=64 attention scores (HANDOFF §49, Test-16): 2 heads x 64 tokens x
// 32 dims, per-head f32 temperatures, (2,64,64) f32 bias (decoded f16).
// TRAP: temp scales each Q element pre-dot (the kernel's 32 mul.f16x2
// sit on the mma-A path); the bias is the mma D-init and is NEVER
// scaled. Mirror and device kernel both run f32 in this exact op order
// (bitwise vs each other); the f64 oracle sits ~1e-4 away at |S|~90
// (tol 0.1). QKV slicing inside tensor [28832,41120) is SIZE INFERENCE
// only -- Test-16 stages the score-GEMM inputs (Q,K) directly.

// Staged score-GEMM inputs: MUST match tools/swin1h_ref.c64_staged_qk
// exactly (LCG stream from 0xC64001, levels [-1,-0.5,0,0.5,1] via f16
// staging quant, Q first then K, head-major [h][i][d]).
static void sw_c64_staged(std::vector<uint8_t>& Qe,
                          std::vector<uint8_t>& Ke) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64001u;
    Qe.resize(4096);
    Ke.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Qe[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Ke[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Staged O-GEMM inputs: MUST match tools/swin1h_ref.c64_staged_pv
// exactly (P stream from 0xC64002, V stream from 0xC64003, levels
// [-1,-0.5,0,0.5,1] via f16 staging quant, head-major).
static void sw_c64ctx_staged(std::vector<uint8_t>& Pqe,
                             std::vector<uint8_t>& Ve) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64002u;
    Pqe.resize(8192);
    Ve.resize(4096);
    for (int i = 0; i < 8192; i++) {
        st = sw_lcg_next(st);
        Pqe[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
    st = 0xC64003u;
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Ve[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-18 oracle recipe in C++ (HANDOFF §78E): two K-halves, f16
// rounding per step via the shared RN converters. Validates the
// transcription twin-vs-oracle (~1e-7); the device kernel does plain
// f32 full-K dots (k_c64scores precedent) at tol 0.1.
static void sw_c64ctx_ref(const float* Pq, const float* V, float* O) {
    for (int h = 0; h < SW_C64_HEADS; h++)
        for (int i = 0; i < SW_C64_TOK; i++)
            for (int n = 0; n < SW_C64_DIM; n++) {
                uint16_t acc = 0;
                for (int kh = 0; kh < 2; kh++) {
                    float s = 0.0f;
                    for (int k = 0; k < 32; k++)
                        s += Pq[((size_t)h * 64 + i) * 64 + kh * 32 + k] *
                             V[((size_t)h * 64 + kh * 32 + k) * 32 + n];
                    acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
                }
                O[((size_t)h * 64 + i) * 32 + n] = sw_f16_to_f32(acc);
            }
}

// Per-head scores in kernel op order: (t*q)*k per element, k-ascending
// dots, unscaled bias add. Shared by the host --c64fnv twin and (as the
// documented transliteration target) the k_c64scores device kernel.
static void sw_c64scores_ref(const float* Q, const float* K,
                             const float* ts, const float* B, float* S) {
    for (int h = 0; h < SW_C64_HEADS; h++)
        for (int i = 0; i < SW_C64_TOK; i++)
            for (int j = 0; j < SW_C64_TOK; j++) {
                float acc = 0.0f;
                for (int d = 0; d < SW_C64_DIM; d++)
                    acc += (ts[h] * Q[h * 2048 + i * 32 + d]) *
                           K[h * 2048 + j * 32 + d];
                S[h * 4096 + i * 64 + j] = acc + B[h * 4096 + i * 64 + j];
            }
}

static void sw_mirror_chain(SwChain& c) {
    std::vector<float> X16, Wq, Wp, W1, W2, Wpt;
    sw_synth(X16, Wq, Wp, W1, W2, Wpt);
    c.Xe.resize(SW_TOK * SW_D_IN);
    for (size_t i = 0; i < c.Xe.size(); i++)
        c.Xe[i] = sw_h2e4m3(sw_f32_to_f16_bits(X16[i]));  // staging mimic
    c.Wpt_bits.resize(SW_C * SW_D_IN);
    for (size_t i = 0; i < c.Wpt_bits.size(); i++)
        c.Wpt_bits[i] = sw_f32_to_f16_bits(Wpt[i]);
    std::vector<float> Wpt_f(SW_C * SW_D_IN);
    for (size_t i = 0; i < Wpt_f.size(); i++)
        Wpt_f[i] = sw_f16_to_f32(c.Wpt_bits[i]);
    std::vector<float> Xe_f(SW_TOK * SW_D_IN);
    for (size_t i = 0; i < Xe_f.size(); i++)
        Xe_f[i] = sw_e4m3_decode(c.Xe[i]);
    c.Ypatch.assign(SW_TOK * SW_C, 0.0f);
    sw_gemm(Xe_f.data(), Wpt_f.data(), c.Ypatch.data(), SW_TOK, SW_D_IN,
            SW_C);
    c.Ye.resize(SW_TOK * SW_C);
    for (size_t i = 0; i < c.Ye.size(); i++)
        c.Ye[i] = sw_h2e4m3(sw_f32_to_f16_bits(c.Ypatch[i]));
    c.Xb.assign(SW_TOK * SW_C, 0.0f);
    for (size_t i = 0; i < c.Xb.size(); i++)
        c.Xb[i] = sw_e4m3_decode(c.Ye[i]);
    // FFN half FIRST (HANDOFF §21: QKV-mma-A = quant(contract-chain-end),
    // i.e. quant(Yf) -- the old attn-first order is voided).
    c.W1_e4.resize(SW_H1 * SW_C);
    std::vector<float> W1_f(SW_H1 * SW_C);
    for (size_t i = 0; i < W1_f.size(); i++) {
        c.W1_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(W1[i]));
        W1_f[i] = sw_e4m3_decode(c.W1_e4[i]);
    }
    c.H.assign(SW_TOK * SW_H1, 0.0f);
    sw_gemm(c.Xb.data(), W1_f.data(), c.H.data(), SW_TOK, SW_C, SW_H1);
    c.Abits.resize(SW_TOK * SW_H1);
    std::vector<float> A(SW_TOK * SW_H1);
    for (size_t i = 0; i < A.size(); i++) {
        c.Abits[i] = sw_mpcubic_silu(sw_f32_to_f16_bits(c.H[i]));
        A[i] = sw_f16_to_f32(c.Abits[i]);
    }
    c.W2_e4.resize(SW_C * SW_H1);
    std::vector<float> W2_f(SW_C * SW_H1);
    for (size_t i = 0; i < W2_f.size(); i++) {
        c.W2_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(W2[i]));
        W2_f[i] = sw_e4m3_decode(c.W2_e4[i]);
    }
    std::vector<float> Yfraw(SW_TOK * SW_C, 0.0f);
    sw_gemm(A.data(), W2_f.data(), Yfraw.data(), SW_TOK, SW_H1, SW_C);
    c.Yf.assign(SW_TOK * SW_C, 0.0f);
    for (int m = 0; m < SW_TOK; m++)
        for (int o = 0; o < SW_C; o++)
            c.Yf[m * SW_C + o] =
                Yfraw[m * SW_C + o] + (float)((o + 1) % 2) * c.Xb[m * SW_C + o];
    c.Yf_e4.resize(SW_TOK * SW_C);
    c.Xh.assign(SW_TOK * SW_C, 0.0f);
    for (size_t i = 0; i < c.Xh.size(); i++) {
        c.Yf_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(c.Yf[i]));
        c.Xh[i] = sw_e4m3_decode(c.Yf_e4[i]);
    }
    // Synthetic bias stand-in: exact zeros (same +0.0 op as the device
    // k_scores add, so the -0.0/+0.0 edge converts identically on all three
    // sides; the real f16 bias only exists in --realweights).
    c.B.assign(SW_TOK * SW_TOK, 0.0f);
    c.Wq_e4.resize(SW_QKV * SW_C);
    std::vector<float> Wq_f(SW_QKV * SW_C);
    for (size_t i = 0; i < Wq_f.size(); i++) {
        c.Wq_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(Wq[i]));
        Wq_f[i] = sw_e4m3_decode(c.Wq_e4[i]);
    }
    c.Yq.assign(SW_TOK * SW_QKV, 0.0f);
    sw_gemm(c.Xh.data(), Wq_f.data(), c.Yq.data(), SW_TOK, SW_C, SW_QKV);
    // QK-norm (HANDOFF §17): per-token sumsq (exact on test vectors),
    // rsqrt factor, Q scaled by f16-rounded temperature.
    c.ssQ.assign(SW_TOK, 0.0f);
    c.ssK.assign(SW_TOK, 0.0f);
    c.Qn.assign(SW_TOK * SW_C, 0.0f);
    c.Kn.assign(SW_TOK * SW_C, 0.0f);
    sw_qknorm_rows(c.Yq.data(),
                   sw_f16_to_f32(sw_f32_to_f16_bits(SW_S_TEST)),
                   c.Qn.data(), c.Kn.data(), c.ssQ.data(), c.ssK.data());
    // Scores: NO 1/sqrt(32) (HANDOFF §19 — the kernel has none; the
    // QK-norm temperature s is the only scale). §21: raw dots + 64x64 f16
    // bias, NO scale (S = QK^T + B, row-major lane (i, j) += B[i*64+j]).
    c.S.assign(SW_TOK * SW_TOK, 0.0f);
    for (int i = 0; i < SW_TOK; i++)
        for (int j = 0; j < SW_TOK; j++) {
            float acc = 0.0f;
            for (int d = 0; d < 32; d++)
                acc += c.Qn[i * SW_C + d] * c.Kn[j * SW_C + d];
            c.S[i * SW_TOK + j] = acc + c.B[i * SW_TOK + j];
        }
    // Fast-exp softmax (§19): per-element f16 u + bit-trick E, k-ascending
    // f16 row sums, f32 RN rcp rounded to f16, f16 normalize. The rcp here
    // is RN; the device uses the hw approximation (probs stay TOL).
    c.Eb.assign(SW_TOK * SW_TOK, 0);
    c.Esum.assign(SW_TOK, 0.0f);
    c.P.assign(SW_TOK * SW_TOK, 0.0f);
    sw_smexp_episode(c.S.data(), c.Eb.data(), c.Esum.data(), c.P.data());
    c.O.assign(SW_TOK * SW_C, 0.0f);
    for (int i = 0; i < SW_TOK; i++)
        for (int d = 0; d < SW_C; d++) {
            float acc = 0.0f;
            for (int j = 0; j < SW_TOK; j++)
                acc += c.P[i * SW_TOK + j] * c.Yq[j * SW_QKV + 64 + d];
            c.O[i * SW_C + d] = acc;
        }
    c.Wp_e4.resize(SW_C * SW_C);
    std::vector<float> Wp_f(SW_C * SW_C);
    for (size_t i = 0; i < Wp_f.size(); i++) {
        c.Wp_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(Wp[i]));
        Wp_f[i] = sw_e4m3_decode(c.Wp_e4[i]);
    }
    std::vector<float> Ypraw(SW_TOK * SW_C, 0.0f);
    sw_gemm(c.O.data(), Wp_f.data(), Ypraw.data(), SW_TOK, SW_C, SW_C);
    // Block output: proj residual rides on Yf (the FFN-half out), not Xb.
    c.Yp.assign(SW_TOK * SW_C, 0.0f);
    for (int m = 0; m < SW_TOK; m++)
        for (int o = 0; o < SW_C; o++)
            c.Yp[m * SW_C + o] =
                Ypraw[m * SW_C + o] + (float)(o % 2) * c.Yf[m * SW_C + o];
}

// ---------------------------------------------------------------------------
// Real-weights chain (HANDOFF §24): the C++ twin of
// tools/swin1h_ref.cmd_realweights. Carves a layer tensor per the §21 table
// and runs the v4 FFN-first chain on the fixed LCG staging input.
// f32 throughout (same rounding as the synthetic mirror); the Python oracle
// runs f64, so final-checksum equality is NOT expected across languages —
// stages maxabs/finite/gates must agree, and the device proves against THIS
// reference (exact stages bitwise, post-rsqrtf stages tol). Known,
// verified cross-language effects (neither a bug): last-ulp f32/f64
// accumulation noise, plus e4m3 boundary-straddle amplification at the
// Yf→Xh quant (24/2048 Yf elements within 1e-4 of a code boundary on
// tensor_000) — maxabs agrees to ~0.1, washed out downstream by
// QK-norm/softmax.
// ---------------------------------------------------------------------------
struct RwChain {
    std::vector<uint8_t> Xe, Ye, Wq_e4, Wp_e4, W1_e4, W2_e4, Yf_e4;
    std::vector<uint16_t> Wpt_bits, Abits, Eb;
    std::vector<float> Ypatch, Xb, H, Yf, Yq, Qn, Kn, ssQ, ssK, S, Esum, P,
        O, Yp, B, ga, gf;
    float s_real;
    bool has_patch;
};

static bool sw_read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t)n);
    size_t got = std::fread(out.data(), 1, (size_t)n, f);
    std::fclose(f);
    return got == (size_t)n;
}

static float sw_maxabs_f(const std::vector<float>& v) {
    float m = 0.0f;  // real-weights runs are finite (see printout)
    for (size_t i = 0; i < v.size(); i++) {
        float a = fabsf(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// FNV-1a over the LE f32 bytes of the finite elements (matches the oracle's
// fnv1a_hex over [v for v in flat(Y) if isfinite(v)]).
static uint32_t sw_fnv_finite_f32(const std::vector<float>& v) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) continue;
        uint32_t u;
        std::memcpy(&u, &v[i], 4);
        for (int b = 0; b < 4; b++) {
            h ^= (uint32_t)((u >> (8 * b)) & 0xFFu);
            h *= 0x01000193u;
        }
    }
    return h;
}

static bool sw_rw_chain(const std::vector<uint8_t>& raw, bool is_pre,
                        RwChain& r) {
    if (is_pre ? raw.size() != 21696 : raw.size() != 20672) return false;
    const size_t o_ffn2 = 4096;  // both tables (§21)
    const size_t o_gate_ffn = is_pre ? 9232 : 8208;
    const size_t o_qkv = is_pre ? 9312 : 8288;
    // §27: bias was transposed between the tables; correct is pre @12384
    // (abuts P-scale @20576), fused @11360 (abuts P-scale @19552).
    const size_t o_bias = is_pre ? 12384 : 11360;
    const size_t o_s = is_pre ? 20576 : 19552;
    const size_t o_proj = is_pre ? 20592 : 19568;
    const size_t o_gate_attn = is_pre ? 21616 : 20592;
    r.has_patch = is_pre;

    r.W1_e4.assign(raw.begin(), raw.begin() + SW_H1 * SW_C);
    std::vector<float> W1_f(SW_H1 * SW_C);
    for (size_t i = 0; i < W1_f.size(); i++)
        W1_f[i] = sw_e4m3_decode(r.W1_e4[i]);
    r.W2_e4.assign(raw.begin() + o_ffn2,
                   raw.begin() + o_ffn2 + SW_C * SW_H1);
    std::vector<float> W2_f(SW_C * SW_H1);
    for (size_t i = 0; i < W2_f.size(); i++)
        W2_f[i] = sw_e4m3_decode(r.W2_e4[i]);
    r.Wq_e4.assign(raw.begin() + o_qkv,
                   raw.begin() + o_qkv + SW_QKV * SW_C);
    std::vector<float> Wq_f(SW_QKV * SW_C);
    for (size_t i = 0; i < Wq_f.size(); i++)
        Wq_f[i] = sw_e4m3_decode(r.Wq_e4[i]);
    r.Wp_e4.assign(raw.begin() + o_proj,
                   raw.begin() + o_proj + SW_C * SW_C);
    std::vector<float> Wp_f(SW_C * SW_C);
    for (size_t i = 0; i < Wp_f.size(); i++)
        Wp_f[i] = sw_e4m3_decode(r.Wp_e4[i]);
    r.gf.assign(SW_C, 0.0f);
    r.ga.assign(SW_C, 0.0f);
    for (int o = 0; o < SW_C; o++) {
        uint16_t b;
        std::memcpy(&b, &raw[o_gate_ffn + 2 * (size_t)o], 2);
        r.gf[o] = sw_f16_to_f32(b);
        std::memcpy(&b, &raw[o_gate_attn + 2 * (size_t)o], 2);
        r.ga[o] = sw_f16_to_f32(b);
    }
    r.B.assign(SW_TOK * SW_TOK, 0.0f);
    for (int i = 0; i < SW_TOK * SW_TOK; i++) {
        uint16_t b;
        std::memcpy(&b, &raw[o_bias + 2 * (size_t)i], 2);
        r.B[i] = sw_f16_to_f32(b);
    }
    std::memcpy(&r.s_real, &raw[o_s], 4);

    // Staging-like e4m3 input: LCG(0xA11CE) mags 0..4 → fixed levels.
    static const float kLevels[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    r.Xe.assign(SW_TOK * SW_D_IN, 0);
    uint32_t st = 0xA11CEu;
    for (size_t i = 0; i < r.Xe.size(); i++) {
        st = sw_lcg_next(st);
        r.Xe[i] = sw_h2e4m3(sw_f32_to_f16_bits(kLevels[(st >> 8) % 5u]));
    }

    r.Xb.assign(SW_TOK * SW_C, 0.0f);
    if (is_pre) {
        r.Wpt_bits.resize(SW_C * SW_D_IN);
        for (size_t i = 0; i < r.Wpt_bits.size(); i++) {
            uint16_t b;
            std::memcpy(&b, &raw[8208 + 2 * i], 2);
            r.Wpt_bits[i] = b;
        }
        std::vector<float> Wpt_f(SW_C * SW_D_IN);
        for (size_t i = 0; i < Wpt_f.size(); i++)
            Wpt_f[i] = sw_f16_to_f32(r.Wpt_bits[i]);
        std::vector<float> Xe_f(SW_TOK * SW_D_IN);
        for (size_t i = 0; i < Xe_f.size(); i++)
            Xe_f[i] = sw_e4m3_decode(r.Xe[i]);
        r.Ypatch.assign(SW_TOK * SW_C, 0.0f);
        sw_gemm(Xe_f.data(), Wpt_f.data(), r.Ypatch.data(), SW_TOK, SW_D_IN,
                SW_C);
        r.Ye.resize(SW_TOK * SW_C);
        for (size_t i = 0; i < r.Ye.size(); i++)
            r.Ye[i] = sw_h2e4m3(sw_f32_to_f16_bits(r.Ypatch[i]));
        for (size_t i = 0; i < r.Xb.size(); i++)
            r.Xb[i] = sw_e4m3_decode(r.Ye[i]);
    } else {
        // No patch region: pair two 16-dim rows into one 32-dim input
        // (CPU-smoke fabrication — no device counterpart; see §24).
        for (int i = 0; i < SW_TOK; i++)
            for (int k = 0; k < SW_D_IN; k++) {
                r.Xb[i * SW_C + k] =
                    sw_e4m3_decode(r.Xe[i * SW_D_IN + k]);
                r.Xb[i * SW_C + SW_D_IN + k] = sw_e4m3_decode(
                    r.Xe[((i + 1) % SW_TOK) * SW_D_IN + k]);
            }
    }

    // FFN half first (§21 order).
    r.H.assign(SW_TOK * SW_H1, 0.0f);
    sw_gemm(r.Xb.data(), W1_f.data(), r.H.data(), SW_TOK, SW_C, SW_H1);
    r.Abits.resize(SW_TOK * SW_H1);
    std::vector<float> A(SW_TOK * SW_H1);
    for (size_t i = 0; i < A.size(); i++) {
        r.Abits[i] = sw_mpcubic_silu(sw_f32_to_f16_bits(r.H[i]));
        A[i] = sw_f16_to_f32(r.Abits[i]);
    }
    std::vector<float> Yfraw(SW_TOK * SW_C, 0.0f);
    sw_gemm(A.data(), W2_f.data(), Yfraw.data(), SW_TOK, SW_H1, SW_C);
    r.Yf.assign(SW_TOK * SW_C, 0.0f);
    for (int m = 0; m < SW_TOK; m++)
        for (int o = 0; o < SW_C; o++)
            r.Yf[m * SW_C + o] =
                Yfraw[m * SW_C + o] + r.gf[o] * r.Xb[m * SW_C + o];
    r.Yf_e4.resize(SW_TOK * SW_C);
    std::vector<float> Xh(SW_TOK * SW_C);
    for (size_t i = 0; i < Xh.size(); i++) {
        r.Yf_e4[i] = sw_h2e4m3(sw_f32_to_f16_bits(r.Yf[i]));
        Xh[i] = sw_e4m3_decode(r.Yf_e4[i]);
    }

    r.Yq.assign(SW_TOK * SW_QKV, 0.0f);
    sw_gemm(Xh.data(), Wq_f.data(), r.Yq.data(), SW_TOK, SW_C, SW_QKV);
    r.ssQ.assign(SW_TOK, 0.0f);
    r.ssK.assign(SW_TOK, 0.0f);
    r.Qn.assign(SW_TOK * SW_C, 0.0f);
    r.Kn.assign(SW_TOK * SW_C, 0.0f);
    sw_qknorm_rows(r.Yq.data(), sw_f16_to_f32(sw_f32_to_f16_bits(r.s_real)),
                   r.Qn.data(), r.Kn.data(), r.ssQ.data(), r.ssK.data());
    r.S.assign(SW_TOK * SW_TOK, 0.0f);
    for (int i = 0; i < SW_TOK; i++)
        for (int j = 0; j < SW_TOK; j++) {
            float acc = 0.0f;
            for (int d = 0; d < 32; d++)
                acc += r.Qn[i * SW_C + d] * r.Kn[j * SW_C + d];
            r.S[i * SW_TOK + j] = acc + r.B[i * SW_TOK + j];
        }
    r.Eb.assign(SW_TOK * SW_TOK, 0);
    r.Esum.assign(SW_TOK, 0.0f);
    r.P.assign(SW_TOK * SW_TOK, 0.0f);
    sw_smexp_episode(r.S.data(), r.Eb.data(), r.Esum.data(), r.P.data());
    r.O.assign(SW_TOK * SW_C, 0.0f);
    for (int i = 0; i < SW_TOK; i++)
        for (int d = 0; d < SW_C; d++) {
            float acc = 0.0f;
            for (int j = 0; j < SW_TOK; j++)
                acc += r.P[i * SW_TOK + j] * r.Yq[j * SW_QKV + 64 + d];
            r.O[i * SW_C + d] = acc;
        }
    std::vector<float> Ypraw(SW_TOK * SW_C, 0.0f);
    sw_gemm(r.O.data(), Wp_f.data(), Ypraw.data(), SW_TOK, SW_C, SW_C);
    r.Yp.assign(SW_TOK * SW_C, 0.0f);
    for (int m = 0; m < SW_TOK; m++)
        for (int o = 0; o < SW_C; o++)
            r.Yp[m * SW_C + o] =
                Ypraw[m * SW_C + o] + r.ga[o] * r.Yf[m * SW_C + o];
    return true;
}

#ifdef SWIN1H_HOST_ONLY
#include "rw_yp_c64s_golden.inc"
#include "rw_yp_c64e_golden.inc"
#include "rw_yp_c64o_golden.inc"
#include "rw_yp_c64p_golden.inc"
#include "rw_yp_c64f_golden.inc"
#include "rw_yp_c64q_golden.inc"
#include "rw_yp_c64c_golden.inc"
#include "rw_yp_c64x_golden.inc"
#include "rw_yp_c64f2_golden.inc"

static void host_checksums() {
    // FNVs of the exact stages for cross-check vs tools/swin1h_ref.py
    // --stage-fnv (must match bitwise).
    SwChain c;
    sw_mirror_chain(c);
    std::printf("patch_raw %08X\n", sw_fnv(c.Ypatch.data(), c.Ypatch.size() * 4));
    std::printf("patch_bytes %08X\n", sw_fnv(c.Ye.data(), c.Ye.size()));
    std::printf("qkv %08X\n", sw_fnv(c.Yq.data(), c.Yq.size() * 4));
    std::printf("sumsqQ %08X\n", sw_fnv(c.ssQ.data(), c.ssQ.size() * 4));
    std::printf("sumsqK %08X\n", sw_fnv(c.ssK.data(), c.ssK.size() * 4));
    std::printf("smexp %08X\n", sw_fnv(c.Eb.data(), c.Eb.size() * 2));
    std::printf("H %08X\n", sw_fnv(c.H.data(), c.H.size() * 4));
    std::printf("actbits %08X\n", sw_fnv(c.Abits.data(), c.Abits.size() * 2));
}

static int sw_gate_in01(const std::vector<float>& g) {
    int n = 0;
    for (size_t i = 0; i < g.size(); i++)
        if (g[i] > 0.0f && g[i] <= 1.0f) n++;
    return n;
}

// Same lines as tools/swin1h_ref --realweights (same order, same formats)
// for eyeball diffing; checksums are f32-side (see RwChain note).
// Host twin for Test-16/17 (HANDOFF §49-§51): the staged-Q/K stream twin
// (must be EXACT vs the golden inputs), mirror scores vs kC64SGolden
// (maxerr predicts the device Test-16 number; f32-vs-f64 ~1e-4), and the
// Test-17 episode on the staged S words: Eb EXACT (integer math,
// straddle-free by construction), P at tol.
static void host_c64fnv_cmd(const char* tensordir) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s", tensordir, "tensor_137.bin");
    std::vector<uint8_t> raw;
    bool okread = sw_read_file(path, raw);
    bool ok = okread && raw.size() == 61760;
    std::printf("tensor_137.bin: %d bytes (expect 61760) %s\n",
                okread ? (int)raw.size() : -1, ok ? "OK" : "SIZE MISMATCH");
    if (!ok) return;
    float t0, t1;
    memcpy(&t0, raw.data() + 57504, 4);
    memcpy(&t1, raw.data() + 57508, 4);
    std::printf("  temps = %.6f %.6f\n", t0, t1);
    std::vector<float> B(8192);
    for (int i = 0; i < 8192; i++) {
        uint16_t u = (uint16_t)raw[(size_t)41120 + (size_t)2 * (size_t)i] |
                   ((uint16_t)raw[(size_t)41120 + (size_t)2 * (size_t)i + 1] << 8);
        B[(size_t)i] = sw_f16_to_f32(u);
    }
    std::vector<uint8_t> Qe, Ke;
    sw_c64_staged(Qe, Ke);
    int qbad = 0, kbad = 0;
    for (int i = 0; i < 4096; i++) {
        if (Qe[(size_t)i] != kC64QeBytes[i]) qbad++;
        if (Ke[(size_t)i] != kC64KeBytes[i]) kbad++;
    }
    std::printf("  staged stream twin: Qe %d/4096 diffs, Ke %d/4096 diffs\n",
                qbad, kbad);
    std::vector<float> Q(4096), K(4096);
    for (int i = 0; i < 4096; i++) {
        Q[(size_t)i] = sw_e4m3_decode(Qe[(size_t)i]);
        K[(size_t)i] = sw_e4m3_decode(Ke[(size_t)i]);
    }
    std::vector<float> S(8192);
    const float ts2[2] = {t0, t1};
    sw_c64scores_ref(Q.data(), K.data(), ts2, B.data(), S.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 8192; i++) {
        float want;
        memcpy(&want, &kC64SGolden[i], 4);
        float d = S[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  S: maxerr %.6g maxabs %.4f nonfinite %d/8192 FNV %08X\n",
                me, mx, nn, sw_fnv(S.data(), S.size() * 4));
    std::vector<float> Sin(8192);
    for (int i = 0; i < 8192; i++) memcpy(&Sin[(size_t)i], &kC64SBytes[i], 4);
    std::vector<uint16_t> Eb(8192);
    std::vector<float> Esum(128), P(8192);
    for (int h = 0; h < 2; h++)
        sw_smexp_episode(Sin.data() + h * 4096, Eb.data() + h * 4096,
                         Esum.data() + h * 64, P.data() + h * 4096);
    int ebad = 0;
    for (int i = 0; i < 8192; i++)
        if (Eb[(size_t)i] != kC64EbGolden[i]) ebad++;
    float mep = 0.0f;
    int nnp = 0;
    for (int i = 0; i < 8192; i++) {
        float want;
        memcpy(&want, &kC64PGolden[i], 4);
        float d = P[(size_t)i] - want;
        if (d != d) { nnp++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > mep) mep = a;
    }
    std::printf("  Eb: %d/8192 mismatches (exact) FNV %08X\n",
                ebad, sw_fnv(Eb.data(), Eb.size() * 2));
    std::printf("  P: maxerr %.6g nonfinite %d/8192 FNV %08X\n",
                mep, nnp, sw_fnv(P.data(), P.size() * 4));
}

// Host twin for Test-18 (HANDOFF §78/§80): the staged-P/V stream twin
// (must be EXACT vs the golden inputs), mirror O vs kC64OGolden (gap
// predicts the device Test-18 number; f32-vs-f64 ~1e-7). No weights.
static void host_c64ofnv_cmd() {
    std::vector<uint8_t> Pqe, Ve;
    sw_c64ctx_staged(Pqe, Ve);
    int pbad = 0, vbad = 0;
    for (int i = 0; i < 8192; i++)
        if (Pqe[(size_t)i] != kC64PqBytes[i]) pbad++;
    for (int i = 0; i < 4096; i++)
        if (Ve[(size_t)i] != kC64VBytes[i]) vbad++;
    std::printf("  staged stream twin: Pqe %d/8192 diffs, Ve %d/4096 diffs\n",
                pbad, vbad);
    std::vector<float> Pq(8192), V(4096);
    for (int i = 0; i < 8192; i++)
        Pq[(size_t)i] = sw_e4m3_decode(Pqe[(size_t)i]);
    for (int i = 0; i < 4096; i++)
        V[(size_t)i] = sw_e4m3_decode(Ve[(size_t)i]);
    std::vector<float> O(4096);
    sw_c64ctx_ref(Pq.data(), V.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 4096; i++) {
        float want;
        memcpy(&want, &kC64OGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  O: maxerr %.6g maxabs %.4f nonfinite %d/4096 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}

// Staged proj inputs: MUST match tools/swin1h_ref.c64_staged_oy
// exactly (Ocat stream from 0xC64004, y stream from 0xC64005, levels
// [-1,-0.5,0,0.5,1] via f16 staging quant, row-major [64][64]).
static void sw_c64proj_staged(std::vector<uint8_t>& Ob,
                              std::vector<uint8_t>& Yb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64004u;
    Ob.resize(4096);
    Yb.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Ob[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
    st = 0xC64005u;
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Yb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-19 oracle recipe in C++ (HANDOFF §78G): plain f64-style sums
// (f32 here; gap ~1e-6) + gate add, mirroring C=32 proj_forward.
static void sw_c64proj_ref(const float* Ocat, const float* Wp,
                           const float* G2, const float* Yb, float* O) {
    for (int m = 0; m < 64; m++)
        for (int o = 0; o < 64; o++) {
            float acc = 0.0f;
            for (int c = 0; c < 64; c++)
                acc += Ocat[(size_t)m * 64 + c] * Wp[(size_t)c * 64 + o];
            O[(size_t)m * 64 + o] = acc + G2[o] * Yb[(size_t)m * 64 + o];
        }
}

// Host twin for Test-19 (HANDOFF §78/§80): staged-O/y stream
// twin (EXACT vs golden inputs) + real Wproj/gate2 dump reads + mirror O
// vs kC64ProjGolden. Needs tensordir (real weights, Test-16 path).
static void host_c64pfnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Ob, Yb;
    sw_c64proj_staged(Ob, Yb);
    int obad = 0, ybad = 0;
    for (int i = 0; i < 4096; i++)
        if (Ob[(size_t)i] != kC64ObBytes[i]) obad++;
    for (int i = 0; i < 4096; i++)
        if (Yb[(size_t)i] != kC64YBytes[i]) ybad++;
    std::printf("  staged stream twin: Ob %d/4096 diffs, Yb %d/4096 diffs\n",
                obad, ybad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> Ocat(4096), Wp(4096), G2(64), Y(4096);
    for (int i = 0; i < 4096; i++)
        Ocat[(size_t)i] = sw_e4m3_decode(Ob[(size_t)i]);
    for (int i = 0; i < 4096; i++)
        Wp[(size_t)i] = sw_e4m3_decode(dump[(size_t)57520 + i]);
    for (int o = 0; o < 64; o++) {
        uint16_t u = (uint16_t)dump[(size_t)61616 + o * 2] |
                     ((uint16_t)dump[(size_t)61616 + o * 2 + 1] << 8);
        G2[(size_t)o] = sw_f16_to_f32(u);
    }
    for (int i = 0; i < 4096; i++)
        Y[(size_t)i] = sw_e4m3_decode(Yb[(size_t)i]);
    std::vector<float> O(4096);
    sw_c64proj_ref(Ocat.data(), Wp.data(), G2.data(), Y.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 4096; i++) {
        float want;
        memcpy(&want, &kC64ProjGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  O: maxerr %.6g maxabs %.4f nonfinite %d/4096 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}

// Staged FFN input: MUST match tools/swin1h_ref.c64_staged_x exactly
// (x stream from 0xC64006, levels [-1,-0.5,0,0.5,1] via f16 staging
// quant, row-major [64][64]).
static void sw_c64ffn_staged(std::vector<uint8_t>& Xb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64006u;
    Xb.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Xb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-20 oracle recipe in C++ (HANDOFF S78H): plain f64-style sums
// (f32 here; gap ~1e-6) + f16-rounded expand + bit-exact act + gate
// add, mirroring C=32 ffn_forward at 64-wide with H1=224.
static void sw_c64ffn_ref(const float* X, const float* W1,
                          const float* W2, const float* G1, float* O) {
    for (int m = 0; m < 64; m++) {
        float Arow[224];
        for (int j = 0; j < 224; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 64; k++)
                acc += X[(size_t)m * 64 + k] * W1[(size_t)j * 64 + k];
            Arow[j] = sw_f16_to_f32(
                sw_mpcubic_silu(sw_f32_to_f16_bits(acc)));
        }
        for (int o = 0; o < 64; o++) {
            float acc = 0.0f;
            for (int j = 0; j < 224; j++)
                acc += Arow[j] * W2[(size_t)o * 224 + j];
            O[(size_t)m * 64 + o] =
                acc + G1[o] * X[(size_t)m * 64 + o];
        }
    }
}

// Host twin for Test-20 (HANDOFF S78H): staged-x stream twin (EXACT
// vs golden inputs) + real w1/w2/gate1 dump reads + mirror Y vs
// kC64FfnGolden. Needs tensordir (real weights, Test-16 path).
static void host_c64ffnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Xb;
    sw_c64ffn_staged(Xb);
    int xbad = 0;
    for (int i = 0; i < 4096; i++)
        if (Xb[(size_t)i] != kC64XBytes[i]) xbad++;
    std::printf("  staged stream twin: Xb %d/4096 diffs\n", xbad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> X(4096), W1(14336), W2(14336), G1(64);
    for (int i = 0; i < 4096; i++)
        X[(size_t)i] = sw_e4m3_decode(Xb[(size_t)i]);
    for (int i = 0; i < 14336; i++)
        W1[(size_t)i] = sw_e4m3_decode(dump[(size_t)i]);
    for (int i = 0; i < 14336; i++)
        W2[(size_t)i] = sw_e4m3_decode(dump[(size_t)14336 + i]);
    for (int o = 0; o < 64; o++) {
        uint16_t u = (uint16_t)dump[(size_t)28688 + o * 2] |
                     ((uint16_t)dump[(size_t)28688 + o * 2 + 1] << 8);
        G1[(size_t)o] = sw_f16_to_f32(u);
    }
    std::vector<float> O(4096);
    sw_c64ffn_ref(X.data(), W1.data(), W2.data(), G1.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 4096; i++) {
        float want;
        memcpy(&want, &kC64FfnGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  Y: maxerr %.6g maxabs %.4f nonfinite %d/4096 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}
// Staged QKV-GEMM input: MUST match tools/swin1h_ref.c64_staged_qin
// exactly (X stream from 0xC64007, levels [-1,-0.5,0,0.5,1] via f16
// staging quant, row-major [64][64]).
static void sw_c64qkv_staged(std::vector<uint8_t>& Xb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64007u;
    Xb.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Xb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-21 oracle recipe in C++ (HANDOFF S90C): per head, two K-halves
// with f16 rounding per step (production mma-D contract, S78E). W is
// flat [kh][h][32][96] e4m3 (K-half-major, S90A). Validates the
// transcription twin-vs-oracle (~1e-7); the device kernel does plain
// f32 full-K dots (k_c64proj precedent) at tol 0.1.
static void sw_c64qkv_ref(const float* X, const float* W, float* O) {
    for (int h = 0; h < 2; h++)
        for (int m = 0; m < 64; m++)
            for (int n = 0; n < 96; n++) {
                uint16_t acc = 0;
                for (int kh = 0; kh < 2; kh++) {
                    float s = 0.0f;
                    for (int k = 0; k < 32; k++)
                        s += X[(size_t)m * 64 + kh * 32 + k] *
                             W[((size_t)kh * 2 + h) * 3072 + k * 96 + n];
                    acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
                }
                O[((size_t)h * 64 + m) * 96 + n] = sw_f16_to_f32(acc);
            }
}

// Host twin for Test-21 (HANDOFF S90): staged-X stream twin (EXACT
// vs golden inputs) + real Wqkv dump read + mirror QKV vs
// kC64QkvGolden. Needs tensordir (real weights, Test-16 path).
static void host_c64qfnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Xb;
    sw_c64qkv_staged(Xb);
    int xbad = 0;
    for (int i = 0; i < 4096; i++)
        if (Xb[(size_t)i] != kC64QinBytes[i]) xbad++;
    std::printf("  staged stream twin: Xb %d/4096 diffs\n", xbad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> X(4096), W(12288);
    for (int i = 0; i < 4096; i++)
        X[(size_t)i] = sw_e4m3_decode(Xb[(size_t)i]);
    for (int i = 0; i < 12288; i++)
        W[(size_t)i] = sw_e4m3_decode(dump[(size_t)28832 + i]);
    std::vector<float> O(12288);
    sw_c64qkv_ref(X.data(), W.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 12288; i++) {
        float want;
        memcpy(&want, &kC64QkvGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  QKV: maxerr %.6g maxabs %.4f nonfinite %d/12288 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}
// Staged FFN-contract input: MUST match tools/swin1h_ref.c64_staged_hin
// exactly (h stream from 0xC64008, levels [-1,-0.5,0,0.5,1] via f16
// staging quant, row-major [64][128]).
static void sw_c64contract_staged(std::vector<uint8_t>& Hb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64008u;
    Hb.resize(8192);
    for (int i = 0; i < 8192; i++) {
        st = sw_lcg_next(st);
        Hb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-22 oracle recipe in C++ (HANDOFF §111): h[64][128] x W2[128][32]
// -> [64][32], K=128 in FOUR k-steps of 32 with f16 rounding per step.
// The 4-step structure is the production D->C chain read off the kernel
// (§110), same contract as Test-21's two-step oracle. W2 is flat
// [128][32] e4m3 (rows = K, row-major). No act/bias/gate/residual.
static void sw_c64contract_ref(const float* H, const float* W, float* O) {
    for (int m = 0; m < 64; m++)
        for (int n = 0; n < 32; n++) {
            uint16_t acc = 0;
            for (int kh = 0; kh < 4; kh++) {
                float s = 0.0f;
                for (int k = 0; k < 32; k++)
                    s += H[(size_t)m * 128 + kh * 32 + k] *
                         W[(size_t)(kh * 32 + k) * 32 + n];
                acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
            }
            O[(size_t)m * 32 + n] = sw_f16_to_f32(acc);
        }
}

// Host twin for Test-22 (HANDOFF §111): staged-h stream twin (EXACT
// vs golden inputs) + real W2 dump read + mirror contract vs
// kC64ContractGolden. Needs tensordir (real weights, Test-16 path).
static void host_c64cfnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Hb;
    sw_c64contract_staged(Hb);
    int hbad = 0;
    for (int i = 0; i < 8192; i++)
        if (Hb[(size_t)i] != kC64HinBytes[i]) hbad++;
    std::printf("  staged stream twin: Hb %d/8192 diffs\n", hbad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> H(8192), W(4096);
    for (int i = 0; i < 8192; i++)
        H[(size_t)i] = sw_e4m3_decode(Hb[(size_t)i]);
    for (int i = 0; i < 4096; i++)
        W[(size_t)i] = sw_e4m3_decode(dump[(size_t)16384 + i]);
    std::vector<float> O(2048);
    sw_c64contract_ref(H.data(), W.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 2048; i++) {
        float want;
        memcpy(&want, &kC64ContractGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  CONTRACT: maxerr %.6g maxabs %.4f nonfinite %d/2048 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}

// Staged connected-FFN input (Test-24, HANDOFF S116): MUST match
// tools/swin1h_ref.c64_staged_ffnin exactly (x stream 0xC6400A, levels
// [-1,-0.5,0,0.5,1] via f16 staging quant, row-major [64][64]).
static void sw_c64ffn2_staged(std::vector<uint8_t>& Xb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC6400Au;
    Xb.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Xb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-24 oracle recipe in C++ (HANDOFF S116): the CONNECTED C=64 FFN.
// x[64][64] -> expand (K=64, TWO f16 k-steps) -> act -> e4m3 -> contract
// (K=128, FOUR f16 k-steps) per pass o<32 / o>=32, + gate1 residual.
// w1a/w1b = dump +0/+8192 [64][128] IN-MAJOR; w2a/w2b = dump
// +16384/+20480 [128][32] IN-MAJOR; G1 = 64 f16 at dump +28688.
static void sw_c64ffn2_ref(const float* X, const float* W1a, const float* W1b,
                           const float* W2a, const float* W2b,
                           const uint16_t* G1, float* O) {
    for (int m = 0; m < 64; m++)
        for (int o = 0; o < 64; o++) {
            const float* W1 = (o < 32) ? W1a : W1b;
            const float* W2 = (o < 32) ? W2a : W2b;
            int n = o % 32;
            float H[128];
            for (int j = 0; j < 128; j++) {
                uint16_t acc = 0;
                for (int kh = 0; kh < 2; kh++) {
                    float s = 0.0f;
                    for (int k = 0; k < 32; k++)
                        s += X[(size_t)m * 64 + kh * 32 + k] *
                             W1[(size_t)(kh * 32 + k) * 128 + j];
                    acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
                }
                H[j] = sw_f16_to_f32(acc);
            }
            uint8_t aq[128];
            for (int j = 0; j < 128; j++)
                aq[j] = sw_h2e4m3(sw_mpcubic_silu(sw_f32_to_f16_bits(H[j])));
            uint16_t acc = 0;
            for (int kh = 0; kh < 4; kh++) {
                float s = 0.0f;
                for (int k = 0; k < 32; k++)
                    s += sw_e4m3_decode(aq[kh * 32 + k]) *
                         W2[(size_t)(kh * 32 + k) * 32 + n];
                acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
            }
            float y = sw_f16_to_f32(acc);
            O[(size_t)m * 64 + o] =
                y + sw_f16_to_f32(G1[o]) * X[(size_t)m * 64 + o];
        }
}

// Host twin for Test-24 (HANDOFF S116): staged-x twin (EXACT vs golden
// input) + real w1/w2/gate1 dump reads + mirror connected FFN vs
// kC64Ffn2Golden. Needs tensordir (real weights, Test-16 path).
static void host_c64f2fnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Xb;
    sw_c64ffn2_staged(Xb);
    int xbad = 0;
    for (int i = 0; i < 4096; i++)
        if (Xb[(size_t)i] != kC64Ffn2Bytes[i]) xbad++;
    std::printf("  staged stream twin: Xb %d/4096 diffs\n", xbad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> X(4096), W1a(8192), W1b(8192), W2a(4096), W2b(4096);
    std::vector<uint16_t> G1(64);
    for (int i = 0; i < 4096; i++) X[(size_t)i] = sw_e4m3_decode(Xb[(size_t)i]);
    for (int i = 0; i < 8192; i++) {
        W1a[(size_t)i] = sw_e4m3_decode(dump[(size_t)i]);
        W1b[(size_t)i] = sw_e4m3_decode(dump[(size_t)8192 + i]);
    }
    for (int i = 0; i < 4096; i++) {
        W2a[(size_t)i] = sw_e4m3_decode(dump[(size_t)16384 + i]);
        W2b[(size_t)i] = sw_e4m3_decode(dump[(size_t)20480 + i]);
    }
    for (int o = 0; o < 64; o++)
        G1[(size_t)o] = (uint16_t)(dump[(size_t)28688 + 2 * o] |
                                   (dump[(size_t)28689 + 2 * o] << 8));
    std::vector<float> O(4096);
    sw_c64ffn2_ref(X.data(), W1a.data(), W1b.data(), W2a.data(), W2b.data(),
                   G1.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 4096; i++) {
        float want;
        memcpy(&want, &kC64Ffn2Golden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  FFN2: maxerr %.6g maxabs %.4f nonfinite %d/4096 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}

// Staged FFN-expand input: MUST match
// tools/swin1h_ref.c64_staged_exin exactly (x stream from 0xC64009,
// levels [-1,-0.5,0,0.5,1] via f16 staging quant, row-major [64][64]).
static void sw_c64expand_staged(std::vector<uint8_t>& Xb) {
    static const float lv[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint32_t st = 0xC64009u;
    Xb.resize(4096);
    for (int i = 0; i < 4096; i++) {
        st = sw_lcg_next(st);
        Xb[(size_t)i] = sw_h2e4m3(sw_f32_to_f16_bits(lv[(st >> 8) % 5u]));
    }
}

// Test-23 oracle recipe in C++ (HANDOFF §113): x[64][64] x w1[64][128]
// -> h[64][128], K=64 in TWO k-steps of 32 with f16 rounding per step.
// The 2-step structure is the production D->C chain read off the kernel
// (§113: 32 chains, all length 2, rooted at the frozen 0.0 accumulator),
// HALF the contract's four steps because w1 is half as deep in K.
// W is flat [64][128] e4m3 IN-MAJOR (rows = K, row-major): W[k*128+n],
// matching Wproj (Test-19), Wqkv (Test-21) and W2-contract (Test-22).
// No act/bias/gate/residual -- all downstream of h (§101 / §111).
static void sw_c64expand_ref(const float* X, const float* W, float* O) {
    for (int m = 0; m < 64; m++)
        for (int n = 0; n < 128; n++) {
            uint16_t acc = 0;
            for (int kh = 0; kh < 2; kh++) {
                float s = 0.0f;
                for (int k = 0; k < 32; k++)
                    s += X[(size_t)m * 64 + kh * 32 + k] *
                         W[(size_t)(kh * 32 + k) * 128 + n];
                acc = sw_f32_to_f16_bits(sw_f16_to_f32(acc) + s);
            }
            O[(size_t)m * 128 + n] = sw_f16_to_f32(acc);
        }
}

// Host twin for Test-23 (HANDOFF §113): staged-x stream twin (EXACT
// vs golden inputs) + real w1 dump read + mirror expand vs
// kC64ExpandGolden. Needs tensordir (real weights, Test-16 path).
static void host_c64xfnv_cmd(const char* tensordir) {
    std::vector<uint8_t> Xb;
    sw_c64expand_staged(Xb);
    int xbad = 0;
    for (int i = 0; i < 4096; i++)
        if (Xb[(size_t)i] != kC64ExinBytes[i]) xbad++;
    std::printf("  staged stream twin: Xb %d/4096 diffs\n", xbad);
    char dp[1024];
    std::snprintf(dp, sizeof(dp), "%s/%s", tensordir, "tensor_137.bin");
    FILE* f = std::fopen(dp, "rb");
    if (!f) { std::printf("  tensor_137.bin not readable\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    std::printf("tensor_137.bin: %ld bytes %s\n", sz,
                sz == 61760 ? "(expect 61760) OK" : "MISMATCH");
    if (sz != 61760) { fclose(f); return; }
    std::vector<uint8_t> dump(61760);
    fseek(f, 0, SEEK_SET);
    if (fread(dump.data(), 1, 61760, f) != 61760) { fclose(f); return; }
    fclose(f);
    std::vector<float> X(4096), W(8192);
    for (int i = 0; i < 4096; i++)
        X[(size_t)i] = sw_e4m3_decode(Xb[(size_t)i]);
    for (int i = 0; i < 8192; i++)
        W[(size_t)i] = sw_e4m3_decode(dump[(size_t)i]);
    std::vector<float> O(8192);
    sw_c64expand_ref(X.data(), W.data(), O.data());
    float me = 0.0f, mx = 0.0f;
    int nn = 0;
    for (int i = 0; i < 8192; i++) {
        float want;
        memcpy(&want, &kC64ExpandGolden[i], 4);
        float d = O[(size_t)i] - want;
        if (d != d) { nn++; continue; }
        float a = d < 0.0f ? -d : d;
        if (a > me) me = a;
        float w = want < 0.0f ? -want : want;
        if (w > mx) mx = w;
    }
    std::printf("  EXPAND: maxerr %.6g maxabs %.4f nonfinite %d/8192 FNV %08X\n",
                me, mx, nn, sw_fnv(O.data(), O.size() * 4));
}

static void host_realweights_cmd(const char* tensordir) {
    const char* names[2] = {"tensor_000.bin", "tensor_001.bin"};
    const int expect[2] = {21696, 20672};
    const int s_off[2] = {20576, 19552};
    for (int t = 0; t < 2; t++) {
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/%s", tensordir, names[t]);
        std::vector<uint8_t> raw;
        bool okread = sw_read_file(path, raw);
        bool ok = okread && (int)raw.size() == expect[t];
        std::printf("%s: %d bytes (expect %d) %s\n", names[t],
                    okread ? (int)raw.size() : -1, expect[t],
                    ok ? "OK" : "SIZE MISMATCH");
        if (!ok) continue;
        RwChain r;
        sw_rw_chain(raw, t == 0, r);
        float gfmn = r.gf[0], gfmx = r.gf[0];
        float gamn = r.ga[0], gamx = r.ga[0];
        for (int o = 1; o < SW_C; o++) {
            if (r.gf[o] < gfmn) gfmn = r.gf[o];
            if (r.gf[o] > gfmx) gfmx = r.gf[o];
            if (r.ga[o] < gamn) gamn = r.ga[o];
            if (r.ga[o] > gamx) gamx = r.ga[o];
        }
        std::printf("  gate_ffn: min %.4f max %.4f in(0,1]=%d/32\n", gfmn,
                    gfmx, sw_gate_in01(r.gf));
        std::printf("  gate_attn: min %.4f max %.4f in(0,1]=%d/32\n", gamn,
                    gamx, sw_gate_in01(r.ga));
        std::printf("  QK-temperature s = %.4f (f32 @+%d)\n", r.s_real,
                    s_off[t]);
        float bmn = r.B[0], bmx = r.B[0];
        int bfin = 0;
        for (size_t i = 0; i < r.B.size(); i++) {
            if (r.B[i] < bmn) bmn = r.B[i];
            if (r.B[i] > bmx) bmx = r.B[i];
            if (std::isfinite(r.B[i])) bfin++;
        }
        std::printf("  bias 64x64 f16: min %.4f max %.4f finite=%d/4096\n",
                    bmn, bmx, bfin);
        if (t == 0) {
            int nnf = 0;
            for (size_t i = 0; i < r.Ypatch.size(); i++)
                if (!std::isfinite(r.Ypatch[i])) nnf++;
            std::printf("  patch: max|out|=%.3f n_nonfinite=%d\n",
                        sw_maxabs_f(r.Ypatch), nnf);
        }
        int fin = 0;
        for (size_t i = 0; i < r.Yp.size(); i++)
            if (std::isfinite(r.Yp[i])) fin++;
        std::printf("  stages maxabs: ffn=%.2f qkv=%.2f scores=%.2f attn=%.2f "
                    "proj=%.2f finite=%d/%d checksum=%08X\n",
                    sw_maxabs_f(r.Yf), sw_maxabs_f(r.Yq), sw_maxabs_f(r.S),
                    sw_maxabs_f(r.O), sw_maxabs_f(r.Yp), fin, SW_TOK * SW_C,
                    sw_fnv_finite_f32(r.Yp));
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && !strcmp(argv[1], "--mpcubic-fnv")) {
        std::printf("%08X\n", host_mpcubic_fnv());
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "--quant-fnv")) {
        std::printf("%08X\n", host_quant_fnv());
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "--f32f16-fnv")) {
        std::printf("%08X\n", host_f32f16_fnv());
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "--fe-fnv")) {
        std::printf("FE-PCG-FNV %08X\n", host_fe_pcg_fnv());
        std::printf("FE-XBITS-FNV %08X\n", host_fe_xbits_fnv());
        std::printf("FE-PATCH-FNV %08X\n", host_fe_patch_fnv());
        host_fe_bm_anchor();
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64fnv")) {
        host_c64fnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "--c64ofnv")) {
        host_c64ofnv_cmd();
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64pfnv")) {
        host_c64pfnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64ffnv")) {
        host_c64ffnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64qfnv")) {
        host_c64qfnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64cfnv")) {
        host_c64cfnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64xfnv")) {
        host_c64xfnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--c64f2fnv")) {
        host_c64f2fnv_cmd(argv[2]);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--realweights")) {
        host_realweights_cmd(argv[2]);
        return 0;
    }
    host_checksums();
    return 0;
}
#endif  // SWIN1H_HOST_ONLY

#ifndef SWIN1H_HOST_ONLY
// ---------------------------------------------------------------------------
// 2. HIP kernels (direct ports of the mirror; k-ascending loop order).
// All GEMMs: one thread per output element. Weights arrive in native formats
// (f16 bits for patch, e4m3 bytes otherwise) and are dequantised inline, so
// the device exercises the same decode as production.
// ---------------------------------------------------------------------------

#define HIPCHECK(e)                                                          \
    do {                                                                     \
        hipError_t _e = (e);                                                 \
        if (_e != hipSuccess) {                                              \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__,          \
                         __LINE__, hipGetErrorString(_e));                   \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// scalar-f16 MpCubicSilu with the stage1_prep idiom (same rounding as the
// packed __half2 version; proven bitwise vs the mirror by the on-device
// exhaustive check in main).
__device__ inline __half sw_mpcubic_half(__half in) {
    const __half n4 = __float2half(-4.0f);
    const __half p4 = __float2half(4.0f);
    const __half c0 = __float2half(-0.055908203125f);
    const __half c1 = __float2half(0.447265625f);
    const __half c2 = __float2half(0.89453125f);
    __half y = __hmin(__hmax(in, n4), p4);
    __half t = __hfma(c0, __habs(y), c1);
    __half u = __hfma(y, t, c2);
    return __hmul(in, u);
}

// patch GEMM: e4m3 X[M][16] x f16 W[32][16] -> f32 Y[M][32].
// The pre-block parameter block the real entry kernel reads (see
// sw_fe_params.inc for the PTX-pinned offsets). Kept OUT of the region
// tools/extract_chain.py copies: the backend feeds that file to hiprtc, which
// compiles it in a temp dir with no include path into hip/mvp1, so an
// #include there cannot resolve -- extract_chain.py emits this file inline.
#include "sw_fe_params.inc"

__global__ void k_patch_gemm(const uint8_t* X, const uint16_t* W, float* Y,
                             int M) {
    int m = (int)(blockIdx.x * blockDim.x + threadIdx.x) / SW_C;
    int o = (int)(blockIdx.x * blockDim.x + threadIdx.x) % SW_C;
    if (m >= M || o >= SW_C) return;
    float acc = 0.0f;
    for (int k = 0; k < SW_D_IN; k++)
        acc += sw_e4m3_decode(X[m * SW_D_IN + k]) *
               __half2float(__ushort_as_half(W[o * SW_D_IN + k]));
    Y[m * SW_C + o] = acc;
}

// f32 -> e4m3 quant (f16 RN via shared integer-ops converter).
__global__ void k_quant_e4m3(const float* X, uint8_t* Y, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) Y[i] = sw_h2e4m3(sw_f32_to_f16_bits(X[i]));
}

// e4m3 -> f32 decode (exact integer math; bit-identical to host
// chain_e4m3_decode -- proven by the tail-run exact-check, zero mismatches).
__global__ void k_decode_e4m3(const uint8_t* X, float* Y, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) Y[i] = sw_e4m3_decode(X[i]);
}

// generic e4m3-GEMM: e4m3 X[M][K] x e4m3 W[N][K] -> f32 Y[M][N].
__global__ void k_gemm_e4m3(const uint8_t* X, const uint8_t* W, float* Y,
                            int M, int K, int N) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = idx / N, o = idx % N;
    if (m >= M || o >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += sw_e4m3_decode(X[m * K + k]) * sw_e4m3_decode(W[o * K + k]);
    Y[m * N + o] = acc;
}

// QK-norm (HANDOFF §17): per-token sumsq (f32, k-ascending — bitwise vs the
// mirror on exact vectors), rsqrt factor, Q scaled by f16-rounded s.
// V passes through untouched (read straight from Yq by k_av).
// Outputs Qn/Kn (normalized) + per-row sumsq (for exact verification).
__global__ void k_qknorm(const float* Yq, float* Qn, float* Kn, float* ssQ,
                         float* ssK, float s) {
    int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (m >= SW_TOK) return;
    float sq = 0.0f, sk = 0.0f;
    for (int d = 0; d < 32; d++) {
        float q = Yq[m * SW_QKV + d], k = Yq[m * SW_QKV + 32 + d];
        sq += q * q;
        sk += k * k;
    }
    ssQ[m] = sq;
    ssK[m] = sk;
    // Allowed divergence: device rsqrtf (hw approx) vs mirror 1/sqrtf.
    float fq = rsqrtf(sq > SW_EPS_NORM ? sq : SW_EPS_NORM);
    float fk = rsqrtf(sk > SW_EPS_NORM ? sk : SW_EPS_NORM);
    float sf = __half2float(__ushort_as_half(sw_f32_to_f16_bits(s)));
    for (int d = 0; d < 32; d++) {
        Qn[m * SW_C + d] = sf * Yq[m * SW_QKV + d] * fq;
        Kn[m * SW_C + d] = Yq[m * SW_QKV + 32 + d] * fk;
    }
}

// QK^T scores over NORMALIZED Q/K. NO 1/sqrt(32) (HANDOFF §19 — the fused
// kernel has none; the QK-norm temperature s is the only scale). §21: raw
// dots + 64x64 f16 bias, NO scale (S = QK^T + B, row-major).
__global__ void k_scores(const float* Qn, const float* Kn, const float* B,
                         float* S) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = idx / SW_TOK, j = idx % SW_TOK;
    if (i >= SW_TOK || j >= SW_TOK) return;
    float acc = 0.0f;
    for (int d = 0; d < 32; d++) acc += Qn[i * SW_C + d] * Kn[j * SW_C + d];
    S[i * SW_TOK + j] = acc + B[i * SW_TOK + j];
}

// C=64 scores (HANDOFF §49, Test-16): 2-head QK^T with per-head f32
// temps and the (2,64,64) f32 bias (decoded f16). TRAP: temp scales each
// Q element pre-dot (the kernel under test does the same: its 32
// mul.f16x2 sit on the mma-A path); the bias is added UNSCALED (D-init).
// One thread per (head,row,col) = 8192 threads; k-ascending f32 dots,
// op-for-op the sw_c64scores_ref mirror (bitwise vs each other; f64
// oracle ~1e-4 away at |S|~90, tol 0.1). Reuses k_smexp/k_smsum/k_smnorm
// unchanged for Test-17 (same 64-wide rows, same constants, §50).
__global__ void k_c64scores(const uint8_t* Qe, const uint8_t* Ke,
                           const float* B, const float* T, float* S) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_HEADS * SW_C64_TOK * SW_C64_TOK) return;
    int h = idx / (SW_C64_TOK * SW_C64_TOK), k = idx % (SW_C64_TOK * SW_C64_TOK);
    int i = k / SW_C64_TOK, j = k % SW_C64_TOK;
    // S132 flagged this as the one place hard-wired to SW_C64_HEADS == 2 (a
    // 2-way select). S146 generalises it: T is a per-head array, so the kernel
    // is head-count-agnostic and the ViT's 32 heads work unchanged. At
    // HEADS == 2 the arithmetic is identical to the old select.
    float t = T[h];
    float acc = 0.0f;
    for (int d = 0; d < SW_C64_DIM; d++)
        acc += (t * sw_e4m3_decode(Qe[h * (SW_C64_TOK * SW_C64_DIM) +
                                       i * SW_C64_DIM + d])) *
               sw_e4m3_decode(Ke[h * (SW_C64_TOK * SW_C64_DIM) +
                                 j * SW_C64_DIM + d]);
    S[idx] = acc + B[idx];
}

// C=64 context O = P x V (HANDOFF §78E, Test-18): staged Pq [2][64][64]
// e4m3 bytes x staged V [2][64][32] e4m3 bytes -> f32 O [2][64][32].
// One thread per (head,row,dim) = 4096 threads; plain f32 full-K dots
// (k_c64scores precedent): production rounds per K-half to f16 D, which
// sits ~1e-3 below at these magnitudes (tol 0.1). O is [2][64][32] =
// 2048 per head: h = idx / 2048 (NOT /1024 -- that mis-index reads
// heads 2-3 out of bounds; fixed after the 70/72 red). Op-for-op the
// sw_c64ctx_ref mirror except the f16 steps (twin-vs-oracle maxerr 0).
__global__ void k_c64ctx(const uint8_t* Pqe, const uint8_t* Ve,
                         float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_HEADS * SW_C64_TOK * SW_C64_DIM) return;
    int h = idx / (SW_C64_TOK * SW_C64_DIM), k = idx % (SW_C64_TOK * SW_C64_DIM);
    int i = k / SW_C64_DIM, n = k % SW_C64_DIM;
    float acc = 0.0f;
    for (int j = 0; j < SW_C64_TOK; j++)
        acc += sw_e4m3_decode(Pqe[h * (SW_C64_TOK * SW_C64_TOK) +
                                  i * SW_C64_TOK + j]) *
               sw_e4m3_decode(Ve[h * (SW_C64_TOK * SW_C64_DIM) +
                                 j * SW_C64_DIM + n]);
    O[idx] = acc;
}

// C=64 proj out = Ocat x Wproj + gate2 * y (HANDOFF §78G, Test-19):
// staged Ocat/y [64][64] e4m3 bytes x real Wproj [64][64] e4m3 +
// real gate2 f16[64] (dump +57520/+61616) -> f32 out [64][64].
// One thread per (m,o) = 4096 threads; plain f32 dots + gate fma
// (C=32 proj_forward precedent). W row-major [c][o] mapping shared
// with oracle/twin (true production orientation is wiring work).
__global__ void k_c64proj(const uint8_t* Ocat, const uint8_t* Wp,
                          const uint8_t* Yb, const float* G2, float* O,
                          int tOut) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_TOK * SW_C64_C) return;   // S162: [TOK][C], not [C][C]
    int m = idx / SW_C64_C, o = idx % SW_C64_C;
    float acc = 0.0f;
    for (int c = 0; c < SW_C64_C; c++)
        acc += sw_e4m3_decode(Ocat[m * SW_C64_C + c]) *
               // S150: out-major [out][in] -- (c,o) lives at o*C + c. Same
               // .row.col reason as the FFN (S148/S149).
               sw_e4m3_decode(tOut ? Wp[(size_t)o * SW_C64_C + c]
                                   : Wp[(size_t)c * SW_C64_C + o]);
    O[idx] = acc + G2[o] * sw_e4m3_decode(Yb[m * SW_C64_C + o]);
}

// C=64 FFN expand + act (HANDOFF S78H, Test-20): staged x [64][64]
// e4m3 x real w1 [224][64] e4m3 (dump +0) -> H f32 -> f16 RN ->
// MpCubicSilu (sw_mpcubic_half, the k_ffn_act combo) -> Abits f16
// [64][224]. 14336 threads, one per (m,j); plain f32 dots
// (k_c64proj precedent).
__global__ void k_c64ffn_act(const uint8_t* Xb, const uint8_t* W1,
                             uint16_t* Abits) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= 14336) return;
    int m = idx / 224, j = idx % 224;
    float acc = 0.0f;
    for (int k = 0; k < 64; k++)
        acc += sw_e4m3_decode(Xb[m * 64 + k]) *
               sw_e4m3_decode(W1[j * 64 + k]);
    __half h = __ushort_as_half(sw_f32_to_f16_bits(acc));
    Abits[idx] = __half_as_ushort(sw_mpcubic_half(h));
}

// C=64 FFN contract + gate1 residual (Test-20): Abits f16 [64][224]
// x real w2 [64][224] e4m3 (dump +14336) + gate1 * x -> f32 Y
// [64][64]. 4096 threads; row-major [o][j] mapping shared with
// oracle/twin (k_ffn2_res precedent, 64-wide).
__global__ void k_c64ffn2(const uint16_t* Abits, const uint8_t* W2,
                          const uint8_t* Xb, const float* G1, float* Y) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= 4096) return;
    int m = idx / 64, o = idx % 64;
    float acc = 0.0f;
    for (int j = 0; j < 224; j++)
        acc += __half2float(__ushort_as_half(Abits[m * 224 + j])) *
               sw_e4m3_decode(W2[o * 224 + j]);
    Y[idx] = acc + G1[o] * sw_e4m3_decode(Xb[m * 64 + o]);
}

// C=64 QKV GEMM (HANDOFF S90, Test-21): staged X [64][64] e4m3 x
// real Wqkv [kh][h][32][96] e4m3 (dump +28832, K-half-major) -> f32
// QKV [2][64][96]. 12288 threads, one per (h,m,n); plain f32
// full-K dots (k_c64proj precedent). N-third order [Q|K|V] by
// convention shared with oracle/twin (wiring work, S90E).
__global__ void k_c64qkv(const uint8_t* Xb, const uint8_t* Wq,
                         float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_HEADS * SW_C64_TOK * (3 * SW_C64_W)) return;  // S162
    int h = idx / (SW_C64_TOK * 3 * SW_C64_W), k = idx % (SW_C64_TOK * 3 * SW_C64_W);
    int m = k / (3 * SW_C64_W), n = k % (3 * SW_C64_W);
    float acc = 0.0f;
    for (int kk = 0; kk < SW_C64_C; kk++)
        acc += sw_e4m3_decode(Xb[m * SW_C64_C + kk]) *
               sw_e4m3_decode(Wq[((kk / SW_C64_W) * SW_C64_HEADS + h) *
                                  (SW_C64_DIM * 3 * SW_C64_W) +
                                  (kk % SW_C64_W) * (3 * SW_C64_W) + n]);
    O[idx] = acc;
}

// Live refinement write-back (S191). The front-end maps 64 tokens to an 8x8
// SPATIAL grid of proxy texels (x = wl & 7, y = (wl>>3) + 4*pas), sampled via
// an affine. So each token IS a sample of the frame, and the block's answer is
// a refined version of that sample. The honest write-back is therefore a blend
// of the frame toward those refined samples -- a real, smooth, image-space edit
// rather than the tiled noise of S181/S182.
//
// The inverse affine recovers the grid cell for a given pixel, and the grid
// cell recovers the token index by inverting the front-end's own mapping:
//   pas = t >> 5, wl = t & 31, x = wl & 7, y = (wl >> 3) + 4*pas
//   => for (gx,gy): pas = gy >= 4 ? 1 : 0 ; wl = gx | ((gy - 4*pas) << 3)
//
// k = 0 leaves the frame untouched, so this is safe to leave wired in.
extern "C" __global__ void k_live_refine(const uint8_t* __restrict__ proxy,
                                         unsigned char* __restrict__ out,
                                         int pitch, int w, int h,
                                         const float* __restrict__ tok, int C,
                                         int TOK, float axa, float axb,
                                         float axm, float aya, float ayb,
                                         float aym, int srcFmt, float k) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= w * h) return;
    if (k <= 0.0f) return;
    int px = idx % w, py = idx / w;

    // inverse affine: cx = (u*axa + axb)*axm  =>  u = (cx/axm - axb)/axa
    float cx = ((float)px + 0.5f) / (float)w;
    float cy = ((float)py + 0.5f) / (float)h;
    float ax = (axa != 0.0f && axm != 0.0f) ? (cx / axm - axb) / axa : 0.0f;
    float ay = (aya != 0.0f && aym != 0.0f) ? (cy / aym - ayb) / aya : 0.0f;
    // S196: BILINEAR across the 8x8 token grid. Nearest-neighbour painted 64
    // hard-edged regions (the "boxes"); a refinement field sampled at 64 points
    // should be smooth. Sample the four surrounding cells and interpolate their
    // tanh'd signals, so the gain varies continuously.
    float fx = ax * 8.0f - 0.5f, fy = ay * 8.0f - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - (float)x0, ty = fy - (float)y0;
    int x1 = x0 + 1, y1 = y0 + 1;

    float r = 0.0f, g = 0.0f, b = 0.0f, wsum = 0.0f;
    for (int j = 0; j < 2; j++) {
        int gy = (j == 0) ? y0 : y1;
        float wy = (j == 0) ? (1.0f - ty) : ty;
        if (gy < 0 || gy > 7) continue;
        int pas = (gy >= 4) ? 1 : 0;
        for (int i = 0; i < 2; i++) {
            int gx = (i == 0) ? x0 : x1;
            if (gx < 0 || gx > 7) continue;
            int wl = gx | (((gy - 4 * pas) & 3) << 3);
            int t = pas * 32 + wl;
            if (t < 0 || t >= TOK || C < 3) continue;
            float w = wy * ((i == 0) ? (1.0f - tx) : tx);
            r += w * tanhf(tok[(size_t)t * C + 0]);
            g += w * tanhf(tok[(size_t)t * C + 1]);
            b += w * tanhf(tok[(size_t)t * C + 2]);
            wsum += w;
        }
    }
    if (wsum > 1e-6f) { r /= wsum; g /= wsum; b /= wsum; }
    else { r = g = b = 0.0f; }                          // outside the grid: no edit
    if (!(r == r) || !(g == g) || !(b == b)) return;    // NaN guard

    // S194: THE MODEL'S OUTPUT IS A LATENT, NOT A PICTURE. The chain here is the
    // pre-block, i.e. the ENCODER's first step, so its channels are activations
    // (tok0 ~ -0.42) and not RGB. Copying them into the model texture produced a
    // negative-luma image, and the resolve's guard for "the model returned an
    // empty frame" (modelLuma <= 1e-5 -> hand the frame back untouched) fired
    // every frame. That guard is CORRECT: it was handed something that is not a
    // picture.
    //
    // So modulate the proxy by the latent instead of replacing it. tanh bounds
    // the per-channel gain to (1-k, 1+k), which keeps the result a well-formed
    // positive picture that the resolve will accept -- and the modulation is
    // still entirely driven by what the network computed.

    size_t o = (size_t)py * (size_t)pitch + (size_t)px * 4u;
    float cr, cg, cb;
    if (srcFmt == 0) {
        cr = (float)proxy[o + 0] / 255.0f;
        cg = (float)proxy[o + 1] / 255.0f;
        cb = (float)proxy[o + 2] / 255.0f;
        cr *= (1.0f + k * r); cg *= (1.0f + k * g); cb *= (1.0f + k * b);
        out[o + 0] = (unsigned char)(cr <= 0.0f ? 0.0f : (cr >= 1.0f ? 255.0f : cr * 255.0f));
        out[o + 1] = (unsigned char)(cg <= 0.0f ? 0.0f : (cg >= 1.0f ? 255.0f : cg * 255.0f));
        out[o + 2] = (unsigned char)(cb <= 0.0f ? 0.0f : (cb >= 1.0f ? 255.0f : cb * 255.0f));
        out[o + 3] = proxy[o + 3];
    } else {
        const uint16_t* src = (const uint16_t*)(proxy + (size_t)py * (size_t)pitch) + (size_t)px * 4u;
        uint16_t* dst = (uint16_t*)(out + (size_t)py * (size_t)pitch) + (size_t)px * 4u;
        if (srcFmt == 1) {
            cr = sw_f16_to_f32(src[0]); cg = sw_f16_to_f32(src[1]); cb = sw_f16_to_f32(src[2]);
            cr *= (1.0f + k * r); cg *= (1.0f + k * g); cb *= (1.0f + k * b);
            // S194: THIS branch ran unclamped, which is how a negative
            // modelLuma reached the resolve and tripped its empty-frame guard.
            cr = cr <= 0.0f ? 0.0f : (cr >= 1.0f ? 1.0f : cr);
            cg = cg <= 0.0f ? 0.0f : (cg >= 1.0f ? 1.0f : cg);
            cb = cb <= 0.0f ? 0.0f : (cb >= 1.0f ? 1.0f : cb);
            dst[0] = sw_f32_to_f16_bits(cr); dst[1] = sw_f32_to_f16_bits(cg);
            dst[2] = sw_f32_to_f16_bits(cb); dst[3] = src[3];
        } else {
            cr = (float)src[0] / 65535.0f; cg = (float)src[1] / 65535.0f; cb = (float)src[2] / 65535.0f;
            cr *= (1.0f + k * r); cg *= (1.0f + k * g); cb *= (1.0f + k * b);
            cr = cr <= 0.0f ? 0.0f : (cr >= 1.0f ? 1.0f : cr);
            cg = cg <= 0.0f ? 0.0f : (cg >= 1.0f ? 1.0f : cg);
            cb = cb <= 0.0f ? 0.0f : (cb >= 1.0f ? 1.0f : cb);
            dst[0] = (uint16_t)(cr * 65535.0f); dst[1] = (uint16_t)(cg * 65535.0f);
            dst[2] = (uint16_t)(cb * 65535.0f); dst[3] = src[3];
        }
    }
}

// Live diagnostic blit (S181): tile a small source buffer into a P x P patch
// CENTRED in a pitch-strided image. One launch, no per-row memcpys -- the
// per-frame path already costs 50% of a frame and I will not add hundreds of
// synchronous copies to it just to make a debug square.
//
// src is SRC x SRC pixels of `bpp` bytes (the block's 32x32 output = 8192 B at
// 8 bpp). dst is the staging buffer. Values are copied verbatim: this is a
// diagnostic overlay of real network output, not a rendering decision.
extern "C" __global__ void k_live_patch(const unsigned char* __restrict__ src,
                                        unsigned char* __restrict__ dst,
                                        int pitch, int w, int h,
                                        int P, int SRC, int bpp) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= P * P) return;
    int r = idx / P, c = idx % P;
    int ox = (w - P) / 2, oy = (h - P) / 2;
    if (ox < 0 || oy < 0) return;
    const unsigned char* s =
        src + ((size_t)(r % SRC) * SRC + (c % SRC)) * (size_t)bpp;
    unsigned char* d =
        dst + (size_t)(oy + r) * (size_t)pitch + (size_t)(ox + c) * (size_t)bpp;
    for (int k = 0; k < bpp; k++) d[k] = s[k];
}

// ViT QKV projection (HANDOFF S147) -- the ViT block's layer2, a plain
// [C][3C] projection, NOT the fused stage's per-head [DIM][3W] layout.
//
// Layout measured from the PTX (cubin_05, cc_vit_qkv_fp8): the kernel
// addresses weights in 32768-byte blocks with 512-byte sub-offsets, so the
// contiguous row is 1024 bytes = C channels. The matrix is therefore stored
// **[3C][C] = [3072][1024], OUT-major**, i.e. row n holds output channel n's
// weights across all C input channels. (3145728 / 1024 = 3072 rows exact.)
// The QKV's weight segment begins 128 B into tensor_052, after the 32 fp32
// per-head coefficients.
//
// One thread per (token, output channel): 64 * 3072 = 196608 threads at the
// ViT's sizes. Plain f32 full-K dot over K = C, the k_c64qkv/k_c64contract
// precedent, so it stays oracle-checkable at the ordinary tol.
__global__ void k_vit_qkv(const uint8_t* Xb, const uint8_t* Wq, float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_TOK * 3 * SW_C64_C) return;
    int m = idx / (3 * SW_C64_C), n = idx % (3 * SW_C64_C);
    float acc = 0.0f;
    for (int k = 0; k < SW_C64_C; k++)
        acc += sw_e4m3_decode(Xb[m * SW_C64_C + k]) *
               sw_e4m3_decode(Wq[(size_t)n * SW_C64_C + k]);
    O[idx] = acc;
}

// C=64 FFN-contract GEMM (HANDOFF §111, Test-22): staged h [64][128]
// e4m3 x real W2 [128][32] e4m3 (dump +16384, pass 0's N-slice of the
// [128][64] region-2 weight) -> f32 [64][32]. 2048 threads, one per
// (m,n); plain f32 full-K dot over K=128 (k_c64qkv precedent). The
// oracle's four f16 k-steps are the production mma-D chain; the
// device's single f32 dot is the same precedent Test-21 settled.
__global__ void k_c64contract(const uint8_t* Hb, const uint8_t* W2,
                              float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_C * SW_C64_W) return;
    int m = idx / SW_C64_W, n = idx % SW_C64_W;
    float acc = 0.0f;
    for (int k = 0; k < SW_C64_H1P; k++)
        acc += sw_e4m3_decode(Hb[m * SW_C64_H1P + k]) *
               sw_e4m3_decode(W2[k * SW_C64_W + n]);
    O[idx] = acc;
}

// C=64 FFN-expand GEMM (HANDOFF §113, Test-23): staged x [64][64]
// e4m3 x real w1 [64][128] e4m3 (dump +0, 8192 B = pass 0's slice of
// the 16384-B region 1) -> f32 [64][128]. 8192 threads, one per (m,n);
// plain f32 full-K dot over K=64 (k_c64contract precedent). The
// oracle's two f16 k-steps are the production mma-D chain (§113: 32
// chains of length 2); the device's single f32 dot is the same
// precedent Test-21/Test-22 settled. w1 is IN-MAJOR W[k*128+n] --
// Test-20's out-major w1 is the §113 defect and is NOT copied here.
// HANDOFF §131 / Phase A: (C, H1) are parameters, not literals. The C=64
// instantiation passes (64, 128) and must reproduce the Test-23 golden exactly
// -- that is what proves the parameterisation is behaviour-preserving.
__global__ void k_c64expand(const uint8_t* Xb, const uint8_t* W1,
                            float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_C * SW_C64_H1P) return;
    int m = idx / SW_C64_H1P, n = idx % SW_C64_H1P;
    float acc = 0.0f;
    for (int k = 0; k < SW_C64_C; k++)
        acc += sw_e4m3_decode(Xb[m * SW_C64_C + k]) *
               sw_e4m3_decode(W1[k * SW_C64_H1P + n]);
    O[idx] = acc;
}

// C=64 CONNECTED FFN (HANDOFF S116, Test-24): staged x[64][64] e4m3 ->
// expand (K=64, TWO f16 k-steps) -> act -> e4m3 -> contract (K=128,
// FOUR f16 k-steps), per pass (o<32 / o>=32), + gate1 residual ->
// f32 [64][64]. Exact mirror of sw_c64ffn2_ref (so device == twin);
// sw_f16_to_f32 is host-only, so the device uses __half2float on the
// same bits. W1 = 16384 B (two [64][128] IN-MAJOR), W2 = 8192 B (two
// [128][32] IN-MAJOR), G1 = 64 f16 (dump +28688).
__global__ void k_c64ffn2c(const uint8_t* Xb, const uint8_t* W1,
                          const uint8_t* W2, const uint16_t* G1, float* O,
                          int tOut) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    // S162: the output is [TOK][C]. The old `C*C` bound conflated the token
    // count with the channel count -- identical at C=64 (=64), 2x wrong at
    // C=128, where it wrote twice the buffer and halved the token extent.
    if (idx >= SW_C64_TOK * SW_C64_C) return;
    int m = idx / SW_C64_C, o = idx % SW_C64_C;
    int pass = o / SW_C64_W, n = o % SW_C64_W;
    const uint8_t* W1p = W1 + (size_t)pass * (SW_C64_C * SW_C64_H1P);
    const uint8_t* W2p = W2 + (size_t)pass * (SW_C64_H1P * SW_C64_W);
    float H[SW_C64_H1P];
    for (int j = 0; j < SW_C64_H1P; j++) {
        uint16_t acc = 0;
        for (int kh = 0; kh < SW_C64_C / SW_C64_K; kh++) {
            float s = 0.0f;
            for (int k = 0; k < SW_C64_K; k++)
                s += sw_e4m3_decode(Xb[m * SW_C64_C + kh * SW_C64_K + k]) *
                     sw_e4m3_decode(tOut
                        // S148: out-major [H1][C] -- (k,j) at j*C + k
                        ? W1p[(size_t)j * SW_C64_C + kh * SW_C64_K + k]
                        // legacy in-major [C][H1] -- (k,j) at k*H1 + j
                        : W1p[(size_t)(kh * SW_C64_K + k) * SW_C64_H1P + j]);
            acc = sw_f32_to_f16_bits(
                __half2float(__ushort_as_half(acc)) + s);
        }
        H[j] = __half2float(__ushort_as_half(acc));
    }
    uint8_t aq[SW_C64_H1P];
    for (int j = 0; j < SW_C64_H1P; j++)
        aq[j] = sw_h2e4m3(__half_as_ushort(sw_mpcubic_half(
            __ushort_as_half(sw_f32_to_f16_bits(H[j])))));
    uint16_t acc = 0;
    for (int kh = 0; kh < SW_C64_H1P / SW_C64_K; kh++) {
        float s = 0.0f;
        for (int k = 0; k < SW_C64_K; k++)
            s += sw_e4m3_decode(aq[kh * SW_C64_K + k]) *
                 sw_e4m3_decode(tOut
                    // S148: out-major [W][H1] -- (k,n) at n*H1 + k
                    ? W2p[(size_t)n * SW_C64_H1P + kh * SW_C64_K + k]
                    // legacy in-major [H1][W] -- (k,n) at k*W + n
                    : W2p[(size_t)(kh * SW_C64_K + k) * SW_C64_W + n]);
        acc = sw_f32_to_f16_bits(__half2float(__ushort_as_half(acc)) + s);
    }
    O[idx] = __half2float(__ushort_as_half(acc)) +
             __half2float(__ushort_as_half(G1[o])) *
                 sw_e4m3_decode(Xb[m * SW_C64_C + o]);
}

// Test-25 glue (HANDOFF S116): split the QKV f32 output [2][64][96]
// (n = Q|K|V thirds) into head-major e4m3 Q/K/V [2][64][32], which is
// what k_c64scores / k_c64ctx read.
__global__ void k_c64qkv_split(const float* QKV, uint8_t* Qe, uint8_t* Ke,
                               uint8_t* Ve) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_HEADS * SW_C64_TOK * (3 * SW_C64_W)) return;  // S162
    int h = idx / (SW_C64_TOK * 3 * SW_C64_W), k = idx % (SW_C64_TOK * 3 * SW_C64_W);
    int m = k / (3 * SW_C64_W), n = k % (3 * SW_C64_W);
    uint8_t b = sw_h2e4m3(sw_f32_to_f16_bits(QKV[idx]));
    if (n < SW_C64_W) Qe[(h * SW_C64_TOK + m) * SW_C64_DIM + n] = b;
    else if (n < 2 * SW_C64_W)
        Ke[(h * SW_C64_TOK + m) * SW_C64_DIM + (n - SW_C64_W)] = b;
    else
        Ve[(h * SW_C64_TOK + m) * SW_C64_DIM + (n - 2 * SW_C64_W)] = b;
}

// Test-25 glue: ctx f32 [2][64][32] -> head-concatenated e4m3 Ocat
// [64][64] (the k_c64proj input). Ocat[m*64 + h*32 + d] = e4m3(O[h][m][d]).
__global__ void k_c64cat_q(const float* O, uint8_t* Ocat) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_C64_HEADS * SW_C64_TOK * SW_C64_DIM) return;
    int h = idx / (SW_C64_TOK * SW_C64_DIM), k = idx % (SW_C64_TOK * SW_C64_DIM);
    int m = k / SW_C64_DIM, d = k % SW_C64_DIM;
    // S164: Ocat is [TOK][C], so the row stride is C. It was SW_C64_TOK --
    // identical at C=64 (TOK == C == 64), so every row at m>0 overwrote the
    // previous one at C=128. NOTE this is the mirror of S162's bug and the
    // S162 audit missed it precisely because this kernel uses ONLY TOK: there
    // was no C-misuse to flag, only a TOK doing a C's job.
    Ocat[(size_t)m * SW_C64_C + h * SW_C64_DIM + d] =
        sw_h2e4m3(sw_f32_to_f16_bits(O[idx]));
}

// Fast-exp E per score (HANDOFF §19): f32 score -> f16 RN -> fma(u) ->
// clamp -> Schraudolph bit-trick on a duplicated word. All f16 ops are
// explicit single-rounding intrinsics, matching the mirror bitwise.
__global__ void k_smexp(const float* S, uint16_t* Eb) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_TOK * SW_TOK) return;
    const __half c1 = __ushort_as_half((uint16_t)SW_EXP_C1);
    const __half c2 = __ushort_as_half((uint16_t)SW_EXP_C2);
    const __half lo = __ushort_as_half((uint16_t)SW_EXP_LO);
    const __half hi = __ushort_as_half((uint16_t)SW_EXP_HI);
    __half x = __float2half_rn(S[idx]);
    __half u = __hfma(x, c1, c2);
    __half v = __hmin(__hmax(u, lo), hi);
    uint32_t w = __half_as_ushort(v);
    uint32_t r = ((w << 16) | w);
    r = (r << 5) + SW_EXP_MAGIC;
    Eb[idx] = (uint16_t)(r & 0xFFFFu);
}

// Per-row f16 sums, k-ascending via explicit __hfma (matches mirror).
__global__ void k_smsum(const uint16_t* Eb, float* Esum) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= SW_TOK) return;
    const __half one = __ushort_as_half(0x3C00u);
    __half acc = __ushort_as_half(0x0000u);
    for (int j = 0; j < SW_TOK; j++)
        acc = __hfma(__ushort_as_half(Eb[i * SW_TOK + j]), one, acc);
    Esum[i] = __half2float(acc);
}

// Normalize: P = f16(E * f16(1/sum)) stored f32. The rcp here is full RN
// division (deliberate: NVIDIA uses rcp.approx — 1-ulp-level divergence,
// noted in HANDOFF §19; RN keeps the port mirror-exact).
__global__ void k_smnorm(const uint16_t* Eb, const float* Esum, float* P) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= SW_TOK * SW_TOK) return;
    int i = idx / SW_TOK;
    float sm = Esum[i] > SW_EPS_NORM ? Esum[i] : SW_EPS_NORM;
    __half g = __float2half_rn(1.0f / sm);
    __half e = __ushort_as_half(Eb[idx]);
    P[idx] = __half2float(__hmul(e, g));
}

// attention x value: P[64][64] x V[64][32] -> O[64][32]
// (V = Yq columns [64:96)).
__global__ void k_av(const float* P, const float* Yq, float* O) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = idx / SW_C, d = idx % SW_C;
    if (i >= SW_TOK || d >= SW_C) return;
    float acc = 0.0f;
    for (int j = 0; j < SW_TOK; j++)
        acc += P[i * SW_TOK + j] * Yq[j * SW_QKV + 64 + d];
    O[i * SW_C + d] = acc;
}

// proj + gate_attn residual: Y = proj(O) + gate[*] * Xin.
__global__ void k_proj_res(const float* O, const uint8_t* Wp,
                           const float* Xin, const float* gate, float* Y) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = idx / SW_C, o = idx % SW_C;
    if (m >= SW_TOK || o >= SW_C) return;
    float acc = 0.0f;
    for (int k = 0; k < SW_C; k++)
        acc += O[m * SW_C + k] * sw_e4m3_decode(Wp[o * SW_C + k]);
    Y[m * SW_C + o] = acc + gate[o] * Xin[m * SW_C + o];
}

// FFN activation: f32 H -> f16 RN -> MpCubicSilu -> f16 bits.
__global__ void k_ffn_act(const float* H, uint16_t* Abits, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        __half h = __ushort_as_half(sw_f32_to_f16_bits(H[i]));
        Abits[i] = __half_as_ushort(sw_mpcubic_half(h));
    }
}

// contract + gate_ffn residual: Y = ffn2(A) + gate[*] * Xin
// (A arrives as f16 bits, dequantised exactly via __half2float).
__global__ void k_ffn2_res(const uint16_t* Abits, const uint8_t* W2,
                           const float* Xin, const float* gate, float* Y) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int m = idx / SW_C, o = idx % SW_C;
    if (m >= SW_TOK || o >= SW_C) return;
    float acc = 0.0f;
    for (int j = 0; j < SW_H1; j++)
        acc += __half2float(__ushort_as_half(Abits[m * SW_H1 + j])) *
               sw_e4m3_decode(W2[o * SW_H1 + j]);
    Y[m * SW_C + o] = acc + gate[o] * Xin[m * SW_C + o];
}

// ---------------------------------------------------------------------------
// Front-end (§13.3 / §33.12): staged RGBA8 proxy -> X e4m3 [64][16].
//
// This is the piece the v4 chain previously took as given ("the chain starts
// at the patch-mma inputs"). Per token q = pas*32 + wl (two passes x 32
// lanes) the kernel fetches pixel (x = wl&7, y = (wl>>3) + 4*pas) of the
// 8x8 window -- the row-major token map proven by tools/o_rowmajor.py R6 --
// and assembles the 16 f16 "patch-A" values exactly as the oracle's
// fe_assemble_X_bits / host fe_assemble_row do:
//
//   [g0, g1, g2, 1.0, n0, n1, n0, n2, n1, n2, hi176, s_sel, s_sel, s_a,
//    s_b, 0]
//
// with n_i = f16((f16(tex_i) - 0.5) * 1.0) -- the exposure scale rides along
// as the s_a/s_b dimensions, it is NOT folded into the normalize -- g* the
// Box-Muller gaussians from the PCG dither streams, and 1.0/0.0 the fixed
// pad slots. The f16 row is then quantised to e4m3, which is what
// k_patch_gemm consumes.
//
// Caveats, documented not fixed: point sampling (real-texture filtering is
// NOT ported, §17), RGBA8/255 as the proxy colour (the encode stage owns
// that convention), and the dither seed is a launch parameter (the kernel
// derives its own from param +200; the port takes it from the caller).
// ---------------------------------------------------------------------------
__device__ inline uint32_t sw_fe_pcg_round(uint32_t st) {
    uint32_t xs = (st >> ((st >> 28) + 4u)) ^ st;
    return xs * 277803737u;
}
__device__ inline uint32_t sw_fe_pcg_out(uint32_t v) {
    return (((v >> 30) ^ (v >> 8)) + 1u);
}
__device__ inline void sw_fe_pcg_streams(uint32_t seed, uint32_t x,
                                         uint32_t y, uint32_t* o) {
    uint32_t hseed = seed * 0x9E3779B9u;
    uint32_t s = (x * (0u - 1918454973u)) ^ hseed;
    uint32_t s1 = (y * (0u - 669632447u)) ^ s;
    uint32_t s2 = s1 ^ 608135816u;
    uint32_t f1 = sw_fe_pcg_round(s2);
    f1 = ((f1 >> 22) ^ f1);
    const uint32_t C0[4] = {747796405u, (0u - 93469191u), (0u - 895109107u),
                            (0u - 2094846927u)};
    const uint32_t C1[4] = {(0u - 1403630843u), 1192405134u, 568162667u,
                            878960812u};
    for (int i = 0; i < 4; i++)
        o[i] = sw_fe_pcg_out(sw_fe_pcg_round(f1 * C0[i] + C1[i]));
}
__device__ inline void sw_fe_boxmuller(float u1, float u2, float u3,
                                       float u4, float* g0, float* g1,
                                       float* g2) {
    const float ln2 = 0.6931471824645996f;    // 0f3F317218
    const float two_pi = 6.283185307179586f;  // 0f40C90FDB
    float r1 = sqrtf(-2.0f * log2f(u1) * ln2);
    float r2 = sqrtf(-2.0f * log2f(u3) * ln2);
    *g0 = r1 * cosf(u2 * two_pi);
    *g1 = r1 * sinf(u2 * two_pi);
    *g2 = r2 * cosf(u4 * two_pi);
}
// f16(tex) - f16(0.5), times f16(mult): matches the host twin on normal
// values (subnormal FTZ caveat, §17).
__device__ inline uint16_t sw_fe_normalize_bits(float tex, float mult) {
    uint16_t t = sw_f32_to_f16_bits(tex);
    uint16_t m = sw_f32_to_f16_bits(mult);
    __half dh = __hsub(__ushort_as_half(t), __ushort_as_half((uint16_t)0x3800u));
    __half r = __hmul(dh, __ushort_as_half(m));
    return __half_as_ushort(r);
}
__device__ inline void sw_fe_assemble_row(const float* tex, const float* gs,
                                          float hi176, float s_sel,
                                          float s_a, float s_b,
                                          uint16_t* out) {
    uint16_t n0 = sw_fe_normalize_bits(tex[0], 1.0f);
    uint16_t n1 = sw_fe_normalize_bits(tex[1], 1.0f);
    uint16_t n2 = sw_fe_normalize_bits(tex[2], 1.0f);
    const uint16_t h[8][2] = {
        {sw_f32_to_f16_bits(gs[0]), sw_f32_to_f16_bits(gs[1])},
        {sw_f32_to_f16_bits(gs[2]), 0x3C00u},
        {n0, n1},
        {n0, n2},
        {n1, n2},
        {sw_f32_to_f16_bits(hi176), sw_f32_to_f16_bits(s_sel)},
        {sw_f32_to_f16_bits(s_sel), sw_f32_to_f16_bits(s_a)},
        {sw_f32_to_f16_bits(s_b), 0x0000u},
    };
    for (int i = 0; i < 8; i++) {
        out[2 * i] = h[i][0];
        out[2 * i + 1] = h[i][1];
    }
}

// proxy: RGBA8 rows of `pitch` bytes. Every sampling parameter comes from the
// real 264-byte block (sw_fe_params.inc), so the port cannot drift from the
// kernel's own layout. The texel is reached through the entry
// kernel's own coordinate map (HANDOFF §13.3, pinned from cubin_00's PTX):
//
//   u = (i + 0.5) / dim            dim = pw or ph
//   coord = (u * a + b) * m        one fma, then one multiply
//   texel = floor(coord * dim)     point sample -- the model uses a POINT
//                                  sampler, so a point fetch is correct here
//
// The three coefficients are the affine sampling triple the real kernel loads
// from its param struct; a=1,b=0,m=1 is the identity (the old direct sample).
// Xb16 may be null (debug only); Xe is the 64x16 e4m3 patch-A input.
__global__ void k_frontend(const uint8_t* __restrict__ proxy, int pitch,
                           SwFeParams pp, int srcFmt,
                           float s_sel, float s_a, float s_b,
                           uint16_t* __restrict__ Xb16,
                           uint8_t* __restrict__ Xe) {
    const int pw = pp.W, ph = pp.H;
    if (pw <= 0 || ph <= 0) return;
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= SW_TOK) return;
    int pas = t >> 5, wl = t & 31;
    int x = wl & 7, y = (wl >> 3) + 4 * pas;
    float u = ((float)x + 0.5f) / (float)pw;
    float v = ((float)y + 0.5f) / (float)ph;
    float cx = (u * pp.ax_a + pp.ax_b) * pp.ax_m;
    float cy = (v * pp.ay_a + pp.ay_b) * pp.ay_m;
    int px = (int)floorf(cx * (float)pw);
    int py = (int)floorf(cy * (float)ph);
    px = px < 0 ? 0 : (px >= pw ? pw - 1 : px);  // edge clamp (point sample)
    py = py < 0 ? 0 : (py >= ph ? ph - 1 : py);
    float tex[3];
    if (srcFmt == 0) {
        // RGBA8 UNORM -- the Test-7 proxy's format.
        const uint8_t* p = proxy + (size_t)py * (size_t)pitch + (size_t)px * 4u;
        tex[0] = (float)p[0] / 255.0f;
        tex[1] = (float)p[1] / 255.0f;
        tex[2] = (float)p[2] / 255.0f;
    } else {
        // R16G16B16A16 -- what the shim actually stages, because the encode
        // proxy carries the FRAME's format (the in-game log says 8bpp).
        // 1 = _FLOAT (half bits), 2 = _UNORM (u16 / 65535).
        const uint16_t* p =
            (const uint16_t*)(proxy + (size_t)py * (size_t)pitch) +
            (size_t)px * 4u;
        for (int c = 0; c < 3; c++)
            tex[c] = (srcFmt == 1) ? sw_f16_to_f32(p[c])
                                   : (float)p[c] / 65535.0f;
    }
    uint32_t o[4];
    sw_fe_pcg_streams((uint32_t)pp.seed, (uint32_t)x, (uint32_t)(wl >> 3), o);
    float u1 = (float)o[0] * 5.960464477539063e-8f;  // * 2^-24, exact
    float u2 = (float)o[1] * 5.960464477539063e-8f;
    float u3 = (float)o[2] * 5.960464477539063e-8f;
    float u4 = (float)o[3] * 5.960464477539063e-8f;
    float gs[3];
    sw_fe_boxmuller(u1, u2, u3, u4, &gs[0], &gs[1], &gs[2]);
    uint16_t row[16];
    sw_fe_assemble_row(tex, gs, pp.hi180, s_sel, s_a, s_b, row);
    if (Xb16) for (int d = 0; d < 16; d++) Xb16[t * 16 + d] = row[d];
    for (int d = 0; d < 16; d++) Xe[t * 16 + d] = sw_h2e4m3(row[d]);
}

// primitive-proof kernels (exhaustive; host checks FNVs against the
// Python/C++-proven literals MPCUBIC_FNV/QUANT_FNV/F32F16_FNV).
__global__ void k_mpcubic_exhaustive(uint16_t* out) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 65536) {
        __half h = __ushort_as_half((uint16_t)i);
        // NOTE: __hmin/__hmax implement fmin/fmax (NaN -> numeric operand),
        // so NaN inputs take y=-4 here while the mirror takes y=x(NaN);
        // both yield NaN outputs, normalised equal by the host.
        out[i] = __half_as_ushort(sw_mpcubic_half(h));
    }
}

__global__ void k_quant_exhaustive(const uint16_t* inb, uint8_t* out) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 65536) out[i] = sw_h2e4m3(inb[i]);
}

__global__ void k_f32f16_sweep(const uint32_t* inb, uint16_t* out, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = sw_f32_to_f16_bits(__uint_as_float(inb[i]));
}



// ---------------------------------------------------------------------------
#define MPCUBIC_FNV 0x84497DFDu
#define QUANT_FNV 0x1FA338C5u
#define F32F16_FNV 0xA15C4893u

static int sw_cmp_exact_f32(const float* gpu, const float* ref, size_t n,
                            const char* name) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (!sw_f32eq(gpu[i], ref[i])) {
            if (bad == 0) first = i;
            bad++;
        }
    if (bad == 0) {
        std::printf("  PASS  %s exact (%zu elems)\n", name, n);
        return 1;
    }
    std::printf("  FAIL  %s: %zu/%zu differ; first [%zu] gpu %a ref %a\n",
                name, bad, n, first, gpu[first], ref[first]);
    return 0;
}

static int sw_cmp_exact_u16(const uint16_t* gpu, const uint16_t* ref,
                            size_t n, const char* name) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++) {
        uint16_t g = gpu[i], r = ref[i];
        bool gn = (g & 0x7C00u) == 0x7C00u && (g & 0x03FFu) != 0;
        bool rn = (r & 0x7C00u) == 0x7C00u && (r & 0x03FFu) != 0;
        if (!((gn && rn) || g == r)) {
            if (bad == 0) first = i;
            bad++;
        }
    }
    if (bad == 0) {
        std::printf("  PASS  %s exact (%zu elems)\n", name, n);
        return 1;
    }
    std::printf("  FAIL  %s: %zu/%zu differ; first [%zu] gpu %04X ref %04X\n",
                name, bad, n, first, gpu[first], ref[first]);
    return 0;
}

static int sw_cmp_exact_u8(const uint8_t* gpu, const uint8_t* ref, size_t n,
                           const char* name) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (gpu[i] != ref[i]) {
            if (bad == 0) first = i;
            bad++;
        }
    if (bad == 0) {
        std::printf("  PASS  %s exact (%zu elems)\n", name, n);
        return 1;
    }
    std::printf("  FAIL  %s: %zu/%zu differ; first [%zu] gpu %02X ref %02X\n",
                name, bad, n, first, gpu[first], ref[first]);
    return 0;
}

static int sw_cmp_tol(const float* gpu, const float* ref, size_t n,
                      float tol, const char* name) {
    double mx = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)gpu[i] - (double)ref[i]);
        if (!(d <= mx)) {
            mx = d;
            at = i;
        }
    }
    if (mx <= tol) {
        std::printf("  PASS  %s tol %.1e (maxerr %.2e at [%zu])\n", name,
                    (double)tol, mx, at);
        return 1;
    }
    std::printf("  FAIL  %s tol %.1e exceeded: maxerr %.3e at [%zu] "
                "(gpu %a ref %a)\n",
                name, (double)tol, mx, at, gpu[at], ref[at]);
    return 0;
}

static uint32_t sw_host_fnv_u16norm(const uint16_t* p, size_t n) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) {
        uint16_t r = p[i];
        if ((r & 0x7C00u) == 0x7C00u && (r & 0x03FFu) != 0) r = 0x7E00u;
        h ^= (r & 0xFFu);
        h *= 0x01000193u;
        h ^= (r >> 8);
        h *= 0x01000193u;
    }
    return h;
}

// Real-weights device proof (HANDOFF §24): full v4 chain on tensor_000
// (pre-block) with real weights, vs the C++ reference above. Stages
// upstream of k_qknorm's rsqrtf are EXACT (same f32 ops/order — the proven
// principle); downstream stages use the synthetic-proven tols. tensor_001
// stays CPU-smoke-only (its paired-row input has no device counterpart).
static int gpu_realweights_cmd(const char* path) {
    std::printf("=== Swin 1H real-weights chain (tensor_000) ===\n");
    std::vector<uint8_t> raw;
    if (!sw_read_file(path, raw) || raw.size() != 21696) {
        std::printf("FAIL cannot read tensor_000 (%s)\n", path);
        return 1;
    }
    RwChain r;
    sw_rw_chain(raw, true, r);
    int passes = 0, total = 0;
    const int B256 = 256;
    const float s_f16 = sw_f16_to_f32(sw_f32_to_f16_bits(r.s_real));
    uint8_t *d_Xe = nullptr, *d_Wq = nullptr, *d_Wp = nullptr;
    uint8_t *d_W1 = nullptr, *d_W2 = nullptr, *d_Ye = nullptr;
    uint8_t *d_Yfe = nullptr;
    uint16_t *d_Wpt = nullptr, *d_Ab = nullptr, *d_Eb = nullptr;
    float *d_Yp0 = nullptr, *d_Yq = nullptr, *d_S = nullptr, *d_P = nullptr;
    float *d_O = nullptr, *d_Yp = nullptr, *d_H = nullptr, *d_Yf = nullptr;
    float *d_Xb = nullptr, *d_Yfr = nullptr, *d_Qn = nullptr, *d_Kn = nullptr;
    float *d_ssQ = nullptr, *d_ssK = nullptr, *d_Esum = nullptr, *d_B = nullptr;
    float *d_ga = nullptr, *d_gf = nullptr;
    HIPCHECK(hipMalloc(&d_Xe, r.Xe.size()));
    HIPCHECK(hipMalloc(&d_Wpt, r.Wpt_bits.size() * 2));
    HIPCHECK(hipMalloc(&d_Yp0, (size_t)SW_TOK * SW_C * 4));
    HIPCHECK(hipMalloc(&d_Ye, r.Ye.size()));
    HIPCHECK(hipMalloc(&d_Xb, r.Xb.size() * 4));
    HIPCHECK(hipMalloc(&d_W1, r.W1_e4.size()));
    HIPCHECK(hipMalloc(&d_H, r.H.size() * 4));
    HIPCHECK(hipMalloc(&d_Ab, r.Abits.size() * 2));
    HIPCHECK(hipMalloc(&d_W2, r.W2_e4.size()));
    HIPCHECK(hipMalloc(&d_gf, (size_t)SW_C * 4));
    HIPCHECK(hipMalloc(&d_Yf, r.Yf.size() * 4));
    HIPCHECK(hipMalloc(&d_Yfe, r.Yf_e4.size()));
    HIPCHECK(hipMalloc(&d_Wq, r.Wq_e4.size()));
    HIPCHECK(hipMalloc(&d_Yq, r.Yq.size() * 4));
    HIPCHECK(hipMalloc(&d_Qn, r.Qn.size() * 4));
    HIPCHECK(hipMalloc(&d_Kn, r.Kn.size() * 4));
    HIPCHECK(hipMalloc(&d_ssQ, r.ssQ.size() * 4));
    HIPCHECK(hipMalloc(&d_ssK, r.ssK.size() * 4));
    HIPCHECK(hipMalloc(&d_B, r.B.size() * 4));
    HIPCHECK(hipMalloc(&d_S, r.S.size() * 4));
    HIPCHECK(hipMalloc(&d_Eb, (size_t)SW_TOK * SW_TOK * 2));
    HIPCHECK(hipMalloc(&d_Esum, (size_t)SW_TOK * 4));
    HIPCHECK(hipMalloc(&d_P, (size_t)SW_TOK * SW_TOK * 4));
    HIPCHECK(hipMalloc(&d_O, r.O.size() * 4));
    HIPCHECK(hipMalloc(&d_Wp, r.Wp_e4.size()));
    HIPCHECK(hipMalloc(&d_ga, (size_t)SW_C * 4));
    HIPCHECK(hipMalloc(&d_Yfr, r.Yf.size() * 4));
    HIPCHECK(hipMalloc(&d_Yp, r.Yp.size() * 4));
    HIPCHECK(hipMemcpy(d_Xe, r.Xe.data(), r.Xe.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wpt, r.Wpt_bits.data(), r.Wpt_bits.size() * 2,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Xb, r.Xb.data(), r.Xb.size() * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_W1, r.W1_e4.data(), r.W1_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_W2, r.W2_e4.data(), r.W2_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_gf, r.gf.data(), (size_t)SW_C * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wq, r.Wq_e4.data(), r.Wq_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_B, r.B.data(), r.B.size() * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wp, r.Wp_e4.data(), r.Wp_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_ga, r.ga.data(), (size_t)SW_C * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Yfr, r.Yf.data(), r.Yf.size() * 4,
                       hipMemcpyHostToDevice));

    std::printf("\n-- rw patch + FFN (exact upstream of rsqrtf) --\n");
    hipLaunchKernelGGL(k_patch_gemm, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Xe, d_Wpt, d_Yp0, SW_TOK);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.Ypatch.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yp0, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), r.Ypatch.data(), h.size(),
                                   "rw_patch_raw");
    }
    hipLaunchKernelGGL(k_quant_e4m3, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Yp0, d_Ye, SW_TOK * SW_C);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint8_t> h(r.Ye.size());
        HIPCHECK(hipMemcpy(h.data(), d_Ye, h.size(), hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u8(h.data(), r.Ye.data(), h.size(),
                                  "rw_patch_bytes");
    }
    hipLaunchKernelGGL(k_gemm_e4m3, dim3((SW_TOK * SW_H1 + 255) / 256),
                       dim3(B256), 0, 0, d_Ye, d_W1, d_H, SW_TOK, SW_C,
                       SW_H1);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.H.size());
        HIPCHECK(hipMemcpy(h.data(), d_H, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), r.H.data(), h.size(), "rw_ffn1");
    }
    hipLaunchKernelGGL(k_ffn_act, dim3((SW_TOK * SW_H1 + 255) / 256),
                       dim3(B256), 0, 0, d_H, d_Ab, SW_TOK * SW_H1);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint16_t> h(r.Abits.size());
        HIPCHECK(hipMemcpy(h.data(), d_Ab, h.size() * 2,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u16(h.data(), r.Abits.data(), h.size(),
                                   "rw_act");
    }
    hipLaunchKernelGGL(k_ffn2_res, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Ab, d_W2, d_Xb, d_gf, d_Yf);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.Yf.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yf, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), r.Yf.data(), h.size(), "rw_ffn");
    }
    hipLaunchKernelGGL(k_quant_e4m3, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Yf, d_Yfe, SW_TOK * SW_C);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint8_t> h(r.Yf_e4.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yfe, h.size(), hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u8(h.data(), r.Yf_e4.data(), h.size(),
                                  "rw_yfe");
    }

    std::printf("\n-- rw QKV + norm + scores --\n");
    hipLaunchKernelGGL(k_gemm_e4m3, dim3((SW_TOK * SW_QKV + 255) / 256),
                       dim3(B256), 0, 0, d_Yfe, d_Wq, d_Yq, SW_TOK, SW_C,
                       SW_QKV);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.Yq.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yq, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), r.Yq.data(), h.size(), "rw_qkv");
    }
    hipLaunchKernelGGL(k_qknorm, dim3((SW_TOK + 255) / 256), dim3(B256), 0, 0,
                       d_Yq, d_Qn, d_Kn, d_ssQ, d_ssK, s_f16);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> hq(SW_TOK), hk(SW_TOK);
        HIPCHECK(hipMemcpy(hq.data(), d_ssQ, hq.size() * 4,
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(hk.data(), d_ssK, hk.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(hq.data(), r.ssQ.data(), hq.size(),
                                   "rw_sumsqQ");
        total++;
        passes += sw_cmp_exact_f32(hk.data(), r.ssK.data(), hk.size(),
                                   "rw_sumsqK");
    }
    {
        std::vector<float> h(r.Qn.size());
        HIPCHECK(hipMemcpy(h.data(), d_Qn, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.Qn.data(), h.size(), 2e-3f,
                             "rw_Qnorm");
    }
    {
        std::vector<float> h(r.Kn.size());
        HIPCHECK(hipMemcpy(h.data(), d_Kn, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.Kn.data(), h.size(), 2e-3f,
                             "rw_Knorm");
    }
    hipLaunchKernelGGL(k_scores, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Qn, d_Kn, d_B, d_S);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.S.size());
        HIPCHECK(hipMemcpy(h.data(), d_S, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.S.data(), h.size(), 1e-2f,
                             "rw_scores");
    }

    std::printf("\n-- rw softmax + AV + proj --\n");
    hipLaunchKernelGGL(k_smexp, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_S, d_Eb);
    hipLaunchKernelGGL(k_smsum, dim3((SW_TOK + 255) / 256), dim3(B256), 0, 0,
                       d_Eb, d_Esum);
    hipLaunchKernelGGL(k_smnorm, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Eb, d_Esum, d_P);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.P.size());
        HIPCHECK(hipMemcpy(h.data(), d_P, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.P.data(), h.size(), 0.1f,
                             "rw_probs");
    }
    hipLaunchKernelGGL(k_av, dim3((SW_TOK * SW_C + 255) / 256), dim3(B256),
                       0, 0, d_P, d_Yq, d_O);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.O.size());
        HIPCHECK(hipMemcpy(h.data(), d_O, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.O.data(), h.size(), 0.5f,
                             "rw_attn_out");
    }
    hipLaunchKernelGGL(k_proj_res, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_O, d_Wp, d_Yfr, d_ga, d_Yp);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(r.Yp.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yp, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), r.Yp.data(), h.size(), 1e-2f,
                             "rw_blockout");
    }

    std::printf("\n=== Summary: %d / %d passes ===\n", passes, total);
    return (passes == total) ? 0 : 1;
}


// ---------------------------------------------------------------------------
// 3. GPU self-test main (RX 9070 XT, gfx1201).
// Primitive proofs (exhaustive FNVs) first, then the linked chain stage by
// stage against the host mirror. Tolerance rationale: scores carry the
// rsqrtf-vs-RN perturbation (tol 1e-2, measured ~1e-8); the fast-exp
// episode is proven bitwise in isolation and re-proven wired (tol 0.1).
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc >= 3 && !strcmp(argv[1], "--realweights"))
        return gpu_realweights_cmd(argv[2]);
    std::printf("=== Swin 1H unfused port (gfx1201) ===\n");
    hipDeviceProp_t prop{};
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (%s, %d CUs)\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount);
    int passes = 0, total = 0;
    const int B256 = 256;

    // -- Test 0: primitive proofs (exhaustive FNVs) --
    std::printf("\n-- Test 0: primitive proofs --\n");
    {
        uint16_t* d_io = nullptr;
        uint8_t* d_e4 = nullptr;
        uint16_t* d_h = nullptr;
        uint32_t* d_b = nullptr;
        HIPCHECK(hipMalloc(&d_io, 65536 * 2));
        HIPCHECK(hipMalloc(&d_e4, 65536));
        HIPCHECK(hipMalloc(&d_h, 65536 * 2));
        std::vector<uint16_t> h_iota(65536);
        for (int i = 0; i < 65536; i++) h_iota[i] = (uint16_t)i;
        HIPCHECK(hipMemcpy(d_io, h_iota.data(), 65536 * 2,
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_mpcubic_exhaustive, dim3(256), dim3(B256), 0, 0,
                           d_io);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint16_t> h_mp(65536);
        HIPCHECK(hipMemcpy(h_mp.data(), d_io, 65536 * 2,
                           hipMemcpyDeviceToHost));
        uint32_t f = sw_host_fnv_u16norm(h_mp.data(), 65536);
        total++;
        if (f == MPCUBIC_FNV) {
            std::printf("  PASS  device mpcubic exhaustive FNV %08X\n", f);
            passes++;
        } else {
            std::printf("  FAIL  device mpcubic FNV %08X want %08X\n", f,
                        MPCUBIC_FNV);
        }
        // re-upload iota (mpcubic ran in-place above)
        HIPCHECK(hipMemcpy(d_io, h_iota.data(), 65536 * 2,
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_quant_exhaustive, dim3(256), dim3(B256), 0, 0,
                           d_io, d_e4);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint8_t> h_q(65536);
        HIPCHECK(hipMemcpy(h_q.data(), d_e4, 65536, hipMemcpyDeviceToHost));
        f = sw_fnv(h_q.data(), 65536);
        total++;
        if (f == QUANT_FNV) {
            std::printf("  PASS  device h2e4m3 exhaustive FNV %08X\n", f);
            passes++;
        } else {
            std::printf("  FAIL  device h2e4m3 FNV %08X want %08X\n", f,
                        QUANT_FNV);
        }
        const int NS = 2000000;
        std::vector<uint32_t> h_bits(NS);
        uint32_t st = 0x5EEDu;
        for (int i = 0; i < NS; i++) {
            st = sw_lcg_next(st);
            h_bits[i] = st ^ (st >> 3);
        }
        HIPCHECK(hipMalloc(&d_b, (size_t)NS * 4));
        HIPCHECK(hipFree(d_h));
        HIPCHECK(hipMalloc(&d_h, (size_t)NS * 2));
        HIPCHECK(hipMemcpy(d_b, h_bits.data(), (size_t)NS * 4,
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_f32f16_sweep, dim3((NS + 255) / 256),
                           dim3(B256), 0, 0, d_b, d_h, NS);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint16_t> h_c(NS);
        HIPCHECK(hipMemcpy(h_c.data(), d_h, (size_t)NS * 2,
                           hipMemcpyDeviceToHost));
        f = sw_host_fnv_u16norm(h_c.data(), NS);
        total++;
        if (f == F32F16_FNV) {
            std::printf("  PASS  device f32->f16 sweep FNV %08X\n", f);
            passes++;
        } else {
            std::printf("  FAIL  device f32->f16 FNV %08X want %08X\n", f,
                        F32F16_FNV);
        }
        HIPCHECK(hipFree(d_io));
        HIPCHECK(hipFree(d_e4));
        HIPCHECK(hipFree(d_h));
        HIPCHECK(hipFree(d_b));
    }

    // -- Tests 1-6: linked chain vs host mirror --
    SwChain m;
    sw_mirror_chain(m);
    std::vector<float> ga(SW_C), gf(SW_C);
    for (int o = 0; o < SW_C; o++) {
        ga[o] = (float)(o % 2);
        gf[o] = (float)((o + 1) % 2);
    }
    uint8_t *d_Xe = nullptr, *d_Wq = nullptr, *d_Wp = nullptr;
    uint8_t *d_W1 = nullptr, *d_W2 = nullptr, *d_Ye = nullptr;
    uint8_t *d_Yfe = nullptr;
    uint16_t*d_Wpt = nullptr, *d_Ab = nullptr;
    float *d_Yp0 = nullptr, *d_Yq = nullptr, *d_S = nullptr, *d_P = nullptr;
    float *d_O = nullptr, *d_Yp = nullptr, *d_H = nullptr, *d_Yf = nullptr;
    float *d_ga = nullptr, *d_gf = nullptr, *d_Xb = nullptr;
    float *d_Qn = nullptr, *d_Kn = nullptr, *d_ssQ = nullptr, *d_ssK = nullptr;
    float *d_Yfr = nullptr, *d_B = nullptr;
    uint16_t* d_Eb = nullptr;
    float* d_Esum = nullptr;
    HIPCHECK(hipMalloc(&d_Xe, m.Xe.size()));
    HIPCHECK(hipMalloc(&d_Wpt, m.Wpt_bits.size() * 2));
    HIPCHECK(hipMalloc(&d_Yp0, (size_t)SW_TOK * SW_C * 4));
    HIPCHECK(hipMalloc(&d_Ye, m.Ye.size()));
    HIPCHECK(hipMalloc(&d_Wq, m.Wq_e4.size()));
    HIPCHECK(hipMalloc(&d_Yq, m.Yq.size() * 4));
    HIPCHECK(hipMalloc(&d_S, m.S.size() * 4));
    HIPCHECK(hipMalloc(&d_Eb, m.Eb.size() * 2));
    HIPCHECK(hipMalloc(&d_Esum, m.Esum.size() * 4));
    HIPCHECK(hipMalloc(&d_P, m.P.size() * 4));
    HIPCHECK(hipMalloc(&d_O, m.O.size() * 4));
    HIPCHECK(hipMalloc(&d_Wp, m.Wp_e4.size()));
    HIPCHECK(hipMalloc(&d_ga, (size_t)SW_C * 4));
    HIPCHECK(hipMalloc(&d_gf, (size_t)SW_C * 4));
    HIPCHECK(hipMalloc(&d_Yp, m.Yp.size() * 4));
    HIPCHECK(hipMalloc(&d_W1, m.W1_e4.size()));
    HIPCHECK(hipMalloc(&d_H, m.H.size() * 4));
    HIPCHECK(hipMalloc(&d_Ab, m.Abits.size() * 2));
    HIPCHECK(hipMalloc(&d_W2, m.W2_e4.size()));
    HIPCHECK(hipMalloc(&d_Yf, m.Yf.size() * 4));
    HIPCHECK(hipMalloc(&d_Yfe, m.Yf_e4.size()));
    // Residual inputs are f32 (host-uploaded; bitwise == device dequant of
    // the view bytes, a path already proven by the QKV stage reading d_Ye).
    // §21 order: d_Xb feeds the gate_ffn residual, d_Yfr the gate_attn one.
    HIPCHECK(hipMalloc(&d_Xb, m.Xb.size() * 4));
    HIPCHECK(hipMemcpy(d_Xb, m.Xb.data(), m.Xb.size() * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMalloc(&d_Yfr, m.Yf.size() * 4));
    HIPCHECK(hipMemcpy(d_Yfr, m.Yf.data(), m.Yf.size() * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMalloc(&d_B, m.B.size() * 4));
    HIPCHECK(hipMemcpy(d_B, m.B.data(), m.B.size() * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMalloc(&d_Qn, m.Qn.size() * 4));
    HIPCHECK(hipMalloc(&d_Kn, m.Kn.size() * 4));
    HIPCHECK(hipMalloc(&d_ssQ, m.ssQ.size() * 4));
    HIPCHECK(hipMalloc(&d_ssK, m.ssK.size() * 4));
    HIPCHECK(hipMemcpy(d_Xe, m.Xe.data(), m.Xe.size(), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wpt, m.Wpt_bits.data(), m.Wpt_bits.size() * 2,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wq, m.Wq_e4.data(), m.Wq_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_Wp, m.Wp_e4.data(), m.Wp_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_W1, m.W1_e4.data(), m.W1_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_W2, m.W2_e4.data(), m.W2_e4.size(),
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_ga, ga.data(), (size_t)SW_C * 4,
                       hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_gf, gf.data(), (size_t)SW_C * 4,
                       hipMemcpyHostToDevice));

    std::printf("\n-- Test 1: patch embed --\n");
    hipLaunchKernelGGL(k_patch_gemm, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Xe, d_Wpt, d_Yp0, SW_TOK);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.Ypatch.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yp0, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), m.Ypatch.data(), h.size(),
                                   "patch_raw");
    }
    hipLaunchKernelGGL(k_quant_e4m3, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Yp0, d_Ye, SW_TOK * SW_C);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint8_t> h(m.Ye.size());
        HIPCHECK(hipMemcpy(h.data(), d_Ye, h.size(), hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u8(h.data(), m.Ye.data(), h.size(),
                                  "patch_bytes");
    }

    // §21 order: FFN half runs first (QKV-mma-A = quant(contract-chain-end)
    // = quant(Yf)); the attn half consumes the FFN-half output.
    std::printf("\n-- Test 2: FFN expand + act --\n");
    hipLaunchKernelGGL(k_gemm_e4m3, dim3((SW_TOK * SW_H1 + 255) / 256),
                       dim3(B256), 0, 0, d_Ye, d_W1, d_H, SW_TOK, SW_C,
                       SW_H1);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.H.size());
        HIPCHECK(hipMemcpy(h.data(), d_H, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), m.H.data(), h.size(), "ffn1");
    }
    hipLaunchKernelGGL(k_ffn_act, dim3((SW_TOK * SW_H1 + 255) / 256),
                       dim3(B256), 0, 0, d_H, d_Ab, SW_TOK * SW_H1);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint16_t> h(m.Abits.size());
        HIPCHECK(hipMemcpy(h.data(), d_Ab, h.size() * 2,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u16(h.data(), m.Abits.data(), h.size(), "act");
    }

    std::printf("\n-- Test 3: FFN contract + gate_ffn residual --\n");
    hipLaunchKernelGGL(k_ffn2_res, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Ab, d_W2, d_Xb, d_gf, d_Yf);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.Yf.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yf, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.Yf.data(), h.size(), 1.0f, "ffn");
    }
    hipLaunchKernelGGL(k_quant_e4m3, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_Yf, d_Yfe, SW_TOK * SW_C);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());

    std::printf("\n-- Test 4: QKV + scores --\n");
    hipLaunchKernelGGL(k_gemm_e4m3, dim3((SW_TOK * SW_QKV + 255) / 256),
                       dim3(B256), 0, 0, d_Yfe, d_Wq, d_Yq, SW_TOK, SW_C,
                       SW_QKV);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.Yq.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yq, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), m.Yq.data(), h.size(), "qkv");
    }
    // QK-norm (HANDOFF §17): sumsq bit-exact; Qn/Kn within rsqrtf hw error.
    hipLaunchKernelGGL(k_qknorm, dim3((SW_TOK + 255) / 256), dim3(B256), 0, 0,
                       d_Yq, d_Qn, d_Kn, d_ssQ, d_ssK,
                       sw_f16_to_f32(sw_f32_to_f16_bits(SW_S_TEST)));
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> hq(SW_TOK), hk(SW_TOK);
        HIPCHECK(hipMemcpy(hq.data(), d_ssQ, hq.size() * 4,
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(hk.data(), d_ssK, hk.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(hq.data(), m.ssQ.data(), hq.size(),
                                   "sumsqQ");
        total++;
        passes += sw_cmp_exact_f32(hk.data(), m.ssK.data(), hk.size(),
                                   "sumsqK");
    }
    {
        std::vector<float> h(m.Qn.size());
        HIPCHECK(hipMemcpy(h.data(), d_Qn, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.Qn.data(), h.size(), 2e-3f, "Qnorm");
    }
    {
        std::vector<float> h(m.Kn.size());
        HIPCHECK(hipMemcpy(h.data(), d_Kn, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.Kn.data(), h.size(), 2e-3f, "Knorm");
    }
    hipLaunchKernelGGL(k_scores, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Qn, d_Kn, d_B, d_S);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        // TOL (not exact): d_Qn/d_Kn carry device-rsqrtf error vs the
        // mirror's 1/sqrtf. Kernel order itself is unchanged (still the
        // exact k-ascending f32 chain proven pre-norm); d_B is exact zeros
        // here (synthetic stand-in for the §21 f16 bias).
        std::vector<float> h(m.S.size());
        HIPCHECK(hipMemcpy(h.data(), d_S, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.S.data(), h.size(), 1e-2f, "scores");
    }

    std::printf("\n-- Test 5: fast-exp softmax + AV --\n");
    // Isolation proofs (mirror inputs -> bitwise recipe proof). d_S currently
    // holds device scores (rsqrtf-perturbed); the chained run follows after.
    HIPCHECK(hipMemcpy(d_S, m.S.data(), m.S.size() * 4,
                       hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_smexp, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_S, d_Eb);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<uint16_t> h(m.Eb.size());
        HIPCHECK(hipMemcpy(h.data(), d_Eb, h.size() * 2,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_u16(h.data(), m.Eb.data(), h.size(), "smexp");
    }
    HIPCHECK(hipMemcpy(d_Eb, m.Eb.data(), m.Eb.size() * 2,
                       hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_smsum, dim3((SW_TOK + 255) / 256), dim3(B256), 0, 0,
                       d_Eb, d_Esum);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.Esum.size());
        HIPCHECK(hipMemcpy(h.data(), d_Esum, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), m.Esum.data(), h.size(),
                                   "Esums");
    }
    HIPCHECK(hipMemcpy(d_Esum, m.Esum.data(), m.Esum.size() * 4,
                       hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_smnorm, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Eb, d_Esum, d_P);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        // EXACT: both sides use RN f32 division + identical f16 roundings.
        // Contingency (see runbook triage): if hw fdiv ever shows 1-ulp
        // diffs vs host division, drop this to tol 2e-3 and report.
        std::vector<float> h(m.P.size());
        HIPCHECK(hipMemcpy(h.data(), d_P, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_exact_f32(h.data(), m.P.data(), h.size(),
                                   "probs_iso");
    }
    // Chained wiring proof: recompute S from the device Qn/Kn, then run the
    // whole episode on device values. tol 0.1 covers the worst-case
    // f16-boundary straddle from the rsqrtf input perturbation (~0.02
    // single-element); wiring bugs show at O(0.3+). Kernels already proven
    // above, so this only needs to prove the buffers flow correctly.
    hipLaunchKernelGGL(k_scores, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Qn, d_Kn, d_B, d_S);
    hipLaunchKernelGGL(k_smexp, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_S, d_Eb);
    hipLaunchKernelGGL(k_smsum, dim3((SW_TOK + 255) / 256), dim3(B256), 0, 0,
                       d_Eb, d_Esum);
    hipLaunchKernelGGL(k_smnorm, dim3((SW_TOK * SW_TOK + 255) / 256),
                       dim3(B256), 0, 0, d_Eb, d_Esum, d_P);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.P.size());
        HIPCHECK(hipMemcpy(h.data(), d_P, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.P.data(), h.size(), 0.1f,
                             "probs_chain");
    }
    hipLaunchKernelGGL(k_av, dim3((SW_TOK * SW_C + 255) / 256), dim3(B256),
                       0, 0, d_P, d_Yq, d_O);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.O.size());
        HIPCHECK(hipMemcpy(h.data(), d_O, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.O.data(), h.size(), 0.5f, "attn_out");
    }

    // Block output: proj residual rides on Yf (d_Yfr), not Xb (§21 order).
    std::printf("\n-- Test 6: proj + gate_attn residual --\n");
    hipLaunchKernelGGL(k_proj_res, dim3((SW_TOK * SW_C + 255) / 256),
                       dim3(B256), 0, 0, d_O, d_Wp, d_Yfr, d_ga, d_Yp);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    {
        std::vector<float> h(m.Yp.size());
        HIPCHECK(hipMemcpy(h.data(), d_Yp, h.size() * 4,
                           hipMemcpyDeviceToHost));
        total++;
        passes += sw_cmp_tol(h.data(), m.Yp.data(), h.size(), 1.0f, "proj");
    }

    // -- Test 7: front-end proxy -> X (HANDOFF §34) --
    // k_frontend on a fixed 8x8 RGBA8 gradient vs the §1b host mirror
    // (fe_pcg_streams + fe_boxmuller_f32 + fe_assemble_row). Xb16 f16 row
    // compared as f32 with tol 1e-2: BM libm noise measures ~2e-7
    // (prep test 8), wiring/pixel/dim bugs show at O(0.1+). Xe bytes are
    // NOT compared exactly here: e4m3 boundary straddle (§5) could flake
    // an exact compare on a correct kernel; quant itself is proven
    // exhaustive in Test 0. Distinct R/G/B gradients + distinct exposure
    // packs make every pixel, channel and dim position-sensitive.
    // s_sel is 0.75 (not the canonical 1.0) to break the dim-3 tie:
    // dims 11/12 (both s_sel by recipe) stay tied by construction --
    // swapping them is value-unobservable no matter the params.
    std::printf("\n-- Test 7: front-end proxy -> X --\n");
    {
        const int PW = 8, PH = 8, PITCH = 32;
        std::vector<uint8_t> proxy((size_t)PH * PITCH, 0);
        for (int y = 0; y < PH; y++)
            for (int x = 0; x < PW; x++) {
                proxy[(size_t)y * PITCH + (size_t)x * 4 + 0] =
                    (uint8_t)((x * 37 + y * 5 + 11) & 255);
                proxy[(size_t)y * PITCH + (size_t)x * 4 + 1] =
                    (uint8_t)((y * 53 + x * 7 + 3) & 255);
                proxy[(size_t)y * PITCH + (size_t)x * 4 + 2] =
                    (uint8_t)((x * 17 + y * 29 + 1) & 255);
                proxy[(size_t)y * PITCH + (size_t)x * 4 + 3] = 255;
            }
        uint8_t* d_proxy = nullptr;
        uint16_t* d_Xb16 = nullptr;
        uint8_t* d_feXe = nullptr;
        HIPCHECK(hipMalloc(&d_proxy, proxy.size()));
        HIPCHECK(hipMalloc(&d_Xb16, (size_t)SW_TOK * SW_D_IN * 2));
        HIPCHECK(hipMalloc(&d_feXe, (size_t)SW_TOK * SW_D_IN));
        HIPCHECK(hipMemcpy(d_proxy, proxy.data(), proxy.size(),
                           hipMemcpyHostToDevice));
        const uint32_t FE_SEED = 0x12345678u;  // oracle FE_SEED
        // The sampling parameters now come from the real 264-byte block. The
        // affine is the identity (a=1,b=0,m=1), so this is still the old
        // direct sample -- the proof is that the comparison below stayed
        // bit-for-bit unchanged when the block replaced the scalars.
        SwFeParams fp;
        memset(&fp, 0, sizeof(fp));
        fp.W = PW; fp.H = PH; fp.seed = (int)FE_SEED; fp.hi180 = 3.0f;
        fp.ax_a = SW_FE_IDENTITY_A; fp.ax_b = SW_FE_IDENTITY_B;
        fp.ax_m = SW_FE_IDENTITY_M;
        fp.ay_a = SW_FE_IDENTITY_A; fp.ay_b = SW_FE_IDENTITY_B;
        fp.ay_m = SW_FE_IDENTITY_M;
        hipLaunchKernelGGL(k_frontend, dim3(1), dim3(B256), 0, 0,
                           d_proxy, PITCH, fp, 0 /* RGBA8 */,
                           0.75f, 0.5f, 0.25f, d_Xb16, d_feXe);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint16_t> h_b16((size_t)SW_TOK * SW_D_IN);
        HIPCHECK(hipMemcpy(h_b16.data(), d_Xb16, h_b16.size() * 2,
                           hipMemcpyDeviceToHost));
        std::vector<float> h_got(h_b16.size()), h_ref(h_b16.size());
        for (size_t i = 0; i < h_b16.size(); i++)
            h_got[i] = sw_f16_to_f32(h_b16[i]);
        for (int t = 0; t < SW_TOK; t++) {
            int pas = t >> 5, wl = t & 31;
            int x = wl & 7, y = (wl >> 3) + 4 * pas;
            // The same coordinate map the kernel uses, so this stays a twin
            // when the affine is not the identity.
            float su = ((float)x + 0.5f) / (float)fp.W;
            float sv = ((float)y + 0.5f) / (float)fp.H;
            int px = (int)floorf((su * fp.ax_a + fp.ax_b) * fp.ax_m * (float)fp.W);
            int py = (int)floorf((sv * fp.ay_a + fp.ay_b) * fp.ay_m * (float)fp.H);
            px = px < 0 ? 0 : (px >= fp.W ? fp.W - 1 : px);
            py = py < 0 ? 0 : (py >= fp.H ? fp.H - 1 : py);
            float tex[3];
            for (int c = 0; c < 3; c++)
                tex[c] = (float)proxy[(size_t)py * PITCH +
                                     (size_t)px * 4 + c] / 255.0f;
            uint32_t o[4];
            fe_pcg_streams(FE_SEED, (uint32_t)x, (uint32_t)(wl >> 3),
                           o);
            const float inv24 = 5.960464477539063e-8f;  // 2^-24
            float g[3];
            fe_boxmuller_f32(o[0] * inv24, o[1] * inv24,
                             o[2] * inv24, o[3] * inv24,
                             &g[0], &g[1], &g[2]);
            uint16_t row[16];
            fe_assemble_row(tex, g, 3.0f, 0.75f, 0.5f, 0.25f, row);
            for (int d = 0; d < SW_D_IN; d++)
                h_ref[(size_t)t * SW_D_IN + d] =
                    sw_f16_to_f32(row[d]);
        }
        total++;
        passes += sw_cmp_tol(h_got.data(), h_ref.data(), h_got.size(),
                             1e-2f, "frontend");
        HIPCHECK(hipFree(d_proxy));
        HIPCHECK(hipFree(d_Xb16));
        HIPCHECK(hipFree(d_feXe));
    }
    std::printf("\n=== Summary: %d / %d passes ===\n", passes, total);
    return (passes == total) ? 0 : 1;
}
#endif  // !SWIN1H_HOST_ONLY
