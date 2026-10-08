#pragma once
// Transposed C64/C128/C256 Swin block (S336): swin_block_fp8's FFN and
// attention in one launch, carried as [feature][token] like k_c32_t (see
// c32_t.hip for the operand layout that makes this work). Each wave owns one
// 16-token tile of the window: the FFN is per token, so it runs entirely in
// that wave's registers; the attention shares only each head's K and V
// (double-buffered in LDS, one barrier a head). Same stage values, rounding
// boundaries and reduction orders as k_swin_ffn + k_swin_attention (and the
// few-window split forms), so the outputs are bit-identical
// (build/swin_t_bench.exe checks). RESIDENT_SWIN_T=0 runs swin_fp8::run.
// Needs swin_block_fp8.hip and c32_t.hip included first.

namespace swin_t {
using swin_fused::Geometry;
using c32t::acc8;
using c32t::op8;
using c32t::pack;
using c32t::mma;
using c32t::arow;
typedef unsigned char u8;

#if defined(H70_NO_FP8)
inline bool enabled = false;
#else
inline bool enabled = [] { const char* v = getenv("RESIDENT_SWIN_T"); return !(v && *v == '0'); }();
#endif

__device__ inline float byte_of(op8 v, int e) {
    return __builtin_amdgcn_cvt_f32_fp8(int(unsigned(e < 4 ? v[0] : v[1]) >> (8*(e % 4))), 0);
}
__device__ inline void load8(const float* p, float* v) {
    const float4 lo = *reinterpret_cast<const float4*>(p), hi = *reinterpret_cast<const float4*>(p + 4);
    v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w; v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
}

template<int C>
__global__ __launch_bounds__(128) void k_swin_t(const float* __restrict__ input, Geometry g,
        const u8* __restrict__ w1, const u8* __restrict__ w2, const u8* __restrict__ w3,
        const float* __restrict__ skip, const u8* __restrict__ wqkv, const float* __restrict__ scales,
        const float* __restrict__ bias, const u8* __restrict__ wp, const float* __restrict__ attention_skip,
        float* output, float* raw_out) {
    constexpr int FT = C/16, KS = C/32, HEADS = C/32, HID = 4*C;
    __shared__ __attribute__((aligned(16))) u8 Ks[2][64*32];  // [key][d], per head parity
    __shared__ __attribute__((aligned(16))) u8 Vt[2][32*64];  // [d][key]
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32, n = lane%16, h = lane/16;
    const int token = wave*16 + n, win = blockIdx.x;
    int px, py;
    const bool valid = g.pixel(win, token, px, py);
    const float* src = input + (size_t(valid ? py : 0)*g.width + (valid ? px : 0))*C;

    // x^T: FT B operands (channels s*16 + h*8 + e).
    op8 xb[FT];
    for (int s = 0; s < FT; ++s) {
        float v[8];
        load8(src + s*16 + h*8, v);  // a valid address either way: no branch
        for (int e = 0; e < 8; ++e) v[e] = valid ? c32f_h(v[e]) : 0.0f;
        xb[s] = pack(v);
    }
    // FFN: hidden 32 rows at a time (C/32-slice sums, gate), each slice
    // folded at once into the middle (HID/32 slices in order, from zero).
    _Float16 mid[FT][8];
    for (int f = 0; f < FT; ++f) for (int e = 0; e < 8; ++e) mid[f][e] = 0;
#pragma unroll 1
    for (int j = 0; j < HID/32; ++j) {
        op8 hb[2];
        for (int u = 0; u < 2; ++u) {
            const u8* wr = w1 + size_t((2*j+u)*16)*C;
            float acc[8] = {};
            for (int k = 0; k < KS; ++k) {
                acc8 part = {};
                part = mma(arow(wr, C, 32*k+16), xb[2*k+1], mma(arow(wr, C, 32*k), xb[2*k], part));
                for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part[e]);
            }
            float v[8];
            for (int e = 0; e < 8; ++e) v[e] = c32t::gate(acc[e]);
            hb[u] = pack(v);
        }
        for (int f = 0; f < FT; ++f) {
            const u8* wr = w2 + size_t(f*16)*HID;
            acc8 part = {};
            part = mma(arow(wr, HID, 32*j+16), hb[1], mma(arow(wr, HID, 32*j), hb[0], part));
            for (int e = 0; e < 8; ++e) mid[f][e] = (_Float16)c32f_h(float(mid[f][e])+part[e]);
        }
    }
    op8 mb[FT];
    for (int f = 0; f < FT; ++f) {
        float v[8];
        for (int e = 0; e < 8; ++e) v[e] = float(mid[f][e]);
        mb[f] = pack(v);
    }
    // W3 onto the skip residual -> the E4M3 feature (B operands for QKV), and
    // the attention residual from that feature.
    op8 fb[FT];
    _Float16 res[FT][8];
    for (int f = 0; f < FT; ++f) {
        float x[8], acc[8];
        load8(src + f*16 + h*8, x);
        for (int e = 0; e < 8; ++e) x[e] = valid ? x[e] : 0.0f;
        for (int e = 0; e < 8; ++e) acc[e] = c32f_h(x[e]*skip[f*16 + h*8 + e]);
        const u8* wr = w3 + size_t(f*16)*C;
        for (int k = 0; k < KS; ++k) {
            acc8 part = {};
            part = mma(arow(wr, C, 32*k+16), mb[2*k+1], mma(arow(wr, C, 32*k), mb[2*k], part));
            for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part[e]);
        }
        fb[f] = pack(acc);
        for (int e = 0; e < 8; ++e) res[f][e] = (_Float16)c32f_h(byte_of(fb[f], e)*attention_skip[f*16 + h*8 + e]);
    }
#pragma unroll 1
    for (int hd = 0; hd < HEADS; ++hd) {
        u8* K = Ks[hd & 1];
        u8* V = Vt[hd & 1];
        op8 qb[2];
#pragma unroll 1
        for (int m = 0; m < 3; ++m) {
            const u8* wr = wqkv + size_t(m*C + hd*32)*C;
            float c0[8] = {}, c1[8] = {};
            for (int k = 0; k < KS; ++k) {
                acc8 p0 = {}, p1 = {};
                p0 = mma(arow(wr, C, 32*k+16), fb[2*k+1], mma(arow(wr, C, 32*k), fb[2*k], p0));
                p1 = mma(arow(wr + 16*C, C, 32*k+16), fb[2*k+1], mma(arow(wr + 16*C, C, 32*k), fb[2*k], p1));
                for (int e = 0; e < 8; ++e) { c0[e] = c32f_h(c0[e]+p0[e]); c1[e] = c32f_h(c1[e]+p1[e]); }
            }
            if (m == 2) {
                for (int e = 0; e < 8; ++e) {
                    V[(h*8+e)*64 + token] = c32f8::q8h(c0[e]);
                    V[(16+h*8+e)*64 + token] = c32f8::q8h(c1[e]);
                }
                continue;
            }
            // split512_norm_inverse_fast as a column reduction (see k_c32_t).
            float s[8], p[4];
            for (int e = 0; e < 8; ++e) s[e] = split512_half_sum(c0[e]*c0[e], c32f_h(c1[e]*c1[e]));
            for (int q = 0; q < 4; ++q) p[q] = c32f_h(s[2*q] + s[2*q+1]);
            for (int q = 0; q < 4; ++q) p[q] = c32f_h(p[q] + __shfl_xor(p[q], 16));
            const float total = c32f_h(c32f_h(p[0] + p[2]) + c32f_h(p[1] + p[3]));
            const float inverse = c32f_h(1.f/sqrtf(fmaxf(total, .00006198883056640625f)));
            const float scale = c32f_h(scales[hd]);
            for (int e = 0; e < 8; ++e) {
                float q0 = c32f_h(c0[e]*inverse), q1 = c32f_h(c1[e]*inverse);
                if (m == 0) { q0 = c32f_h(q0*scale); q1 = c32f_h(q1*scale); }
                c0[e] = q0; c1[e] = q1;
            }
            if (m == 0) {
                qb[0] = pack(c0); qb[1] = pack(c1);
            } else {
                *reinterpret_cast<op8*>(K + token*32 + h*8) = pack(c0);
                *reinterpret_cast<op8*>(K + token*32 + 16 + h*8) = pack(c1);
            }
        }
        __syncthreads();
        float ex[4][8];
        for (int k = 0; k < 4; ++k) {
            acc8 c = {};
            c = mma(arow(K + k*16*32, 32, 16), qb[1], mma(arow(K + k*16*32, 32, 0), qb[0], c));
            float bv[8];
            load8(bias + (size_t(hd*64 + token))*64 + k*16 + h*8, bv);
            for (int e = 0; e < 8; ++e) ex[k][e] = c32t::expo(c[e], bv[e]);
        }
        float den;
        {   // split512_den order (see k_c32_t).
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
        op8 cb[2];
        for (int d = 0; d < 2; ++d) {
            float acc[8] = {};
            for (int s = 0; s < 2; ++s) {
                acc8 part = {};
                part = mma(arow(V + d*16*64, 64, (2*s+1)*16), pb[2*s+1], mma(arow(V + d*16*64, 64, (2*s)*16), pb[2*s], part));
                for (int e = 0; e < 8; ++e) acc[e] = c32f_h(acc[e]+part[e]);
            }
            cb[d] = pack(acc);
        }
        // Projection K slice `hd` onto the residual, heads in order.
        for (int f = 0; f < FT; ++f) {
            const u8* wr = wp + size_t(f*16)*C;
            acc8 part = {};
            part = mma(arow(wr, C, hd*32+16), cb[1], mma(arow(wr, C, hd*32), cb[0], part));
            for (int e = 0; e < 8; ++e) res[f][e] = (_Float16)c32f_h(float(res[f][e])+part[e]);
            }
    }
    if (!valid) return;
    float* out = output + (size_t(py)*g.width + px)*C;
    float* raw = raw_out ? raw_out + (size_t(py)*g.width + px)*C : nullptr;
    for (int f = 0; f < FT; ++f) {
        float v[8], q[8];
        for (int e = 0; e < 8; ++e) { v[e] = float(res[f][e]); q[e] = split512_fp8(v[e]); }
        float4* o = reinterpret_cast<float4*>(out + f*16 + h*8);
        o[0] = make_float4(q[0], q[1], q[2], q[3]); o[1] = make_float4(q[4], q[5], q[6], q[7]);
        if (raw) {
            float4* r = reinterpret_cast<float4*>(raw + f*16 + h*8);
            r[0] = make_float4(v[0], v[1], v[2], v[3]); r[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
    }
}

// swin_fp8::run's contract; false means "run the reference kernels".
template<int C>
inline bool run(const float* input, float* output, float* raw_out, int width, int height, int shift,
                const swin_fused::Block& w, void* scratch) {
    if (c512_resident::exact_math) return false;
    if (!enabled) return swin_fp8::run<C>(input, output, raw_out, width, height, shift, w, scratch);
    hipStream_t s = c512_resident::stream;
    Geometry g{width, height, (shift&1) ? 4 : 0, (shift&2) ? 4 : 0, 0};
    g.pw = ((width+g.px+7)/8)*8;
    int ph = ((height+g.py+7)/8)*8, windows = (g.pw/8)*(ph/8);
    auto f8w = [&](const float* W, size_t n) { return swin_fp8::weights(s, W, n); };
    hipLaunchKernelGGL(k_swin_t<C>, dim3(windows), dim3(128), 0, s, input, g,
                       f8w(w.w1, size_t(4*C)*C), f8w(w.w2, size_t(C)*4*C), f8w(w.w3, size_t(C)*C), w.skip,
                       f8w(w.qkv, size_t(3*C)*C), w.scales, w.bias, f8w(w.projection, size_t(C)*C),
                       w.attention_skip, output, raw_out);
    HIP_CHECK(hipGetLastError());
    if (c512_resident::launch_hook) c512_resident::launch_hook(C == 64 ? "k_swin_t<64>" : C == 128 ? "k_swin_t<128>" : "k_swin_t<256>");
    return true;
}
}  // namespace swin_t
