#pragma once
// The C512 block's grouped FFN in one launch (S339): for each of the eight
// 64-channel groups, the 64 -> 256 expand with its gate and the 256 -> 64
// contract, E4M3 bytes in (mixed8) and out (branch8). The two transposed
// GEMMs it replaces (k_wmma_gemm8t GATE then FP8, groups 8) wrote and read
// the 15 MB hidden tensor at 5120x1440 and ran at ~14 TFLOPS (K = 64 per
// expand output: launch and staging overheads dominate).
// Same arithmetic as those GEMMs: 32-deep slices (two 16-deep FP8 WMMAs from
// zero) summed onto the accumulator from zero (wmma_t::acc_add when the GEMM
// takes its SR form, else f32-add-then-half), the GATE epilogue's gate (the
// f16 form, wmma_t::gate_pack: equal for every half) and the FP8 epilogue's
// E4M3 bytes. Transposed operands as in c32_t.hip: an expand accumulator
// tile is the contract's B operand.
// Each wave: one group, four 16-token tiles (the weight fragments serve
// four products); hidden tiles 2s, 2s+1 are made just before contract slice s.
// Needs wmma_t.hip.

namespace split512_ffn {
// RESIDENT_C512_FFN=0: the two grouped GEMMs instead.
inline bool enabled = [] { const char* v = getenv("RESIDENT_C512_FFN"); return !(v && *v == '0'); }();
using wmma_t::acc8;
using wmma_t::op8;
using wmma_t::h8;

template<bool SR>
__global__ __launch_bounds__(128) void k_split512_ffn_t(const unsigned char* __restrict__ x8,
        const unsigned char* __restrict__ w1, const unsigned char* __restrict__ w2,
        unsigned char* out8, int tokens) {
    const int lane = threadIdx.x % 32, n = lane % 16, h = lane / 16;
    const int item = blockIdx.x*4 + threadIdx.x/32;  // (token block, group)
    const int group = item % 8, t0 = (item / 8)*64;
    if (t0 >= tokens) return;  // whole waves only (the WMMAs need every lane)
    const unsigned char* W1 = w1 + size_t(group)*256*64;   // [256 hidden][64]
    const unsigned char* W2 = w2 + size_t(group)*64*256;   // [64 out][256 hidden]
    auto add = [](h8& acc, const acc8& part) {
        if constexpr (SR) wmma_t::acc_add(acc, part); else wmma_t::acc_add2(acc, part);
    };
    // x^T: four 16-deep B operands a tile (channels group*64 + k*16 + h*8 + e).
    op8 xb[4][4];
    for (int u = 0; u < 4; ++u) {
        const int t = t0 + u*16 + n;
        for (int k = 0; k < 4; ++k) {
            uint2 v = uint2{0, 0};
            if (t < tokens) v = *reinterpret_cast<const uint2*>(x8 + size_t(t)*512 + group*64 + k*16 + h*8);
            xb[u][k] = op8{int(v.x), int(v.y)};
        }
    }
    h8 out[4][4];  // [tile][output tile f]: halves
    for (int u = 0; u < 4; ++u) for (int f = 0; f < 4; ++f) out[u][f] = h8{};
    for (int s = 0; s < 8; ++s) {
        // Hidden tiles 2s and 2s+1 (rows 16t..): two 32-deep slices, the gate.
        op8 hb[4][2];
        for (int q = 0; q < 2; ++q) {
            const int t = 2*s + q;
            const op8 a0 = wmma_t::arow(W1 + t*16*64, 64, 0), a1 = wmma_t::arow(W1 + t*16*64, 64, 16);
            const op8 a2 = wmma_t::arow(W1 + t*16*64, 64, 32), a3 = wmma_t::arow(W1 + t*16*64, 64, 48);
            for (int u = 0; u < 4; ++u) {
                h8 acc = {};
                acc8 part = {};
                part = wmma_t::mma(a1, xb[u][1], wmma_t::mma(a0, xb[u][0], part));
                add(acc, part);
                part = acc8{};
                part = wmma_t::mma(a3, xb[u][3], wmma_t::mma(a2, xb[u][2], part));
                add(acc, part);
                hb[u][q] = wmma_t::gate_pack(acc);
            }
        }
        // Contract slice s (hidden 32s..32s+31) onto the four output tiles.
        for (int f = 0; f < 4; ++f) {
            const op8 a0 = wmma_t::arow(W2 + f*16*256, 256, 32*s), a1 = wmma_t::arow(W2 + f*16*256, 256, 32*s + 16);
            for (int u = 0; u < 4; ++u) {
                acc8 part = {};
                part = wmma_t::mma(a1, hb[u][1], wmma_t::mma(a0, hb[u][0], part));
                add(out[u][f], part);
            }
        }
    }
    for (int u = 0; u < 4; ++u) {
        const int t = t0 + u*16 + n;
        if (t >= tokens) continue;
        for (int f = 0; f < 4; ++f) {
            float v[8];
            for (int e = 0; e < 8; ++e) v[e] = float(out[u][f][e]);
            const op8 b = wmma_t::pack(v);  // the FP8 epilogue's bytes
            *reinterpret_cast<uint2*>(out8 + size_t(t)*512 + group*64 + f*16 + h*8) = uint2{unsigned(b[0]), unsigned(b[1])};
        }
    }
}

inline void run(hipStream_t stream, const unsigned char* x8, const unsigned char* w1, const unsigned char* w2,
                unsigned char* out8, int tokens, bool sr) {
    const int items = ((tokens + 63)/64)*8;
    if (sr) hipLaunchKernelGGL(k_split512_ffn_t<true>, dim3((items + 3)/4), dim3(128), 0, stream, x8, w1, w2, out8, tokens);
    else hipLaunchKernelGGL(k_split512_ffn_t<false>, dim3((items + 3)/4), dim3(128), 0, stream, x8, w1, w2, out8, tokens);
    HIP_CHECK(hipGetLastError());
}
}  // namespace split512_ffn
