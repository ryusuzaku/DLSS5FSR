#pragma once
// Tiled form of the resident "32-product half boundary" matrix product.
// out[t][o] = epilogue(acc), acc = init; for every 32-wide K slice:
//   part = sum_{j<32} load(X[t][k+j]) * W[o][k+j]   (float32, index order)
//   acc  = half(acc + part)
// Each output keeps exactly this operation order (no fused multiply-add;
// build with -ffp-contract=off), so results are bit-identical to the
// one-thread-per-output kernels. Tiles only change memory traffic: a 256-thread
// group computes 64 tokens x 64 outputs, staging 32-wide slices in LDS.
#include "split512_block.hip"
#include "resident_stream.h"
#include "wmma_gemm.hip"
#include <cstdio>

namespace tiled {
enum Epilogue : int {
    RAW = 0,      // out = acc
    FP8 = 1,      // out = fp8(acc); out2 (optional) = acc
    GATE = 2,     // out (optional) = acc; out2 = fp8(half(acc * poly(acc)))
    DOUBLE_RES = 3,  // K == 32: out = hd(double(part) + double(half(R*S)))  (C32 projection)
};

// Rounding policy: the split512 family (C64-C512, ViT) or the head70/C32 family.
struct Split {
    __device__ static float half(float x) { return split512_half(x); }
    __device__ static float fp8(float x) { return split512_fp8(x); }
    __device__ static float load(float x) { return x; }
};

constexpr int TT = 64, TO = 64, TK = 32, PAD = 1;

// Grid: x over output tiles, y over token tiles, z over groups (offsets gx/gw/go).
template<class P, int EPI, bool RESIDUAL, int BT = TT, int BO = TO>
__global__ __launch_bounds__(256) void k_gemm(
        const float* __restrict__ X, int ldx, const float* __restrict__ W, int K,
        const float* __restrict__ R, int ldr, const float* __restrict__ S,
        float* out, float* out2, int ldo, int T, int O,
        long long gx, long long gw, long long go, int parts) {
    static_assert((BT/4)*(BO/4) == 256, "tile must give 4x4 outputs to each of 256 threads");
    __shared__ float xs[TK][BT+PAD];
    __shared__ float ws[TK][BO+PAD];
    const int tid = threadIdx.x, tx = tid % (BO/4), ty = tid / (BO/4);
    const int t0 = blockIdx.y*BT, o0 = blockIdx.x*BO;
    X += blockIdx.z*gx; W += blockIdx.z*gw;
    if (out) out += blockIdx.z*go;
    if (out2) out2 += blockIdx.z*go;
    if (RESIDUAL) R += blockIdx.z*go;
    // parts > 1: K splits into equal ranges accumulated separately (the
    // residual only in the first) and combined as total = half(total + acc).
    const int span = K / parts;
    float acc[4][4], res[4][4], total[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int t = t0+ty*4+i, o = o0+tx*4+j;
            acc[i][j] = 0.0f; res[i][j] = 0.0f;
            if (RESIDUAL && t < T && o < O) {
                float product = R[size_t(t)*ldr+o]*S[o];
                if (EPI == DOUBLE_RES) res[i][j] = P::half(product);
                else acc[i][j] = P::half(product);
            }
        }
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int r = 0; r < (BT*TK)/256; ++r) {
            int i = tid + r*256, row = i / TK, col = i % TK, t = t0+row;
            xs[col][row] = t < T ? P::load(X[size_t(t)*ldx+k0+col]) : 0.0f;
        }
        for (int r = 0; r < (BO*TK)/256; ++r) {
            int i = tid + r*256, row = i / TK, col = i % TK, o = o0+row;
            ws[col][row] = o < O ? W[size_t(o)*K+k0+col] : 0.0f;
        }
        __syncthreads();
        float part[4][4] = {};
        for (int col = 0; col < TK; ++col) {
            float xv[4], wv[4];
            for (int i = 0; i < 4; ++i) xv[i] = xs[col][ty*4+i];
            for (int j = 0; j < 4; ++j) wv[j] = ws[col][tx*4+j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) part[i][j] += xv[i]*wv[j];
        }
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                acc[i][j] = EPI == DOUBLE_RES ? part[i][j] : P::half(acc[i][j]+part[i][j]);
        if (EPI != DOUBLE_RES && (k0+TK) % span == 0) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    total[i][j] = k0+TK == span ? acc[i][j] : P::half(total[i][j]+acc[i][j]);
                    acc[i][j] = 0.0f;
                }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            int t = t0+ty*4+i, o = o0+tx*4+j;
            if (t >= T || o >= O) continue;
            size_t at = size_t(t)*ldo+o;
            float a = EPI == DOUBLE_RES ? acc[i][j] : total[i][j];
            if constexpr (EPI == RAW) {
                out[at] = a;
            } else if constexpr (EPI == FP8) {
                out[at] = P::fp8(a);
                if (out2) out2[at] = a;
            } else if constexpr (EPI == DOUBLE_RES) {
                out[at] = P::hd(double(a)+double(res[i][j]));
            } else {
                if (out) out[at] = a;
                float gate = fminf(4.0f, fmaxf(-4.0f, a));
                float inner = P::half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                float poly = P::half(gate*inner+0.89453125f);
                out2[at] = P::fp8(P::half(a*poly));
            }
        }
}

template<class P, int EPI, bool RESIDUAL>
inline void gemm(hipStream_t stream, const float* X, int ldx, c512_resident::WeightRef W, int K,
                 const float* R, int ldr, const float* S, float* out, float* out2, int ldo,
                 int T, int O, int groups = 1, long long gx = 0, long long gw = 0, long long go = 0,
                 int parts = 1, const unsigned char* X8 = nullptr, unsigned char* out8 = nullptr) {
    // X8 / out8: E4M3 byte copies of X / the FP8 output (wmma_gemm::gemm).
    // out8 is always written: by the GEMM when it can, else from the floats.
    bool wrote8 = false;
    const bool wmma = wmma_gemm::enabled && !c512_resident::exact_math &&
        wmma_gemm::gemm<P,EPI,RESIDUAL>(stream, X, ldx, W, K, R, ldr, S, out, out2, ldo, T, O, groups, gx, gw, go, parts,
                                        X8, out8, &wrote8);
    // Narrow outputs use taller tiles so no thread idles; the arithmetic per
    // output is identical for every tile shape.
    if (wmma) {
    } else if (O <= 32) {
        dim3 grid((O+31)/32, (T+127)/128, groups);
        hipLaunchKernelGGL((k_gemm<P,EPI,RESIDUAL,128,32>), grid, dim3(256), 0, stream,
                           X, ldx, W, K, R, ldr, S, out, out2, ldo, T, O, gx, gw, go, parts);
    } else if (O % 128 == 0 && K == 32) {
        dim3 grid(O/128, (T+31)/32, groups);
        hipLaunchKernelGGL((k_gemm<P,EPI,RESIDUAL,32,128>), grid, dim3(256), 0, stream,
                           X, ldx, W, K, R, ldr, S, out, out2, ldo, T, O, gx, gw, go, parts);
    } else {
        dim3 grid((O+TO-1)/TO, (T+TT-1)/TT, groups);
        hipLaunchKernelGGL((k_gemm<P,EPI,RESIDUAL>), grid, dim3(256), 0, stream,
                           X, ldx, W, K, R, ldr, S, out, out2, ldo, T, O, gx, gw, go, parts);
    }
    if (out8 && !wrote8 && (EPI == FP8 || EPI == GATE)) {
        const float* src = EPI == FP8 ? out : out2;
        const size_t n = size_t(groups-1)*go + size_t(T)*ldo;
        hipLaunchKernelGGL(wmma_gemm::k_to_e4m3, dim3((n+255)/256), dim3(256), 0, stream, src, out8, n);
    }
    if (c512_resident::launch_hook) {
        char name[96];
        snprintf(name, sizeof(name), "%s_%s%s_K%d_O%d_G%d_P%d", wmma ? "wmma" : "tiled", EPI == GATE ? "gate" : EPI == DOUBLE_RES ? "dres" :
                 EPI == FP8 ? "fp8" : "raw", RESIDUAL ? "_res" : "", K, O, groups, parts);
        c512_resident::launch_hook(name);
    }
}
} // namespace tiled
