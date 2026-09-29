#pragma once
// WMMA FP8 form of the resident "32-product half boundary" GEMM (gfx12).
// Same contract as tiled::k_gemm: per 32-wide K slice the products are
// accumulated in fp32 (two 16x16x16 FP8 WMMA steps), then acc = half(acc +
// slice); residual/partition/epilogue semantics are identical. All GEMM
// operands in the network are FP8 values, so the byte conversion is exact.
// The WMMA accumulation order differs from the scalar kernel's sequential
// sum; the byte-exact regressions decide whether results agree.
#include <rocwmma/rocwmma.hpp>
#include <map>
#include <memory>

namespace wmma_gemm {
using fp8 = rocwmma::float8_t;
// Epilogue ids, identical to tiled::Epilogue.
enum : int { RAW = 0, FP8 = 1, GATE = 2, DOUBLE_RES = 3 };
struct Identity { __device__ static float load(float x) { return x; } };
constexpr int BT = 128, BO = 128, KS = 32;  // block tile and K slice
constexpr int WT = 32, WO = 64;             // wave tile: 2 x 4 fragments

template<class P>
__global__ void k_to_fp8(const float* in, fp8* out, int rows, int cols, int ld) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i >= size_t(rows)*cols) return;
    size_t r = i / cols, c = i % cols;
    out[r*ld + c] = fp8(P::load(in[r*ld + c]));
}

// Grid: x over output tiles, y over token tiles, z over groups.
template<class P, int EPI, bool RESIDUAL>
__global__ __launch_bounds__(256) void k_wmma_gemm(
        const fp8* __restrict__ X, int ldx, const fp8* __restrict__ W, int K,
        const float* __restrict__ R, int ldr, const float* __restrict__ S,
        float* out, float* out2, int ldo, int T, int O,
        long long gx, long long gw, long long go, int parts) {
    using namespace rocwmma;
    __shared__ __attribute__((aligned(16))) fp8 xs[BT*KS];
    __shared__ __attribute__((aligned(16))) fp8 ws[BO*KS];
    __shared__ float scratch[8][16*16];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int t0 = blockIdx.y*BT, o0 = blockIdx.x*BO;
    const int wt = (wave / 2) * WT, wo = (wave % 2) * WO;
    X += blockIdx.z*gx; W += blockIdx.z*gw;
    if (out) out += blockIdx.z*go;
    if (out2) out2 += blockIdx.z*go;
    if (RESIDUAL) R += blockIdx.z*go;
    float* sc = scratch[wave];
    const int span = K / parts;
    fragment<accumulator,16,16,16,float> acc[2][4], total[2][4], res[2][4];
    // Residual start: half(R*S) per element, staged through the wave scratch.
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) {
            fill_fragment(acc[i][j], 0.0f);
            fill_fragment(res[i][j], 0.0f);
            if (RESIDUAL) {
                for (int e = lane; e < 256; e += 32) {
                    int t = t0+wt+i*16+e/16, o = o0+wo+j*16+e%16;
                    sc[e] = t < T && o < O ? P::half(R[size_t(t)*ldr+o]*S[o]) : 0.0f;
                }
                __syncthreads();
                if constexpr (EPI == DOUBLE_RES) load_matrix_sync(res[i][j], sc, 16, mem_row_major);
                else load_matrix_sync(acc[i][j], sc, 16, mem_row_major);
                __syncthreads();
            }
        }
    for (int k0 = 0; k0 < K; k0 += KS) {
        // Stage 128x32 FP8 slices of X and W with 16-byte loads (zero padded).
        for (int v = tid; v < (BT*KS)/16; v += 256) {
            int row = v / (KS/16), part = v % (KS/16), t = t0+row;
            uint4 value = {0,0,0,0};
            if (t < T) value = *reinterpret_cast<const uint4*>(X + size_t(t)*ldx + k0 + part*16);
            *reinterpret_cast<uint4*>(xs + row*KS + part*16) = value;
        }
        for (int v = tid; v < (BO*KS)/16; v += 256) {
            int row = v / (KS/16), part = v % (KS/16), o = o0+row;
            uint4 value = {0,0,0,0};
            if (o < O) value = *reinterpret_cast<const uint4*>(W + size_t(o)*K + k0 + part*16);
            *reinterpret_cast<uint4*>(ws + row*KS + part*16) = value;
        }
        __syncthreads();
        fragment<matrix_a,16,16,16,fp8,row_major> a[2][2];
        fragment<matrix_b,16,16,16,fp8,col_major> b[2][4];
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 2; ++i) load_matrix_sync(a[h][i], xs + (wt+i*16)*KS + h*16, KS);
            for (int j = 0; j < 4; ++j) load_matrix_sync(b[h][j], ws + (wo+j*16)*KS + h*16, KS);
        }
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 4; ++j) {
                fragment<accumulator,16,16,16,float> slice;
                fill_fragment(slice, 0.0f);
                mma_sync(slice, a[0][i], b[0][j], slice);
                mma_sync(slice, a[1][i], b[1][j], slice);
                for (int e = 0; e < slice.num_elements; ++e)
                    acc[i][j].x[e] = EPI == DOUBLE_RES ? slice.x[e] : P::half(acc[i][j].x[e] + slice.x[e]);
            }
        if (EPI != DOUBLE_RES && (k0+KS) % span == 0) {
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int e = 0; e < acc[i][j].num_elements; ++e) {
                        total[i][j].x[e] = k0+KS == span ? acc[i][j].x[e] : P::half(total[i][j].x[e] + acc[i][j].x[e]);
                        acc[i][j].x[e] = 0.0f;
                    }
        }
        __syncthreads();
    }
    // Epilogue with coordinates through the wave scratch, one fragment at a time.
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) {
            if constexpr (EPI == DOUBLE_RES) {
                for (int e = 0; e < acc[i][j].num_elements; ++e)
                    acc[i][j].x[e] = P::hd(double(acc[i][j].x[e]) + double(res[i][j].x[e]));
                store_matrix_sync(sc, acc[i][j], 16, mem_row_major);
            } else {
                store_matrix_sync(sc, total[i][j], 16, mem_row_major);
            }
            __syncthreads();
            for (int e = lane; e < 256; e += 32) {
                int t = t0+wt+i*16+e/16, o = o0+wo+j*16+e%16;
                if (t >= T || o >= O) continue;
                size_t at = size_t(t)*ldo+o;
                float a = sc[e];
                if constexpr (EPI == RAW || EPI == DOUBLE_RES) {
                    out[at] = a;
                } else if constexpr (EPI == FP8) {
                    out[at] = P::fp8(a);
                    if (out2) out2[at] = a;
                } else {
                    if (out) out[at] = a;
                    float gate = fminf(4.0f, fmaxf(-4.0f, a));
                    float inner = P::half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                    float poly = P::half(gate*inner+0.89453125f);
                    out2[at] = P::fp8(P::half(a*poly));
                }
            }
            __syncthreads();
        }
}

// Weight cache (FP8 bytes per source pointer) and activation scratch.
struct State {
    std::map<const float*, std::pair<fp8*, size_t>> weights;
    fp8* scratch = nullptr;
    size_t scratch_bytes = 0;
    ~State() { clear(); if (scratch) (void)hipFree(scratch); }
    void clear() { for (auto& w : weights) (void)hipFree(w.second.first); weights.clear(); }
};
inline State& state() { static State s; return s; }
// RESIDENT_WMMA=1 selects this path (engines may also set it directly).
inline bool enabled = [] { const char* v = getenv("RESIDENT_WMMA"); return v && *v == '1'; }();

template<class P, int EPI, bool RESIDUAL>
inline bool gemm(hipStream_t stream, const float* X, int ldx, const float* W, int K,
                 const float* R, int ldr, const float* S, float* out, float* out2, int ldo,
                 int T, int O, int groups, long long gx, long long gw, long long go, int parts) {
    // Vector staging needs 16-byte aligned rows and slices.
    if (K % 32 || ldx % 16 || gx % 16 || gw % 16) return false;
    auto& s = state();
    size_t wcount = size_t(groups-1)*gw + size_t(O)*K;
    auto found = s.weights.find(W);
    if (found == s.weights.end() || found->second.second != wcount) {
        fp8* packed = nullptr;
        HIP_CHECK(hipMalloc(&packed, wcount));
        hipLaunchKernelGGL((k_to_fp8<Identity>), dim3((wcount+255)/256), dim3(256), 0, stream,
                           W, packed, 1, int(wcount), int(wcount));
        if (found != s.weights.end()) (void)hipFree(found->second.first);
        s.weights[W] = {packed, wcount};
        found = s.weights.find(W);
    }
    // Groups offset columns within a row, so X spans T rows of the row stride.
    if (size_t(groups-1)*gx + K > size_t(ldx)) return false;
    size_t xcount = size_t(T)*ldx;
    if (xcount > s.scratch_bytes) {
        if (s.scratch) HIP_CHECK(hipFree(s.scratch));
        HIP_CHECK(hipMalloc(&s.scratch, xcount));
        s.scratch_bytes = xcount;
    }
    hipLaunchKernelGGL((k_to_fp8<P>), dim3((xcount+255)/256), dim3(256), 0, stream,
                       X, s.scratch, T, ldx, ldx);
    dim3 grid((O+BO-1)/BO, (T+BT-1)/BT, groups);
    hipLaunchKernelGGL((k_wmma_gemm<P,EPI,RESIDUAL>), grid, dim3(256), 0, stream,
                       s.scratch, ldx, found->second.first, K, R, ldr, S, out, out2, ldo, T, O, gx, gw, go, parts);
    return true;
}
} // namespace wmma_gemm
