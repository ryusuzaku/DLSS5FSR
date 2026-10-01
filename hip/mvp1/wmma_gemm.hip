#pragma once
// WMMA (f16 in, fp32 accumulate, gfx12) form of the resident "32-product
// half boundary" GEMM. Same contract as tiled::k_gemm: per 32-wide K slice
// the products are summed in fp32 (two 16x16x16 steps), then acc =
// half(acc + slice); residual/partition/epilogue semantics are identical.
// GEMM operands in the network are FP8 values, so converting them to half is
// exact and so is every product; only the fp32 summation order inside a
// slice differs from the sequential kernel. That order rarely matters (FP8
// products carry <= 8 significant bits), but it is not guaranteed, so the
// path is off under c512_resident::exact_math (RESIDENT_EXACT=1).
#include <rocwmma/rocwmma.hpp>
#include <cstdlib>
#include <map>

namespace wmma_gemm {
using h16 = rocwmma::float16_t;
// Epilogue ids, identical to tiled::Epilogue.
enum : int { RAW = 0, FP8 = 1, GATE = 2, DOUBLE_RES = 3 };
constexpr int KS = 32, LDS_LD = KS + 8;  // K slice; padded LDS row (halves)

__global__ void k_to_half(const float* in, h16* out, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = split512_to_half(in[i]);
}

// Epilogue of one output (a: the combined total; res: DOUBLE_RES residual).
template<class P, int EPI>
__device__ inline void store(float a, float res, size_t at, float* out, float* out2) {
    if constexpr (EPI == RAW) {
        out[at] = a;
    } else if constexpr (EPI == FP8) {
        out[at] = P::fp8(a);
        if (out2) out2[at] = a;
    } else if constexpr (EPI == DOUBLE_RES) {
        out[at] = P::hd(double(a)+double(res));
    } else {
        if (out) out[at] = a;
        float gate = fminf(4.0f, fmaxf(-4.0f, a));
        float inner = P::half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
        float poly = P::half(gate*inner+0.89453125f);
        out2[at] = P::fp8(P::half(a*poly));
    }
}

// Block tile BT tokens x BO outputs, 8 waves, each FM x FN 16x16 fragments.
// partial != null (one group, parts > 1): blockIdx.z is the K partition; the
// block writes that partition's total to partial[z][T][O] for k_wmma_combine.
// Accumulator layout (gfx12): lane l, element e -> row (l/16)*8+e, col l%16.
template<class P, int EPI, bool RESIDUAL, int BT, int BO, int FM, int FN>
__global__ __launch_bounds__(256) void k_wmma_gemm(
        const float* __restrict__ X, int ldx, const h16* __restrict__ W, int K,
        const float* __restrict__ R, int ldr, const float* __restrict__ S,
        float* out, float* out2, int ldo, int T, int O,
        long long gx, long long gw, long long go, int parts, float* partial) {
    using namespace rocwmma;
    constexpr int WR = BT/16/FM, WC = BO/16/FN;
    static_assert(WR*WC == 8, "8 waves per block");
    // Double-buffered slices: slice k+1 is fetched into registers while
    // slice k is multiplied, then stored to the other buffer.
    __shared__ __attribute__((aligned(16))) h16 xs[2][BT*LDS_LD];
    __shared__ __attribute__((aligned(16))) h16 ws[2][BO*LDS_LD];
    constexpr int XN = (BT*KS + 255)/256, WN = (BO*(KS/8) + 255)/256;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    const int t0 = blockIdx.y*BT, o0 = blockIdx.x*BO;
    const int wt = t0 + (wave/WC)*FM*16, wo = o0 + (wave%WC)*FN*16;
    const int part = partial ? blockIdx.z : 0, group = partial ? 0 : blockIdx.z;
    X += group*gx; W += group*gw;
    if (out) out += group*go;
    if (out2) out2 += group*go;
    if (RESIDUAL) R += group*go;
    const int span = K / parts, kbegin = partial ? part*span : 0, kend = partial ? kbegin+span : K;
    float acc[FM][FN][8], total[FM][FN][8], res[FM][FN][8];
    for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
            for (int e = 0; e < 8; ++e) {
                int t = wt+i*16+lr+e, o = wo+j*16+lc;
                acc[i][j][e] = 0.0f; res[i][j][e] = 0.0f; total[i][j][e] = 0.0f;
                if (RESIDUAL && part == 0 && t < T && o < O) {
                    float product = R[size_t(t)*ldr+o]*S[o];
                    if (EPI == DOUBLE_RES) res[i][j][e] = P::half(product);
                    else acc[i][j][e] = P::half(product);
                }
            }
    float xr[XN];
    uint4 wr[WN];
    auto fetch = [&](int k0) {
        for (int j = 0; j < XN; ++j) {
            int i = tid + j*256;
            if (i < BT*KS) {
                int row = i / KS, col = i % KS, t = t0+row;
                xr[j] = t < T ? P::load(X[size_t(t)*ldx+k0+col]) : 0.0f;
            }
        }
        for (int j = 0; j < WN; ++j) {
            int v = tid + j*256;
            wr[j] = uint4{0,0,0,0};
            if (v < BO*(KS/8)) {
                int row = v / (KS/8), piece = v % (KS/8), o = o0+row;
                if (o < O) wr[j] = *reinterpret_cast<const uint4*>(W + size_t(o)*K + k0 + piece*8);
            }
        }
    };
    auto stash = [&](int b) {
        for (int j = 0; j < XN; ++j) {
            int i = tid + j*256;
            if (i < BT*KS) xs[b][(i/KS)*LDS_LD + i%KS] = split512_to_half(xr[j]);
        }
        for (int j = 0; j < WN; ++j) {
            int v = tid + j*256;
            if (v < BO*(KS/8)) *reinterpret_cast<uint4*>(ws[b] + (v/(KS/8))*LDS_LD + (v%(KS/8))*8) = wr[j];
        }
    };
    fetch(kbegin);
    stash(0);
    __syncthreads();
    int cur = 0;
    for (int k0 = kbegin; k0 < kend; k0 += KS) {
        const bool more = k0+KS < kend;
        if (more) fetch(k0+KS);  // in flight while this slice is multiplied
        fragment<matrix_a,16,16,16,h16,row_major> a[2][FM];
        fragment<matrix_b,16,16,16,h16,col_major> b[2][FN];
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < FM; ++i) load_matrix_sync(a[h][i], xs[cur] + (wt-t0+i*16)*LDS_LD + h*16, LDS_LD);
            for (int j = 0; j < FN; ++j) load_matrix_sync(b[h][j], ws[cur] + (wo-o0+j*16)*LDS_LD + h*16, LDS_LD);
        }
        for (int i = 0; i < FM; ++i)
            for (int j = 0; j < FN; ++j) {
                fragment<accumulator,16,16,16,float> slice;
                fill_fragment(slice, 0.0f);
                mma_sync(slice, a[0][i], b[0][j], slice);
                mma_sync(slice, a[1][i], b[1][j], slice);
                for (int e = 0; e < 8; ++e)
                    acc[i][j][e] = EPI == DOUBLE_RES ? slice.x[e] : P::half(acc[i][j][e] + slice.x[e]);
            }
        if (EPI != DOUBLE_RES && (k0+KS) % span == 0) {
            for (int i = 0; i < FM; ++i)
                for (int j = 0; j < FN; ++j)
                    for (int e = 0; e < 8; ++e) {
                        total[i][j][e] = k0+KS == kbegin+span ? acc[i][j][e] : P::half(total[i][j][e] + acc[i][j][e]);
                        acc[i][j][e] = 0.0f;
                    }
        }
        if (more) stash(cur ^ 1);
        __syncthreads();
        cur ^= 1;
    }
    for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
            for (int e = 0; e < 8; ++e) {
                int t = wt+i*16+lr+e, o = wo+j*16+lc;
                if (t >= T || o >= O) continue;
                if (partial) { partial[(size_t(part)*T + t)*O + o] = total[i][j][e]; continue; }
                store<P,EPI>(EPI == DOUBLE_RES ? acc[i][j][e] : total[i][j][e], res[i][j][e], size_t(t)*ldo+o, out, out2);
            }
}

// Partition totals in order: total = half(total + part), then the epilogue.
template<class P, int EPI>
__global__ void k_wmma_combine(const float* partial, int parts, int T, int O, float* out, float* out2, int ldo) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x, n = size_t(T)*O;
    if (i >= n) return;
    float a = partial[i];
    for (int p = 1; p < parts; ++p) a = P::half(a + partial[p*n + i]);
    store<P,EPI>(a, 0.0f, (i/O)*ldo + i%O, out, out2);
}

// Half copies of weight matrices, keyed by source pointer and size. Weights
// stay resident for an engine's lifetime; clear() when they are freed.
struct State {
    std::map<const float*, std::pair<h16*, size_t>> weights;
    float* partial = nullptr;  // split-K partition totals
    size_t partial_count = 0;
    ~State() { clear(); }
    void clear() {
        for (auto& w : weights) (void)hipFree(w.second.first);
        weights.clear();
        if (partial) (void)hipFree(partial);
        partial = nullptr; partial_count = 0;
    }
};
inline State& state() { static State s; return s; }
// RESIDENT_WMMA=0 disables this path; exact_math always does.
inline bool enabled = [] { const char* v = getenv("RESIDENT_WMMA"); return !(v && *v == '0'); }();

// Cached half copy of `count` weights at W (converted once on `stream`).
inline const h16* half_weights(hipStream_t stream, const float* W, size_t count) {
    auto& s = state();
    auto found = s.weights.find(W);
    if (found == s.weights.end() || found->second.second != count) {
        h16* packed = nullptr;
        HIP_CHECK(hipMalloc(&packed, count*sizeof(h16)));
        hipLaunchKernelGGL(k_to_half, dim3((count+255)/256), dim3(256), 0, stream, W, packed, count);
        if (found != s.weights.end()) (void)hipFree(found->second.first);
        s.weights[W] = {packed, count};
        return packed;
    }
    return found->second.first;
}

template<class P, int EPI, bool RESIDUAL>
inline bool gemm(hipStream_t stream, const float* X, int ldx, const float* W, int K,
                 const float* R, int ldr, const float* S, float* out, float* out2, int ldo,
                 int T, int O, int groups, long long gx, long long gw, long long go, int parts) {
    // 16-byte weight rows/slices; K whole slices in every partition.
    if (K % 32 || (K/parts) % 32 || gw % 8) return false;
    const h16* hw = half_weights(stream, W, size_t(groups-1)*gw + size_t(O)*K);
    // Few tiles and a partitioned K (the ViT contract/projection): run the
    // partitions as separate blocks and combine them in order afterwards.
    float* partial = nullptr;
    const int bo = O <= 32 ? 32 : (O <= 64 || O % 128) ? 64 : 128, bt = bo == 128 ? 64 : 128;
    if (EPI != DOUBLE_RES && groups == 1 && parts > 1 && size_t((O+bo-1)/bo)*((T+bt-1)/bt) < 256) {
        auto& s = state();
        size_t need = size_t(parts)*T*O;
        if (need > s.partial_count) {
            if (s.partial) HIP_CHECK(hipFree(s.partial));
            s.partial = nullptr; s.partial_count = 0;
            HIP_CHECK(hipMalloc(&s.partial, need*sizeof(float)));
            s.partial_count = need;
        }
        partial = s.partial;
    }
    const int zdim = partial ? parts : groups;
#define WMMA_LAUNCH(BT, BO, FM, FN) \
    hipLaunchKernelGGL((k_wmma_gemm<P,EPI,RESIDUAL,BT,BO,FM,FN>), dim3((O+BO-1)/BO, (T+BT-1)/BT, zdim), dim3(256), 0, \
                       stream, X, ldx, hw, K, R, ldr, S, out, out2, ldo, T, O, gx, gw, go, parts, partial)
    // Few blocks with the usual tile (small token counts): 32x64 tiles give
    // four times as many; the per-output arithmetic is the same.
    const size_t blocks = size_t((O+bo-1)/bo)*((T+bt-1)/bt)*zdim;
    if (O > 32 && O % 64 == 0 && blocks < 128) WMMA_LAUNCH(32, 64, 1, 1);
    else if (O <= 32) WMMA_LAUNCH(128, 32, 1, 2);
    else if (O <= 64 || O % 128) WMMA_LAUNCH(128, 64, 2, 2);
    else WMMA_LAUNCH(64, 128, 2, 2);
#undef WMMA_LAUNCH
    if (partial)
        hipLaunchKernelGGL((k_wmma_combine<P,EPI>), dim3((size_t(T)*O+255)/256), dim3(256), 0, stream,
                           partial, parts, T, O, out, out2, ldo);
    HIP_CHECK(hipGetLastError());
    return true;
}
} // namespace wmma_gemm
