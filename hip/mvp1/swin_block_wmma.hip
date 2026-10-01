#pragma once
// Fused fast (non-exact) C64/C128/C256 Swin blocks: two launches per block
// instead of gather, w1+gate, w2, w3, qkv, attention, projection and scatter.
// One workgroup per shifted 64-token window.
//   k_swin_ffn:       gather -> w1 + gate (hidden in 128-column chunks, fed
//                     straight into w2's accumulators in K order) -> w2 (FP8)
//                     -> w3 with the gathered skip -> FP8 feature (half, window order)
//   k_swin_attention: per head qkv -> window attention -> one 32-wide K slice
//                     of the projection (head h is slice h) -> FP8 output and
//                     optional raw projection, scattered to HWC.
// Products are f16 WMMA on FP8/half values (exact); only the fp32 summation
// order inside each 32-product slice differs from the sequential reference.
// Rounding boundaries, epilogues and the denominator order are unchanged.
// Needs split512_attention.hip, split512_attention_wmma.hip and wmma_gemm.hip.

namespace swin_fused {
using h16 = rocwmma::float16_t;
using FA = rocwmma::fragment<rocwmma::matrix_a,16,16,16,h16,rocwmma::row_major>;
using FBc = rocwmma::fragment<rocwmma::matrix_b,16,16,16,h16,rocwmma::col_major>;
using FBr = rocwmma::fragment<rocwmma::matrix_b,16,16,16,h16,rocwmma::row_major>;
using FC = rocwmma::fragment<rocwmma::accumulator,16,16,16,float>;

// One 32-wide K slice of a 16x16 tile; B halves are bstep apart.
template<class FB>
__device__ inline void slice(FC& part, const h16* a, int lda, const h16* b, int ldb, int bstep) {
    FA a0, a1; FB b0, b1;
    rocwmma::load_matrix_sync(a0, a, lda);
    rocwmma::load_matrix_sync(a1, a+16, lda);
    rocwmma::load_matrix_sync(b0, b, ldb);
    rocwmma::load_matrix_sync(b1, b+bstep, ldb);
    rocwmma::fill_fragment(part, 0.0f);
    rocwmma::mma_sync(part, a0, b0, part);
    rocwmma::mma_sync(part, a1, b1, part);
}

// Window geometry of a shifted 8x8 partition.
struct Geometry {
    int width, height, px, py, pw;
    __device__ bool pixel(int win, int t, int& x, int& y) const {
        x = (win % (pw/8))*8 + t%8 - px;
        y = (win / (pw/8))*8 + t/8 - py;
        return x >= 0 && x < width && y >= 0 && y < height;
    }
};

// C-wide outputs of a window: 4 x C/16 tiles, PER per wave.
template<int C> struct Tiles {
    static constexpr int FN = C/16, PER = 4*FN/8;
    __device__ static int rt(int wave, int p) { return (wave*PER+p) / FN; }
    __device__ static int ct(int wave, int p) { return (wave*PER+p) % FN; }
};

template<int C>
__global__ __launch_bounds__(256) void k_swin_ffn(const float* __restrict__ input, Geometry g,
        const h16* __restrict__ w1, const h16* __restrict__ w2, const h16* __restrict__ w3,
        const float* __restrict__ skip, h16* __restrict__ feature) {
    constexpr int LX = C+8, LH = 136, HID = 4*C, PER = Tiles<C>::PER;
    __shared__ __attribute__((aligned(16))) h16 xs[64*LX];  // gathered input, then FP8 middle
    __shared__ __attribute__((aligned(16))) h16 hs[64*LH];  // one 128-column hidden chunk
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x;
    for (int i = tid; i < 64*C; i += 256) {
        int t = i/C, c = i%C, x, y;
        xs[t*LX+c] = split512_to_half(g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f);
    }
    __syncthreads();
    float mid[PER][8];
    for (int p = 0; p < PER; ++p) for (int e = 0; e < 8; ++e) mid[p][e] = 0.0f;
    for (int h0 = 0; h0 < HID; h0 += 128) {
        {   // Hidden chunk 64x128 (K=C): 32 tiles, wave -> row tile, 4 column tiles.
            const int rt = wave%4;
            for (int j = 0; j < 4; ++j) {
                const int ct = wave/4*4+j;
                float acc[8] = {};
                for (int k0 = 0; k0 < C; k0 += 32) {
                    FC part; slice<FBc>(part, xs+rt*16*LX+k0, LX, w1+size_t(h0+ct*16)*C+k0, C, 16);
                    for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
                }
                for (int e = 0; e < 8; ++e) {
                    float a = acc[e];
                    float gate = fminf(4.0f, fmaxf(-4.0f, a));
                    float inner = split512_half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
                    float poly = split512_half(gate*inner+0.89453125f);
                    hs[(rt*16+lr+e)*LH+ct*16+lc] = split512_to_half(split512_fp8(split512_half(a*poly)));
                }
            }
        }
        __syncthreads();
        for (int p = 0; p < PER; ++p) {  // w2: this chunk's four K slices, in order
            const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
            for (int s = 0; s < 4; ++s) {
                FC part; slice<FBc>(part, hs+rt*16*LH+s*32, LH, w2+size_t(ct*16)*HID+h0+s*32, HID, 16);
                for (int e = 0; e < 8; ++e) mid[p][e] = split512_half(mid[p][e]+part.x[e]);
            }
        }
        __syncthreads();
    }
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) xs[(rt*16+lr+e)*LX+ct*16+lc] = split512_to_half(split512_fp8(mid[p][e]));
    }
    __syncthreads();
    for (int p = 0; p < PER; ++p) {  // w3 with the gathered skip
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        float acc[8];
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = ct*16+lc, x, y;
            float v = g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f;
            acc[e] = split512_half(v*skip[c]);
        }
        for (int k0 = 0; k0 < C; k0 += 32) {
            FC part; slice<FBc>(part, xs+rt*16*LX+k0, LX, w3+size_t(ct*16)*C+k0, C, 16);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e)
            feature[(size_t(win)*64+rt*16+lr+e)*C+ct*16+lc] = split512_to_half(split512_fp8(acc[e]));
    }
}

template<int C, bool SPLIT>
__global__ __launch_bounds__(256) void k_swin_attention(const h16* __restrict__ feature, Geometry g,
        const h16* __restrict__ wqkv, const float* __restrict__ scales, const float* __restrict__ bias,
        const h16* __restrict__ wp, const float* __restrict__ attention_skip,
        float* output, float* raw_out, float* partial) {
    constexpr int LX = C+8, LD = 40, LS = 72, PER = Tiles<C>::PER, HEADS = C/32;
    // The feature stays in global memory (contiguous per window, L2
    // resident): the products read their A fragments from there, which keeps
    // LDS small enough for more than one workgroup per CU.
    __shared__ __attribute__((aligned(16))) h16 qs[64*LD], ks[64*LD], vs[64*LD];  // qs holds the context after scores
    __shared__ __attribute__((aligned(16))) h16 sc[64*LS];
    __shared__ float inv[128];
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x;
    const h16* src = feature + size_t(win)*64*C;
    float acc[PER][8];
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) {
            int c = ct*16+lc;
            acc[p][e] = split512_half(float(src[(rt*16+lr+e)*C+c])*attention_skip[c]);
        }
    }
    // SPLIT: one head per block (blockIdx.y); the block writes its
    // projection slice (fp32, before the half boundary) for k_swin_combine.
    const int head0 = SPLIT ? blockIdx.y : 0, head1 = SPLIT ? head0 + 1 : HEADS;
    for (int head = head0; head < head1; ++head) {
        for (int j = 0; j < 3; ++j) {  // qkv of this head: 24 tiles, 3 per wave
            const int k = wave*3+j, m = k/8, rt = k%8/2, ct = k%2;
            float q[8] = {};
            for (int k0 = 0; k0 < C; k0 += 32) {
                FC part; slice<FBc>(part, src+rt*16*C+k0, C, wqkv+size_t(m*C+head*32+ct*16)*C+k0, C, 16);
                for (int e = 0; e < 8; ++e) q[e] = split512_half(q[e]+part.x[e]);
            }
            h16* dst = m == 0 ? qs : m == 1 ? ks : vs;
            for (int e = 0; e < 8; ++e) dst[(rt*16+lr+e)*LD+ct*16+lc] = split512_to_half(q[e]);
        }
        __syncthreads();
        if (tid < 128) {
            const h16* row = (tid < 64 ? qs : ks) + (tid%64)*LD;
            float v[32];
            for (int c = 0; c < 32; ++c) v[c] = float(row[c]);
            inv[tid] = split512_norm_inverse_fast(v);
        }
        __syncthreads();
        const float scale = split512_half(scales[head]);
        for (int i = tid; i < 3*64*32; i += 256) {
            int m = i/2048, t = i/32%64, c = i%32;
            h16* p = (m == 0 ? qs : m == 1 ? ks : vs) + t*LD + c;
            float value = float(*p);
            if (m < 2) {
                value = split512_half(value*inv[m*64+t]);
                if (m == 0) value = split512_half(value*scale);
            }
            *p = split512_to_half(split512_fp8(value));
        }
        __syncthreads();
        {   // Scores + bias + exponent transform: 16 tiles, 2 per wave.
            const int rt = wave%4;
            FA a0, a1;
            rocwmma::load_matrix_sync(a0, qs + rt*16*LD, LD);
            rocwmma::load_matrix_sync(a1, qs + rt*16*LD + 16, LD);
            for (int j = 0; j < 2; ++j) {
                const int ct = wave/4*2+j;
                FBc b0, b1; FC s;
                rocwmma::load_matrix_sync(b0, ks + ct*16*LD, LD);
                rocwmma::load_matrix_sync(b1, ks + ct*16*LD + 16, LD);
                rocwmma::fill_fragment(s, 0.0f);
                rocwmma::mma_sync(s, a0, b0, s);
                rocwmma::mma_sync(s, a1, b1, s);
                for (int e = 0; e < 8; ++e) {
                    int q = rt*16+lr+e, k = ct*16+lc;
                    float score = split512_half(s.x[e]+bias[(head*64+q)*64+k]);
                    float b = score*.044921875f+1.30078125f;
                    float clipped = fminf(1.5693359375f, fmaxf(1.03125f, split512_half(b)));
                    _Float16 hc = split512_to_half(clipped);
                    unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
                    bits = (unsigned short)((unsigned(bits)<<5)+0x8000u);
                    __builtin_memcpy(&hc, &bits, 2);
                    sc[q*LS+k] = hc;
                }
            }
        }
        __syncthreads();
        if (tid < 64) {  // split512_den order on the half row
            const h16* e = sc + tid*LS;
            float parity[2]; const int lanes[4] = {0,2,8,10}; const int offs[3] = {4,32,36};
            for (int odd = 0; odd < 2; ++odd) {
                float partial[4];
                for (int l = 0; l < 4; ++l) {
                    int b = odd+lanes[l];
                    float value = split512_half(float(e[b])+float(e[b+16]));
                    for (int t = 0; t < 3; ++t)
                        value = split512_half(value+split512_half(float(e[b+offs[t]])+float(e[b+offs[t]+16])));
                    partial[l] = value;
                }
                float total = partial[0];
                for (int l = 1; l < 4; ++l) total = split512_half(total+partial[l]);
                parity[odd] = total;
            }
            inv[tid] = split512_half(1.0f/split512_half(parity[0]+parity[1]));
        }
        __syncthreads();
        for (int i = tid; i < 64*64; i += 256) {
            int q = i/64, k = i%64;
            sc[q*LS+k] = split512_to_half(split512_fp8(split512_half(float(sc[q*LS+k])*inv[q])));
        }
        __syncthreads();
        {   // Context (two 32-key slices) -> FP8 into qs: 8 tiles, one per wave.
            const int rt = wave/2, ct = wave%2;
            float c8[8] = {};
            for (int k0 = 0; k0 < 64; k0 += 32) {
                FC part; slice<FBr>(part, sc+rt*16*LS+k0, LS, vs+k0*LD+ct*16, LD, 16*LD);
                for (int e = 0; e < 8; ++e) c8[e] = split512_half(c8[e]+part.x[e]);
            }
            for (int e = 0; e < 8; ++e) qs[(rt*16+lr+e)*LD+ct*16+lc] = split512_to_half(split512_fp8(c8[e]));
        }
        __syncthreads();
        for (int p = 0; p < PER; ++p) {  // projection K slice `head`
            const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
            FC part; slice<FBc>(part, qs+rt*16*LD, LD, wp+size_t(ct*16)*C+head*32, C, 16);
            if constexpr (SPLIT) {
                float* dst = partial + ((size_t(win)*HEADS + head)*64)*C;
                for (int e = 0; e < 8; ++e) dst[size_t(rt*16+lr+e)*C + ct*16+lc] = part.x[e];
            } else {
                for (int e = 0; e < 8; ++e) acc[p][e] = split512_half(acc[p][e]+part.x[e]);
            }
        }
        __syncthreads();
    }
    if constexpr (SPLIT) return;
    for (int p = 0; p < PER; ++p) {
        const int rt = Tiles<C>::rt(wave, p), ct = Tiles<C>::ct(wave, p);
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = ct*16+lc, x, y;
            if (!g.pixel(win, t, x, y)) continue;
            size_t at = (size_t(y)*g.width+x)*C+c;
            output[at] = split512_fp8(acc[p][e]);
            if (raw_out) raw_out[at] = acc[p][e];
        }
    }
}

// The FFN in three exact stages for levels with few windows (one fused
// block per window leaves most of the GPU idle there). Every output keeps
// its own K order; only the work is spread over more blocks.
//   k_swin_hidden: grid (window, 128-column chunk): w1 + gate -> FP8 hidden (global)
//   k_swin_w2:     grid (window, 64-column group): middle = FP8(sum of HID/32 slices)
//   k_swin_w3:     grid (window, 64-column group): w3 with the gathered skip -> feature
template<int C>
__global__ __launch_bounds__(256) void k_swin_hidden(const float* __restrict__ input, Geometry g,
        const h16* __restrict__ w1, h16* __restrict__ hidden) {
    constexpr int LX = C+8, HID = 4*C;
    __shared__ __attribute__((aligned(16))) h16 xs[64*LX];
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, h0 = blockIdx.y*128;
    for (int i = tid; i < 64*C; i += 256) {
        int t = i/C, c = i%C, x, y;
        xs[t*LX+c] = split512_to_half(g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f);
    }
    __syncthreads();
    const int rt = wave%4;
    for (int j = 0; j < 4; ++j) {
        const int ct = wave/4*4+j;
        float acc[8] = {};
        for (int k0 = 0; k0 < C; k0 += 32) {
            FC part; slice<FBc>(part, xs+rt*16*LX+k0, LX, w1+size_t(h0+ct*16)*C+k0, C, 16);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e) {
            float a = acc[e];
            float gate = fminf(4.0f, fmaxf(-4.0f, a));
            float inner = split512_half(fabsf(gate)*(-0.055908203125f)+0.447265625f);
            float poly = split512_half(gate*inner+0.89453125f);
            hidden[(size_t(win)*64+rt*16+lr+e)*HID + h0+ct*16+lc] = split512_to_half(split512_fp8(split512_half(a*poly)));
        }
    }
}

template<int C>
__global__ __launch_bounds__(256) void k_swin_w2(const h16* __restrict__ hidden, const h16* __restrict__ w2,
                                                 h16* __restrict__ middle) {
    constexpr int HID = 4*C;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, o0 = blockIdx.y*64;
    // 64 rows x 64 columns: 16 tiles, 2 per wave (row tile wave%4, column tiles).
    const int rt = wave%4;
    const h16* a = hidden + size_t(win)*64*HID;
    for (int j = 0; j < 2; ++j) {
        const int ct = wave/4*2+j;
        float acc[8] = {};
        for (int k0 = 0; k0 < HID; k0 += 32) {
            FC part; slice<FBc>(part, a+rt*16*HID+k0, HID, w2+size_t(o0+ct*16)*HID+k0, HID, 16);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e)
            middle[(size_t(win)*64+rt*16+lr+e)*C + o0+ct*16+lc] = split512_to_half(split512_fp8(acc[e]));
    }
}

template<int C>
__global__ __launch_bounds__(256) void k_swin_w3(const float* __restrict__ input, Geometry g,
        const h16* __restrict__ middle, const h16* __restrict__ w3, const float* __restrict__ skip,
        h16* __restrict__ feature) {
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, win = blockIdx.x, o0 = blockIdx.y*64;
    const int rt = wave%4;
    const h16* a = middle + size_t(win)*64*C;
    for (int j = 0; j < 2; ++j) {
        const int ct = wave/4*2+j;
        float acc[8];
        for (int e = 0; e < 8; ++e) {
            int t = rt*16+lr+e, c = o0+ct*16+lc, x, y;
            float v = g.pixel(win, t, x, y) ? input[(size_t(y)*g.width+x)*C+c] : 0.0f;
            acc[e] = split512_half(v*skip[c]);
        }
        for (int k0 = 0; k0 < C; k0 += 32) {
            FC part; slice<FBc>(part, a+rt*16*C+k0, C, w3+size_t(o0+ct*16)*C+k0, C, 16);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e)
            feature[(size_t(win)*64+rt*16+lr+e)*C + o0+ct*16+lc] = split512_to_half(split512_fp8(acc[e]));
    }
}

// Scratch for the split FFN (hidden, then middle), grown on demand.
inline h16* ffn_scratch(size_t halves) {
    static h16* data = nullptr;
    static size_t capacity = 0;
    if (capacity < halves) {
        if (data) HIP_CHECK(hipFree(data));
        data = nullptr; capacity = 0;
        HIP_CHECK(hipMalloc(&data, halves*sizeof(h16)));
        capacity = halves;
    }
    return data;
}

// Head-split projection: acc = half(feature * skip), then each head's slice
// in head order (the order of the fused kernel), FP8 output and raw, HWC.
template<int C>
__global__ void k_swin_combine(const h16* __restrict__ feature, const float* __restrict__ partial,
                               const float* __restrict__ attention_skip, Geometry g, int windows,
                               float* output, float* raw_out) {
    constexpr int HEADS = C/32;
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i >= size_t(windows)*64*C) return;
    int c = int(i % C), t = int(i / C % 64), win = int(i / C / 64), x, y;
    if (!g.pixel(win, t, x, y)) return;
    float acc = split512_half(float(feature[i])*attention_skip[c]);
    for (int h = 0; h < HEADS; ++h)
        acc = split512_half(acc + partial[((size_t(win)*HEADS + h)*64 + t)*C + c]);
    size_t at = (size_t(y)*g.width + x)*C + c;
    output[at] = split512_fp8(acc);
    if (raw_out) raw_out[at] = acc;
}

// Head-split scratch (fp32 projection slices), grown on demand.
inline float* head_partials(size_t count) {
    static float* data = nullptr;
    static size_t capacity = 0;
    if (capacity < count) {
        if (data) HIP_CHECK(hipFree(data));
        data = nullptr; capacity = 0;
        HIP_CHECK(hipMalloc(&data, count*sizeof(float)));
        capacity = count;
    }
    return data;
}

struct Block { const float *w1, *w2, *w3, *skip, *qkv, *scales, *bias, *projection, *attention_skip; };

// One fused block unless exact_math; scratch holds 64*C halves per window.
// false means "run the reference kernels".
template<int C>
inline bool run(const float* input, float* output, float* raw_out, int width, int height, int shift,
                const Block& w, void* scratch) {
    if (c512_resident::exact_math) return false;
    hipStream_t s = c512_resident::stream;
    Geometry g{width, height, (shift&1) ? 4 : 0, (shift&2) ? 4 : 0, 0};
    g.pw = ((width+g.px+7)/8)*8;
    int ph = ((height+g.py+7)/8)*8, windows = (g.pw/8)*(ph/8);
    auto half = [&](const float* W, size_t n) { return wmma_gemm::half_weights(s, W, n); };
    auto* feature = reinterpret_cast<h16*>(scratch);
    if (windows < 128 && C >= 64) {
        // Few windows: the FFN as three wider launches (see k_swin_hidden).
        h16* hidden = ffn_scratch(size_t(windows)*64*(4*C + C));
        h16* middle = hidden + size_t(windows)*64*4*C;
        hipLaunchKernelGGL(k_swin_hidden<C>, dim3(windows, 4*C/128), dim3(256), 0, s, input, g,
                           half(w.w1, size_t(4*C)*C), hidden);
        hipLaunchKernelGGL(k_swin_w2<C>, dim3(windows, C/64), dim3(256), 0, s, hidden,
                           half(w.w2, size_t(C)*4*C), middle);
        hipLaunchKernelGGL(k_swin_w3<C>, dim3(windows, C/64), dim3(256), 0, s, input, g, middle,
                           half(w.w3, size_t(C)*C), w.skip, feature);
    } else {
        hipLaunchKernelGGL(k_swin_ffn<C>, dim3(windows), dim3(256), 0, s, input, g,
                           half(w.w1, size_t(4*C)*C), half(w.w2, size_t(C)*4*C), half(w.w3, size_t(C)*C), w.skip, feature);
    }
    HIP_CHECK(hipGetLastError());
    if (c512_resident::launch_hook) c512_resident::launch_hook(C == 64 ? "k_swin_ffn<64>" : C == 128 ? "k_swin_ffn<128>" : "k_swin_ffn<256>");
    // Few windows leave most of the GPU idle while each block walks its heads:
    // then each (window, head) is its own block and a combine pass follows.
    const bool split = windows < 128 && C > 32;
    float* partial = split ? head_partials(size_t(windows)*64*C*(C/32)) : nullptr;
    if (split)
        hipLaunchKernelGGL((k_swin_attention<C, true>), dim3(windows, C/32), dim3(256), 0, s, feature, g,
                           half(w.qkv, size_t(3*C)*C), w.scales, w.bias, half(w.projection, size_t(C)*C),
                           w.attention_skip, output, raw_out, partial);
    else
        hipLaunchKernelGGL((k_swin_attention<C, false>), dim3(windows), dim3(256), 0, s, feature, g,
                           half(w.qkv, size_t(3*C)*C), w.scales, w.bias, half(w.projection, size_t(C)*C),
                           w.attention_skip, output, raw_out, nullptr);
    HIP_CHECK(hipGetLastError());
    if (split) {
        size_t n = size_t(windows)*64*C;
        hipLaunchKernelGGL(k_swin_combine<C>, dim3((n+255)/256), dim3(256), 0, s, feature, partial,
                           w.attention_skip, g, windows, output, raw_out);
        HIP_CHECK(hipGetLastError());
    }
    if (c512_resident::launch_hook) c512_resident::launch_hook(C == 64 ? "k_swin_attention<64>" : C == 128 ? "k_swin_attention<128>" : "k_swin_attention<256>");
    return true;
}
} // namespace swin_fused
