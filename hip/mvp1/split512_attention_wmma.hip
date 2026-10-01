#pragma once
// Fused fast (non-exact) window attention of the C64-C512 blocks: one
// workgroup runs one 64-token window x one 32-channel head from raw qkv to
// the FP8 context, replacing qknorm_inv/apply, scores, exp, inv_rows,
// prob_rows and context. Scores and context are f16 WMMA products (FP8
// operands, exact products, fp32 sums inside each 32-product slice instead
// of the sequential order); every rounding boundary, the exponent transform
// and the fixed denominator order are unchanged.
// Layouts as the per-level kernels: qkv [token][3C] (Q, K, V; head*32 + c),
// bias [head][64][64], context [token][C]. Needs split512_attention.hip and
// rocwmma (tiled_gemm.hip) first.

// Where window tokens live: window order (hwc = 0, row win*64 + t), or an
// HWC image (qkv rows y*width + x, context written there) through a shifted
// 8x8 partition, padding tokens reading zero qkv (what the gather produced)
// and not written. Extents above 4 only (no wrap-around).
struct WindowIo {
    int hwc = 0, width = 0, height = 0, px = 0, py = 0, pw = 0;
    __device__ long long row(int win, int t) const {
        if (!hwc) return (long long)win*64 + t;
        int x = (win % (pw/8))*8 + t%8 - px, y = (win / (pw/8))*8 + t/8 - py;
        return x >= 0 && x < width && y >= 0 && y < height ? (long long)y*width + x : -1;
    }
};

__global__ __launch_bounds__(256) void k_split512_attention_wmma(
        const float* __restrict__ qkv, const float* __restrict__ scales, const float* __restrict__ bias,
        float* context, int channels, WindowIo io) {
    using namespace rocwmma;
    using h16 = float16_t;
    constexpr int LD = 40, LS = 72;  // padded LDS rows (halves)
    __shared__ __attribute__((aligned(16))) h16 qs[64*LD], ks[64*LD], vs[64*LD];
    __shared__ __attribute__((aligned(16))) h16 sc[64*LS];
    __shared__ float inv[128];
    const int win = blockIdx.x, head = blockIdx.y;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    // Raw qkv is half-valued (RAW GEMM epilogue), so half storage is exact.
    for (int i = tid; i < 3*64*32; i += 256) {
        int m = i/2048, t = i/32%64, c = i%32;
        h16* dst = m == 0 ? qs : m == 1 ? ks : vs;
        long long r = io.row(win, t);
        dst[t*LD+c] = split512_to_half(r < 0 ? 0.0f : qkv[r*3*channels + m*channels + head*32 + c]);
    }
    __syncthreads();
    if (tid < 128) {
        const h16* src = (tid < 64 ? qs : ks) + (tid%64)*LD;
        float v[32];
        for (int c = 0; c < 32; ++c) v[c] = float(src[c]);
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
    {   // Scores (K=32 raw sum) + bias and exponent transform: 16 tiles, 2 per wave.
        const int rt = wave%4;
        fragment<matrix_a,16,16,16,h16,row_major> a0, a1;
        load_matrix_sync(a0, qs + rt*16*LD, LD);
        load_matrix_sync(a1, qs + rt*16*LD + 16, LD);
        for (int j = 0; j < 2; ++j) {
            const int ct = wave/4*2+j;
            fragment<matrix_b,16,16,16,h16,col_major> b0, b1;
            fragment<accumulator,16,16,16,float> s;
            load_matrix_sync(b0, ks + ct*16*LD, LD);
            load_matrix_sync(b1, ks + ct*16*LD + 16, LD);
            fill_fragment(s, 0.0f);
            mma_sync(s, a0, b0, s);
            mma_sync(s, a1, b1, s);
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
    {   // Context: two 32-key slices, half boundary after each: 8 tiles.
        const int rt = wave/2, ct = wave%2;
        float acc[8] = {};
        for (int k0 = 0; k0 < 64; k0 += 32) {
            fragment<matrix_a,16,16,16,h16,row_major> a0, a1;
            fragment<matrix_b,16,16,16,h16,row_major> b0, b1;
            fragment<accumulator,16,16,16,float> part;
            load_matrix_sync(a0, sc + rt*16*LS + k0, LS);
            load_matrix_sync(a1, sc + rt*16*LS + k0 + 16, LS);
            load_matrix_sync(b0, vs + k0*LD + ct*16, LD);
            load_matrix_sync(b1, vs + (k0+16)*LD + ct*16, LD);
            fill_fragment(part, 0.0f);
            mma_sync(part, a0, b0, part);
            mma_sync(part, a1, b1, part);
            for (int e = 0; e < 8; ++e) acc[e] = split512_half(acc[e]+part.x[e]);
        }
        for (int e = 0; e < 8; ++e) {
            int q = rt*16+lr+e, c = ct*16+lc;
            long long r = io.row(win, q);
            if (r >= 0) context[r*channels + head*32 + c] = split512_fp8(acc[e]);
        }
    }
}

namespace c512_resident {
// Launches the fused attention unless exact_math; false means "run the
// reference kernels". tokens: whole 64-token windows in window order.
// io (optional): qkv and context in HWC order instead (tokens = the padded
// window token count, which sets the grid).
inline bool fused_attention(const float* qkv, const float* scales, const float* bias,
                            float* context, int tokens, int channels, const WindowIo* io = nullptr) {
    if (exact_math || tokens % 64 || channels % 32) return false;
    hipLaunchKernelGGL(k_split512_attention_wmma, dim3(tokens/64, channels/32), dim3(256), 0, stream,
                       qkv, scales, bias, context, channels, io ? *io : WindowIo());
    HIP_CHECK(hipGetLastError());
    if (launch_hook) launch_hook("k_split512_attention_wmma");
    return true;
}
} // namespace c512_resident
