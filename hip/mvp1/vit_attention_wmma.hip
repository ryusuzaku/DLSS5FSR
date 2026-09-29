#pragma once
// Fast (non-exact) form of k_vit_attention_chunks: the numerator is a
// (64 queries x 32 keys) x (32 keys x 32 channels) WMMA product per slice.
// fp8(exponent) and V (FP8) are exact in half, so every product is exact;
// the 32-key part is summed in fp32 (reference: double, then float) before
// the same half boundary. Denominators keep the reference reduction order and
// are computed once per query instead of once per output channel.
// Needs vit_expand_chain.hip and rocwmma (via tiled_gemm.hip) first.

// Grid (tokens/64, 32 heads), 256 threads; tokens a multiple of 64.
__global__ __launch_bounds__(256) void k_vit_attention_wmma(const float* __restrict__ qkv,
                                                            const float* __restrict__ exponents,
                                                            float* output, int tokens) {
    using namespace rocwmma;
    using h16 = float16_t;
    constexpr int LD = 40;  // padded LDS rows (halves)
    __shared__ __attribute__((aligned(16))) h16 ps[64*LD];  // fp8(exponent) [query][key]
    __shared__ __attribute__((aligned(16))) h16 vs[32*LD];  // V [key][channel]
    __shared__ float den[64];
    const int head = blockIdx.y, q0 = blockIdx.x*64;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16, rt = wave/2, ct = wave%2;
    const float* E = exponents + (size_t(head)*tokens + q0)*tokens;
    if (tid < 64) {
        const float* e = E + size_t(tid)*tokens;
        float d = 0.0f;
        for (int chunk = 0; chunk < tokens; chunk += 64) {
            float part = vit_denominator(e+chunk);
            d = chunk ? split512_half(d+part) : part;
        }
        den[tid] = d;
    }
    float acc[8];
    for (int base = 0; base < tokens; base += 32) {
        for (int i = tid; i < 64*32; i += 256) {
            int q = i/32, k = i%32;
            ps[q*LD+k] = split512_to_half(split512_fp8(E[size_t(q)*tokens+base+k]));
        }
        for (int i = tid; i < 32*32; i += 256) {
            int k = i/32, c = i%32;
            vs[k*LD+c] = split512_to_half(qkv[size_t(base+k)*3072+2048+head*32+c]);
        }
        __syncthreads();
        fragment<matrix_a,16,16,16,h16,row_major> a0, a1;
        fragment<matrix_b,16,16,16,h16,row_major> b0, b1;
        fragment<accumulator,16,16,16,float> part;
        load_matrix_sync(a0, ps + rt*16*LD, LD);
        load_matrix_sync(a1, ps + rt*16*LD + 16, LD);
        load_matrix_sync(b0, vs + ct*16, LD);
        load_matrix_sync(b1, vs + 16*LD + ct*16, LD);
        fill_fragment(part, 0.0f);
        mma_sync(part, a0, b0, part);
        mma_sync(part, a1, b1, part);
        for (int e = 0; e < 8; ++e) acc[e] = split512_half(base ? acc[e]+part.x[e] : part.x[e]);
        __syncthreads();
    }
    for (int e = 0; e < 8; ++e) {
        int q = rt*16+lr+e, c = ct*16+lc;
        output[size_t(q0+q)*1024+head*32+c] = split512_fp8(split512_half(acc[e]*split512_half(1.0f/den[q])));
    }
}
