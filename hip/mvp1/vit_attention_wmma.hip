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

// Fast fused ViT attention: from the raw qkv projection to the FP8 output in
// one launch (replaces k_vit_qkv_normalize, k_vit_scores, k_vit_exponents and
// k_vit_attention_wmma). Grid (tokens/64 query tiles, 32 heads); keys are
// walked in chunks of 64 in the reference's own structure: the chunk
// denominator (vit_denominator order) is half-accumulated across chunks and
// the numerator's 32-key slices are half-accumulated in key order, so no
// running-max rescaling is needed (the exponent transform is clamped). Scores
// and the numerator are WMMA products of FP8 operands (exact products, fp32
// sums inside a 32-product slice). Keys at or after `valid` (padding tokens
// of an extent whose token count is not a multiple of 64) get a zero
// exponent: they drop out of the denominator and the numerator.
__device__ inline float vit_inverse_norm(const float* raw) {
    float ordered[32];
    for (int j = 0; j < 32; ++j) ordered[j] = raw[vit_qkv_inverse_order(j)];
    float parts[16];
    for (int j = 0; j < 16; ++j)
        parts[j] = split512_half_sum(ordered[j]*ordered[j], split512_half(ordered[j+16]*ordered[j+16]));
    for (int j = 0; j < 8; ++j) parts[j] = split512_half(parts[2*j]+parts[2*j+1]);
    for (int width = 4; width; width /= 2)
        for (int j = 0; j < width; ++j) parts[j] = split512_half(parts[j]+parts[j+width]);
    return split512_half(1.0f/sqrtf(fmaxf(parts[0], .00006198883056640625f)));
}

__global__ __launch_bounds__(256) void k_vit_attention_fused(const float* __restrict__ projected,
                                                             const float* __restrict__ scales,
                                                             float* output, int tokens, int valid) {
    using namespace rocwmma;
    using h16 = float16_t;
    constexpr int LD = 40, LS = 72;
    __shared__ __attribute__((aligned(16))) h16 qs[64*LD], ks[64*LD], vs[64*LD];
    __shared__ __attribute__((aligned(16))) h16 ps[64*LS];  // exponents, then fp8(exponents)
    __shared__ float den[64];
    const int head = blockIdx.y, q0 = blockIdx.x*64;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
    const int lr = lane/16*8, lc = lane%16;
    // Q rows: normalized, scaled (5.65625, then the head scale), FP8.
    if (tid < 64) {
        float raw[32];
        const float* src = projected + size_t(q0+tid)*3072 + head*32;
        for (int c = 0; c < 32; ++c) raw[c] = src[c];
        const float inv = vit_inverse_norm(raw), scale = split512_half(scales[head]);
        for (int c = 0; c < 32; ++c) {
            float v = split512_half(split512_half(split512_half(raw[c]*inv)*5.65625f)*scale);
            qs[tid*LD+c] = split512_to_half(split512_fp8(v));
        }
    }
    float d = 0.0f;           // tid < 64: this query's running denominator
    float acc[8] = {};        // numerator tile (rt = wave/2, ct = wave%2)
    for (int k0 = 0; k0 < tokens; k0 += 64) {
        if (tid < 64) {        // K row: normalized, FP8
            float raw[32];
            const float* src = projected + size_t(k0+tid)*3072 + 1024 + head*32;
            for (int c = 0; c < 32; ++c) raw[c] = src[c];
            const float inv = vit_inverse_norm(raw);
            for (int c = 0; c < 32; ++c) ks[tid*LD+c] = split512_to_half(split512_fp8(split512_half(raw[c]*inv)));
        } else if (tid < 128) {  // V row: FP8
            const float* src = projected + size_t(k0+tid-64)*3072 + 2048 + head*32;
            for (int c = 0; c < 32; ++c) vs[(tid-64)*LD+c] = split512_to_half(split512_fp8(src[c]));
        }
        __syncthreads();
        {   // Scores (raw sum, half) and the exponent transform: 16 tiles, 2 per wave.
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
                    float score = split512_half(s.x[e]);
                    float affine = fminf(1.9775390625f, fmaxf(1.439453125f, split512_half(score*.08953857421875f+1.708984375f)));
                    _Float16 hc = split512_to_half(affine);
                    unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
                    bits = (unsigned short)((unsigned(bits)<<4)+0x4000u);
                    __builtin_memcpy(&hc, &bits, 2);
                    if (k0+ct*16+lc >= valid) hc = _Float16(0.0f);
                    ps[(rt*16+lr+e)*LS + ct*16+lc] = hc;
                }
            }
        }
        __syncthreads();
        if (tid < 64) {        // this chunk's denominator, the reference order
            float e[64];
            for (int k = 0; k < 64; ++k) e[k] = float(ps[tid*LS+k]);
            float part = vit_denominator(e);
            d = k0 ? split512_half(d+part) : part;
        }
        __syncthreads();
        for (int i = tid; i < 64*64; i += 256) {
            h16* p = ps + (i/64)*LS + i%64;
            *p = split512_to_half(split512_fp8(float(*p)));
        }
        __syncthreads();
        {   // Numerator: the chunk's two 32-key slices, in key order.
            const int rt = wave/2, ct = wave%2;
            for (int s0 = 0; s0 < 64; s0 += 32) {
                fragment<matrix_a,16,16,16,h16,row_major> a0, a1;
                fragment<matrix_b,16,16,16,h16,row_major> b0, b1;
                fragment<accumulator,16,16,16,float> part;
                load_matrix_sync(a0, ps + rt*16*LS + s0, LS);
                load_matrix_sync(a1, ps + rt*16*LS + s0 + 16, LS);
                load_matrix_sync(b0, vs + s0*LD + ct*16, LD);
                load_matrix_sync(b1, vs + (s0+16)*LD + ct*16, LD);
                fill_fragment(part, 0.0f);
                mma_sync(part, a0, b0, part);
                mma_sync(part, a1, b1, part);
                const bool first = k0 == 0 && s0 == 0;
                for (int e = 0; e < 8; ++e) acc[e] = split512_half(first ? part.x[e] : acc[e]+part.x[e]);
            }
        }
        __syncthreads();
    }
    if (tid < 64) den[tid] = d;
    __syncthreads();
    const int rt = wave/2, ct = wave%2;
    for (int e = 0; e < 8; ++e) {
        int q = rt*16+lr+e, c = ct*16+lc;
        output[size_t(q0+q)*1024+head*32+c] = split512_fp8(split512_half(acc[e]*split512_half(1.0f/den[q])));
    }
}

// Transposed form of k_vit_attention_fused (S336d), same values and orders.
// k_vit_kv8 first normalizes K (vit order) and quantizes K and V once
// per launch: K8 [head][token][32], V8 [head][32][tokens] (E4M3 bytes).
// k_vit_attention_t: grid (tokens/64, 32 heads), 4 waves of 16 queries,
// [d][query] FP8 WMMA with K and V^T read as A operands straight from
// those bytes; the permuted-order reductions (the Q norm, each chunk's
// denominator) go through a small per-wave LDS row, one lane a query.
// out8 (optional): the output's E4M3 bytes for the projection GEMM.
__global__ void k_vit_kv8(const float* __restrict__ projected, unsigned char* k8, unsigned char* v8, int tokens) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= tokens*32) return;
    const int token = i % tokens, head = i / tokens;  // consecutive tokens: coalesced V^T bytes
    const float4* k = reinterpret_cast<const float4*>(projected + size_t(token)*3072 + 1024 + head*32);
    const float4* v = reinterpret_cast<const float4*>(projected + size_t(token)*3072 + 2048 + head*32);
    float raw[32], vv[32];
    for (int c = 0; c < 8; ++c) {
        const float4 a = k[c], b = v[c];
        raw[4*c] = a.x; raw[4*c+1] = a.y; raw[4*c+2] = a.z; raw[4*c+3] = a.w;
        vv[4*c] = b.x; vv[4*c+1] = b.y; vv[4*c+2] = b.z; vv[4*c+3] = b.w;
    }
    const float inv = vit_inverse_norm(raw);
    for (int c = 0; c < 32; c += 8) {
        float q[8];
        for (int e = 0; e < 8; ++e) q[e] = split512_half(raw[c+e]*inv);
        *reinterpret_cast<wmma_t::op8*>(k8 + (size_t(head)*tokens + token)*32 + c) = wmma_t::pack(q);
    }
    for (int c = 0; c < 32; ++c) v8[(size_t(head)*32 + c)*tokens + token] = wmma_t::byte8(split512_half(vv[c]));
}

__global__ __launch_bounds__(128) void k_vit_attention_t(const float* __restrict__ projected,
        const unsigned char* __restrict__ k8, const unsigned char* __restrict__ v8,
        const float* __restrict__ scales, float* output, unsigned char* out8, int tokens, int valid) {
    using namespace wmma_t;
    __shared__ float row[4][16][33];  // per wave: Q values, then a chunk's exponents (as floats)
    __shared__ float erow[4][16][65];
    const int head = blockIdx.y, q0 = blockIdx.x*64;
    const int tid = threadIdx.x, wave = tid/32, lane = tid%32, n = lane%16, h = lane/16;
    const int query = q0 + wave*16 + n;
    // Q: raw values (half), the norm in vit order by one lane a query.
    float qv[2][8];
    {
        const float* src = projected + size_t(query)*3072 + head*32;
        for (int t = 0; t < 2; ++t)
            for (int e = 0; e < 8; ++e) { qv[t][e] = src[t*16 + h*8 + e]; row[wave][n][t*16 + h*8 + e] = qv[t][e]; }
    }
    __syncthreads();
    float inv = 0.0f;
    if (h == 0) {
        float raw[32];
        for (int c = 0; c < 32; ++c) raw[c] = row[wave][n][c];
        inv = vit_inverse_norm(raw);
    }
    inv = __shfl(inv, n);  // lanes n and n+16 share the query
    op8 qb[2];
    {
        const float scale = hf(scales[head]);
        for (int t = 0; t < 2; ++t) {
            float v[8];
            for (int e = 0; e < 8; ++e) v[e] = hf(hf(hf(qv[t][e]*inv)*5.65625f)*scale);
            qb[t] = pack(v);
        }
    }
    const unsigned char* K = k8 + size_t(head)*tokens*32;
    const unsigned char* V = v8 + size_t(head)*32*tokens;
    float d = 0.0f, acc[2][8] = {};
    for (int k0 = 0; k0 < tokens; k0 += 64) {
        float ex[4][8];
        for (int kt = 0; kt < 4; ++kt) {
            const unsigned char* kr = K + size_t(k0 + kt*16)*32;
            acc8 c = {};
            c = mma(arow(kr, 32, 16), qb[1], mma(arow(kr, 32, 0), qb[0], c));
            for (int e = 0; e < 8; ++e) {
                float score = hf(c[e]);
                float affine = fminf(1.9775390625f, fmaxf(1.439453125f, hf(score*.08953857421875f+1.708984375f)));
                _Float16 hc = split512_to_half(affine);
                unsigned short bits; __builtin_memcpy(&bits, &hc, 2);
                bits = (unsigned short)((unsigned(bits)<<4)+0x4000u);
                __builtin_memcpy(&hc, &bits, 2);
                const int key = k0 + kt*16 + h*8 + e;
                ex[kt][e] = key >= valid ? 0.0f : float(hc);
                erow[wave][n][kt*16 + h*8 + e] = ex[kt][e];
            }
        }
        __syncthreads();  // (wave-local rows; a block barrier keeps it simple)
        float part = 0.0f;
        if (h == 0) part = vit_denominator(&erow[wave][n][0]);
        part = __shfl(part, n);
        d = k0 ? hf(d + part) : part;
        __syncthreads();  // erow is rewritten by the next chunk
        op8 pb[4];
        for (int kt = 0; kt < 4; ++kt) pb[kt] = pack(ex[kt]);  // fp8(exponent): exact halves
        for (int dt = 0; dt < 2; ++dt)
            for (int s0 = 0; s0 < 2; ++s0) {
                const unsigned char* vr = V + size_t(dt*16)*tokens + k0 + s0*32;
                acc8 p = {};
                p = mma(arow(vr, tokens, 16), pb[2*s0+1], mma(arow(vr, tokens, 0), pb[2*s0], p));
                const bool first = k0 == 0 && s0 == 0;
                for (int e = 0; e < 8; ++e) acc[dt][e] = hf(first ? p[e] : acc[dt][e] + p[e]);
            }
    }
    const float r = hf(1.0f/d);
    for (int dt = 0; dt < 2; ++dt) {
        float v[8], q[8];
        for (int e = 0; e < 8; ++e) { v[e] = hf(acc[dt][e]*r); q[e] = split512_fp8(v[e]); }
        const size_t at = size_t(query)*1024 + head*32 + dt*16 + h*8;
        if (output) {  // null: the bytes (out8) only
            reinterpret_cast<float4*>(output + at)[0] = make_float4(q[0], q[1], q[2], q[3]);
            reinterpret_cast<float4*>(output + at)[1] = make_float4(q[4], q[5], q[6], q[7]);
        }
        if (out8) *reinterpret_cast<op8*>(out8 + at) = pack(v);
    }
}
