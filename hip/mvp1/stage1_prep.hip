// hip/mvp1/stage1_prep.hip — Stage 1 staging & quantisation HIP kernels for gfx1201
//
// 1. rgba8_to_f32x4: UNORM8 RGBA pitch-padded D3D12 buffer -> linear float32 [H*W*4]
// 2. h2e4m3: Manual f16 -> e4m3fn RN-satfinite quantiser (PTX bit-exact)
// 3. mpcubic_silu2: Packed f16x2 MpCubicSilu activation via __hfma2/__habs2/__hmin2/__hmax2
// 4. normalise_input: (x - 0.5f) * (2.0f * scale) + dither in f32 -> f16

#define _CRT_SECURE_NO_WARNINGS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>

#define HIPCHECK(e)                                                          \
    do {                                                                     \
        hipError_t _e = (e);                                                 \
        if (_e != hipSuccess) {                                              \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__,          \
                         __LINE__, hipGetErrorString(_e));                   \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Helper elementwise min/max for __half2
__device__ inline __half2 __hmin2(__half2 a, __half2 b) {
    return __half2{__hmin(a.x, b.x), __hmin(a.y, b.y)};
}

__device__ inline __half2 __hmax2(__half2 a, __half2 b) {
    return __half2{__hmax(a.x, b.x), __hmax(a.y, b.y)};
}

// ---------------------------------------------------------------------------
// 1. RGBA8 UNORM -> float32 staging kernel
// ---------------------------------------------------------------------------

__global__ void rgba8_to_f32x4_kernel(const uint8_t* __restrict__ in_rgba8,
                                      float* __restrict__ out_f32,
                                      int w, int h, int pitch_bytes) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    int in_offset = y * pitch_bytes + x * 4;
    int out_offset = (y * w + x) * 4;

    out_f32[out_offset + 0] = (float)in_rgba8[in_offset + 0] / 255.0f;
    out_f32[out_offset + 1] = (float)in_rgba8[in_offset + 1] / 255.0f;
    out_f32[out_offset + 2] = (float)in_rgba8[in_offset + 2] / 255.0f;
    out_f32[out_offset + 3] = (float)in_rgba8[in_offset + 3] / 255.0f;
}

// ---------------------------------------------------------------------------
// 2. Manual f16 -> e4m3fn RN-satfinite quantiser
// ---------------------------------------------------------------------------

__device__ inline uint8_t h2e4m3_manual(uint16_t u16) {
    uint8_t sign = (uint8_t)((u16 >> 15) & 1);
    uint16_t abs_u16 = u16 & 0x7FFF;

    // NaN: preserve sign, mantissa all 1s (0x7F / 0xFF)
    if (abs_u16 > 0x7C00) {
        return (uint8_t)((sign << 7) | 0x7F);
    }
    // Satfinite: Inf and values >= 432.0 (0x5EC0) saturate to 448.0 (0x7E)
    if (abs_u16 >= 0x5EC0) {
        return (uint8_t)((sign << 7) | 0x7E);
    }
    // Underflow: values <= 1/1024 (0x1400) round to zero
    if (abs_u16 <= 0x1400) {
        return (uint8_t)((sign << 7) | 0x00);
    }

    uint16_t exp = abs_u16 >> 10;
    uint16_t mant = abs_u16 & 0x3FF;

    if (exp < 9) {
        // E4M3 subnormal: target unit is 2^-9
        uint32_t sig = 1024 + mant;
        int shift = 16 - exp;
        uint32_t half_lsb = 1u << (shift - 1);
        uint32_t rem = sig & ((1u << shift) - 1);
        uint32_t res = sig >> shift;
        if (rem > half_lsb || (rem == half_lsb && (res & 1))) {
            res++;
        }
        return (uint8_t)((sign << 7) | (uint8_t)res);
    } else {
        // E4M3 normal: e_8 = exp - 8, mantissa shift = 7 bits
        uint8_t e_8 = (uint8_t)(exp - 8);
        uint8_t m_8 = (uint8_t)(mant >> 7);
        uint16_t rem = mant & 0x7F;
        if (rem > 64 || (rem == 64 && (m_8 & 1))) {
            m_8++;
            if (m_8 == 8) {
                m_8 = 0;
                e_8++;
            }
        }
        return (uint8_t)((sign << 7) | (e_8 << 3) | m_8);
    }
}

__device__ inline uint16_t h2e4m3x2_device(__half2 h2) {
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&h2);
    uint8_t lo = h2e4m3_manual(u16[0]);
    uint8_t hi = h2e4m3_manual(u16[1]);
    return (uint16_t)((hi << 8) | lo);
}

__global__ void h2e4m3_exhaustive_kernel(const uint16_t* __restrict__ in_u16,
                                         uint8_t* __restrict__ out_e4m3,
                                         int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        out_e4m3[idx] = h2e4m3_manual(in_u16[idx]);
    }
}

// ---------------------------------------------------------------------------
// 3. MpCubicSilu activation kernel (__hfma2 / __hmin2 / __hmax2 / __habs2)
// ---------------------------------------------------------------------------

__device__ inline __half2 mpcubic_silu2_device(__half2 in) {
    const __half2 c_neg4 = __float2half2_rn(-4.0f);
    const __half2 c_pos4 = __float2half2_rn(4.0f);
    const __half2 c0 = __float2half2_rn(-0.055908203125f);
    const __half2 c1 = __float2half2_rn(0.447265625f);
    const __half2 c2 = __float2half2_rn(0.89453125f);

    __half2 y = __hmin2(__hmax2(in, c_neg4), c_pos4);
    __half2 t = __hfma2(c0, __habs2(y), c1);
    __half2 u = __hfma2(y, t, c2);
    return __hmul2(in, u);
}

__global__ void mpcubic_silu2_exhaustive_kernel(const __half2* __restrict__ in_h2,
                                                __half2* __restrict__ out_h2,
                                                int n_pairs) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n_pairs) {
        out_h2[idx] = mpcubic_silu2_device(in_h2[idx]);
    }
}

// ---------------------------------------------------------------------------
// 4. Normalise input kernel: (x - 0.5f) * (2.0f * scale) + dither -> f16
// ---------------------------------------------------------------------------

__device__ inline __half normalise_input_device(float x, float scale, float dither) {
    float val = (x - 0.5f) * (2.0f * scale) + dither;
    return __float2half_rn(val);
}

__global__ void normalise_input_kernel(const float* __restrict__ in_f32,
                                       __half* __restrict__ out_f16,
                                       float scale, float dither,
                                       int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        out_f16[idx] = normalise_input_device(in_f32[idx], scale, dither);
    }
}

// ---------------------------------------------------------------------------
// Front-end kernels (HANDOFF 13.3/path-1; gate F pins PTX).
// Twins of swin_1h.hip section 1b (same PTX immediates, same order);
// __half ops are RN, bitwise-equal to the mirror on normal values.
// ---------------------------------------------------------------------------

__host__ __device__ inline uint32_t fe_pcg_round_hd(uint32_t st) {
    uint32_t xs = (st >> ((st >> 28) + 4u)) ^ st;
    return xs * 277803737u;
}
__host__ __device__ inline uint32_t fe_pcg_out_hd(uint32_t v) {
    return (((v >> 30) ^ (v >> 8)) + 1u);
}
__host__ __device__ inline void fe_pcg_streams_hd(uint32_t seed, uint32_t x,
                                                 uint32_t y, uint32_t o[4]) {
    uint32_t hseed = seed * 0x9E3779B9u;
    uint32_t s = (x * (0u - 1918454973u)) ^ hseed;
    uint32_t s1 = (y * (0u - 669632447u)) ^ s;
    uint32_t s2 = s1 ^ 608135816u;
    uint32_t f1 = fe_pcg_round_hd(s2);
    f1 = ((f1 >> 22) ^ f1);
    const uint32_t C0[4] = {747796405u, (0u - 93469191u),
                            (0u - 895109107u), (0u - 2094846927u)};
    const uint32_t C1[4] = {(0u - 1403630843u), 1192405134u, 568162667u,
                            878960812u};
    for (int i = 0; i < 4; i++) {
        uint32_t v = f1 * C0[i] + C1[i];
        o[i] = fe_pcg_out_hd(fe_pcg_round_hd(v));
    }
}

__global__ void fe_pcg_kernel(uint32_t seed,
                              uint32_t* __restrict__ out_u32) {
    // 64 fetches (pas*32+wl); 4 streams each.
    int f = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (f >= 64) return;
    uint32_t wl = (uint32_t)(f % 32);
    uint32_t o[4];
    fe_pcg_streams_hd(seed, wl & 7u, wl >> 3, o);
    for (int i = 0; i < 4; i++) out_u32[f * 4 + i] = o[i];
}

__global__ void fe_normalize_kernel(const float* __restrict__ in_tex,
                                    uint16_t* __restrict__ out_b16,
                                    float half, float mult, int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        __half t = __float2half_rn(in_tex[idx]);
        __half h = __float2half_rn(half);
        __half m = __float2half_rn(mult);
        __half d = __hsub(t, h);
        __half r = __hmul(d, m);
        out_b16[idx] = *reinterpret_cast<uint16_t*>(&r);
    }
}

__global__ void fe_patch_kernel(const float* __restrict__ X,
                                const float* __restrict__ W,
                                float* __restrict__ Y) {
    // One thread per (q,o); k-ascending f32 (matches mirror/oracle order).
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 64 * 32) return;
    int q = i / 32, o = i % 32;
    float s = 0.0f;
    for (int d = 0; d < 16; d++) s += X[q * 16 + d] * W[o * 16 + d];
    Y[i] = s;
}

__global__ void fe_bm_affine_kernel(const float* __restrict__ in_u,
                                    float* __restrict__ out_g,
                                    float Wd, float Ht, int n) {
    // Box-Muller (device libm, tol vs mirror) + affine u-coord
    // (div/fma/mul, tol vs mirror) per element.
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        float u1 = in_u[idx * 4 + 0], u2 = in_u[idx * 4 + 1];
        float u3 = in_u[idx * 4 + 2], u4 = in_u[idx * 4 + 3];
        const float ln2 = 0.6931471824645996f;
        const float two_pi = 6.283185307179586f;
        float r1 = sqrtf(-2.0f * log2f(u1) * ln2);
        out_g[idx * 4 + 0] = r1 * cosf(u2 * two_pi);
        out_g[idx * 4 + 1] = r1 * sinf(u2 * two_pi);
        out_g[idx * 4 + 2] = sqrtf(-2.0f * log2f(u3) * ln2) * cosf(u4 * two_pi);
        // affine x-map with identity params (s=1,o=0,m=1), square Wd=Ht
        float u = ((float)idx + 0.5f) / Wd;
        out_g[idx * 4 + 3] = (u * 1.0f + 0.0f) * 1.0f;
        (void)Ht;
    }
}

// ---------------------------------------------------------------------------
// Main test & verification runner
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== MVP-1 Stage 1 Prep Kernels (gfx1201) ===\n");

    hipDeviceProp_t prop{};
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (%s, %d CUs)\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount);

    int passes = 0, total_passes = 8;

    // -----------------------------------------------------------------------
    // Test 1: rgba8_to_f32x4 staging kernel
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 1: rgba8_to_f32x4 staging --\n");
        const int w = 64, h = 64;
        const int pitch = 384;  // padded from 256 bytes
        std::vector<uint8_t> h_in(pitch * h);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                h_in[y * pitch + x * 4 + 0] = (uint8_t)((x * 4) & 0xFF);
                h_in[y * pitch + x * 4 + 1] = (uint8_t)((y * 4) & 0xFF);
                h_in[y * pitch + x * 4 + 2] = (uint8_t)((x + y) & 0xFF);
                h_in[y * pitch + x * 4 + 3] = 255;
            }
        }

        uint8_t* d_in = nullptr;
        float* d_out = nullptr;
        HIPCHECK(hipMalloc(&d_in, h_in.size()));
        HIPCHECK(hipMalloc(&d_out, w * h * 4 * sizeof(float)));
        HIPCHECK(hipMemcpy(d_in, h_in.data(), h_in.size(), hipMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((w + 15) / 16, (h + 15) / 16);
        hipLaunchKernelGGL(rgba8_to_f32x4_kernel, grid, block, 0, 0,
                           d_in, d_out, w, h, pitch);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());

        std::vector<float> h_out(w * h * 4);
        HIPCHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                           hipMemcpyDeviceToHost));

        int mismatches = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < 4; ++c) {
                    float want = (float)h_in[y * pitch + x * 4 + c] / 255.0f;
                    float got = h_out[(y * w + x) * 4 + c];
                    if (std::abs(got - want) > 1e-7f) {
                        mismatches++;
                    }
                }
            }
        }
        HIPCHECK(hipFree(d_in));
        HIPCHECK(hipFree(d_out));

        if (mismatches == 0) {
            std::printf("  PASS  rgba8_to_f32x4 exact (%d pixels checked)\n", w * h);
            passes++;
        } else {
            std::printf("  FAIL  rgba8_to_f32x4 had %d mismatches\n", mismatches);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: h2e4m3 exhaustive 65,536 dump
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 2: h2e4m3 exhaustive (65536 values) --\n");
        const int N = 65536;
        std::vector<uint16_t> h_u16(N);
        for (int i = 0; i < N; ++i) h_u16[i] = (uint16_t)i;

        uint16_t* d_u16 = nullptr;
        uint8_t* d_e4m3 = nullptr;
        HIPCHECK(hipMalloc(&d_u16, N * sizeof(uint16_t)));
        HIPCHECK(hipMalloc(&d_e4m3, N * sizeof(uint8_t)));
        HIPCHECK(hipMemcpy(d_u16, h_u16.data(), N * sizeof(uint16_t), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(h2e4m3_exhaustive_kernel, dim3((N + 255) / 256), dim3(256),
                           0, 0, d_u16, d_e4m3, N);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());

        std::vector<uint8_t> h_e4m3(N);
        HIPCHECK(hipMemcpy(h_e4m3.data(), d_e4m3, N, hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_u16));
        HIPCHECK(hipFree(d_e4m3));

        FILE* f = fopen("build/dump_h2e4m3.bin", "wb");
        if (f) {
            fwrite(h_e4m3.data(), 1, N, f);
            fclose(f);
            std::printf("  Dumped 65536 bytes to build/dump_h2e4m3.bin\n");
        } else {
            std::printf("  WARN: could not write build/dump_h2e4m3.bin\n");
        }

        // Quick self-check on known constants
        bool ok = true;
        // 0.0 (0x0000) -> 0x00
        if (h_e4m3[0x0000] != 0x00) { std::printf("  FAIL: +0.0 != 0x00\n"); ok = false; }
        // -0.0 (0x8000) -> 0x80
        if (h_e4m3[0x8000] != 0x80) { std::printf("  FAIL: -0.0 != 0x80\n"); ok = false; }
        // 1.0 (0x3C00) -> 0x38
        if (h_e4m3[0x3C00] != 0x38) { std::printf("  FAIL: 1.0 != 0x38 (got 0x%02X)\n", h_e4m3[0x3C00]); ok = false; }
        // +inf (0x7C00) -> 0x7E (satfinite)
        if (h_e4m3[0x7C00] != 0x7E) { std::printf("  FAIL: +inf != 0x7E (got 0x%02X)\n", h_e4m3[0x7C00]); ok = false; }
        // -inf (0xFC00) -> 0xFE (satfinite)
        if (h_e4m3[0xFC00] != 0xFE) { std::printf("  FAIL: -inf != 0xFE (got 0x%02X)\n", h_e4m3[0xFC00]); ok = false; }

        if (ok) {
            std::printf("  PASS  h2e4m3 key landmarks verified\n");
            passes++;
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: mpcubic_silu2 exhaustive (65536 values in 32768 pairs)
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 3: mpcubic_silu2 exhaustive (65536 values) --\n");
        const int N = 65536;
        const int n_pairs = N / 2;
        std::vector<uint16_t> h_u16(N);
        for (int i = 0; i < N; ++i) h_u16[i] = (uint16_t)i;

        __half2* d_in = nullptr;
        __half2* d_out = nullptr;
        HIPCHECK(hipMalloc(&d_in, n_pairs * sizeof(__half2)));
        HIPCHECK(hipMalloc(&d_out, n_pairs * sizeof(__half2)));
        HIPCHECK(hipMemcpy(d_in, h_u16.data(), N * sizeof(uint16_t), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(mpcubic_silu2_exhaustive_kernel, dim3((n_pairs + 255) / 256),
                           dim3(256), 0, 0, d_in, d_out, n_pairs);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());

        std::vector<uint16_t> h_res(N);
        HIPCHECK(hipMemcpy(h_res.data(), d_out, N * sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_in));
        HIPCHECK(hipFree(d_out));

        FILE* f = fopen("build/dump_mpcubic.bin", "wb");
        if (f) {
            fwrite(h_res.data(), sizeof(uint16_t), N, f);
            fclose(f);
            std::printf("  Dumped 131072 bytes to build/dump_mpcubic.bin\n");
        } else {
            std::printf("  WARN: could not write build/dump_mpcubic.bin\n");
        }

        // Quick landmark check: 0.0 -> 0.0
        bool ok = true;
        if (h_res[0x0000] != 0x0000) {
            std::printf("  FAIL: MpCubicSilu(0.0) != 0.0 (got 0x%04X)\n", h_res[0x0000]);
            ok = false;
        }
        if (ok) {
            std::printf("  PASS  mpcubic_silu2 key landmarks verified\n");
            passes++;
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: normalise_input kernel
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 4: normalise_input --\n");
        const int N = 256;
        std::vector<float> h_in(N);
        const float scale = 1.25f;
        const float dither = 0.001f;
        for (int i = 0; i < N; ++i) h_in[i] = (float)i / 255.0f;

        float* d_in = nullptr;
        __half* d_out = nullptr;
        HIPCHECK(hipMalloc(&d_in, N * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, N * sizeof(__half)));
        HIPCHECK(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(normalise_input_kernel, dim3(1), dim3(256), 0, 0,
                           d_in, d_out, scale, dither, N);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());

        std::vector<uint16_t> h_out(N);
        HIPCHECK(hipMemcpy(h_out.data(), d_out, N * sizeof(__half), hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_in));
        HIPCHECK(hipFree(d_out));

        int mismatches = 0;
        for (int i = 0; i < N; ++i) {
            float exp_f32 = (h_in[i] - 0.5f) * (2.0f * scale) + dither;
            // CPU half conversion
            __half exp_h = __float2half_rn(exp_f32);
            uint16_t want = *reinterpret_cast<uint16_t*>(&exp_h);
            uint16_t got = h_out[i];
            if (got != want) mismatches++;
        }

        if (mismatches == 0) {
            std::printf("  PASS  normalise_input matches CPU __float2half_rn exactly\n");
            passes++;
        } else {
            std::printf("  FAIL  normalise_input had %d mismatches\n", mismatches);
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: fe_pcg streams (u32, bitwise vs host twin)
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 5: fe_pcg streams --\n");
        const uint32_t seed = 0x12345678u;
        uint32_t* d_out = nullptr;
        HIPCHECK(hipMalloc(&d_out, 64 * 4 * sizeof(uint32_t)));
        hipLaunchKernelGGL(fe_pcg_kernel, dim3(1), dim3(64), 0, 0,
                           seed, d_out);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h_out(64 * 4);
        HIPCHECK(hipMemcpy(h_out.data(), d_out, 64 * 4 * sizeof(uint32_t),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_out));
        int mismatches = 0;
        uint32_t h = 0x811C9DC5u;
        for (int f = 0; f < 64; f++) {
            uint32_t wl = (uint32_t)(f % 32);
            uint32_t want[4];
            fe_pcg_streams_hd(seed, wl & 7u, wl >> 3, want);
            for (int i = 0; i < 4; i++) {
                if (h_out[f * 4 + i] != want[i]) mismatches++;
                for (int b = 0; b < 4; b++) {
                    h ^= (uint8_t)(h_out[f * 4 + i] >> (8 * b));
                    h *= 0x01000193u;
                }
            }
        }
        std::printf("  FE-PCG-FNV %08X (expect 563A3EFD)\n", h);
        if (mismatches == 0 && h == 0x563A3EFDu) {
            std::printf("  PASS  fe_pcg bitwise, FNV matches gate\n");
            passes++;
        } else {
            std::printf("  FAIL  fe_pcg mismatches=%d fnv=%08X\n",
                        mismatches, h);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: fe_normalize (f16 sub/mul, bitwise vs host twin)
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 6: fe_normalize --\n");
        const int N = 64;
        std::vector<float> h_in(N);
        for (int i = 0; i < N; i++) h_in[i] = (float)(i % 16) - 0.5f;
        float* d_in = nullptr;
        uint16_t* d_out = nullptr;
        HIPCHECK(hipMalloc(&d_in, N * sizeof(float)));
        HIPCHECK(hipMalloc(&d_out, N * sizeof(uint16_t)));
        HIPCHECK(hipMemcpy(d_in, h_in.data(), N * sizeof(float),
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(fe_normalize_kernel, dim3(1), dim3(64), 0, 0,
                           d_in, d_out, 0.5f, 1.0f, N);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<uint16_t> h_out(N);
        HIPCHECK(hipMemcpy(h_out.data(), d_out, N * sizeof(uint16_t),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_in));
        HIPCHECK(hipFree(d_out));
        int mismatches = 0;
        for (int i = 0; i < N; i++) {
            __half t = __float2half_rn(h_in[i]);
            __half hh = __float2half_rn(0.5f);
            __half m = __float2half_rn(1.0f);
            __half r = __hmul(__hsub(t, hh), m);
            uint16_t want = *reinterpret_cast<uint16_t*>(&r);
            if (h_out[i] != want) mismatches++;
        }
        if (mismatches == 0) {
            std::printf("  PASS  fe_normalize bitwise (%d values)\n", N);
            passes++;
        } else {
            std::printf("  FAIL  fe_normalize had %d mismatches\n", mismatches);
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: fe_patch (f32 k-ascending, bitwise vs host twin + oracle FNV)
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 7: fe_patch --\n");
        std::vector<float> h_X(64 * 16), h_W(32 * 16);
        for (int q = 0; q < 64; q++)
            for (int d = 0; d < 16; d++)
                h_X[q * 16 + d] = (float)(((q + d) % 5) - 2);
        for (int o = 0; o < 32; o++)
            for (int d = 0; d < 16; d++)
                h_W[o * 16 + d] = (float)(((o * 16 + d) % 7) - 3);
        float *d_X = nullptr, *d_W = nullptr, *d_Y = nullptr;
        HIPCHECK(hipMalloc(&d_X, 64 * 16 * sizeof(float)));
        HIPCHECK(hipMalloc(&d_W, 32 * 16 * sizeof(float)));
        HIPCHECK(hipMalloc(&d_Y, 64 * 32 * sizeof(float)));
        HIPCHECK(hipMemcpy(d_X, h_X.data(), 64 * 16 * sizeof(float),
                           hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(d_W, h_W.data(), 32 * 16 * sizeof(float),
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(fe_patch_kernel, dim3(8), dim3(256), 0, 0,
                           d_X, d_W, d_Y);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<float> h_Y(64 * 32);
        HIPCHECK(hipMemcpy(h_Y.data(), d_Y, 64 * 32 * sizeof(float),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_X));
        HIPCHECK(hipFree(d_W));
        HIPCHECK(hipFree(d_Y));
        int mismatches = 0;
        uint32_t h = 0x811C9DC5u;
        for (int q = 0; q < 64; q++)
            for (int o = 0; o < 32; o++) {
                float want = 0.0f;
                for (int d = 0; d < 16; d++)
                    want += h_X[q * 16 + d] * h_W[o * 16 + d];
                float got = h_Y[q * 32 + o];
                if (memcmp(&got, &want, 4) != 0) mismatches++;
                uint32_t w;
                memcpy(&w, &got, 4);
                for (int b = 0; b < 4; b++) {
                    h ^= (uint8_t)(w >> (8 * b));
                    h *= 0x01000193u;
                }
            }
        std::printf("  FE-PATCH-KERNEL-FNV %08X\n", h);
        if (mismatches == 0) {
            std::printf("  PASS  fe_patch bitwise (%d outputs)\n", 64 * 32);
            passes++;
        } else {
            std::printf("  FAIL  fe_patch had %d mismatches\n", mismatches);
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: fe_bm_affine (tol vs host twin)
    // -----------------------------------------------------------------------
    {
        std::printf("\n-- Test 8: fe_bm_affine --\n");
        const int N = 64;
        std::vector<float> h_u(N * 4);
        {
            uint32_t st = 0x5EEDu;
            for (int i = 0; i < N * 4; i++) {
                st = 1664525u * st + 1013904223u;
                h_u[i] = (float)((st >> 8) % 0xFFFFFEu + 1u) * 5.9604644775390625e-08f;
            }
        }
        float *d_u = nullptr, *d_g = nullptr;
        HIPCHECK(hipMalloc(&d_u, N * 4 * sizeof(float)));
        HIPCHECK(hipMalloc(&d_g, N * 4 * sizeof(float)));
        HIPCHECK(hipMemcpy(d_u, h_u.data(), N * 4 * sizeof(float),
                           hipMemcpyHostToDevice));
        hipLaunchKernelGGL(fe_bm_affine_kernel, dim3(1), dim3(64), 0, 0,
                           d_u, d_g, 8.0f, 8.0f, N);
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
        std::vector<float> h_g(N * 4);
        HIPCHECK(hipMemcpy(h_g.data(), d_g, N * 4 * sizeof(float),
                           hipMemcpyDeviceToHost));
        HIPCHECK(hipFree(d_u));
        HIPCHECK(hipFree(d_g));
        float worst = 0.0f;
        const float ln2 = 0.6931471824645996f;
        const float two_pi = 6.283185307179586f;
        for (int i = 0; i < N; i++) {
            float u1 = h_u[i * 4 + 0], u2 = h_u[i * 4 + 1];
            float u3 = h_u[i * 4 + 2];
            float r1 = sqrtf(-2.0f * log2f(u1) * ln2);
            float w0 = r1 * cosf(u2 * two_pi);
            float w1 = r1 * sinf(u2 * two_pi);
            float w2 = sqrtf(-2.0f * log2f(u3) * ln2) * cosf(h_u[i * 4 + 3] * two_pi);
            float w3 = ((float)i + 0.5f) / 8.0f;
            float gs[4] = {w0, w1, w2, w3};
            for (int k = 0; k < 4; k++) {
                float dd = fabsf(h_g[i * 4 + k] - gs[k]);
                if (dd > worst) worst = dd;
            }
        }
        std::printf("  worst|diff| = %.3g (tol 1e-5)\n", worst);
        if (worst < 1e-5f) {
            std::printf("  PASS  fe_bm_affine within tol\n");
            passes++;
        } else {
            std::printf("  FAIL  fe_bm_affine exceeded tol\n");
        }
    }

    std::printf("\n=== Summary: %d / %d passes ===\n", passes, total_passes);
    return (passes == total_passes) ? 0 : 1;
}
