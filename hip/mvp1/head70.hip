// Block 70 outer passes. Logical float32 HWC buffers, origin (0,0).
// Equations: tools/head70_reference.py; derived from Kien's MIT reference
// ref/dlss5-port/Development/native_post70_reference.py (post and aligned).
// Body execution and the original kernel's packed surface ABI are separate.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

__device__ static float head70_half(float x) {
    return __half2float(__float2half_rn(x));
}

// Write directly into [window][64][32] for an 8x8-window body.
extern "C" __global__ void k_head70_merge(
    const float* main, const float* skip, const float* sm, const float* ss,
    float* tiles, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height * 32) return;
    int c = i % 32, p = i / 32, x = p % width, y = p / width;
    int m = ((y / 2) * (width / 2) + x / 2) * 32 + c;
    int t = (((y / 8) * (width / 8) + x / 8) * 64
             + (y % 8) * 8 + x % 8) * 32 + c;
    // Preserve the reference's float32 boundaries before each half conversion.
    volatile float main_product = main[m] * sm[c];
    volatile float skip_product = skip[i] * ss[c];
    volatile float combined = head70_half(main_product) + skip_product;
    tiles[t] = head70_half(combined);
}

// Reference aligned(): 16 products on a 27-bit exponent grid, then f16.
// Use double for the reference's exact products/truncation; this is a scalar
// correctness implementation, not a claim of a fast WMMA implementation.
__device__ static float head70_aligned(const float* a, const float* b, float acc) {
    int e = -1000;
    for (int k = 0; k < 16; ++k) {
        if (a[k] != 0.0f && b[k] != 0.0f) {
            int ae, be;
            frexpf(fabsf(a[k]), &ae);
            frexpf(fabsf(b[k]), &be);
            e = max(e, ae + be);
        }
    }
    if (acc != 0.0f) {
        int ae;
        frexpf(fabsf(acc), &ae);
        e = max(e, ae + 1);
    }
    if (e == -1000) e = 0;
    double quantum = ldexp(1.0, e - 27), sum = 0.0;
    for (int k = 0; k < 16; ++k)
        sum += trunc((double(a[k]) * double(b[k])) / quantum);
    // Convert directly from double to half: a float intermediate can double-round.
    return float(static_cast<_Float16>(sum * quantum + double(acc)));
}

// Input [window][64][32] is the RAW float32 body output (not fp8).
// Coefficients are logical [3][32], extracted rows 0,2,4. Color/output HWC RGB.
extern "C" __global__ void k_head70_finish(
    const float* tiles, const float* coeff, const float* color,
    float* rgb, int width, int height, float input_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height * 3) return;
    int c = i % 3, p = i / 3, x = p % width, y = p / width;
    int t = (((y / 8) * (width / 8) + x / 8) * 64
             + (y % 8) * 8 + x % 8) * 32;
    float value = head70_aligned(tiles + t, coeff + c * 32, 0.0f);
    value = head70_aligned(tiles + t + 16, coeff + c * 32 + 16, value);
    float base = float(double(color[i]) * 0.125 - 0.0625);
    float encoded = float(double(value) * double(input_scale) + double(base));
    rgb[i] = fminf(1.0f, fmaxf(0.0f, float(double(encoded) * 8.0 + 0.5)));
}
