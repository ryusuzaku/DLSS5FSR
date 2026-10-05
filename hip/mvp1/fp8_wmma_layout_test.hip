// Checks the gfx12 wave32 FP8 WMMA operand layout the C32 FP8 kernel assumes:
//   A (16x16, row-major M x K): lane l holds row l%16, K = (l/16)*8 .. +8 (bytes 0..7 of an int2)
//   B (16x16, K x N):           lane l holds column l%16, K = (l/16)*8 .. +8
//   D (16x16 f32):              lane l holds column l%16, rows (l/16)*8 .. +8
// against a CPU product of the same E4M3 values. Prints PASS/FAIL.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

typedef float float8v __attribute__((ext_vector_type(8)));
typedef int int2v __attribute__((ext_vector_type(2)));

__device__ inline unsigned pack4(float a, float b, float c, float d) {
    unsigned lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
}

__global__ void k_test(const float* A, const float* B, float* D) {
    const int l = threadIdx.x, r = l % 16, k0 = (l / 16) * 8;
    int2v a, b;
    a.x = (int)pack4(A[r*16+k0+0], A[r*16+k0+1], A[r*16+k0+2], A[r*16+k0+3]);
    a.y = (int)pack4(A[r*16+k0+4], A[r*16+k0+5], A[r*16+k0+6], A[r*16+k0+7]);
    b.x = (int)pack4(B[(k0+0)*16+r], B[(k0+1)*16+r], B[(k0+2)*16+r], B[(k0+3)*16+r]);
    b.y = (int)pack4(B[(k0+4)*16+r], B[(k0+5)*16+r], B[(k0+6)*16+r], B[(k0+7)*16+r]);
    float8v c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    for (int e = 0; e < 8; ++e) D[(k0+e)*16 + r] = c[e];
}

__global__ void k_round(const float* in, float* out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = __builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_pk_fp8_f32(in[i], 0.0f, 0, false), 0);
}

int main() {
    float A[256], B[256], Ar[256], Br[256], D[256];
    for (int i = 0; i < 256; ++i) { A[i] = float((i * 37) % 29 - 14) / 8.0f; B[i] = float((i * 53) % 23 - 11) / 4.0f; }
    float *dA, *dB, *dD, *dR;
    (void)hipMalloc(&dA, 1024); (void)hipMalloc(&dB, 1024); (void)hipMalloc(&dD, 1024); (void)hipMalloc(&dR, 1024);
    (void)hipMemcpy(dA, A, 1024, hipMemcpyHostToDevice); (void)hipMemcpy(dB, B, 1024, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(k_round, dim3(1), dim3(256), 0, 0, dA, dR, 256); (void)hipMemcpy(Ar, dR, 1024, hipMemcpyDeviceToHost);
    hipLaunchKernelGGL(k_round, dim3(1), dim3(256), 0, 0, dB, dR, 256); (void)hipMemcpy(Br, dR, 1024, hipMemcpyDeviceToHost);
    hipLaunchKernelGGL(k_test, dim3(1), dim3(32), 0, 0, dA, dB, dD);
    (void)hipMemcpy(D, dD, 1024, hipMemcpyDeviceToHost);
    double worst = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            double s = 0;
            for (int k = 0; k < 16; ++k) s += double(Ar[m*16+k]) * Br[k*16+n];
            worst = fmax(worst, fabs(s - D[m*16+n]));
        }
    printf("fp8 wmma layout: max |err| %g -> %s\n", worst, worst < 1e-3 ? "PASS" : "FAIL");
    return worst < 1e-3 ? 0 : 1;
}
