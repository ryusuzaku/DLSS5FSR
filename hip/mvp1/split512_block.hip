// Standalone first C512 bottleneck projection. Input is already FP8-decoded.
// Each 32-wide product rounds to half, as in native_split_reference.multiply.
#include <hip/hip_runtime.h>

__device__ inline float split512_half(float x) {
    volatile _Float16 v = (_Float16)x;
    return (float)v;
}

// F(x) in the upstream reference: saturating E4M3FN, ties to even.
__device__ inline float split512_fp8(float x) {
    float a = fabsf(x);
    if (a >= 448.0f) return copysignf(448.0f, x);
    if (a < 0.015625f) {
        float q = nearbyintf(a * 512.0f) / 512.0f;
        return copysignf(q, x);
    }
    int e;
    frexpf(a, &e);
    --e;
    float m = nearbyintf((ldexpf(a, -e) - 1.0f) * 8.0f);
    if (m == 8.0f) { ++e; m = 0.0f; }
    if (e >= 8) { e = 8; m = fminf(m, 6.0f); }
    return copysignf(ldexpf(1.0f + m / 8.0f, e), x);
}

__global__ void k_split512_pre(const float* input, const float* matrix,
                               float* output, int tokens) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tokens * 512) return;
    int token = id / 512, row = id % 512;
    float acc = 0.0f;
    for (int base = 0; base < 512; base += 32) {
        float part = 0.0f;
        for (int col = 0; col < 32; ++col)
            part += input[token * 512 + base + col] *
                    matrix[row * 512 + base + col];
        acc = split512_half(acc + part);
    }
    output[id] = acc;
}

__global__ void k_split512_quant(const float* input, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = split512_fp8(input[i]);
}

__global__ void k_split512_expand(const float* mixed, const float* weights,
                                  float* hidden, int tokens) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tokens * 2048) return;
    int token = id / 2048, group = id % 2048 / 256, row = id % 256;
    const float* x = mixed + token * 512 + group * 64;
    const float* w = weights + group * (256 * 64) + row * 64;
    float acc = 0.0f;
    for (int base = 0; base < 64; base += 32) {
        float part = 0.0f;
        for (int col = 0; col < 32; ++col)
            part += x[base + col] * w[base + col];
        acc = split512_half(acc + part);
    }
    float gate = fminf(4.0f, fmaxf(-4.0f, acc));
    float poly = split512_half(gate * split512_half(fabsf(gate) *
                 (-0.055908203125f) + 0.447265625f) + 0.89453125f);
    hidden[id] = split512_fp8(split512_half(acc * poly));
}

__global__ void k_split512_contract(const float* hidden, const float* weights,
                                    float* output, int tokens) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tokens * 512) return;
    int token = id / 512, group = id % 512 / 64, row = id % 64;
    const float* x = hidden + token * 2048 + group * 256;
    const float* w = weights + group * (64 * 256) + row * 256;
    float acc = 0.0f;
    for (int base = 0; base < 256; base += 32) {
        float part = 0.0f;
        for (int col = 0; col < 32; ++col)
            part += x[base + col] * w[base + col];
        acc = split512_half(acc + part);
    }
    output[id] = split512_fp8(acc);
}

__global__ void k_split512_ffn_projection(const float* branch,
                                          const float* original,
                                          const float* weights,
                                          const float* skip,
                                          float* raw_output,
                                          float* feature, int tokens) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tokens * 512) return;
    int token = id / 512, row = id % 512;
    float acc = split512_half(original[id] * skip[row]);
    const float* x = branch + token * 512;
    const float* w = weights + row * 512;
    for (int base = 0; base < 512; base += 32) {
        float part = 0.0f;
        for (int col = 0; col < 32; ++col)
            part += x[base + col] * w[base + col];
        acc = split512_half(acc + part);
    }
    if (raw_output) raw_output[id] = acc;
    feature[id] = split512_fp8(acc);
}

__global__ void k_split512_qkv(const float* feature, const float* weights,
                               float* output, int tokens) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= tokens * 3 * 512) return;
    int token = id / (3 * 512), matrix = id % (3 * 512) / 512;
    int row = id % 512;
    const float* x = feature + token * 512;
    const float* w = weights + (matrix * 512 + row) * 512;
    float acc = 0.0f;
    for (int base = 0; base < 512; base += 32) {
        float part = 0.0f;
        for (int col = 0; col < 32; ++col)
            part += x[base + col] * w[base + col];
        acc = split512_half(acc + part);
    }
    output[id] = acc;
}
