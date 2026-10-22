#pragma once
#include <hip/hip_runtime.h>
#include <cmath>

// Backward-flow conditional Gaussian splitting, not the full GWTF algorithm.
// If d destinations select independent parent Q, let Zi be fresh independent
// standard normals and return Q/sqrt(d) + Zi - mean(Z). Each child has variance
// one, and distinct children have covariance 1/d - 1/d = 0. Distinct parent
// groups are independent. Mapping/validity may depend on guides, never noise.
// A supersampled grid improves subpixel correspondence; disjoint SxS blocks
// reduce by sum/S, preserving independent standard-normal network samples.
namespace motion_noise {
struct alignas(8) Group { unsigned long long sum[3]; unsigned count, pad; };
static_assert(sizeof(Group) == 32, "group storage size");
constexpr float quant = 1048576.0f;

__device__ inline unsigned hash(unsigned v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16);
}
__device__ inline float uniform(unsigned v) { return (float(hash(v) >> 8) + 0.5f) * 0x1p-24f; }
__device__ inline float3 gaussian(unsigned pixel, unsigned seed) {
    const unsigned h = hash(pixel ^ hash(seed + 0x632be5abu));
    const float a = sqrtf(-2.0f*logf(uniform(h ^ 0x824d13b5u)));
    const float b = sqrtf(-2.0f*logf(uniform(h ^ 0x7fe92d63u)));
    const float u = uniform(h ^ 0x945c891du)*6.28318530718f;
    const float v = uniform(h ^ 0x376b2fe9u)*6.28318530718f;
    return make_float3(a*cosf(u), a*sinf(u), b*cosf(v));
}
__device__ inline long long zi(float v) { return llrintf(v*quant); }

__global__ void initialize(float3* out, int n, unsigned seed) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = gaussian(unsigned(i), seed);
}
// map/gather has no floating-point atomics: fixed-point innovation sums make
// reset/replay deterministic. Quantization error is below 5e-7 per sample.
__global__ void route(const float2* mv, const float* depth, const float* old_depth,
                      int fw, int fh, int sw, int sh, int width, int height,
                      int scale, float2 ms, int* parents, Group* groups, unsigned seed) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= width*height) return;
    int parent = -1;
    const int x = i % width, y = i / width;
    if (x < sw*scale && y < sh*scale) {
        const int px = min(int((x+0.5f)*fw/(sw*scale)), fw-1);
        const int py = min(int((y+0.5f)*fh/(sh*scale)), fh-1);
        const float2 m = mv[py*fw+px];
        const float fx = x + m.x*ms.x*(float(sw)*scale/fw);
        const float fy = y + m.y*ms.y*(float(sh)*scale/fh);
        if (isfinite(fx) && isfinite(fy) && fx >= -0.5f && fy >= -0.5f &&
            fx < sw*scale-0.5f && fy < sh*scale-0.5f) {
            const int sx = int(floorf(fx+0.5f)), sy = int(floorf(fy+0.5f));
            const int qx = min(int((sx+0.5f)*fw/(sw*scale)), fw-1);
            const int qy = min(int((sy+0.5f)*fh/(sh*scale)), fh-1);
            const float d = depth[py*fw+px], q = old_depth[qy*fw+qx];
            if (isfinite(d) && isfinite(q) && d >= 0 && q >= 0 &&
                fabsf(d-q) <= 0.05f*fmaxf(fmaxf(d,q), 1e-6f)) parent = sy*width+sx;
        }
    }
    parents[i] = parent;
    if (parent >= 0) {
        const float3 z = gaussian(unsigned(i), seed ^ 0xa461b39du);
        atomicAdd(&groups[parent].count, 1u);
        atomicAdd(&groups[parent].sum[0], static_cast<unsigned long long>(zi(z.x)));
        atomicAdd(&groups[parent].sum[1], static_cast<unsigned long long>(zi(z.y)));
        atomicAdd(&groups[parent].sum[2], static_cast<unsigned long long>(zi(z.z)));
    }
}
__global__ void split(const float3* previous, float3* out, const int* parents,
                      const Group* groups, int n, unsigned seed) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = parents[i];
    if (p < 0) { out[i] = gaussian(unsigned(i), seed ^ 0x8ed924f3u); return; }
    const Group g = groups[p];
    if (g.count == 1) { out[i] = previous[p]; return; }
    const float3 z = gaussian(unsigned(i), seed ^ 0xa461b39du), q = previous[p];
    const float r = 1.0f/sqrtf(float(g.count)), t = 1.0f/(quant*float(g.count));
    out[i] = make_float3(q.x*r + float(zi(z.x))/quant - float(static_cast<long long>(g.sum[0]))*t,
                        q.y*r + float(zi(z.y))/quant - float(static_cast<long long>(g.sum[1]))*t,
                        q.z*r + float(zi(z.z))/quant - float(static_cast<long long>(g.sum[2]))*t);
}
__global__ void reduce(const float3* high, float* out, int w, int h, int scale) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= w*h) return;
    const int x = i % w, y = i / w, hw = w*scale;
    float3 s = make_float3(0,0,0);
    for (int dy = 0; dy < scale; ++dy) for (int dx = 0; dx < scale; ++dx) {
        const float3 v = high[(y*scale+dy)*hw+x*scale+dx];
        s.x += v.x; s.y += v.y; s.z += v.z;
    }
    out[i*3] = s.x/scale; out[i*3+1] = s.y/scale; out[i*3+2] = s.z/scale;
}
} // namespace motion_noise
