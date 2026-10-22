#pragma once
#include <hip/hip_runtime.h>

// Repair the extra pass's residual rather than fading it to zero or blurring
// the first pass's picture. No settled support leaves the current map alone.
__global__ void k_uncover_extra(const float4* map, const float4* first, const float* age,
                                const float2* mv, const float* depth, float4* out,
                                int fw, int fh, float2 ms, int radius, float limit) {
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= fw*fh) return;
    float4 o = map[p];
    float m = fminf(fmaxf(1.0f - age[p]/limit, 0.0f), 1.0f);
    if (m > 0.0f) {
        const int x = p % fw, y = p / fw;
        float2 v = mv[p];
        v.x *= ms.x; v.y *= ms.y;
        if (!isfinite(v.x) || !isfinite(v.y)) v = make_float2(0.0f, 0.0f);
        const float vs = hypotf(v.x, v.y), dp = depth[p];
        m *= fminf(fmaxf((12.0f - vs)/8.0f, 0.0f), 1.0f);
        if (m <= 0.0f) { out[p] = o; return; }
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        float sw = 0.0f, full = 0.0f;
        for (int i = 0; i < 48; ++i) {
            const float f = (i + 0.5f)/48.0f, r = radius*sqrtf(f), a = 2.3999632f*i;
            const float g = expf(-2.0f*f);
            full += g;
            const int qx = x + int(lroundf(r*cosf(a))), qy = y + int(lroundf(r*sinf(a)));
            if (qx < 0 || qx >= fw || qy < 0 || qy >= fh) continue;
            const size_t q = size_t(qy)*fw + qx;
            if (age[q] < limit) continue;
            float2 u = mv[q];
            u.x *= ms.x; u.y *= ms.y;
            if (!isfinite(u.x) || !isfinite(u.y)) continue;
            if (hypotf(u.x-v.x, u.y-v.y) > fmaxf(1.0f, 0.25f*fmaxf(vs,hypotf(u.x,u.y)))) continue;
            if (fabsf(depth[q]-dp) > 0.06f*dp) continue;
            const float4 n = map[q], b = first[q];
            sum.x += g*(n.x-b.x); sum.y += g*(n.y-b.y); sum.z += g*(n.z-b.z); sw += g;
        }
        const float k = m*fminf(sw/(0.25f*full),1.0f);
        if (k > 0.0f) {
            const float4 b = first[p];
            const float cap = 0.1f;
            o.x += fminf(fmaxf((b.x+sum.x/sw-o.x)*k,-cap),cap);
            o.y += fminf(fmaxf((b.y+sum.y/sw-o.y)*k,-cap),cap);
            o.z += fminf(fmaxf((b.z+sum.z/sw-o.z)*k,-cap),cap);
        }
    }
    out[p] = o;
}
