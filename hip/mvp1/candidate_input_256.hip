// Candidate 256x256 input from the already-staged, sRGB-encoded NGX proxy.
// This matches tools/prepare_candidate_input.py's diagnostic crop/color path.
// It does not claim the original NVIDIA frontend's geometry or color contract.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float candidate_proxy_channel(
        const unsigned char* src, int pitch, int bpp, int x, int y, int c) {
    const unsigned char* pixel = src + (size_t)y * pitch + (size_t)x * bpp;
    if (bpp == 4) return (float)pixel[c] / 255.0f;
    const uint16_t bits = (uint16_t)pixel[2*c] | ((uint16_t)pixel[2*c+1] << 8);
    return __half2float(__ushort_as_half(bits));
}

extern "C" __global__ void k_candidate_input_256(const unsigned char* src, float* output,
                                       int width, int height, int pitch, int bpp) {
    const int pixel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (pixel >= 256*256) return;
    const int ox = pixel & 255;
    const int oy = pixel >> 8;
    const int side = width < height ? width : height;
    const int left = (width-side)/2;
    const int top = (height-side)/2;
    const float step = (float)side / 256.0f;
    float fx = ((float)ox + 0.5f)*step - 0.5f;
    float fy = ((float)oy + 0.5f)*step - 0.5f;
    fx = fminf(fmaxf(fx, 0.0f), (float)(side-1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(side-1));
    const int ix = (int)floorf(fx);
    const int iy = (int)floorf(fy);
    const int x0 = left + ix;
    const int y0 = top + iy;
    const int x1 = left + (ix+1 < side ? ix+1 : side-1);
    const int y1 = top + (iy+1 < side ? iy+1 : side-1);
    const float tx = fx - (float)ix;
    const float ty = fy - (float)iy;
    for (int c=0; c<3; ++c) {
        const float a = candidate_proxy_channel(src,pitch,bpp,x0,y0,c);
        const float b = candidate_proxy_channel(src,pitch,bpp,x1,y0,c);
        const float d = candidate_proxy_channel(src,pitch,bpp,x0,y1,c);
        const float e = candidate_proxy_channel(src,pitch,bpp,x1,y1,c);
        const float upper = a*(1.0f-tx) + b*tx;
        const float lower = d*(1.0f-tx) + e*tx;
        float srgb = upper*(1.0f-ty) + lower*ty;
        srgb = fminf(fmaxf(srgb,0.0f),1.0f);
        output[pixel*3+c] = srgb <= 0.04045f ? srgb/12.92f :
            powf((srgb+0.055f)/1.055f,2.4f);
    }
}
