#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void k_vit_bridge_gather(const float* hwc,const int32_t* indices,
                                    float* vit,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)vit[i]=hwc[indices[i]];
}
