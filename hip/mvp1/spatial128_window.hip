// Logical C128 HWC <-> zero-padded 8x8 window-major storage.
// The upstream native_upsample48_reference.upsample uses this spatial geometry for block56.
#include <hip/hip_runtime.h>

__global__ void k_spatial128_gather(const float* input,float* windows,
                                    int width,int height,int shift) {
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int padded_width=((width+px+7)/8)*8;
    int padded_height=((height+py+7)/8)*8;
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=padded_width*padded_height*128)return;
    int channel=id%128,token=id/128;
    int win=token/64,within=token%64;
    int tile_x=win%(padded_width/8),tile_y=win/(padded_width/8);
    int x=tile_x*8+within%8-px,y=tile_y*8+within/8-py;
    windows[id]=(x>=0&&x<width&&y>=0&&y<height)
        ?input[(y*width+x)*128+channel]:0.0f;
}

__global__ void k_spatial128_scatter(const float* windows,float* output,
                                     int width,int height,int shift) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=width*height*128)return;
    int channel=id%128,x=id/128%width,y=id/128/width;
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int padded_width=((width+px+7)/8)*8;
    int wx=x+px,wy=y+py;
    int win=(wy/8)*(padded_width/8)+wx/8;
    int within=(wy%8)*8+wx%8;
    output[id]=windows[(win*64+within)*128+channel];
}
