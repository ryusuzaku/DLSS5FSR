// Native C32 HWC <-> peer C32 channel basis in 8x8 shifted windows.
// peer_to_native() is checked against all 2048 block66 projection entries.
#include <hip/hip_runtime.h>

__device__ inline int peer_to_native32(int c) {
    return (c&~15)|(c&1)|((c&8)>>2)|((c&2)<<1)|((c&4)<<1);
}

__global__ void k_spatial32_peer_gather(const float* native,float* windows,
                                         int width,int height,int shift) {
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int pw=((width+px+7)/8)*8,ph=((height+py+7)/8)*8;
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=pw*ph*32)return;
    int channel=id%32,token=id/32,win=token/64,within=token%64;
    int tile_x=win%(pw/8),tile_y=win/(pw/8);
    int x=tile_x*8+within%8-px,y=tile_y*8+within/8-py;
    windows[id]=(x>=0&&x<width&&y>=0&&y<height)
        ?native[(y*width+x)*32+peer_to_native32(channel)]:0.0f;
}

__global__ void k_spatial32_peer_scatter(const float* windows,float* native,
                                          int width,int height,int shift) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=width*height*32)return;
    int channel=id%32,x=id/32%width,y=id/32/width;
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int pw=((width+px+7)/8)*8;
    int wx=x+px,wy=y+py;
    int win=(wy/8)*(pw/8)+wx/8,within=(wy%8)*8+wx%8;
    native[(y*width+x)*32+peer_to_native32(channel)]=windows[(win*64+within)*32+channel];
}
