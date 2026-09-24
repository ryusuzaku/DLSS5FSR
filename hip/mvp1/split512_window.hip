#include <hip/hip_runtime.h>

// Logical HWC -> window-major [window][64][512], matching attention_window.
__global__ void k_split512_window_gather(const float* feature,float* windows,
                                         int width,int height,int shift) {
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int padded_width=((width+px+7)/8)*8;
    int padded_height=((height+py+7)/8)*8;
    int n=padded_width*padded_height*512;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int c=i%512,token=i/512;
    int win=token/64,within=token%64;
    int tile_x=win%(padded_width/8),tile_y=win/(padded_width/8);
    int sx=tile_x*8+within%8-px,sy=tile_y*8+within/8-py;
    bool valid_x=width==4 || (sx>=0 && sx<width);
    bool valid_y=height==4 || (sy>=0 && sy<height);
    int x=width==4?((sx%4+4)%4):max(0,min(sx,width-1));
    int y=height==4?((sy%4+4)%4):max(0,min(sy,height-1));
    windows[i]=(valid_x&&valid_y)?feature[(y*width+x)*512+c]:0.0f;
}

// Window-major attention result -> cropped logical HWC.
__global__ void k_split512_window_scatter(const float* windows,float* feature,
                                          int width,int height,int shift) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=width*height*512)return;
    int c=i%512,x=i/512%width,y=i/512/width;
    int px=(shift&1)?4:0,py=(shift&2)?4:0;
    int padded_width=((width+px+7)/8)*8;
    int wx=x+px,wy=y+py;
    int win=(wy/8)*(padded_width/8)+wx/8;
    int within=(wy%8)*8+wx%8;
    feature[i]=windows[(win*64+within)*512+c];
}
