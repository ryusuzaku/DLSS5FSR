#include "stabilize_guided.hip"
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cstring>
static void hc(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> struct D { T* p; explicit D(size_t n){hc(hipMalloc(&p,n*sizeof(T)));} ~D(){hipFree(p);} void put(const std::vector<T>& v){hc(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice));} };
static int checks=0;
static void check(bool v,const char* message) { if(!v)throw std::runtime_error(message); printf("PASS %s\n",message); ++checks; }
int main(){
    try {
        constexpr int w=16,h=16,n=w*h,at=8*w+8;
        D<float4> map(n),low(n),old(n),next(n),out(n),full(n),nextfull(n),g(n),og(n);
        D<float2> mv(n); D<float> depth(n),od(n);
        std::vector<float4> a(n,make_float4(0.01f,0.02f,0.03f,0.42f)),b=a,hist(n,make_float4(0.03f,0.04f,0.05f,0));
        std::vector<float4> guide(n,make_float4(-2,-2,-2,0)),oldguide=guide;
        std::vector<float> depths(n,0.5f),olddepth=depths; std::vector<float2> motion(n,make_float2(0,0));
        b[at].x=0.0104f; b[at-1].x=-0.02f; b[at+1].x=0.05f;
        // Whole edit equals low band at the test center for direct expectations.
        a[at]=b[at];
        auto run=[&](int valid,float still=0){
            map.put(a);low.put(b);old.put(hist);full.put(hist);g.put(guide);og.put(oldguide);depth.put(depths);od.put(olddepth);mv.put(motion);
            hipLaunchKernelGGL(stabilize_guided::apply,dim3(1),dim3(256),0,0,map.p,low.p,old.p,next.p,out.p,mv.p,depth.p,od.p,w,h,make_float2(1,1),0.66f,0.04f,valid,full.p,nextfull.p,still,0.02f,g.p,og.p);
            hc(hipGetLastError());std::vector<float4> r(n);hc(hipMemcpy(r.data(),out.p,n*sizeof(float4),hipMemcpyDeviceToHost));return r;
        };
        auto r=run(0);check(!memcmp(r.data(),a.data(),sizeof(float4)*n),"invalid/reset history passes current map exactly");
        r=run(1);check(r[at].x>b[at].x+0.005f&&r[at].x<hist[at].x,"stable appearance permits bounded accumulation");
        check(r[at].w==a[at].w,"input luminance guide remains unchanged");
        oldguide.assign(n,make_float4(-2.5f,-2.5f,-2.5f,0));r=run(1,0.7f);check(r[at].x==a[at].x&&r[at].y==a[at].y,"real lighting change rejects low and full-band history");
        oldguide=guide;oldguide.assign(n,make_float4(-2,-2.5f,-2,0));r=run(1);check(r[at].x==a[at].x,"chroma-only input change rejects history");
        oldguide=guide;oldguide.assign(n,make_float4(-2.4f,-2.4f,-2.4f,0));guide[at-1]=make_float4(-2.5f,-2.5f,-2.5f,0);r=run(1);check(r[at].x>b[at].x+0.005f,"subpixel appearance inside current range remains eligible");
        depths[at-1]=0.8f;r=run(1);check(r[at].x<a[at].x+0.002f,"different-depth neighbor cannot widen appearance range");
        depths.assign(n,0.5f);guide.assign(n,make_float4(-2,-2,-2,0));oldguide=guide;olddepth.assign(n,0.8f);r=run(1);check(r[at].x==a[at].x,"disocclusion rejects old surface");
        olddepth=depths;motion[at]=make_float2(NAN,0);r=run(1);check(r[at].x==a[at].x,"non-finite motion cannot reuse history");
        motion[at]=make_float2(1000,0);r=run(1);check(r[at].x==a[at].x,"offscreen history rejected");
        motion[at]=make_float2(0.25f,0);olddepth[at+1]=0.8f;r=run(1);check(r[at].x>a[at].x&&r[at].x<0.0235f,"individual bilinear depth rejection reduces confidence");
        motion[at]=make_float2(0,0);olddepth=depths;guide[at]=make_float4(NAN,-2,-2,0);r=run(1);check(r[at].x==a[at].x,"non-finite current appearance cannot reuse history");
        guide[at]=make_float4(-2,-2,-2,0);hist.assign(n,make_float4(1,1,1,0));b.assign(n,make_float4(0,0,0,0));a=b;r=run(1);check(fabsf(r[at].x)<0.001f,"flat edit neighborhood bounds stale history without broad blur");
        b[at+1].x=0.1f;hist.assign(n,make_float4(0.1f,0,0,0));r=run(1,0.7f);
        check(fabsf(r[at].x-0.0124f)<1e-6f,"full-band change cap wins over a disjoint neighborhood range");
        printf("stabilize_guided: %d/%d PASS\n",checks,checks);return 0;
    }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
