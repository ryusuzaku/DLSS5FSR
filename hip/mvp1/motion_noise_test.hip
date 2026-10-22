#include "motion_noise.hip"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

static int checks = 0;
static void check(bool v, const char* what) { if (!v) throw std::runtime_error(what); ++checks; printf("PASS %s\n", what); }
static void hc(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> struct Device {
    T* p; size_t n;
    explicit Device(size_t count): n(count) { hc(hipMalloc(&p,n*sizeof(T))); }
    ~Device() { hipFree(p); }
    void put(const std::vector<T>& v) { hc(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> v(n); hc(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};
static void stats(const std::vector<float3>& v, int w, const char* label) {
    double sum=0,sq=0,cx=0,cy=0,cc=0; size_t pairs=0;
    for (size_t i=0;i<v.size();++i) {
        const auto a=v[i]; sum+=a.x+a.y+a.z; sq+=a.x*a.x+a.y*a.y+a.z*a.z; cc+=a.x*a.y;
        if (i%w && i>=size_t(w)) { cx+=a.x*v[i-1].x; cy+=a.x*v[i-w].x; ++pairs; }
    }
    const double mean=sum/(v.size()*3), variance=sq/(v.size()*3)-mean*mean;
    printf("STATS %s mean=%.6f variance=%.6f corrx=%.6f corry=%.6f cross=%.6f\n",label,mean,variance,cx/pairs/variance,cy/pairs/variance,cc/v.size()/variance);
    check(fabs(mean)<0.012 && fabs(variance-1)<0.025 && fabs(cx/pairs/variance)<0.018 && fabs(cy/pairs/variance)<0.018 && fabs(cc/v.size()/variance)<0.018,label);
}
int main(int argc, char** argv) {
    try {
        const bool bench=argc>1 && std::string(argv[1])=="--bench";
        const int w=bench?1280:384,h=bench?832:256,S=argc>2?std::stoi(argv[2]):2,hw=w*S,hh=h*S,n=hw*hh;
        if(S!=1&&S!=2&&S!=4) throw std::runtime_error("scale must be1,2or4");
        Device<float3> a(n),b(n); Device<int> parents(n); Device<motion_noise::Group> groups(n);
        Device<float2> mv(w*h); Device<float> d(w*h),old_d(w*h),reduced(w*h*3);
        std::vector<float2> m(w*h,make_float2(0,0)); std::vector<float> depths(w*h,0.5f);
        mv.put(m); d.put(depths); old_d.put(depths);
        auto init=[&](unsigned seed) { hipLaunchKernelGGL(motion_noise::initialize,dim3((n+255)/256),dim3(256),0,0,a.p,n,seed); hc(hipGetLastError()); };
        auto step=[&](unsigned seed) {
            hc(hipMemset(groups.p,0,n*sizeof(motion_noise::Group)));
            hipLaunchKernelGGL(motion_noise::route,dim3((n+255)/256),dim3(256),0,0,mv.p,d.p,old_d.p,w,h,w,h,hw,hh,S,make_float2(1,1),parents.p,groups.p,seed);
            hipLaunchKernelGGL(motion_noise::split,dim3((n+255)/256),dim3(256),0,0,a.p,b.p,parents.p,groups.p,n,seed);
            hc(hipGetLastError()); std::swap(a.p,b.p);
        };
        if(bench) {
            for(auto& v:m) v=make_float2(1.25f,0.5f); mv.put(m); init(0);
            for(int k=0;k<3;++k)step(k+1);
            hipEvent_t begin,end;hc(hipEventCreate(&begin));hc(hipEventCreate(&end));hc(hipEventRecord(begin));
            for(int k=0;k<12;++k) { step(k+4); hipLaunchKernelGGL(motion_noise::reduce,dim3((w*h+255)/256),dim3(256),0,0,a.p,reduced.p,w,h,S); }
            hc(hipEventRecord(end));hc(hipEventSynchronize(end));float ms=0;hc(hipEventElapsedTime(&ms,begin,end));
            printf("BENCH transport-only %dx%d scale%d warm %.4fms allocated %.1fMiB\n",w,h,S,ms/12,double(n)*(sizeof(float3)*2+sizeof(int)+sizeof(motion_noise::Group))/(1024*1024)+double(w*h)*(3*sizeof(float)+sizeof(float2)+2*sizeof(float))/(1024*1024));
            hipEventDestroy(begin);hipEventDestroy(end);return 0;
        }
        init(0); const auto first=a.get(); stats(first,hw,"initial Gaussian statistics");
        step(1); const auto still=a.get(); check(!memcmp(first.data(),still.data(),n*sizeof(float3)),"identity carries every sample exactly");
        for (auto& v:m) v=make_float2(1,0); mv.put(m); step(2); const auto moved=a.get();
        bool exact=true; for (int y=0;y<hh;++y) for(int x=0;x<hw-S;++x) exact &= !memcmp(&moved[y*hw+x],&first[y*hw+x+S],sizeof(float3));
        check(exact,"integer translation follows corresponding surface exactly"); stats(moved,hw,"integer translation statistics");
        init(0);for(auto& v:m)v=make_float2(0.5f,0);mv.put(m);step(3);const auto half=a.get();
        exact=true;for(int y=0;y<hh;++y)for(int x=0;x<hw-1;++x)exact &= !memcmp(&half[y*hw+x],&first[y*hw+x+1],sizeof(float3));
        check(exact,"half-pixel surface translation follows finer-grid correspondence exactly");
        for (int y=0;y<h;++y) for(int x=0;x<w;++x) m[y*w+x]=make_float2(-0.5f*x,0); mv.put(m); init(0); step(3);
        const auto expanded=a.get(); stats(expanded,hw,"expansion splits preserve spatial statistics");
        init(0); step(3); const auto repeat=a.get(); check(!memcmp(expanded.data(),repeat.data(),n*sizeof(float3)),"fixed-point innovation makes split deterministic");
        hipLaunchKernelGGL(motion_noise::reduce,dim3((w*h+255)/256),dim3(256),0,0,a.p,reduced.p,w,h,S);
        const auto low=reduced.get(); std::vector<float3> low3(w*h); memcpy(low3.data(),low.data(),low.size()*sizeof(float)); stats(low3,w,"reduced network samples preserve statistics");
        for (int y=0;y<h;++y) for(int x=0;x<w;++x) m[y*w+x]=make_float2(0.5f*x,0.25f); mv.put(m); step(4); stats(a.get(),hw,"contraction and subpixel statistics");
        for (int f=5;f<37;++f) {
            for (int y=0;y<h;++y) for(int x=0;x<w;++x) m[y*w+x]=make_float2((f&1 ? -0.08f:0.08f)*(x-w/2),0.25f);
            mv.put(m); step(unsigned(f));
        }
        stats(a.get(),hw,"32-frame expansion/contraction drift");
        for(auto& v:m) v=make_float2(100000,0); mv.put(m); step(38);
        auto ps=parents.get(); bool rejected=true; for(int p:ps) rejected &= p==-1; check(rejected,"out-of-bounds motion uses fresh noise"); stats(a.get(),hw,"out-of-bounds fresh statistics");
        for(auto& v:m) v=make_float2(NAN,0); mv.put(m); step(39); ps=parents.get(); rejected=true; for(int p:ps) rejected &= p==-1; check(rejected,"non-finite motion rejected");
        for(auto& v:m) v=make_float2(0,0); mv.put(m); for(auto& v:depths)v=0.25f; d.put(depths); step(40); ps=parents.get(); rejected=true; for(int p:ps) rejected &= p==-1; check(rejected,"depth disocclusion uses fresh noise");
        init(0); const auto reset=a.get(); check(!memcmp(first.data(),reset.data(),n*sizeof(float3)),"reset restores seed-zero field");
        printf("motion_noise: %d/%d PASS\n",checks,checks); return 0;
    } catch (const std::exception& e) { fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
