// Asset-free contracts for guided extra-pass repair. Optional --bench times
// this kernel alone on a warm 1280x800 field with 10% recently uncovered.
#include "uncover_extra.hip"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void hc(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct Field {
    int w, h;
    size_t n;
    std::vector<float4> map, first, out;
    std::vector<float> age, depth;
    std::vector<float2> mv;
    float4 *dm = nullptr, *df = nullptr, *dout = nullptr;
    float *da = nullptr, *dd = nullptr;
    float2* dv = nullptr;
    Field(int width, int height) : w(width), h(height), n(size_t(w)*h),
        map(n), first(n), out(n), age(n), depth(n), mv(n) {
        hc(hipMalloc(&dm,n*sizeof(float4))); hc(hipMalloc(&df,n*sizeof(float4)));
        hc(hipMalloc(&dout,n*sizeof(float4))); hc(hipMalloc(&da,n*sizeof(float)));
        hc(hipMalloc(&dd,n*sizeof(float))); hc(hipMalloc(&dv,n*sizeof(float2)));
        reset();
    }
    ~Field() {
        (void)hipFree(dm); (void)hipFree(df); (void)hipFree(dout);
        (void)hipFree(da); (void)hipFree(dd); (void)hipFree(dv);
    }
    void reset() {
        for (size_t p = 0; p < n; ++p) {
            const float x = (p%w)&1 ? 1.0f/64 : -1.0f/64;
            first[p] = make_float4(x,0.0f,-x,0.25f+float(p%w)/(4*w));
            map[p] = make_float4(x+3.0f/64,-2.0f/64,-x+4.0f/64,first[p].w);
            age[p] = 5.0f; depth[p] = 0.5f; mv[p] = make_float2(0,0);
        }
    }
    void upload() {
        hc(hipMemcpy(dm,map.data(),n*sizeof(float4),hipMemcpyHostToDevice));
        hc(hipMemcpy(df,first.data(),n*sizeof(float4),hipMemcpyHostToDevice));
        hc(hipMemcpy(da,age.data(),n*sizeof(float),hipMemcpyHostToDevice));
        hc(hipMemcpy(dd,depth.data(),n*sizeof(float),hipMemcpyHostToDevice));
        hc(hipMemcpy(dv,mv.data(),n*sizeof(float2),hipMemcpyHostToDevice));
    }
    void launch() {
        hipLaunchKernelGGL(k_uncover_extra,dim3((n+255)/256),dim3(256),0,0,
                           dm,df,da,dv,dd,dout,w,h,make_float2(float(w),float(h)),10,5.0f);
        hc(hipGetLastError());
    }
    void download() { hc(hipMemcpy(out.data(),dout,n*sizeof(float4),hipMemcpyDeviceToHost)); }
};

int main(int argc, char** argv) {
    const bool bench = argc==2 && !strcmp(argv[1],"--bench");
    if (argc>1 && !bench) { fputs("usage: uncover_extra_test [--bench]\n",stderr); return 2; }
    try {
        Field f(64,64);
        const size_t c = size_t(f.h/2)*f.w+f.w/2;
        auto unchanged = [&] { return !memcmp(f.map.data(),f.out.data(),f.n*sizeof(float4)); };
        auto run = [&](const char* name, auto verify) {
            f.upload(); f.launch(); f.download();
            bool ok = verify();
            for (size_t p=0;p<f.n;++p) ok = ok && !memcmp(&f.map[p].w,&f.out[p].w,sizeof(float));
            printf("%s: %s\n",name,ok ? "PASS" : "FAIL");
            if (!ok) throw std::runtime_error(name);
        };
        // A textured first-pass field plus a uniform extra lighting correction
        // must retain both, even where the output needs repair.
        for (int y=30;y<=34;++y) for(int x=30;x<=34;++x) f.age[size_t(y)*f.w+x]=0;
        run("uniform extra preserves first-pass texture",[&] {
            for(size_t p=0;p<f.n;++p) {
                const float4 a=f.map[p], b=f.out[p];
                if(fabsf(a.x-b.x)>1e-6f || fabsf(a.y-b.y)>1e-6f || fabsf(a.z-b.z)>1e-6f) return false;
            }
            return true;
        });
        f.reset(); f.age[c]=0;
        f.map[c].x+=0.25f; f.map[c].y-=0.25f; f.map[c].z+=0.25f;
        run("supported outlier corrected with channel cap",[&] {
            if(fabsf(f.out[c].x-(f.map[c].x-0.1f))>1e-6f ||
               fabsf(f.out[c].y-(f.map[c].y+0.1f))>1e-6f ||
               fabsf(f.out[c].z-(f.map[c].z-0.1f))>1e-6f) return false;
            for(size_t p=0;p<f.n;++p) if(p!=c && memcmp(&f.map[p],&f.out[p],sizeof(float4))) return false;
            return true;
        });
        for(auto& a:f.age) a=0;
        run("no settled support retains the extra edit",unchanged);
        f.reset(); f.age[c]=0; f.depth[c]=0.1f;
        run("depth boundary blocks borrowed correction",unchanged);
        f.reset(); f.age[c]=0;
        for(auto& v:f.mv) v=make_float2(8.0f/f.w,0);
        f.mv[c]=make_float2(0,0);
        run("motion boundary blocks borrowed correction",unchanged);
        f.reset(); f.age[c]=0;
        for(auto& v:f.mv) v=make_float2(16.0f/f.w,0);
        run("fast surface motion retains the current edit",unchanged);
        if(bench) {
            Field b(1280,800);
            for(size_t p=0;p<b.n;++p) b.age[p]=(p/b.w)%80<8 ? 0.0f : 5.0f;
            b.upload();
            for(int i=0;i<10;++i) b.launch();
            hipEvent_t start,stop; hc(hipEventCreate(&start)); hc(hipEventCreate(&stop));
            hc(hipEventRecord(start));
            for(int i=0;i<100;++i) b.launch();
            hc(hipEventRecord(stop)); hc(hipEventSynchronize(stop));
            float ms=0; hc(hipEventElapsedTime(&ms,start,stop));
            hc(hipEventDestroy(start)); hc(hipEventDestroy(stop));
            printf("warm kernel only, 1280x800, 10%% uncovered, radius 10: %.4f ms\n",ms/100);
        }
    } catch(const std::exception& e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
