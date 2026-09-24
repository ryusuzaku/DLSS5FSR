// Diagnostic: merge -> EXISTING SHIPPED C32 BODY -> finish, on head weights.
// Not the upstream head function; see tools/head70_shipped_body.py.
#include "head70.hip"
#include "head70_test_common.h"
#include "swin_1h_chain_c32.hip"
#include <cstdint>
#include <cstring>

static std::vector<unsigned char> body_bytes(const std::string& dir) {
    FILE* f = fopen((dir + "/body.bin").c_str(), "rb");
    if (!f) { fprintf(stderr, "missing body.bin\n"); exit(1); }
    std::vector<unsigned char> v(20672);
    bool ok = fread(v.data(), 1, v.size(), f) == v.size() && fgetc(f) == EOF;
    fclose(f);
    if (!ok) { fprintf(stderr, "bad body.bin size\n"); exit(1); }
    return v;
}

static uint32_t body_fnv(const std::vector<unsigned char>& v) {
    uint32_t h = 2166136261u;
    for (auto b : v) h = (h ^ b) * 16777619u;
    return h;
}

// Independent C++ rewrite, same formula as TinRewriteStageFFN at C32.
static void body_rewrite(std::vector<unsigned char>& v) {
    const auto raw = v;
    for (int region = 0; region < 2; ++region) {
        int K = region ? 128 : 32, N = region ? 32 : 128, base = region * 4096;
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
            int lane = 4 * (n % 8) + (k % 16) / 4;
            int e = 8 * ((n % 16) / 8) + 4 * ((k % 32) / 16) + k % 4;
            int sub = (k / 32) * (N / 16) + n / 16;
            v[base + n * K + k] = raw[base + sub * 512 + lane * 16 + e];
        }
    }
}

template<class T> static T* alloc(size_t n) {
    T* p; HIP_CHECK(hipMalloc(&p, n * sizeof(T)));
    // NaNs in float buffers expose any unwritten output; inputs are overwritten.
    HIP_CHECK(hipMemset(p, 0xff, n * sizeof(T))); return p;
}

static bool stage(const std::string& dir, const char* name, float* p, size_t n, size_t win,
                  double tolerance) {
    auto all = read(dir, name, n * win);
    std::vector<float> got(all.size());
    HIP_CHECK(hipMemcpy(got.data(), p, got.size() * 4, hipMemcpyDeviceToHost));
    size_t bad = 0, nf = 0; double me = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(all[i])) ++nf;
        double e = std::fabs(double(got[i]) - all[i]);
        me = std::fmax(me, e); if (e > tolerance) ++bad;
    }
    printf("%s: %s (%zu values, maxerr %.9g, tol %.9g, %zu outside, %zu nonfinite)\n",
           name, !bad && !nf ? "PASS" : "FAIL", got.size(), me, tolerance, bad, nf);
    return !bad && !nf;
}

#define LAUNCH(kernel, n, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((n)+255)/256), dim3(256), 0, 0, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); } while (0)

int main(int argc, char** argv) {
    if (argc != 6) { fprintf(stderr, "usage: head70_body_test fixture width height scale expected-fnv\n"); return 2; }
    std::string dir = argv[1]; int w = atoi(argv[2]), h = atoi(argv[3]);
    float scale = strtof(argv[4], nullptr);
    if (w <= 0 || h <= 0 || w % 8 || h % 8 || w > 4096 || h > 4096 || !std::isfinite(scale)) return 2;
    auto raw = body_bytes(dir); uint32_t before = body_fnv(raw); body_rewrite(raw);
    uint32_t after = body_fnv(raw), expected = uint32_t(strtoul(argv[5], nullptr, 16));
    printf("shipped-mode2 body FNV %08X -> %08X (expected %08X)\n", before, after, expected);
    if (after != expected) return 1;
    auto dw = alloc<unsigned char>(raw.size());
    HIP_CHECK(hipMemcpy(dw, raw.data(), raw.size(), hipMemcpyHostToDevice));
    std::vector<float> bias(4096), gate(32);
    for (size_t i = 0; i < bias.size(); ++i) { unsigned short b; memcpy(&b, raw.data()+11360+i*2,2); bias[i]=__half2float(__ushort_as_half(b)); }
    for (size_t i = 0; i < gate.size(); ++i) { unsigned short b; memcpy(&b, raw.data()+20592+i*2,2); gate[i]=__half2float(__ushort_as_half(b)); }
    float temp; memcpy(&temp,raw.data()+19552,4);
    auto db = upload(bias), dg = upload(gate), dt = upload(std::vector<float>{temp});
    size_t windows = size_t(w)*h/64, n = windows*2048;
    auto dm = upload(read(dir,"main",n/4)), ds = upload(read(dir,"skip",n));
    auto sm = upload(read(dir,"sm",32)), ss = upload(read(dir,"ss",32));
    auto cf = upload(read(dir,"coeff",96)), color = upload(read(dir,"color",size_t(w)*h*3));
    auto merged = alloc<float>(n), out = alloc<float>(n), rgb = alloc<float>(size_t(w)*h*3);
    auto yf = alloc<float>(n), qkv = alloc<float>(windows*6144), scores = alloc<float>(windows*4096);
    auto prob = alloc<float>(windows*4096), ctx = alloc<float>(n);
    auto xb = alloc<unsigned char>(2048), ye = alloc<unsigned char>(2048);
    auto qe = alloc<unsigned char>(2048), ke = alloc<unsigned char>(2048), ve = alloc<unsigned char>(2048);
    auto pq = alloc<unsigned char>(4096), oc = alloc<unsigned char>(2048);
    auto eb = alloc<unsigned short>(4096); auto es = alloc<float>(64);
    LAUNCH(k_head70_merge,n,dm,ds,sm,ss,merged,w,h);
    bool ok = compare("merge/window",merged,read(dir,"merged",n));
    for (size_t i=0;i<windows;++i) {
        auto f=yf+i*2048, q=qkv+i*6144, s=scores+i*4096, p=prob+i*4096, c=ctx+i*2048;
        LAUNCH(k_quant_e4m3,2048,merged+i*2048,xb,2048);
        LAUNCH(k_c64ffn2c,2048,xb,dw,dw+4096,reinterpret_cast<unsigned short*>(dw+8208),f,1);
        LAUNCH(k_quant_e4m3,2048,f,ye,2048);
        LAUNCH(k_c64qkv,6144,ye,dw+8288,q);
        LAUNCH(k_c64qkv_split,6144,q,qe,ke,ve);
        LAUNCH(k_c64scores,4096,qe,ke,db,dt,s);
        LAUNCH(k_smexp,4096,s,eb); LAUNCH(k_smsum,64,eb,es); LAUNCH(k_smnorm,4096,eb,es,p);
        LAUNCH(k_quant_e4m3,4096,p,pq,4096); LAUNCH(k_c64ctx,2048,pq,ve,c);
        LAUNCH(k_c64cat_q,2048,c,oc);
        LAUNCH(k_c64proj,2048,oc,dw+19568,ye,dg,out+i*2048,1);
    }
    // Predeclared: existing full-block tolerance 0.1 for body stage diagnostics;
    // RGB limit 0.001. Outer-pass unit tests retain exact equality separately.
    ok = stage(dir,"ffn",yf,2048,windows,.1) && ok;
    ok = stage(dir,"qkv",qkv,6144,windows,.1) && ok;
    ok = stage(dir,"scores",scores,4096,windows,.1) && ok;
    ok = stage(dir,"prob",prob,4096,windows,.1) && ok;
    ok = stage(dir,"context",ctx,2048,windows,.1) && ok;
    ok = stage(dir,"body",out,2048,windows,.1) && ok;
    LAUNCH(k_head70_finish,size_t(w)*h*3,out,cf,color,rgb,w,h,scale);
    ok = stage(dir,"rgb",rgb,size_t(w)*h*3,1,.001) && ok;
    for (void* p : std::vector<void*>{dw,db,dg,dt,dm,ds,sm,ss,cf,color,merged,out,rgb,yf,qkv,scores,prob,ctx,xb,ye,qe,ke,ve,pq,oc,eb,es}) HIP_CHECK(hipFree(p));
    return ok ? 0 : 1;
}
