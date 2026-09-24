#include "head70_test_common.h"
#include "split512_attention.hip"
#include "split512_window.hip"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 6) {
        fprintf(stderr, "usage: split512_block_test <fixture-dir> <tokens> [width height shift]\n");
        return 2;
    }
    int tokens = atoi(argv[2]);
    bool spatial = argc == 6;
    int width = spatial ? atoi(argv[3]) : 0;
    int height = spatial ? atoi(argv[4]) : 0;
    int shift = spatial ? atoi(argv[5]) : 0;
    if (tokens <= 0 || tokens > 4096) return 2;
    if (spatial && (width < 4 || height < 4 || width % 4 || height % 4 ||
                    width * height != tokens || shift < 0 || shift > 3)) return 2;
    if (!spatial && tokens % 64) return 2;
    int px = spatial && (shift & 1) ? 4 : 0;
    int py = spatial && (shift & 2) ? 4 : 0;
    int window_tokens = spatial ? ((width + px + 7) / 8) * ((height + py + 7) / 8) * 64 : tokens;
    int windows = window_tokens / 64;
    std::string dir = argv[1];
    auto x = read(dir, "input", size_t(tokens) * 512);
    auto w = read(dir, "matrix", 512 * 512);
    auto want = read(dir, "expected", size_t(tokens) * 512);
    auto ew = read(dir, "expand", 8 * 256 * 64);
    auto cw = read(dir, "contract", 8 * 64 * 256);
    auto branch = read(dir, "branch", size_t(tokens) * 512);
    auto pw = read(dir, "ffn_projection", 512 * 512);
    auto skip = read(dir, "ffn_skip", 512);
    auto feature = read(dir, "feature", size_t(tokens) * 512);
    auto feature_window = spatial ? read(dir, "feature_window", size_t(window_tokens) * 512)
                                  : std::vector<float>();
    auto qkv_weights = read(dir, "qkv_weights", 3 * 512 * 512);
    auto qkv = read(dir, "qkv", size_t(window_tokens) * 3 * 512);
    auto scales = read(dir, "scales", 16);
    auto bias = read(dir, "bias", 16 * 4096);
    auto normalized = read(dir, "normalized", size_t(window_tokens) * 3 * 512);
    auto scores = read(dir, "scores", size_t(windows) * 16 * 4096);
    auto exponents = read(dir, "exponents", size_t(windows) * 16 * 4096);
    auto probabilities = read(dir, "probabilities", size_t(windows) * 16 * 4096);
    auto context = read(dir, "context", size_t(window_tokens) * 512);
    auto context_hwc = spatial ? read(dir, "context_hwc", size_t(tokens) * 512)
                               : std::vector<float>();
    auto final_weights = read(dir, "final_weights", 512 * 512);
    auto final_skip = read(dir, "final_skip", 512);
    auto final_raw = read(dir, "final_raw", size_t(tokens) * 512);
    auto final = read(dir, "final", size_t(tokens) * 512);
    float *dx = upload(x), *dw = upload(w), *de = upload(ew), *dc = upload(cw);
    float *dp = upload(pw), *ds = upload(skip), *dq = upload(qkv_weights);
    float *dsc = upload(scales), *dbias = upload(bias);
    float *dfw = upload(final_weights), *dfs = upload(final_skip);
    float *dy, *dm, *dh, *db, *df, *dwindow = nullptr, *doq, *dn, *dscores, *dex, *dprob, *dcontext, *dcrop = nullptr, *draw, *dfinal;
    HIP_CHECK(hipMalloc(&dy, want.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dm, want.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dh, size_t(tokens) * 2048 * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, branch.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&df, feature.size() * sizeof(float)));
    if (spatial) HIP_CHECK(hipMalloc(&dwindow, size_t(window_tokens) * 512 * sizeof(float)));
    HIP_CHECK(hipMalloc(&doq, qkv.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dn, normalized.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dscores, scores.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dex, exponents.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dprob, probabilities.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dcontext, context.size() * sizeof(float)));
    if (spatial) HIP_CHECK(hipMalloc(&dcrop, context_hwc.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dfinal, final.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&draw, final_raw.size() * sizeof(float)));
    hipLaunchKernelGGL(k_split512_pre, dim3((want.size() + 255) / 256),
                       dim3(256), 0, 0, dx, dw, dy, tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    bool ok = compare("split512 pre", dy, want);
    hipLaunchKernelGGL(k_split512_quant, dim3((want.size() + 255) / 256),
                       dim3(256), 0, 0, dy, dm, int(want.size()));
    hipLaunchKernelGGL(k_split512_expand, dim3((size_t(tokens) * 2048 + 255) / 256),
                       dim3(256), 0, 0, dm, de, dh, tokens);
    hipLaunchKernelGGL(k_split512_contract, dim3((branch.size() + 255) / 256),
                       dim3(256), 0, 0, dh, dc, db, tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    ok = compare("split512 branch", db, branch) && ok;
    hipLaunchKernelGGL(k_split512_ffn_projection,
                       dim3((feature.size() + 255) / 256), dim3(256), 0, 0,
                       db, dx, dp, ds, nullptr, df, tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    ok = compare("split512 feature", df, feature) && ok;
    if (spatial) {
        hipLaunchKernelGGL(k_split512_window_gather,
                           dim3((size_t(window_tokens) * 512 + 255) / 256),dim3(256),0,0,
                           df,dwindow,width,height,shift);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
        ok = compare("split512 feature window",dwindow,feature_window) && ok;
    }
    hipLaunchKernelGGL(k_split512_qkv, dim3((qkv.size() + 255) / 256),
                       dim3(256), 0, 0, spatial ? dwindow : df, dq, doq, window_tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    ok = compare("split512 qkv", doq, qkv) && ok;
    hipLaunchKernelGGL(k_split512_qknorm,dim3((normalized.size()+255)/256),dim3(256),0,0,
                       doq,dsc,dn,windows);
    hipLaunchKernelGGL(k_split512_scores,dim3((scores.size()+255)/256),dim3(256),0,0,
                       dn,dbias,dscores,windows);
    hipLaunchKernelGGL(k_split512_exp,dim3((exponents.size()+255)/256),dim3(256),0,0,
                       dscores,dex,int(exponents.size()));
    hipLaunchKernelGGL(k_split512_prob,dim3((probabilities.size()+255)/256),dim3(256),0,0,
                       dex,dprob,int(probabilities.size()));
    hipLaunchKernelGGL(k_split512_context,dim3((context.size()+255)/256),dim3(256),0,0,
                       dprob,dn,dcontext,windows);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    ok = compare("split512 normalized",dn,normalized) && ok;
    ok = compare("split512 scores",dscores,scores) && ok;
    ok = compare("split512 exponents",dex,exponents) && ok;
    ok = compare("split512 probabilities",dprob,probabilities) && ok;
    ok = compare("split512 context",dcontext,context) && ok;
    if (spatial) {
        hipLaunchKernelGGL(k_split512_window_scatter,
                           dim3((context_hwc.size()+255)/256),dim3(256),0,0,
                           dcontext,dcrop,width,height,shift);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
        ok = compare("split512 context HWC",dcrop,context_hwc) && ok;
    }
    hipLaunchKernelGGL(k_split512_ffn_projection,
                       dim3((final.size()+255)/256),dim3(256),0,0,
                       spatial ? dcrop : dcontext,df,dfw,dfs,draw,dfinal,tokens);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    ok = compare("split512 final raw",draw,final_raw) && ok;
    ok = compare("split512 final",dfinal,final) && ok;
    if (spatial && ok) {
        std::vector<float> device_final(final.size());
        HIP_CHECK(hipMemcpy(device_final.data(),dfinal,final.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::vector<float> device_raw(final_raw.size());
        HIP_CHECK(hipMemcpy(device_raw.data(),draw,final_raw.size()*sizeof(float),hipMemcpyDeviceToHost));
        std::string path=dir+"/final_device.f32";
        FILE* file=fopen(path.c_str(),"wb");
        if (!file || fwrite(device_final.data(),sizeof(float),device_final.size(),file)!=device_final.size()) {
            fprintf(stderr,"cannot write %s\n",path.c_str());
            if(file)fclose(file);
            return 1;
        }
        if (fclose(file)) {fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
        path=dir+"/final_raw_device.f32";
        file=fopen(path.c_str(),"wb");
        if (!file || fwrite(device_raw.data(),sizeof(float),device_raw.size(),file)!=device_raw.size()) {
            fprintf(stderr,"cannot write %s\n",path.c_str());
            if(file)fclose(file);
            return 1;
        }
        if (fclose(file)) {fprintf(stderr,"cannot close %s\n",path.c_str());return 1;}
    }
    for (float* p : {dx,dw,de,dc,dp,ds,dq,dsc,dbias,dfw,dfs,dy,dm,dh,db,df,doq,
                     dn,dscores,dex,dprob,dcontext,draw,dfinal}) HIP_CHECK(hipFree(p));
    if (spatial) {HIP_CHECK(hipFree(dwindow));HIP_CHECK(hipFree(dcrop));}
    return ok ? 0 : 1;
}
