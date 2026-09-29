// Checks ViT scores, exponents and chunked attention for any multiple of 64
// tokens against saved scalar references (tools/check_vit_attention_chunks.py).
#include "head70_test_common.h"
#include "vit_expand_chain.hip"

int main(int argc, char** argv) {
    if (argc != 3) { fputs("usage: vit_attention_chunks_test dir tokens\n", stderr); return 2; }
    std::string dir = argv[1];
    int tokens = atoi(argv[2]);
    if (tokens <= 0 || tokens % 64) return 2;
    size_t n = size_t(tokens)*1024, sn = size_t(32)*tokens*tokens;
    auto qkv = read(dir, "qkv", 3*n);
    auto scores = read(dir, "scores", sn), exponents = read(dir, "exponents", sn);
    auto attention = read(dir, "attention", n);
    float *dq = upload(qkv), *ds, *de, *da;
    HIP_CHECK(hipMalloc(&ds, sn*sizeof(float)));
    HIP_CHECK(hipMalloc(&de, sn*sizeof(float)));
    HIP_CHECK(hipMalloc(&da, n*sizeof(float)));
    hipLaunchKernelGGL(k_vit_scores, dim3((sn+255)/256), dim3(256), 0, 0, dq, ds, tokens);
    hipLaunchKernelGGL(k_vit_exponents, dim3((sn+255)/256), dim3(256), 0, 0, ds, de, int(sn));
    hipLaunchKernelGGL(k_vit_attention_chunks, dim3((n+255)/256), dim3(256), 0, 0, dq, de, da, tokens);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    bool ok = compare("vit scores", ds, scores);
    ok = compare("vit exponents", de, exponents) && ok;
    ok = compare("vit chunked attention", da, attention) && ok;
    return ok ? 0 : 1;
}
