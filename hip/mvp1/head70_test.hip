// Standalone device check for head70 outer passes. See tools/check_head70.py.
#include "head70.hip"
#include "head70_test_common.h"

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: head70_test fixture width height scale\n"); return 2; }
    int w = atoi(argv[2]), h = atoi(argv[3]);
    float scale = strtof(argv[4], nullptr);
    if (w <= 0 || h <= 0 || w % 8 || h % 8 || w > 4096 || h > 4096 || !std::isfinite(scale)) return 2;
    std::string dir = argv[1];
    size_t n = size_t(w) * h * 32, nrgb = size_t(w) * h * 3;
    float *main = upload(read(dir, "main", n / 4)), *skip = upload(read(dir, "skip", n));
    float *sm = upload(read(dir, "sm", 32)), *ss = upload(read(dir, "ss", 32));
    float *coeff = upload(read(dir, "coeff", 96)), *color = upload(read(dir, "color", nrgb));
    float *features = upload(read(dir, "features", n)), *merged, *rgb;
    // Sentinels expose omitted output lanes and launches rounded down.
    merged = upload(std::vector<float>(n, std::numeric_limits<float>::quiet_NaN()));
    rgb = upload(std::vector<float>(nrgb, std::numeric_limits<float>::quiet_NaN()));
    hipLaunchKernelGGL(k_head70_merge, dim3((n + 255) / 256), dim3(256), 0, 0,
                      main, skip, sm, ss, merged, w, h);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    bool ok = compare("merge/window", merged, read(dir, "merged", n));
    hipLaunchKernelGGL(k_head70_finish, dim3((nrgb + 255) / 256), dim3(256), 0, 0,
                      features, coeff, color, rgb, w, h, scale);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    ok = compare("finish/dewindow", rgb, read(dir, "rgb", nrgb)) && ok;
    for (float* p : {main, skip, sm, ss, coeff, color, features, merged, rgb}) HIP_CHECK(hipFree(p));
    return ok ? 0 : 1;
}
