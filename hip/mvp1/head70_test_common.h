// Shared host helpers for standalone head diagnostics.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <limits>

// Test drivers exit on failure. An in-process engine defines HIP_CHECK,
// HEAD70_FATAL and HEAD70_INVALID before including these headers to throw.
#ifndef HIP_CHECK
#define HIP_CHECK(expr) do { hipError_t e = (expr); if (e != hipSuccess) { \
    fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(e)); exit(1); } } while (0)
#endif
#ifndef HEAD70_FATAL
#define HEAD70_FATAL(message) do { fprintf(stderr, "%s\n", std::string(message).c_str()); exit(1); } while (0)
#endif
#ifndef HEAD70_INVALID
#define HEAD70_INVALID(message) do { fputs(message "\n", stderr); exit(2); } while (0)
#endif

static std::vector<float> read(const std::string& dir, const char* name, size_t n) {
    std::string path = dir + "/" + name + ".f32";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) HEAD70_FATAL("cannot read " + path);
    std::vector<float> v(n);
    bool ok = fread(v.data(), sizeof(float), n, f) == n && fgetc(f) == EOF;
    fclose(f);
    if (!ok) HEAD70_FATAL("wrong size: " + path);
    return v;
}

static float* upload(const std::vector<float>& v) {
    float* p;
    HIP_CHECK(hipMalloc(&p, v.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
    return p;
}

static bool compare(const char* label, float* p, const std::vector<float>& want) {
    std::vector<float> got(want.size());
    HIP_CHECK(hipMemcpy(got.data(), p, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    size_t bad = 0, nonfinite = 0;
    double maxerr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) ++nonfinite;
        if (got[i] != want[i]) {
            if (bad < 5) printf("  %s[%zu]: got %.9g want %.9g\n", label, i, got[i], want[i]);
            ++bad;
        }
        maxerr = std::fmax(maxerr, std::fabs(double(got[i]) - want[i]));
    }
    // Exact numeric equality was chosen before device execution. No tolerance.
    printf("%s: %s (%zu values, %zu unequal, %zu nonfinite, maxerr %.9g)\n",
           label, bad == 0 && nonfinite == 0 ? "PASS" : "FAIL",
           got.size(), bad, nonfinite, maxerr);
    return bad == 0 && nonfinite == 0;
}
