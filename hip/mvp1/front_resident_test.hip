// Standalone check of the resident C32 front end against saved references.
#include "front_resident.h"
#include <cstring>

using namespace c512_resident;

static bool exact(const std::string& dir, const char* name, DeviceTensor view) {
    auto expected = read(dir, name, view.count);
    std::vector<float> got(view.count);
    HIP_CHECK(hipMemcpy(got.data(), view.data, got.size()*sizeof(float), hipMemcpyDeviceToHost));
    bool ok = !memcmp(got.data(), expected.data(), got.size()*sizeof(float));
    printf("%s repeat %s: %s (byte-exact)\n", dir.c_str(), name, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 5) { fputs("usage: front_resident_test fixture width height repeats\n", stderr); return 2; }
    std::string dir = argv[1];
    int width = atoi(argv[2]), height = atoi(argv[3]), repeats = atoi(argv[4]);
    if (repeats < 1 || repeats > 100) return 2;
    front_resident::FrontEnd front(width, height, dir);
    Buffer rgb(dir, "rgb", size_t(width)*height*3);
    if (!front.run_from_device({rgb.data, rgb.count}, true)) return 1;
    size_t checks = front.comparisons;
    HIP_CHECK(hipDeviceSynchronize());
    Traffic before = traffic;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    for (int i = 0; i < repeats; ++i) {
        HIP_CHECK(hipEventRecord(start));
        if (!front.run_from_device({rgb.data, rgb.count})) return 1;
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        ms.push_back(elapsed);
    }
    if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
        traffic.d2h_bytes != before.d2h_bytes || front.comparisons != checks) return 1;
    bool ok = exact(dir, "c64_input", front.c64_view()) && exact(dir, "skip4", front.skip4_view()) &&
              exact(dir, "skip0", front.skip0_view());
    printf("{\"stage_comparisons\":%zu,\"repeats\":%d,\"frame_ms\":[", checks, repeats);
    for (size_t i = 0; i < ms.size(); ++i) printf("%s%.6f", i ? "," : "", ms[i]);
    printf("]}\n");
    return ok ? 0 : 1;
}
