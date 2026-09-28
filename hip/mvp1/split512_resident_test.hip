// Standalone validation and repeat timing for a device-resident C512 chain.
#include "split512_resident.h"
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: split512_resident_test list width height repeats output-dir [head-fixture]\n");
        return 2;
    }
    int width = atoi(argv[2]), height = atoi(argv[3]), repeats = atoi(argv[4]);
    if (width < 4 || height < 4 || width > 1024 || height > 1024 || width%4 || height%4 ||
        width*height > 4096 || repeats < 1 || repeats > 1000) return 2;
    std::ifstream list(argv[1]);
    if (!list) return 2;
    std::vector<std::pair<std::string, int>> blocks;
    std::string line;
    while (std::getline(list, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3 || line[1] != '\t' || line[0] < '0' || line[0] > '3' || blocks.size() >= 8)
            return 2;
        blocks.emplace_back(line.substr(2), line[0]-'0');
    }
    if (blocks.empty()) return 2;
    using namespace c512_resident;
    auto begin = std::chrono::steady_clock::now();
    Chain chain(width, height, blocks, argc == 7 ? argv[6] : "");
    HIP_CHECK(hipDeviceSynchronize());
    double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
    if (!chain.run(true)) return 1;
    HIP_CHECK(hipDeviceSynchronize());
    size_t diagnostic_comparisons = chain.comparisons;
    Traffic before = traffic;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < repeats; ++i) if (!chain.run(false)) return 1;
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    Traffic after = traffic;
    if (after.allocations != before.allocations || after.h2d_bytes != before.h2d_bytes ||
        after.d2h_bytes != before.d2h_bytes || chain.comparisons != diagnostic_comparisons) {
        fputs("FAIL: host transfer/allocation during resident repeats\n", stderr); return 1;
    }
    if (!chain.save(argv[5])) { fputs("cannot save finite chain outputs\n", stderr); return 1; }
    std::string path = std::string(argv[5]) + "/resident_metrics.json";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return 1;
    int written = fprintf(f,
        "{\n\"blocks\":%zu,\"repeats\":%d,\"diagnostic_comparisons\":%zu,"
        "\"setup_ms\":%.6f,\"resident_ms_per_repeat\":%.6f,"
        "\"setup_allocations\":%zu,\"setup_h2d_bytes\":%zu,"
        "\"repeat_allocations\":%zu,\"repeat_h2d_bytes\":%zu,\"repeat_d2h_bytes\":%zu,"
        "\"repeat_d2d_bytes\":%zu,\"activation_uploads\":1,\"device_handoffs\":%zu,"
        "\"original_kernel_executed\":false,\"production_wiring\":false\n}\n",
        blocks.size(), repeats, diagnostic_comparisons, setup_ms, elapsed/repeats,
        before.allocations, before.h2d_bytes, after.allocations-before.allocations,
        after.h2d_bytes-before.h2d_bytes, after.d2h_bytes-before.d2h_bytes,
        after.d2d_bytes-before.d2d_bytes, blocks.size()-1+(argc==7 ? 1 : 0));
    bool ok = written > 0;
    if (fclose(f)) ok = false;
    if (!ok) return 1;
    printf("resident chain: PASS (%zu blocks, %d repeats, %.6f ms/repeat)\n", blocks.size(), repeats, elapsed/repeats);
    return 0;
}
