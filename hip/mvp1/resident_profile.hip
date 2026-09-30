// Per-stage GPU timing of the resident network at a chosen extent.
// Usage: resident_profile config.txt capture.bin [repeats]
// Compiles the engine's own Network so the measured code is the shipped code.
#include "resident_engine.hip"
#include <cstdio>
#include <map>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) { fputs("usage: resident_profile config.txt capture.bin [repeats]\n", stderr); return 2; }
    int repeats = argc > 3 ? atoi(argv[3]) : 3;
    try {
        std::ifstream file(argv[1]);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        std::ifstream capture(argv[2], std::ios::binary);
        std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(capture)), std::istreambuf_iterator<char>());
        unsigned int f[4];
        memcpy(f, bytes.data()+8, sizeof(f));
        int fw = int(f[0]), fh = int(f[1]), bpp = int(f[2]), pitch = int(f[3]);
        int W = 0, H = 0;
        network_extent(fw, fh, W, H);
        hipStream_t stream;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        c512_resident::stream = stream;
        Network net(lines, W, H);
        Buffer raw((size_t(pitch)*fh+3)/4);
        HIP_CHECK(hipMemcpy(raw.data, bytes.data()+36, size_t(pitch)*fh, hipMemcpyHostToDevice));
        const unsigned char* src = reinterpret_cast<const unsigned char*>(raw.data);
        std::vector<std::pair<const char*, std::function<bool()>>> stages = {
            {"input", [&]{ hipLaunchKernelGGL(k_candidate_input_full, dim3((W*H+255)/256), dim3(256), 0, stream,
                                              src, net.color->data, fw, fh, pitch, bpp, W, H); return true; }},
            {"front C32 (stem, block0-4, down4)", [&]{ return net.front->run_from_device({net.color->data,net.color->count}); }},
            {"encoder5-8 C64", [&]{ return net.encoder8->run_from_device(net.front->c64_view()); }},
            {"down8", [&]{ return net.down8->run_from_device(net.encoder8->raw_view()); }},
            {"encoder9-14 C128", [&]{ return net.encoder14->run_from_device(net.down8->final_view()); }},
            {"down14", [&]{ return net.down14->run_from_device(net.encoder14->raw_view()); }},
            {"encoder15-22 C256", [&]{ return net.encoder22->run_from_device(net.down14->final_view()); }},
            {"down22", [&]{ return net.down22->run_from_device(net.encoder22->raw_view()); }},
            {"encoder23-30 C512 + head", [&]{ return net.chain->run_from_device(net.down22->final_view()); }},
            {"gather", [&]{ auto head = net.chain->head_view();
                hipLaunchKernelGGL(k_vit_bridge_gather, dim3((net.hn+255)/256), dim3(256), 0, stream,
                                   head.data, net.map->data, net.bridge->data, int(net.hn)); return true; }},
            {"ViT31-38", [&]{ return net.vit->run_from_device({net.bridge->data,net.hn}); }},
            {"decoder39 entry", [&]{ return net.entry->run_from_device(net.vit->final_view(), net.chain->final_view()); }},
            {"decoder40-47 C512", [&]{ return net.decoder->run_from_device(net.entry->final_view()); }},
            {"prefix48", [&]{ return net.prefix48->run_from_device(net.decoder->final_view(), net.encoder22->final_view()); }},
            {"decoder48-55 C256", [&]{ return net.c256->run_from_device(net.prefix48->final_view()); }},
            {"prefix56", [&]{ return net.prefix56->run_from_device(net.c256->final_view(), net.encoder14->final_view()); }},
            {"decoder56-61 C128", [&]{ return net.c128->run_from_device(net.prefix56->final_view()); }},
            {"prefix62", [&]{ return net.prefix62->run_from_device(net.c128->final_view(), net.encoder8->final_view()); }},
            {"decoder62-65 C64", [&]{ return net.c64->run_from_device(net.prefix62->final_view()); }},
            {"prefix66", [&]{ return net.prefix66->run_from_device(net.c64->final_view(), net.front->skip4_view()); }},
            {"decoder66-69 C32", [&]{ return net.c32->run_from_device(net.prefix66->final_view()); }},
            {"head70", [&]{ return net.head70->run_from_device(net.c32->final_view(), net.front->skip0_view(),
                                                                 {net.color->data,net.color->count}); }},
        };
        std::vector<hipEvent_t> events(stages.size()+1);
        for (auto& e : events) HIP_CHECK(hipEventCreate(&e));
        std::vector<double> total(stages.size(), 0.0);
        for (int r = 0; r < repeats+1; ++r) {
            HIP_CHECK(hipEventRecord(events[0], stream));
            for (size_t i = 0; i < stages.size(); ++i) {
                if (!stages[i].second()) throw std::runtime_error(std::string("stage failed: ") + stages[i].first);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipEventRecord(events[i+1], stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            if (r == 0) continue;  // warm-up
            for (size_t i = 0; i < stages.size(); ++i) {
                float ms = 0;
                HIP_CHECK(hipEventElapsedTime(&ms, events[i], events[i+1]));
                total[i] += ms;
            }
        }
        // Per-kernel: one more run with an event after every resident launch.
        static std::vector<std::pair<std::string, hipEvent_t>> marks;
        static hipStream_t mark_stream;
        mark_stream = stream;
        c512_resident::launch_hook = [](const char* kernel) {
            hipEvent_t e;
            if (hipEventCreate(&e) == hipSuccess && hipEventRecord(e, mark_stream) == hipSuccess)
                marks.emplace_back(kernel, e);
        };
        hipEvent_t begin;
        HIP_CHECK(hipEventCreate(&begin));
        HIP_CHECK(hipEventRecord(begin, stream));
        for (auto& stage : stages) if (!stage.second()) throw std::runtime_error("stage failed");
        HIP_CHECK(hipStreamSynchronize(stream));
        c512_resident::launch_hook = nullptr;
        std::map<std::string, std::pair<double,int>> kernels;
        hipEvent_t previous = begin;
        for (auto& m : marks) {
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, previous, m.second));
            kernels[m.first].first += ms; kernels[m.first].second += 1;
            previous = m.second;
        }
        std::vector<std::pair<double, std::string>> ranked;
        double kernel_sum = 0;
        for (auto& k : kernels) { ranked.emplace_back(k.second.first, k.first); kernel_sum += k.second.first; }
        std::sort(ranked.rbegin(), ranked.rend());
        printf("per-kernel (one run, %zu launches, %.1f ms incl. unhooked work):\n", marks.size(), kernel_sum);
        for (size_t i = 0; i < ranked.size() && i < 80; ++i)
            printf("  %-40s %8.2f ms  %4d launches\n", ranked[i].second.c_str(), ranked[i].first,
                   kernels[ranked[i].second].second);
        double sum = 0;
        for (double t : total) sum += t;
        printf("network %dx%d for %dx%d frame, mean of %d runs: %.1f ms\n", W, H, fw, fh, repeats, sum/repeats);
        for (size_t i = 0; i < stages.size(); ++i)
            printf("  %-36s %8.2f ms  %5.1f%%\n", stages[i].first, total[i]/repeats, 100.0*total[i]/sum);
    } catch (const std::exception& e) {
        fprintf(stderr, "profile failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
