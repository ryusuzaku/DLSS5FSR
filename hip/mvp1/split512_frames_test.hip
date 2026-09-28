// Changing-frame component test; setup is outside all measured frame regions.
#include "split512_resident.h"
#include "vit_bridge_gather.hip"
#include "vit_resident.h"
#include <fstream>
#include <cstring>

using namespace c512_resident;

static bool check_save(const std::string& fixture, const std::string& output,
                       const char* name, const Buffer& buffer) {
    auto expected = read(fixture, name, buffer.count);
    std::vector<float> got(buffer.count);
    HIP_CHECK(hipMemcpy(got.data(), buffer.data, got.size()*sizeof(float), hipMemcpyDeviceToHost));
    traffic.d2h_bytes += got.size()*sizeof(float);
    bool ok = !memcmp(got.data(), expected.data(), got.size()*sizeof(float));
    for (size_t i = 0; i < got.size(); ++i)
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) ok = false;
    printf("%s %s: %s (byte-exact)\n", output.c_str(), name, ok ? "PASS" : "FAIL");
    if (!ok) return false;
    std::string path = output + "/" + name + ".f32";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    ok = fwrite(got.data(), sizeof(float), got.size(), f) == got.size();
    if (fclose(f)) ok = false;
    return ok;
}

static void consume(DeviceTensor view, Buffer& destination) {
    if (!view.data || view.count != destination.count) {
        fputs("invalid borrowed output view\n", stderr); exit(1);
    }
    HIP_CHECK(hipMemcpyAsync(destination.data, view.data, view.count*sizeof(float), hipMemcpyDeviceToDevice));
    traffic.d2d_bytes += view.count*sizeof(float);
}

struct IndexMap {
    int32_t* data = nullptr;
    IndexMap(const char* path, size_t count) {
        std::vector<int32_t> values(count);
        FILE* f = fopen(path, "rb");
        if (!f) { fputs("cannot read bridge map\n", stderr); exit(2); }
        bool ok = fread(values.data(), sizeof(int32_t), count, f) == count && fgetc(f) == EOF;
        fclose(f);
        std::vector<bool> seen(count);
        for (int32_t value : values) {
            if (value < 0 || size_t(value) >= count || seen[value]) { ok = false; break; }
            seen[value] = true;
        }
        if (!ok) { fputs("invalid bridge map\n", stderr); exit(2); }
        HIP_CHECK(hipMalloc(&data, count*sizeof(int32_t)));
        ++traffic.allocations;
        HIP_CHECK(hipMemcpy(data, values.data(), count*sizeof(int32_t), hipMemcpyHostToDevice));
        traffic.h2d_bytes += count*sizeof(int32_t);
    }
    ~IndexMap() { HIP_CHECK(hipFree(data)); }
};

int main(int argc, char** argv) {
    if (argc != 6 && argc != 8 && argc != 9) {
        fputs("usage: split512_frames_test blocks width height frames output [head-fixture bridge-map [vit-blocks]]\n", stderr);
        return 2;
    }
    int width = atoi(argv[2]), height = atoi(argv[3]);
    if ((width != 8 && width != 16 && width != 32) || height != 8) return 2;
    std::ifstream list(argv[1]), frames_file(argv[4]);
    if (!list || !frames_file) return 2;
    std::vector<std::pair<std::string, int>> blocks;
    std::string line;
    while (std::getline(list, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3 || line[1] != '\t' || line[0] < '0' || line[0] > '3' || blocks.size() >= 8) return 2;
        blocks.emplace_back(line.substr(2), line[0]-'0');
    }
    std::vector<std::string> frames;
    while (std::getline(frames_file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || frames.size() >= 32) return 2;
        frames.push_back(line);
    }
    if (blocks.size() != 8 || frames.size() < 3) return 2;
    bool encoder = argc >= 8, with_vit = argc == 9;
    if (with_vit && width == 16) return 2; // No 32-token attention contract.
    size_t n = size_t(width)*height*512, hn = size_t(width)*height/4*1024;
    Chain chain(width, height, blocks, encoder ? argv[6] : "");
    Buffer incoming(n), final_consumer(n), raw_consumer(n);
    std::unique_ptr<Buffer> head_consumer, bridge_output;
    std::unique_ptr<IndexMap> map;
    if (encoder) {
        head_consumer.reset(new Buffer(hn));
        bridge_output.reset(new Buffer(hn));
        map.reset(new IndexMap(argv[7], hn));
    }
    std::unique_ptr<vit_resident::Chain> vit;
    std::unique_ptr<Buffer> vit_consumer;
    if (with_vit) {
        std::ifstream vit_file(argv[8]);
        if (!vit_file) return 2;
        std::vector<std::string> folders;
        while (std::getline(vit_file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || folders.size() >= 8) return 2;
            folders.push_back(line);
        }
        if (folders.size() != 8) return 2;
        vit.reset(new vit_resident::Chain(width*height/4, folders));
        vit_consumer.reset(new Buffer(hn));
        if (vit->final_view().data) return 1;
    }
    if (chain.final_view().data || chain.raw_view().data || chain.head_view().data) return 1;
    HIP_CHECK(hipDeviceSynchronize());
    Traffic setup = traffic;
    std::vector<float> frame_ms;
    size_t compute_d2d = 0, frame_upload_bytes = 0;
    int rejected = 0;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    for (size_t index = 0; index < frames.size(); ++index) {
        if (index == 2) {
            Traffic before = traffic;
            if (chain.run_from_device({nullptr, n}) || chain.final_view().data ||
                chain.raw_view().data || chain.head_view().data || chain.save(argv[5])) return 1;
            ++rejected;
            if (chain.run_from_device({incoming.data, n-1}) || chain.final_view().data ||
                chain.raw_view().data || chain.head_view().data) return 1;
            ++rejected;
            if (vit) {
                if (vit->run_from_device({nullptr, hn}) || vit->final_view().data) return 1;
                ++rejected;
                if (vit->run_from_device({bridge_output->data, hn-1}) || vit->final_view().data) return 1;
                ++rejected;
            }
            if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
                traffic.d2h_bytes != before.d2h_bytes || traffic.d2d_bytes != before.d2d_bytes) return 1;
        }
        auto input = read(frames[index], "input", n);
        for (float value : input) if (!std::isfinite(value)) return 2;
        HIP_CHECK(hipMemcpy(incoming.data, input.data(), n*sizeof(float), hipMemcpyHostToDevice));
        traffic.h2d_bytes += n*sizeof(float);
        frame_upload_bytes += n*sizeof(float);
        // Check every ViT stage on the actual GPU-produced first-frame bridge.
        // Diagnostic readbacks are outside all measured execution regions.
        if (vit && index == 0) {
            if (!chain.run_from_device({incoming.data, n})) return 1;
            hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, 0,
                               chain.head_view().data, map->data, bridge_output->data, int(hn));
            HIP_CHECK(hipGetLastError());
            if (!vit->run_from_device({bridge_output->data, hn}, true) || vit->comparisons != 80)
                return 1;
        }
        Traffic before = traffic;
        HIP_CHECK(hipEventRecord(start));
        if (!chain.run_from_device({incoming.data, n})) return 1;
        consume(chain.final_view(), final_consumer);
        consume(chain.raw_view(), raw_consumer);
        if (encoder) {
            consume(chain.head_view(), *head_consumer);
            // A real adjacent gather kernel consumes the borrowed head pointer.
            auto view = chain.head_view();
            hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, 0,
                               view.data, map->data, bridge_output->data, int(hn));
            HIP_CHECK(hipGetLastError());
        }
        if (vit) {
            if (!vit->run_from_device({bridge_output->data, hn})) return 1;
            consume(vit->final_view(), *vit_consumer);
        }
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        frame_ms.push_back(elapsed);
        if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
            traffic.d2h_bytes != before.d2h_bytes || chain.comparisons ||
            (vit && vit->comparisons != 80)) return 1;
        size_t expected_d2d = (3*n + (encoder ? hn : 0) + (vit ? 2*hn : 0))*sizeof(float);
        if (traffic.d2d_bytes-before.d2d_bytes != expected_d2d) return 1;
        compute_d2d += expected_d2d;
        std::string output = std::string(argv[5]) + "/frame_" + std::to_string(index);
        if (!check_save(frames[index], output, "final", final_consumer) ||
            !check_save(frames[index], output, "final_raw", raw_consumer) ||
            (encoder && (!check_save(frames[index], output, "head", *head_consumer) ||
                         !check_save(frames[index], output, "bridge", *bridge_output)))) return 1;
        if (vit && !check_save(frames[index], output, "vit", *vit_consumer)) return 1;
        // A borrowed input must not be overwritten by chain or consumer work.
        std::vector<float> unchanged(n);
        HIP_CHECK(hipMemcpy(unchanged.data(), incoming.data, n*sizeof(float), hipMemcpyDeviceToHost));
        traffic.d2h_bytes += n*sizeof(float);
        if (memcmp(unchanged.data(), input.data(), n*sizeof(float))) return 1;
    }
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    if (traffic.allocations != setup.allocations || traffic.h2d_bytes-setup.h2d_bytes != frame_upload_bytes)
        return 1;
    std::string path = std::string(argv[5]) + "/frames_metrics.json";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return 1;
    bool ok = fprintf(f, "{\"frames\":%zu,\"invalid_views_rejected\":%d,"
        "\"post_setup_allocations\":0,\"frame_upload_bytes\":%zu,\"compute_h2d_bytes\":0,"
        "\"compute_d2h_bytes\":0,\"compute_d2d_bytes\":%zu,\"weights_loaded_once\":true,"
        "\"input_preserved\":true,\"adjacent_vit_bridge\":%s,\"resident_vit\":%s,"
        "\"vit_stage_comparisons\":%zu,\"frame_ms\":[",
        frames.size(), rejected, frame_upload_bytes, compute_d2d, encoder ? "true" : "false",
        vit ? "true" : "false", vit ? vit->comparisons : size_t(0)) > 0;
    for (size_t i = 0; i < frame_ms.size(); ++i)
        ok = (fprintf(f, "%s%.6f", i ? "," : "", frame_ms[i]) > 0) && ok;
    ok = (fprintf(f, "]}\n") > 0) && ok;
    if (fclose(f)) ok = false;
    return ok ? 0 : 1;
}
