// Changing-frame component test; setup is outside all measured frame regions.
#include "split512_resident.h"
#include "vit_bridge_gather.hip"
#include "vit_resident.h"
#include "decoder39_resident.h"
#include "c256_resident.h"
#include "c128_resident.h"
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

static bool input_unchanged(DeviceTensor view, const Buffer& snapshot) {
    if (!view.data || view.count != snapshot.count) return false;
    std::vector<float> after(view.count), before(view.count);
    HIP_CHECK(hipMemcpy(after.data(), view.data, view.count*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(before.data(), snapshot.data, view.count*sizeof(float), hipMemcpyDeviceToHost));
    traffic.d2h_bytes += 2*view.count*sizeof(float);
    return !memcmp(after.data(), before.data(), view.count*sizeof(float));
}
int main(int argc, char** argv) {
    if (argc != 6 && argc != 8 && argc != 9 && argc != 11 && argc != 13 && argc != 15) {
        fputs("usage: split512_frames_test blocks width height frames output [head-fixture bridge-map [vit-blocks [decoder39-fixture decoder-blocks [prefix48-fixture c256-blocks [prefix56-fixture c128-blocks]]]]]\n", stderr);
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
    bool encoder = argc >= 8, with_vit = argc >= 9, with_decoder = argc >= 11,
         with_c256 = argc >= 13, with_c128 = argc == 15;
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
    std::unique_ptr<decoder39_resident::Entry> entry;
    std::unique_ptr<Chain> decoder;
    std::unique_ptr<Buffer> inverse_consumer, projected_consumer, entry_consumer,
                            decoder_consumer, decoder_raw_consumer;
    if (with_decoder) {
        std::ifstream decoder_file(argv[10]);
        if (!decoder_file) return 2;
        std::vector<std::pair<std::string, int>> decoder_blocks;
        while (std::getline(decoder_file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.size() < 3 || line[1] != '\t' || line[0] < '0' || line[0] > '3' ||
                decoder_blocks.size() >= 8) return 2;
            decoder_blocks.emplace_back(line.substr(2), line[0]-'0');
        }
        if (decoder_blocks.size() != 8) return 2;
        entry.reset(new decoder39_resident::Entry(width/2, height/2, argv[9]));
        decoder.reset(new Chain(width, height, decoder_blocks));
        inverse_consumer.reset(new Buffer(hn));
        projected_consumer.reset(new Buffer(hn/2));
        entry_consumer.reset(new Buffer(n));
        decoder_consumer.reset(new Buffer(n));
        decoder_raw_consumer.reset(new Buffer(n));
        if (entry->final_view().data || entry->main_view().data || entry->projected_view().data ||
            decoder->final_view().data || decoder->raw_view().data) return 1;
    }
    std::unique_ptr<c256_resident::Prefix> prefix48;
    std::unique_ptr<c256_resident::Chain> c256;
    std::unique_ptr<Buffer> skip22, low48_consumer, merge48_consumer, c256_consumer;
    if (with_c256) {
        std::ifstream file(argv[12]);
        if (!file) return 2;
        std::vector<std::pair<std::string,int>> c256_blocks;
        while (std::getline(file,line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.size()<3 || line[1]!='\t' || line[0]<'0' || line[0]>'3' || c256_blocks.size()>=8) return 2;
            c256_blocks.emplace_back(line.substr(2),line[0]-'0');
        }
        if (c256_blocks.size()!=8) return 2;
        prefix48.reset(new c256_resident::Prefix(width,height,argv[11]));
        c256.reset(new c256_resident::Chain(width*2,height*2,c256_blocks));
        skip22.reset(new Buffer(2*n));
        low48_consumer.reset(new Buffer(n/2));
        merge48_consumer.reset(new Buffer(2*n));
        c256_consumer.reset(new Buffer(2*n));
        if (prefix48->final_view().data || prefix48->low_view().data || c256->final_view().data) return 1;
    }
    std::unique_ptr<c128_resident::Prefix> prefix56;
    std::unique_ptr<c128_resident::Chain> c128;
    std::unique_ptr<Buffer> skip14, low56_consumer, merge56_consumer, c128_consumer;
    if (with_c128) {
        std::ifstream file(argv[14]);
        if (!file) return 2;
        std::vector<std::pair<std::string,int>> blocks128;
        while (std::getline(file,line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.size()<3 || line[1]!='\t' || line[0]<'0' || line[0]>'3' || blocks128.size()>=6) return 2;
            blocks128.emplace_back(line.substr(2),line[0]-'0');
        }
        if (blocks128.size()!=6) return 2;
        prefix56.reset(new c128_resident::Prefix(width*2,height*2,argv[13]));
        c128.reset(new c128_resident::Chain(width*4,height*4,blocks128));
        skip14.reset(new Buffer(4*n));
        low56_consumer.reset(new Buffer(n));
        merge56_consumer.reset(new Buffer(4*n));
        c128_consumer.reset(new Buffer(4*n));
        if (prefix56->final_view().data || prefix56->low_view().data || c128->final_view().data) return 1;
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
            if (entry) {
                DeviceTensor v{vit_consumer->data, hn}, s{final_consumer.data, n};
                auto reject = [&](DeviceTensor main, DeviceTensor skip) {
                    return !entry->run_from_device(main, skip) && !entry->final_view().data &&
                           !entry->main_view().data && !entry->projected_view().data;
                };
                if (!reject({nullptr, hn}, s) || !reject({v.data, hn-1}, s) ||
                    !reject(v, {nullptr, n}) || !reject(v, {s.data, n-1})) return 1;
                rejected += 4;
                if (decoder->run_from_device({nullptr, n}) || decoder->final_view().data ||
                    decoder->raw_view().data || decoder->save(argv[5])) return 1;
                ++rejected;
                if (decoder->run_from_device({entry_consumer->data, n-1}) || decoder->final_view().data ||
                    decoder->raw_view().data) return 1;
                ++rejected;
            }
            if (prefix48) {
                DeviceTensor main{decoder_consumer->data,n}, skip{skip22->data,2*n};
                auto reject = [&](DeviceTensor a, DeviceTensor b) {
                    return !prefix48->run_from_device(a,b) && !prefix48->final_view().data && !prefix48->low_view().data;
                };
                if (!reject({nullptr,n},skip) || !reject({main.data,n-1},skip) ||
                    !reject(main,{nullptr,2*n}) || !reject(main,{skip.data,2*n-1})) return 1;
                rejected += 4;
                if (c256->run_from_device({nullptr,2*n}) || c256->final_view().data ||
                    c256->run_from_device({merge48_consumer->data,2*n-1}) || c256->final_view().data) return 1;
                rejected += 2;
            }
            if (prefix56) {
                DeviceTensor main{c256_consumer->data,2*n}, skip{skip14->data,4*n};
                auto reject = [&](DeviceTensor a, DeviceTensor b) {
                    return !prefix56->run_from_device(a,b) && !prefix56->final_view().data && !prefix56->low_view().data;
                };
                if (!reject({nullptr,2*n},skip) || !reject({main.data,2*n-1},skip) ||
                    !reject(main,{nullptr,4*n}) || !reject(main,{skip.data,4*n-1})) return 1;
                rejected += 4;
                if (c128->run_from_device({nullptr,4*n}) || c128->final_view().data ||
                    c128->run_from_device({merge56_consumer->data,4*n-1}) || c128->final_view().data) return 1;
                rejected += 2;
            }
            if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
                traffic.d2h_bytes != before.d2h_bytes || traffic.d2d_bytes != before.d2d_bytes) return 1;
        }
        auto input = read(frames[index], "input", n);
        for (float value : input) if (!std::isfinite(value)) return 2;
        HIP_CHECK(hipMemcpy(incoming.data, input.data(), n*sizeof(float), hipMemcpyHostToDevice));
        traffic.h2d_bytes += n*sizeof(float);
        frame_upload_bytes += n*sizeof(float);
        std::vector<float> skip_input;
        if (skip22) {
            skip_input = read(frames[index],"skip22",2*n);
            for (float value:skip_input) if (!std::isfinite(value)) return 2;
            HIP_CHECK(hipMemcpy(skip22->data,skip_input.data(),2*n*sizeof(float),hipMemcpyHostToDevice));
            traffic.h2d_bytes += 2*n*sizeof(float);
            frame_upload_bytes += 2*n*sizeof(float);
        }
        std::vector<float> skip14_input;
        if (skip14) {
            skip14_input = read(frames[index],"skip14",4*n);
            for (float value:skip14_input) if (!std::isfinite(value)) return 2;
            HIP_CHECK(hipMemcpy(skip14->data,skip14_input.data(),4*n*sizeof(float),hipMemcpyHostToDevice));
            traffic.h2d_bytes += 4*n*sizeof(float);
            frame_upload_bytes += 4*n*sizeof(float);
        }
        // Check every ViT stage on the actual GPU-produced first-frame bridge.
        // Diagnostic readbacks are outside all measured execution regions.
        if (vit && index == 0) {
            if (!chain.run_from_device({incoming.data, n})) return 1;
            hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, 0,
                               chain.head_view().data, map->data, bridge_output->data, int(hn));
            HIP_CHECK(hipGetLastError());
            if (!vit->run_from_device({bridge_output->data, hn}, true) || vit->comparisons != 80)
                return 1;
            if (entry && (!entry->run_from_device(vit->final_view(), chain.final_view(), true) ||
                          entry->comparisons != 5 ||
                          !decoder->run_from_device(entry->final_view(), true) || decoder->comparisons != 112))
                return 1;
            if (prefix48 && (!prefix48->run_from_device(decoder->final_view(),{skip22->data,2*n},true) ||
                             prefix48->comparisons != 4 || !c256->run_from_device(prefix48->final_view(),true) ||
                             c256->comparisons != 120)) return 1;
            if (prefix56 && (!prefix56->run_from_device(c256->final_view(),{skip14->data,4*n},true) ||
                             prefix56->comparisons != 4 || !c128->run_from_device(prefix56->final_view(),true) ||
                             c128->comparisons != 90)) return 1;
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
        if (entry) {
            // Preserve the encoder30 skip until decoder39 consumes this same frame.
            if (!entry->run_from_device(vit->final_view(), chain.final_view()) ||
                !decoder->run_from_device(entry->final_view())) return 1;
            consume(entry->main_view(), *inverse_consumer);
            consume(entry->projected_view(), *projected_consumer);
            consume(entry->final_view(), *entry_consumer);
            consume(decoder->final_view(), *decoder_consumer);
            consume(decoder->raw_view(), *decoder_raw_consumer);
        }
        if (prefix48) {
            if (!prefix48->run_from_device(decoder->final_view(),{skip22->data,2*n}) ||
                !c256->run_from_device(prefix48->final_view())) return 1;
            consume(prefix48->low_view(),*low48_consumer);
            consume(prefix48->final_view(),*merge48_consumer);
            consume(c256->final_view(),*c256_consumer);
        }
        if (prefix56) {
            if (!prefix56->run_from_device(c256->final_view(),{skip14->data,4*n}) ||
                !c128->run_from_device(prefix56->final_view())) return 1;
            consume(prefix56->low_view(),*low56_consumer);
            consume(prefix56->final_view(),*merge56_consumer);
            consume(c128->final_view(),*c128_consumer);
        }
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        frame_ms.push_back(elapsed);
        if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
            traffic.d2h_bytes != before.d2h_bytes || chain.comparisons ||
            (vit && vit->comparisons != 80) ||
            (entry && (entry->comparisons != 5 || decoder->comparisons != 112)) ||
            (prefix48 && (prefix48->comparisons != 4 || c256->comparisons != 120)) ||
            (prefix56 && (prefix56->comparisons != 4 || c128->comparisons != 90))) return 1;
        size_t expected_d2d = (3*n + (encoder ? hn : 0) + (vit ? 2*hn : 0) +
                               (entry ? 4*n + hn + hn/2 : 0) + (prefix48 ? 6*n+n/2 : 0) +
                               (prefix56 ? 13*n : 0))*sizeof(float);
        if (traffic.d2d_bytes-before.d2d_bytes != expected_d2d) return 1;
        compute_d2d += expected_d2d;
        std::string output = std::string(argv[5]) + "/frame_" + std::to_string(index);
        if (!check_save(frames[index], output, "final", final_consumer) ||
            !check_save(frames[index], output, "final_raw", raw_consumer) ||
            (encoder && (!check_save(frames[index], output, "head", *head_consumer) ||
                         !check_save(frames[index], output, "bridge", *bridge_output)))) return 1;
        if (vit && !check_save(frames[index], output, "vit", *vit_consumer)) return 1;
        if (entry && (!check_save(frames[index], output, "main39", *inverse_consumer) ||
                      !check_save(frames[index], output, "projected39", *projected_consumer) ||
                      !check_save(frames[index], output, "decoder39", *entry_consumer) ||
                      !check_save(frames[index], output, "decoder47", *decoder_consumer) ||
                      !check_save(frames[index], output, "decoder47_raw", *decoder_raw_consumer))) return 1;
        if (entry && (!input_unchanged(vit->final_view(), *vit_consumer) ||
                      !input_unchanged(chain.final_view(), final_consumer))) return 1;
        if (prefix48) {
            if (!check_save(frames[index],output,"low48",*low48_consumer) ||
                !check_save(frames[index],output,"merge48",*merge48_consumer) ||
                !check_save(frames[index],output,"decoder55",*c256_consumer) ||
                !input_unchanged(decoder->final_view(),*decoder_consumer)) return 1;
            std::vector<float> after(2*n);
            HIP_CHECK(hipMemcpy(after.data(),skip22->data,2*n*sizeof(float),hipMemcpyDeviceToHost));
            traffic.d2h_bytes += 2*n*sizeof(float);
            if (memcmp(after.data(),skip_input.data(),2*n*sizeof(float))) return 1;
        }
        if (prefix56) {
            if (!check_save(frames[index],output,"low56",*low56_consumer) ||
                !check_save(frames[index],output,"merge56",*merge56_consumer) ||
                !check_save(frames[index],output,"decoder61",*c128_consumer) ||
                !input_unchanged(c256->final_view(),*c256_consumer)) return 1;
            std::vector<float> after(4*n);
            HIP_CHECK(hipMemcpy(after.data(),skip14->data,4*n*sizeof(float),hipMemcpyDeviceToHost));
            traffic.d2h_bytes += 4*n*sizeof(float);
            if (memcmp(after.data(),skip14_input.data(),4*n*sizeof(float))) return 1;
        }
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
        "\"vit_stage_comparisons\":%zu,\"resident_decoder\":%s,"
        "\"decoder39_stage_comparisons\":%zu,\"decoder_stage_comparisons\":%zu,"
        "\"resident_c256\":%s,\"prefix48_stage_comparisons\":%zu,\"c256_stage_comparisons\":%zu,"
        "\"resident_c128\":%s,\"prefix56_stage_comparisons\":%zu,\"c128_stage_comparisons\":%zu,\"frame_ms\":[",
        frames.size(), rejected, frame_upload_bytes, compute_d2d, encoder ? "true" : "false",
        vit ? "true" : "false", vit ? vit->comparisons : size_t(0), entry ? "true" : "false",
        entry ? entry->comparisons : size_t(0), decoder ? decoder->comparisons : size_t(0),
        c256 ? "true" : "false", prefix48 ? prefix48->comparisons : size_t(0), c256 ? c256->comparisons : size_t(0),
        c128 ? "true" : "false", prefix56 ? prefix56->comparisons : size_t(0), c128 ? c128->comparisons : size_t(0)) > 0;
    for (size_t i = 0; i < frame_ms.size(); ++i)
        ok = (fprintf(f, "%s%.6f", i ? "," : "", frame_ms[i]) > 0) && ok;
    ok = (fprintf(f, "]}\n") > 0) && ok;
    if (fclose(f)) ok = false;
    return ok ? 0 : 1;
}
