// Persistent resident candidate: encoder5-30, ViT31-38, decoder39-69, head70.
// Weights and workspace load once. Each request directory supplies the C64
// boundary4 input, block4/preblock0 skips and linear RGB; the server writes
// native- and public-gain enhanced RGB. Arithmetic is the unchanged resident
// kernels validated by split512_frames_test's 26-argument mode.
#include "split512_resident.h"
#include "vit_bridge_gather.hip"
#include "vit_resident.h"
#include "decoder39_resident.h"
#include "c256_resident.h"
#include "c128_resident.h"
#include "c64_resident.h"
#include "c32_resident.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace c512_resident;

static bool read_list(const char* path, size_t count, std::vector<std::pair<std::string,int>>& out) {
    std::ifstream file(path);
    std::string line;
    if (!file) return false;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3 || line[1] != '\t' || line[0] < '0' || line[0] > '3' || out.size() >= count) return false;
        out.emplace_back(line.substr(2), line[0]-'0');
    }
    return out.size() == count;
}

static bool read_folders(const char* path, size_t count, std::vector<std::string>& out) {
    std::ifstream file(path);
    std::string line;
    if (!file) return false;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || out.size() >= count) return false;
        out.push_back(line);
    }
    return out.size() == count;
}

// Reads exactly count finite floats; false on any size or value problem.
static bool load(const std::string& path, std::vector<float>& values, size_t count) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    values.resize(count);
    bool ok = fread(values.data(), sizeof(float), count, f) == count && fgetc(f) == EOF;
    fclose(f);
    if (!ok) return false;
    for (float v : values) if (!std::isfinite(v)) return false;
    return true;
}

static bool store(const std::string& path, const Buffer& buffer) {
    std::vector<float> values(buffer.count);
    HIP_CHECK(hipMemcpy(values.data(), buffer.data, values.size()*sizeof(float), hipMemcpyDeviceToHost));
    std::string temp = path + ".tmp";
    FILE* f = fopen(temp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(values.data(), sizeof(float), values.size(), f) == values.size();
    if (fclose(f)) ok = false;
    if (!ok) return false;
    remove(path.c_str());
    return rename(temp.c_str(), path.c_str()) == 0;
}

static void consume(DeviceTensor view, Buffer& destination) {
    if (!view.data || view.count != destination.count) throw std::runtime_error("invalid resident output view");
    HIP_CHECK(hipMemcpyAsync(destination.data, view.data, view.count*sizeof(float), hipMemcpyDeviceToDevice));
}

int main(int argc, char** argv) {
    if (argc != 24) {
        fputs("usage: resident_frame_server c512-blocks width height head-fixture bridge-map vit-blocks "
              "decoder39-fixture decoder-blocks prefix48 c256-blocks prefix56 c128-blocks prefix62 c64-blocks "
              "prefix66 c32-blocks head70 encoder15-blocks downsample22 encoder9-blocks downsample14 "
              "encoder5-blocks downsample8\n"
              "then one request directory per stdin line (c64_input, skip4, skip0, color .f32);\n"
              "writes rgb_native.f32 and rgb_public.f32 and answers OK <gpu-ms> or ERR <reason>\n", stderr);
        return 2;
    }
    int width = atoi(argv[2]), height = atoi(argv[3]);
    if ((width != 8 && width != 32) || height != 8) return 2;
    std::vector<std::pair<std::string,int>> c512_blocks, decoder_blocks, c256_blocks, c128_blocks,
        c64_blocks, c32_blocks, enc22_blocks, enc14_blocks, enc8_blocks;
    std::vector<std::string> vit_folders;
    if (!read_list(argv[1],8,c512_blocks) || !read_folders(argv[6],8,vit_folders) ||
        !read_list(argv[8],8,decoder_blocks) || !read_list(argv[10],8,c256_blocks) ||
        !read_list(argv[12],6,c128_blocks) || !read_list(argv[14],4,c64_blocks) ||
        !read_list(argv[16],4,c32_blocks) || !read_list(argv[18],8,enc22_blocks) ||
        !read_list(argv[20],6,enc14_blocks) || !read_list(argv[22],4,enc8_blocks)) {
        fputs("invalid block list\n", stderr);
        return 2;
    }
    size_t n = size_t(width)*height*512, hn = size_t(width)*height/4*1024;
    try {
        c64_resident::Chain encoder8(width*8,height*8,enc8_blocks,true);
        c64_resident::Downsample down8(width*8,height*8,argv[23]);
        c128_resident::Chain encoder14(width*4,height*4,enc14_blocks,true);
        c128_resident::Downsample down14(width*4,height*4,argv[21]);
        c256_resident::Chain encoder22(width*2,height*2,enc22_blocks,true);
        c256_resident::Downsample down22(width*2,height*2,argv[19]);
        Chain chain(width,height,c512_blocks,argv[4]);
        IndexMap map(argv[5],hn);
        Buffer bridge(hn);
        vit_resident::Chain vit(width*height/4,vit_folders);
        decoder39_resident::Entry entry(width/2,height/2,argv[7]);
        Chain decoder(width,height,decoder_blocks);
        c256_resident::Prefix prefix48(width,height,argv[9]);
        c256_resident::Chain c256(width*2,height*2,c256_blocks);
        c128_resident::Prefix prefix56(width*2,height*2,argv[11]);
        c128_resident::Chain c128(width*4,height*4,c128_blocks);
        c64_resident::Prefix prefix62(width*4,height*4,argv[13]);
        c64_resident::Chain c64(width*8,height*8,c64_blocks);
        c32_resident::Prefix prefix66(width*8,height*8,argv[15]);
        c32_resident::Chain c32(width*16,height*16,c32_blocks);
        c32_resident::Head head70(width*32,height*32,argv[17]);
        Buffer c64_input(8*n), skip4(16*n), skip0(64*n), color(6*n), rgb_native(6*n), rgb_public(6*n);
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipDeviceSynchronize());
        const Traffic setup = traffic;
        printf("READY width=%d height=%d allocations=%zu\n", width, height, setup.allocations);
        fflush(stdout);
        std::string request;
        std::vector<float> host;
        while (std::getline(std::cin, request)) {
            if (!request.empty() && request.back() == '\r') request.pop_back();
            if (request.empty()) continue;
            if (request == "QUIT") break;
            struct Input { const char* name; Buffer* buffer; } inputs[] = {
                {"c64_input",&c64_input},{"skip4",&skip4},{"skip0",&skip0},{"color",&color}};
            bool valid = true;
            for (const auto& item : inputs) {
                if (!load(request+"/"+item.name+".f32", host, item.buffer->count)) {
                    printf("ERR missing, wrong-size or nonfinite %s\n", item.name);
                    valid = false;
                    break;
                }
                HIP_CHECK(hipMemcpy(item.buffer->data, host.data(), host.size()*sizeof(float), hipMemcpyHostToDevice));
            }
            if (!valid) { fflush(stdout); continue; }
            const Traffic before = traffic;
            HIP_CHECK(hipEventRecord(start));
            // Skips stay borrowed on the device until their decoder prefixes consume them.
            bool ok = encoder8.run_from_device({c64_input.data,8*n}) &&
                down8.run_from_device(encoder8.raw_view()) &&
                encoder14.run_from_device(down8.final_view()) &&
                down14.run_from_device(encoder14.raw_view()) &&
                encoder22.run_from_device(down14.final_view()) &&
                down22.run_from_device(encoder22.raw_view()) &&
                chain.run_from_device(down22.final_view());
            if (ok) {
                auto head = chain.head_view();
                hipLaunchKernelGGL(k_vit_bridge_gather, dim3((hn+255)/256), dim3(256), 0, 0,
                                   head.data, map.data, bridge.data, int(hn));
                HIP_CHECK(hipGetLastError());
                ok = vit.run_from_device({bridge.data,hn}) &&
                    entry.run_from_device(vit.final_view(), chain.final_view()) &&
                    decoder.run_from_device(entry.final_view()) &&
                    prefix48.run_from_device(decoder.final_view(), encoder22.final_view()) &&
                    c256.run_from_device(prefix48.final_view()) &&
                    prefix56.run_from_device(c256.final_view(), encoder14.final_view()) &&
                    c128.run_from_device(prefix56.final_view()) &&
                    prefix62.run_from_device(c128.final_view(), encoder8.final_view()) &&
                    c64.run_from_device(prefix62.final_view()) &&
                    prefix66.run_from_device(c64.final_view(), {skip4.data,16*n}) &&
                    c32.run_from_device(prefix66.final_view()) &&
                    head70.run_from_device(c32.final_view(), {skip0.data,64*n}, {color.data,6*n});
            }
            if (!ok) { printf("ERR resident execution rejected a view\n"); fflush(stdout); continue; }
            consume(head70.final_view(), rgb_native);
            consume(head70.public_view(), rgb_public);
            HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
            float elapsed = 0;
            HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
            if (traffic.allocations != before.allocations || traffic.h2d_bytes != before.h2d_bytes ||
                traffic.d2h_bytes != before.d2h_bytes) {
                printf("ERR host transfer or allocation inside resident execution\n");
                fflush(stdout);
                continue;
            }
            if (!store(request+"/rgb_native.f32", rgb_native) || !store(request+"/rgb_public.f32", rgb_public)) {
                printf("ERR cannot write RGB outputs\n");
                fflush(stdout);
                continue;
            }
            printf("OK %.6f\n", elapsed);
            fflush(stdout);
        }
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    } catch (const std::exception& error) {
        fprintf(stderr, "resident setup failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
