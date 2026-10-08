// Runs the resident network on one frame with the original's input/output
// contract, for comparison with NVIDIA's own output (tools/nvidia_parity.py):
// the 8-bit colour as is (display-encoded, as the DLL is fed), the controls
// as the NGX settings map them, the stem noise seed, the frame at its own
// resolution padded to the network extent; writes the head's output.
// Usage: network_parity config.txt in.rgb out.f32 [style tone structure skin automask seed extent]
//   in.rgb: uint32 width, height, then width*height RGB bytes
//   out.f32: uint32 width, height, then width*height*3 floats (the head output, display domain)
//   extent: compact (default) or legacy
#include <array>
#include <functional>
#include "resident_engine.hip"

int main(int argc, char** argv) {
    if (argc < 4) {
        fputs("usage: network_parity config.txt in.rgb out.f32 [style tone structure skin automask seed extent]\n", stderr);
        return 2;
    }
    const int style = argc > 4 ? atoi(argv[4]) : 0;
    const float tone = argc > 5 ? float(atof(argv[5])) : 1.0f;
    const float structure = argc > 6 ? float(atof(argv[6])) : 1.0f;
    const float skin = argc > 7 ? float(atof(argv[7])) : -1.0f;
    const bool automask = argc > 8 ? atoi(argv[8]) != 0 : true;
    const unsigned seed = argc > 9 ? unsigned(atoi(argv[9])) : 0u;
    const bool compact = argc > 10 ? std::string(argv[10]) != "legacy" : true;
    std::vector<std::string> config;
    {
        std::ifstream f(argv[1]);
        std::string line;
        while (std::getline(f, line) && config.size() < 24) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            config.push_back(line);
        }
    }
    if (config.size() < 24) { fputs("config needs 24 path lines\n", stderr); return 1; }
    FILE* f = fopen(argv[2], "rb");
    if (!f) return 1;
    unsigned dims[2];
    if (fread(dims, 4, 2, f) != 2) return 1;
    const int fw = int(dims[0]), fh = int(dims[1]);
    std::vector<unsigned char> rgb(size_t(fw)*fh*3), rgba(size_t(fw)*fh*4);
    if (fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) return 1;
    fclose(f);
    for (size_t i = 0; i < size_t(fw)*fh; ++i) {
        rgba[i*4] = rgb[i*3]; rgba[i*4+1] = rgb[i*3+1]; rgba[i*4+2] = rgb[i*3+2]; rgba[i*4+3] = 255;
    }
    // The NGX controls as the original maps them (peer DLSSNR-AMD, nr_graph.cpp):
    // the automatic mask with no external mask texture turns structure/skin/other on.
    const float mask_skin = automask ? (skin < 0.0f ? structure : skin) : -1.0f;
    const float mask_other = automask ? structure : -1.0f;
    const bool any_mask = std::max(mask_skin, mask_other) >= 0.0f;
    const float controls[5] = {
        float(style)/128.0f, tone, any_mask ? 1.0f : structure,
        any_mask ? (mask_skin < 0.0f ? structure : mask_skin) : -1.0f,
        any_mask ? (mask_other < 0.0f ? structure : mask_other) : -1.0f};
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    c512_resident::stream = stream;
    int W = 256, H = 256;
    network_extent(fw, fh, W, H, compact);
    printf("frame %dx%d, network extent %dx%d (%s); controls [%g %g %g %g %g], seed %u\n", fw, fh, W, H,
           compact ? "compact" : "legacy", controls[0], controls[1], controls[2], controls[3], controls[4], seed);
    Network net(config, W, H);
    for (int k = 0; k < 5; ++k) net.front->controls.v[k] = controls[k];
    net.front->noise_seed = seed;
    net.front->noise_function = getenv("PARITY_NOISE_FN") && *getenv("PARITY_NOISE_FN") == '1';
    void* dev = nullptr;
    HIP_CHECK(hipMalloc(&dev, rgba.size()));
    HIP_CHECK(hipMemcpy(dev, rgba.data(), rgba.size(), hipMemcpyHostToDevice));
    // encoded = 1: the network sees the display-encoded values, as the original.
    net.run(reinterpret_cast<const unsigned char*>(dev), fw, fh, fw*4, 4, false, stream, fw, fh, nullptr, 1);
    HIP_CHECK(hipStreamSynchronize(stream));
    // PARITY_DUMP=dir: every stage's activations as raw floats (dir/NN_name.f32),
    // in this implementation's own layout, for comparison with a reference.
    if (const char* dir = getenv("PARITY_DUMP")) {
        struct Stage { const char* name; DeviceTensor t; };
        const Stage stages[] = {
            {"00_input", {net.color->data, net.color->count}}, {"01_front_c64", net.front->c64_view()},
            {"02_skip4", net.front->skip4_view()}, {"03_skip0", net.front->skip0_view()},
            {"04_encoder8", net.encoder8->final_view()}, {"05_down8", net.down8->final_view()},
            {"06_encoder14", net.encoder14->final_view()}, {"07_down14", net.down14->final_view()},
            {"08_encoder22", net.encoder22->final_view()}, {"09_down22", net.down22->final_view()},
            {"10_chain", net.chain->final_view()}, {"11_vit", net.vit->final_view()},
            {"12_entry", net.entry->final_view()}, {"13_decoder", net.decoder->final_view()},
            {"14_c256", net.c256->final_view()}, {"15_c128", net.c128->final_view()},
            {"16_c64", net.c64->final_view()}, {"17_c32", net.c32->final_view()},
            {"18_head", {net.rgb_native->data, net.rgb_native->count}}};
        for (const Stage& st : stages) {
            std::vector<float> v(st.t.count);
            if (st.t.data && st.t.count) HIP_CHECK(hipMemcpy(v.data(), st.t.data, v.size()*sizeof(float), hipMemcpyDeviceToHost));
            const std::string path = std::string(dir) + "/" + st.name + ".f32";
            if (FILE* d = fopen(path.c_str(), "wb")) { fwrite(v.data(), 4, v.size(), d); fclose(d); }
            printf("dump %s: %zu floats\n", st.name, v.size());
        }
    }
    std::vector<float> out(size_t(W)*H*3);
    HIP_CHECK(hipMemcpy(out.data(), net.rgb_native->data, out.size()*sizeof(float), hipMemcpyDeviceToHost));
    FILE* o = fopen(argv[3], "wb");
    if (!o) return 1;
    const unsigned odims[2] = {unsigned(fw), unsigned(fh)};
    fwrite(odims, 4, 2, o);
    for (int y = 0; y < fh; ++y) fwrite(out.data() + size_t(y)*W*3, sizeof(float), size_t(fw)*3, o);
    fclose(o);
    return 0;
}
