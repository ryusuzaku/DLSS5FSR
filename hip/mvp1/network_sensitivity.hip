// Where does the network turn a tiny input change into a full-size output
// change? Runs the resident network (as the engine's luma path does) on a
// capture and on the same capture with a tiny perturbation, and prints, per
// stage, how much the activations differ: relative mean |a-b| / mean |a| and
// the fraction of elements that differ at all.
// Usage: network_sensitivity config.txt capture.bin [amplitude [patch]]
//   amplitude: uniform noise added to the colour (display units, default 0.0005)
//   patch: 0 = the whole frame (default), N = only an N x N patch at (20, 20)
#include <array>
#include <functional>
#include "resident_engine.hip"

static std::vector<float> host(DeviceTensor t) {
    std::vector<float> v(t.count);
    if (t.data && t.count) HIP_CHECK(hipMemcpy(v.data(), t.data, t.count*sizeof(float), hipMemcpyDeviceToHost));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fputs("usage: network_sensitivity config.txt capture.bin [amplitude [patch]]\n", stderr); return 2; }
    const float amp = argc > 3 ? float(atof(argv[3])) : 0.0005f;
    const int patch = argc > 4 ? atoi(argv[4]) : 0;
    std::vector<std::string> config;
    {
        std::ifstream f(argv[1]);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            config.push_back(line);
        }
    }
    if (config.size() < 24) { fputs("config needs 24 path lines\n", stderr); return 1; }
    float scale = 1.0f, budget = 0.0f;
    int compact = 0;
    std::array<float, 5> controls{0, 0, 0, 0, 0};
    for (size_t i = 24; i < config.size(); ++i) {
        const std::string& l = config[i];
        const size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = l.substr(0, eq), v = l.substr(eq + 1);
        if (k == "scale") scale = std::stof(v);
        else if (k == "budget") budget = std::stof(v);
        else if (k == "extent") compact = v == "compact";
        else if (k == "controls") sscanf(v.c_str(), "%f,%f,%f,%f,%f", &controls[0], &controls[1], &controls[2], &controls[3], &controls[4]);
    }
    std::vector<unsigned char> cap;
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) return 1;
        unsigned char buf[1 << 16];
        size_t got;
        while ((got = fread(buf, 1, sizeof(buf), f)) > 0) cap.insert(cap.end(), buf, buf + got);
        fclose(f);
    }
    unsigned int hdr[4];
    memcpy(hdr, cap.data() + 8, sizeof(hdr));
    const int fw = int(hdr[0]), fh = int(hdr[1]), bpp = int(hdr[2]), pitch = int(hdr[3]);
    if (bpp != 8) { fputs("RGBA16F capture needed\n", stderr); return 1; }
    std::vector<unsigned char> a(cap.begin() + 36, cap.begin() + 36 + size_t(pitch)*fh), b = a;
    unsigned s = 12345u;
    for (int y = 0; y < fh; ++y)
        for (int x = 0; x < fw; ++x) {
            if (patch && !(x >= 20 && x < 20 + patch && y >= 20 && y < 20 + patch)) continue;
            for (int c = 0; c < 3; ++c) {
                _Float16* v = reinterpret_cast<_Float16*>(&b[size_t(y)*pitch + size_t(x)*8 + c*2]);
                s = s*1664525u + 1013904223u;
                const float u = float(s >> 8) / 16777216.0f - 0.5f;
                *v = (_Float16)std::max(0.0f, float(*v) + 2.0f*amp*u);
            }
        }
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    c512_resident::stream = stream;
    float sc = scale;
    if (budget > 0.0f) sc = std::min(sc, std::sqrt(budget/(float(fw)*float(fh))));
    int sw = fw, sh = fh;
    if (sc < 1.0f) { sw = std::max(64, int(fw*sc + 0.5f)); sh = std::max(64, int(fh*sc + 0.5f)); }
    int W = 256, H = 256;
    network_extent(sw, sh, W, H, compact);
    printf("frame %dx%d -> network content %dx%d, extent %dx%d; noise %g %s\n", fw, fh, sw, sh, W, H, amp,
           patch ? "in a patch" : "everywhere");
    Network net(config, W, H);
    for (int k = 0; k < 5; ++k) net.front->controls.v[k] = controls[k];
    void* dev = nullptr;
    HIP_CHECK(hipMalloc(&dev, a.size()));
    struct Stage { const char* name; std::function<DeviceTensor()> view; };
    std::vector<Stage> stages = {
        {"input (network colour)", [&] { return DeviceTensor{net.color->data, net.color->count}; }},
        {"front c64", [&] { return net.front->c64_view(); }},
        {"encoder8", [&] { return net.encoder8->final_view(); }},
        {"down8", [&] { return net.down8->final_view(); }},
        {"encoder14", [&] { return net.encoder14->final_view(); }},
        {"down14", [&] { return net.down14->final_view(); }},
        {"encoder22", [&] { return net.encoder22->final_view(); }},
        {"down22", [&] { return net.down22->final_view(); }},
        {"chain (C512)", [&] { return net.chain->final_view(); }},
        {"vit", [&] { return net.vit->final_view(); }},
        {"entry", [&] { return net.entry->final_view(); }},
        {"decoder (C512)", [&] { return net.decoder->final_view(); }},
        {"c256", [&] { return net.c256->final_view(); }},
        {"c128", [&] { return net.c128->final_view(); }},
        {"c64", [&] { return net.c64->final_view(); }},
        {"c32", [&] { return net.c32->final_view(); }},
        {"head (rgb native)", [&] { return DeviceTensor{net.rgb_native->data, net.rgb_native->count}; }},
    };
    std::vector<std::vector<float>> ra, rb;
    for (int run = 0; run < 3; ++run) {
        const auto& src = run == 1 ? b : a;  // a, b, a again (determinism check)
        HIP_CHECK(hipMemcpy(dev, src.data(), src.size(), hipMemcpyHostToDevice));
        net.run(reinterpret_cast<const unsigned char*>(dev), fw, fh, pitch, bpp, false, stream, sw, sh, nullptr, 1);
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<std::vector<float>> r;
        for (auto& st : stages) r.push_back(host(st.view()));
        if (run == 0) ra = r;
        else if (run == 1) rb = r;
        else {
            size_t same = 0;
            for (size_t k = 0; k < r.size(); ++k) same += r[k] == ra[k];
            printf("determinism: %zu of %zu stages identical on a repeated run\n", same, r.size());
        }
    }
    printf("%-24s %12s %12s %10s\n", "stage", "rel |a-b|", "differing", "elements");
    for (size_t k = 0; k < stages.size(); ++k) {
        const auto& x = ra[k];
        const auto& y = rb[k];
        double sa = 0, sd = 0;
        size_t diff = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            sa += std::fabs(x[i]);
            const double d = std::fabs(double(x[i]) - y[i]);
            sd += d;
            diff += d > 0;
        }
        printf("%-24s %12.3e %11.1f%% %10zu\n", stages[k].name, sa > 0 ? sd/sa : 0.0,
               x.empty() ? 0.0 : 100.0*diff/x.size(), x.size());
    }
    return 0;
}
