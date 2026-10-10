// The original's temporal pipeline around the resident network, over a
// frame sequence (tools/nvidia_sequence.py): for comparison with NVIDIA's own
// behaviour on the peer DLSSNR-AMD's moving sequences.
//
// Per frame k, as the original (peer DLSSNR-AMD fswin_t.comp NR_TEMPORAL_HISTORY,
// image_input.glsl, temporal_history.glsl, transcribed from NVIDIA's PTX):
//   - the stem's noise seed is k (0 on the first frame / a reset);
//   - pre: from k = 1, stem features 7..9 are the previous output sampled at
//     uv + mv, mv read at whichever of the pixel and its four diagonal
//     neighbours is nearest the camera; five-tap Catmull-Rom reconstruction;
//   - post: rgb = clamp(input + 0.25 * net.rgb); from k = 1,
//     w = clamp(sigmoid(net row 3) * 0.7397) and rgb = lerp(rgb, history5(uv + mv), w)
//     with the pixel's own vector; that rgb is the output (Intensity 1) and
//     the next frame's history.
// Usage: network_sequence config.txt frames_dir out_dir [count]
//   frames_dir/NNNN.seq: uint32 w, h; w*h RGB bytes; w*h float2 motion in pixels
//   (previous = current + mv); w*h float depth (reversed-Z).
//   out_dir/NNNN.f32: uint32 w, h; w*h*3 floats (the output, display domain).
#include <array>
#include <functional>
#include "resident_engine.hip"

static bool slurp(const std::string& path, std::vector<unsigned char>& data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END); const long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    data.resize(size_t(n));
    const bool ok = fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 4) { fputs("usage: network_sequence config.txt frames_dir out_dir [count]\n", stderr); return 2; }
    std::vector<std::string> config;
    {
        std::ifstream f(argv[1]);
        std::string line;
        while (std::getline(f, line) && config.size() < 24) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            config.push_back(line);
        }
    }
    const int count = argc > 4 ? atoi(argv[4]) : 10;
    std::vector<float> row(32);
    {
        FILE* f = fopen((config[16] + "/coeff_history.f32").c_str(), "rb");
        if (!f || fread(row.data(), 4, 32, f) != 32) { fputs("needs coeff_history.f32 in the head fixture\n", stderr); return 1; }
        fclose(f);
    }
    // NGX defaults: Style 0, LocalTone 1, LocalStructure 1, Skin -1, automatic mask (no mask texture).
    const float controls[5] = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float blend_scale = 0.7397f;  // the model's history blend scale
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    c512_resident::stream = stream;
    std::unique_ptr<Network> net;
    int fw = 0, fh = 0, W = 0, H = 0;
    float *d_rgb = nullptr, *d_hist = nullptr, *d_out = nullptr, *d_feat = nullptr, *d_depth = nullptr, *d_row = nullptr;
    float2* d_mv = nullptr;
    unsigned char* d_rgba = nullptr;
    std::vector<unsigned char> data;
    for (int k = 0; k < count; ++k) {
        char name[64];
        snprintf(name, sizeof(name), "/%04d.seq", k);
        if (!slurp(std::string(argv[2]) + name, data) || data.size() < 8) { fprintf(stderr, "missing frame %d\n", k); return 1; }
        unsigned dims[2]; memcpy(dims, data.data(), 8);
        if (!net) {
            fw = int(dims[0]); fh = int(dims[1]);
            network_extent(fw, fh, W, H, true);
            net.reset(new Network(config, W, H));
            for (int c = 0; c < 5; ++c) net->front->controls.v[c] = controls[c];
            net->front->noise_function = true;
            HIP_CHECK(hipMalloc(&d_rgba, size_t(fw)*fh*4));
            HIP_CHECK(hipMalloc(&d_mv, size_t(fw)*fh*8));
            HIP_CHECK(hipMalloc(&d_depth, size_t(fw)*fh*4));
            HIP_CHECK(hipMalloc(&d_hist, size_t(fw)*fh*12));
            HIP_CHECK(hipMalloc(&d_out, size_t(fw)*fh*12));
            HIP_CHECK(hipMalloc(&d_feat, size_t(W)*H*12));
            HIP_CHECK(hipMalloc(&d_row, 32*4));
            HIP_CHECK(hipMemcpy(d_row, row.data(), 32*4, hipMemcpyHostToDevice));
            printf("frames %dx%d, network extent %dx%d\n", fw, fh, W, H);
        }
        const size_t px = size_t(fw)*fh;
        if (data.size() != 8 + px*3 + px*8 + px*4) { fprintf(stderr, "frame %d has the wrong size\n", k); return 1; }
        std::vector<unsigned char> rgba(px*4);
        for (size_t i = 0; i < px; ++i) {
            rgba[i*4] = data[8+i*3]; rgba[i*4+1] = data[8+i*3+1]; rgba[i*4+2] = data[8+i*3+2]; rgba[i*4+3] = 255;
        }
        HIP_CHECK(hipMemcpy(d_rgba, rgba.data(), rgba.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_mv, data.data() + 8 + px*3, px*8, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_depth, data.data() + 8 + px*11, px*4, hipMemcpyHostToDevice));
        const bool temporal = k > 0;
        net->front->noise_seed = unsigned(k);
        if (temporal) {
            hipLaunchKernelGGL(k_seq_pre, dim3((W*H+255)/256), dim3(256), 0, stream, d_hist, d_mv, d_depth, d_feat,
                               fw, fh, W, H);
            HIP_CHECK(hipGetLastError());
        }
        net->front->history = temporal ? d_feat : nullptr;
        net->run(d_rgba, fw, fh, fw*4, 4, false, stream, fw, fh, nullptr, 1);
        hipLaunchKernelGGL(k_seq_post, dim3((fw*fh+255)/256), dim3(256), 0, stream,
                           net->rgb_native->data, net->head70->native_view().data, d_row, d_hist, d_mv, d_out,
                           fw, fh, W, blend_scale, int(temporal));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpyAsync(d_hist, d_out, px*12, hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<float> out(px*3);
        HIP_CHECK(hipMemcpy(out.data(), d_out, out.size()*4, hipMemcpyDeviceToHost));
        snprintf(name, sizeof(name), "/%04d.f32", k);
        FILE* o = fopen((std::string(argv[3]) + name).c_str(), "wb");
        if (!o) return 1;
        const unsigned od[2] = {unsigned(fw), unsigned(fh)};
        fwrite(od, 4, 2, o); fwrite(out.data(), 4, out.size(), o); fclose(o);
    }
    printf("%d frames done\n", count);
    return 0;
}
