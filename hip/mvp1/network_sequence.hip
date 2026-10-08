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

// Five-tap Catmull-Rom of `hist` (fw x fh RGB, texel centres at i + 0.5) at
// pos (texel units), taps clamped per axis to [0.5, extent - 0.5], each tap a
// bilinear fetch, normalised by the weight sum: the original's form.
__device__ inline float3 hist_bilinear(const float* h, int fw, int fh, float px, float py) {
    const float fx = px - 0.5f, fy = py - 0.5f;
    const int x0 = int(floorf(fx)), y0 = int(floorf(fy));
    const float tx = fx - x0, ty = fy - y0;
    auto at = [&](int x, int y) {
        x = min(max(x, 0), fw-1); y = min(max(y, 0), fh-1);
        const float* p = h + (size_t(y)*fw + x)*3;
        return make_float3(p[0], p[1], p[2]);
    };
    const float3 a = at(x0, y0), b = at(x0+1, y0), c = at(x0, y0+1), d = at(x0+1, y0+1);
    auto mix = [](float3 u, float3 v, float t) { return make_float3(u.x+(v.x-u.x)*t, u.y+(v.y-u.y)*t, u.z+(v.z-u.z)*t); };
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}
__device__ inline float3 hist_5tap(const float* h, int fw, int fh, float px, float py) {
    const float cx = floorf(px - 0.5f) + 0.5f, cy = floorf(py - 0.5f) + 0.5f;
    const float tx = fminf(fmaxf(px - cx, 0.0f), 1.0f), ty = fminf(fmaxf(py - cy, 0.0f), 1.0f);
    auto w = [](float t, float& w0, float& w1, float& w2, float& w3) {
        const float t2 = t*t, t3 = t2*t;
        w0 = (t + t3) * -0.5f + t2; w1 = 1.5f*t3 - 2.5f*t2 + 1.0f; w3 = 0.5f*(t3 - t2); w2 = 1.0f - w0 - w1 - w3;
    };
    float w0x, w1x, w2x, w3x, w0y, w1y, w2y, w3y;
    w(tx, w0x, w1x, w2x, w3x); w(ty, w0y, w1y, w2y, w3y);
    const float wcx = w1x + w2x, wcy = w1y + w2y;
    auto cl = [](float v, float hi) { return fminf(fmaxf(v, 0.5f), hi); };
    const float hx = fw - 0.5f, hy = fh - 0.5f;
    const float p0x = cl(cx - 1.0f, hx), p0y = cl(cy - 1.0f, hy);
    const float pmx = cl(cx + w2x/wcx, hx), pmy = cl(cy + w2y/wcy, hy);
    const float p2x = cl(cx + 2.0f, hx), p2y = cl(cy + 2.0f, hy);
    const float3 left = hist_bilinear(h, fw, fh, p0x, pmy), up = hist_bilinear(h, fw, fh, pmx, p0y);
    const float3 centre = hist_bilinear(h, fw, fh, pmx, pmy), down = hist_bilinear(h, fw, fh, pmx, p2y);
    const float3 right = hist_bilinear(h, fw, fh, p2x, pmy);
    const float kl = w0x*wcy, ku = w0y*wcx, kc = wcx*wcy, kd = w3y*wcx, kr = w3x*wcy;
    const float s = kl + ku + kc + kd + kr;
    return make_float3((kl*left.x + ku*up.x + kc*centre.x + kd*down.x + kr*right.x)/s,
                       (kl*left.y + ku*up.y + kc*centre.y + kd*down.y + kr*right.y)/s,
                       (kl*left.z + ku*up.z + kc*centre.z + kd*down.z + kr*right.z)/s);
}

// pre: the history features for every network pixel (W x H, the frame mirrored
// past its edge as the colour is).
__global__ void k_seq_pre(const float* hist, const float2* mv, const float* depth, float* out,
                          int fw, int fh, int W, int H) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= W*H) return;
    const int x = i % W, y = i / W;
    const int sx = max(0, x < fw ? x : 2*fw - x - 2), sy = max(0, y < fh ? y : 2*fh - y - 2);
    int mx = sx, my = sy;
    float best = depth[size_t(sy)*fw + sx];
    for (int dy = -1; dy <= 1; dy += 2)
        for (int dx = -1; dx <= 1; dx += 2) {
            const int qx = min(max(sx + dx, 0), fw-1), qy = min(max(sy + dy, 0), fh-1);
            const float d = depth[size_t(qy)*fw + qx];
            if (d > best) { best = d; mx = qx; my = qy; }  // reversed-Z: larger is nearer
        }
    const float2 v = mv[size_t(my)*fw + mx];
    const float3 h = hist_5tap(hist, fw, fh, sx + 0.5f + v.x, sy + 0.5f + v.y);
    out[size_t(i)*3] = h.x; out[size_t(i)*3+1] = h.y; out[size_t(i)*3+2] = h.z;
}

// post: blend and store. rgb_native is clamp(input + 0.25 * net.rgb) (W stride);
// tiles the head's C32 features (8x8 window order, W wide); row the projection's
// fourth row (the head's coefficient row 6).
__global__ void k_seq_post(const float* rgb_native, const float* tiles, const float* row, const float* hist,
                           const float2* mv, float* out, int fw, int fh, int W, float scale, int temporal) {
    const int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= fw*fh) return;
    const int x = p % fw, y = p / fw;
    const float* r = rgb_native + (size_t(y)*W + x)*3;
    float3 rgb = make_float3(r[0], r[1], r[2]);
    if (temporal) {
        const float* t = tiles + size_t(((y/8)*(W/8) + x/8)*64 + (y%8)*8 + x%8)*32;
        float o = 0.0f;
        for (int k = 0; k < 32; ++k) o += t[k]*row[k];
        const float w = fminf(fmaxf(scale / (1.0f + expf(-o)), 0.0f), 1.0f);
        const float2 v = mv[p];
        const float3 h = hist_5tap(hist, fw, fh, x + 0.5f + v.x, y + 0.5f + v.y);
        rgb = make_float3(rgb.x + (h.x - rgb.x)*w, rgb.y + (h.y - rgb.y)*w, rgb.z + (h.z - rgb.z)*w);
    }
    out[size_t(p)*3] = rgb.x; out[size_t(p)*3+1] = rgb.y; out[size_t(p)*3+2] = rgb.z;
}

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
