#pragma once
// Host-side persistent C512 chain. Arithmetic remains in existing kernels.
#include "head70_test_common.h"
#include "split512_attention.hip"
#include "split512_window.hip"
#include "split512_bridge.hip"
#include "resident_stream.h"
#include "tiled_gemm.hip"
#include <memory>
#include <utility>

namespace c512_resident {
struct Traffic {
    size_t allocations = 0, h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
};
inline Traffic traffic;


// Borrowed default-stream device view; it never owns or frees its pointer.
struct DeviceTensor {
    const float* data = nullptr;
    size_t count = 0;
};

struct Buffer {
    float* data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        HIP_CHECK(hipMalloc(&data, n * sizeof(float)));
        ++traffic.allocations;
    }
    Buffer(const std::string& dir, const char* name, size_t n) : Buffer(n) {
        auto values = read(dir, name, n);
        HIP_CHECK(hipMemcpy(data, values.data(), n * sizeof(float), hipMemcpyHostToDevice));
        traffic.h2d_bytes += n * sizeof(float);
    }
    ~Buffer() { if (data) (void)hipFree(data); }  // destructors never throw or exit
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Weights {
    std::string dir;
    int shift;
    Buffer matrix, expand, contract, projection, skip, qkv, scales, bias, final, final_skip;
    Weights(const std::string& folder, int s) : dir(folder), shift(s),
        matrix(dir, "matrix", 512*512), expand(dir, "expand", 8*256*64),
        contract(dir, "contract", 8*64*256), projection(dir, "ffn_projection", 512*512),
        skip(dir, "ffn_skip", 512), qkv(dir, "qkv_weights", 3*512*512),
        scales(dir, "scales", 16), bias(dir, "bias", 16*4096),
        final(dir, "final_weights", 512*512), final_skip(dir, "final_skip", 512) {}
};

#define C512_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+255)/256), dim3(256), 0, c512_resident::stream, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while (0)

class Chain {
    int width, height, tokens, max_window_tokens;
    size_t n;
    std::unique_ptr<Buffer> seed;  // standalone replay input, loaded on first run()
    std::string seed_dir;
    Buffer ping, pong, pre, mixed, hidden, branch, feature, window, qkv,
           normalized, scores, exponents, probabilities, inverse, context, crop;
    std::vector<std::unique_ptr<Weights>> weights;
    std::unique_ptr<Buffer> head_weights, pooled, head_output;
    std::string head_dir;
    float* result = nullptr;
    Buffer raw_output;

    bool check(const std::string& dir, const char* name, float* device, size_t count, bool verify) {
        if (!verify) return true;
        ++comparisons;
        traffic.d2h_bytes += count * sizeof(float);
        auto expected = read(dir, name, count);
        return compare((dir + ": " + name).c_str(), device, expected);
    }

public:
    size_t comparisons = 0;
    Chain(int w, int h, const std::vector<std::pair<std::string, int>>& blocks,
          const std::string& head = "") :
        width(w), height(h), tokens(w*h),
        max_window_tokens(((w+11)/8)*8*((h+11)/8)*8), n(size_t(tokens)*512),
        seed_dir(blocks.front().first), ping(n), pong(n), pre(n), mixed(n),
        hidden(size_t(tokens)*2048), branch(n), feature(n), window(size_t(max_window_tokens)*512),
        qkv(size_t(max_window_tokens)*1536), normalized(size_t(max_window_tokens)*1536),
        scores(size_t(max_window_tokens)*1024), exponents(size_t(max_window_tokens)*1024),
        probabilities(size_t(max_window_tokens)*1024), inverse(size_t(max_window_tokens)*32),
        context(size_t(max_window_tokens)*512),
        crop(n), head_dir(head), raw_output(n) {
        for (const auto& block : blocks)
            weights.emplace_back(new Weights(block.first, block.second));
        if (!head.empty()) {
            head_weights.reset(new Buffer(head, "weights", 1024*512));
            pooled.reset(new Buffer(size_t(tokens/4)*512));
            head_output.reset(new Buffer(size_t(tokens/4)*1024));
        }
    }

    bool run(bool verify) {
        if (!seed) seed.reset(new Buffer(seed_dir, "input", n));
        return run_from_device({seed->data, n}, verify);
    }

    // Input must remain valid through the ordered copy on this HIP device's
    // default stream. Outputs are borrowed until the next run or destruction.
    // The copy isolates caller-owned input from ping-pong writes.
    bool run_from_device(DeviceTensor source, bool verify = false) {
        result = nullptr;
        if (!source.data || source.count != n) return false;
        if (source.data != ping.data) {
            HIP_CHECK(hipMemcpyAsync(ping.data, source.data, n*sizeof(float), hipMemcpyDeviceToDevice,c512_resident::stream));
            traffic.d2d_bytes += n*sizeof(float);
        }
        float* input = ping.data;
        float* output = pong.data;
        for (const auto& entry : weights) {
            const Weights& w = *entry;
            int px = (w.shift & 1) ? 4 : 0, py = (w.shift & 2) ? 4 : 0;
            int wt = ((width+px+7)/8)*8*((height+py+7)/8)*8, windows = wt/64;
            size_t wn = size_t(wt)*512, qn = size_t(wt)*1536, sn = size_t(windows)*16*4096;
            // This is a readback assertion only. No later input fixture is uploaded.
            if (!check(w.dir, "input", input, n, verify)) return false;
            // Few tokens keep the one-thread-per-output kernels; larger extents tile.
            const bool small = tokens <= 64;
            if (small) {
                C512_LAUNCH(k_split512_pre, n, input, w.matrix.data, pre.data, tokens);
            } else {
                tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream, input, 512, w.matrix.data, 512, nullptr, 0, nullptr,
                    pre.data, nullptr, 512, tokens, 512);
            }
            if (!check(w.dir, "expected", pre.data, n, verify)) return false;
            C512_LAUNCH(k_split512_quant, n, pre.data, mixed.data, int(n));
            // Eight groups: 64 -> 256 expand with gate, 256 -> 64 contract.
            if (small) {
                C512_LAUNCH(k_split512_expand, size_t(tokens)*2048, mixed.data, w.expand.data, hidden.data, tokens);
                C512_LAUNCH(k_split512_contract, n, hidden.data, w.contract.data, branch.data, tokens);
            } else {
                tiled::gemm<tiled::Split,tiled::GATE,false>(c512_resident::stream, mixed.data, 512, w.expand.data, 64, nullptr, 0, nullptr,
                    nullptr, hidden.data, 2048, tokens, 256, 8, 64, 256*64, 256);
                tiled::gemm<tiled::Split,tiled::FP8,false>(c512_resident::stream, hidden.data, 2048, w.contract.data, 256, nullptr, 0, nullptr,
                    branch.data, nullptr, 512, tokens, 64, 8, 256, 64*256, 64);
                HIP_CHECK(hipGetLastError());
            }
            if (!check(w.dir, "branch", branch.data, n, verify)) return false;
            if (small) {
                C512_LAUNCH(k_split512_ffn_projection, n, branch.data, input, w.projection.data,
                            w.skip.data, nullptr, feature.data, tokens);
            } else {
                tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream, branch.data, 512, w.projection.data, 512, input, 512,
                    w.skip.data, feature.data, nullptr, 512, tokens, 512);
            }
            if (!check(w.dir, "feature", feature.data, n, verify)) return false;
            C512_LAUNCH(k_split512_window_gather, wn, feature.data, window.data, width, height, w.shift);
            if (!check(w.dir, "feature_window", window.data, wn, verify)) return false;
            if (small) {
                C512_LAUNCH(k_split512_qkv, qn, window.data, w.qkv.data, qkv.data, wt);
            } else {
                tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream, window.data, 512, w.qkv.data, 512, nullptr, 0, nullptr,
                    qkv.data, nullptr, 1536, wt, 1536);
            }
            if (!check(w.dir, "qkv", qkv.data, qn, verify)) return false;
            C512_LAUNCH(k_split512_qknorm_inv, size_t(wt)*2*16, qkv.data, inverse.data, wt, 512);
            C512_LAUNCH(k_split512_qknorm_apply, size_t(wt)*3*512, qkv.data, w.scales.data, inverse.data, normalized.data, wt, 512);
            C512_LAUNCH(k_split512_scores, sn, normalized.data, w.bias.data, scores.data, windows);
            C512_LAUNCH(k_split512_exp, sn, scores.data, exponents.data, int(sn));
            C512_LAUNCH(k_split512_inv_rows, sn/64, exponents.data, inverse.data, int(sn/64));
            C512_LAUNCH(k_split512_prob_rows, sn, exponents.data, inverse.data, probabilities.data, int(sn));
            C512_LAUNCH(k_split512_context, wn, probabilities.data, normalized.data, context.data, windows);
            if (!check(w.dir, "normalized", normalized.data, qn, verify) ||
                !check(w.dir, "scores", scores.data, sn, verify) ||
                !check(w.dir, "exponents", exponents.data, sn, verify) ||
                !check(w.dir, "probabilities", probabilities.data, sn, verify) ||
                !check(w.dir, "context", context.data, wn, verify)) return false;
            C512_LAUNCH(k_split512_window_scatter, n, context.data, crop.data, width, height, w.shift);
            if (!check(w.dir, "context_hwc", crop.data, n, verify)) return false;
            if (small) {
                C512_LAUNCH(k_split512_ffn_projection, n, crop.data, feature.data, w.final.data,
                            w.final_skip.data, raw_output.data, output, tokens);
            } else {
                tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream, crop.data, 512, w.final.data, 512, feature.data, 512,
                    w.final_skip.data, output, raw_output.data, 512, tokens, 512);
                HIP_CHECK(hipGetLastError());
            }
            if (!check(w.dir, "final_raw", raw_output.data, n, verify) ||
                !check(w.dir, "final", output, n, verify)) return false;
            std::swap(input, output);
        }
        if (head_weights) {
            if (!check(head_dir, "raw", raw_output.data, n, verify)) return false;
            C512_LAUNCH(k_split512_pool, pooled->count, raw_output.data, pooled->data, width, height);
            C512_LAUNCH(k_split512_head, head_output->count, pooled->data, head_weights->data,
                        head_output->data, tokens/4);
            if (!check(head_dir, "pool", pooled->data, pooled->count, verify) ||
                !check(head_dir, "head", head_output->data, head_output->count, verify)) return false;
        }
        result = input;
        return true;
    }

    DeviceTensor final_view() const { return result ? DeviceTensor{result, n} : DeviceTensor{}; }
    DeviceTensor raw_view() const { return result ? DeviceTensor{raw_output.data, n} : DeviceTensor{}; }
    DeviceTensor head_view() const {
        return result && head_output ? DeviceTensor{head_output->data, head_output->count} : DeviceTensor{};
    }

    bool save(const std::string& directory) {
        if (!result) return false;
        auto write = [&](const char* name, float* data, size_t count) {
            std::vector<float> values(count);
            HIP_CHECK(hipMemcpy(values.data(), data, count*sizeof(float), hipMemcpyDeviceToHost));
            traffic.d2h_bytes += count*sizeof(float);
            for (float value : values) if (!std::isfinite(value)) return false;
            std::string path = directory + "/" + name + ".f32";
            FILE* f = fopen(path.c_str(), "wb");
            if (!f) return false;
            bool ok = fwrite(values.data(), sizeof(float), count, f) == count;
            if (fclose(f)) ok = false;
            return ok;
        };
        return write("final", result, n) && write("final_raw", raw_output.data, n) &&
               (!head_output || write("head", head_output->data, head_output->count));
    }
};
#undef C512_LAUNCH
} // namespace c512_resident
