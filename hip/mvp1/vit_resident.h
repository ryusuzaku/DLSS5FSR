#pragma once
// Persistent ViT31-38 workspace; arithmetic stays in the standalone kernels.
#include "split512_resident.h"
#include "vit_expand_chain.hip"
#include "tiled_gemm.hip"
#include "vit_attention_wmma.hip"
#include <stdexcept>

namespace vit_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::traffic;

struct Weights {
    std::string dir;
    Buffer expand, contract, skip, qkv, scales, projection, projection_skip;
    explicit Weights(const std::string& folder) : dir(folder),
        expand(dir, "weights", 4096*1024), contract(dir, "contract_weights", 1024*4096),
        skip(dir, "contract_skip", 1024), qkv(dir, "qkv_weights", 3*1024*1024),
        scales(dir, "qkv_scales", 32), projection(dir, "projection_weights", 1024*1024),
        projection_skip(dir, "projection_skip", 1024) {}
};

#define VIT_LAUNCH(kernel, count, ...) do { \
    hipLaunchKernelGGL(kernel, dim3(((count)+255)/256), dim3(256), 0, c512_resident::stream, __VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); if (c512_resident::launch_hook) c512_resident::launch_hook(#kernel); } while (0)

class Chain {
    static int checked_tokens(int tokens, size_t blocks) {
        if ((tokens != 16 && (tokens <= 0 || tokens % 64)) || blocks != 8)
            throw std::invalid_argument("resident ViT requires 16 or a multiple of 64 tokens and eight blocks");
        return tokens;
    }
    int tokens, valid;  // valid < tokens: padding tokens, masked as keys (fast path only)
    size_t n, sn;
    Buffer ping, pong, expanded, hidden, contract, projected, qkv, scores, exponents, attention;
    std::vector<std::unique_ptr<Weights>> weights;
    float* result = nullptr;
    // E4M3 byte copies of the FP8 activations between the GEMMs (S336).
    Buffer hidden8, contract8, ping8, pong8, kv8, attention8;
    // RESIDENT_ATTN_T=0: k_vit_attention_fused instead of k_vit_kv8 + k_vit_attention_t.
    static inline const bool attention_t = [] { const char* v = getenv("RESIDENT_ATTN_T"); return !(v && *v == '0'); }();
    static unsigned char* bytes(Buffer& b) { return reinterpret_cast<unsigned char*>(b.data); }

    bool check(const std::string& dir, const char* name, float* device, size_t count, bool verify) {
        if (!verify) return true;
        ++comparisons;
        traffic.d2h_bytes += count*sizeof(float);
        return compare((dir + ": " + name).c_str(), device, read(dir, name, count));
    }

public:
    size_t comparisons = 0;
    Chain(int count, const std::vector<std::string>& blocks, int valid_tokens = 0) :
        tokens(checked_tokens(count, blocks.size())), valid(valid_tokens > 0 ? valid_tokens : count),
        n(size_t(tokens)*1024),
        sn(size_t(32)*tokens*tokens), ping(n), pong(n), expanded(4*n), hidden(4*n),
        contract(n), projected(3*n), qkv(3*n), scores(sn), exponents(sn), attention(n),
        hidden8(n), contract8(n/4), ping8(n/4), pong8(n/4), kv8(n/2), attention8(n/4) {
        for (const auto& dir : blocks) weights.emplace_back(new Weights(dir));
    }

    // Borrowed same-device/default-stream input. Output lasts until next submission.
    // verify=true compares only against the construction fixtures, outside timing.
    bool run_from_device(DeviceTensor source, bool verify = false) {
        result = nullptr;
        if (!source.data || source.count != n) return false;
        if (source.data != ping.data) {
            HIP_CHECK(hipMemcpyAsync(ping.data, source.data, n*sizeof(float), hipMemcpyDeviceToDevice,c512_resident::stream));
            traffic.d2d_bytes += n*sizeof(float);
        }
        float *input = ping.data, *output = pong.data;
        const bool b8 = !c512_resident::exact_math && !verify && wmma_gemm::fp8;  // FP8 build only
        unsigned char *input8 = nullptr, *output8 = b8 ? bytes(pong8) : nullptr, *spare8 = b8 ? bytes(ping8) : nullptr;
        for (const auto& owned : weights) {
            const auto& w = *owned;
            if (!check(w.dir, "input", input, n, verify)) return false;
            // Few tokens keep the one-thread-per-output kernels in exact mode (more parallel);
            // larger counts use the tiled, bit-identical forms (4-/2-part partitions).
            const bool small = tokens <= 64 && c512_resident::exact_math;  // the fast GEMMs split K instead
            if (small) {
                VIT_LAUNCH(k_vit_expand, 4*n, input, w.expand.data, expanded.data, hidden.data, tokens);
                VIT_LAUNCH(k_vit_residual_projection, n, hidden.data, input, w.contract.data,
                           w.skip.data, contract.data, tokens, 4096);
                VIT_LAUNCH(k_vit_qkv_projection, 3*n, contract.data, w.qkv.data, projected.data, tokens);
            } else {
                tiled::gemm<tiled::Split,tiled::GATE,false>(c512_resident::stream, input, 1024, w.expand.data, 1024, nullptr, 0, nullptr,
                    verify ? expanded.data : nullptr, hidden.data, 4096, tokens, 4096, 1, 0, 0, 0, 1,
                    input8, b8 ? bytes(hidden8) : nullptr);
                tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream, hidden.data, 4096, w.contract.data, 4096, input, 1024,
                    w.skip.data, contract.data, nullptr, 1024, tokens, 1024, 1, 0, 0, 0, 4,
                    b8 ? bytes(hidden8) : nullptr, b8 ? bytes(contract8) : nullptr);
                tiled::gemm<tiled::Split,tiled::RAW,false>(c512_resident::stream, contract.data, 1024, w.qkv.data, 1024, nullptr, 0, nullptr,
                    projected.data, nullptr, 3072, tokens, 3072, 1, 0, 0, 0, 2, b8 ? bytes(contract8) : nullptr);
                HIP_CHECK(hipGetLastError());
            }
            const bool fused = tokens % 64 == 0 && !c512_resident::exact_math && !verify;
            if (valid < tokens && !fused) return false;  // only the fused attention masks padding
            bool attention_bytes = false;
            if (fused && attention_t && b8) {
                // K/V bytes once, then the transposed attention (S336d).
                hipLaunchKernelGGL(k_vit_kv8, dim3((tokens*32+255)/256), dim3(256), 0, c512_resident::stream,
                                   projected.data, bytes(kv8), bytes(kv8) + n, tokens);
                if (c512_resident::launch_hook) c512_resident::launch_hook("k_vit_kv8");
                hipLaunchKernelGGL(k_vit_attention_t, dim3(tokens/64, 32), dim3(128), 0, c512_resident::stream,
                                   projected.data, bytes(kv8), bytes(kv8) + n, w.scales.data, attention.data,
                                   bytes(attention8), tokens, valid);
                HIP_CHECK(hipGetLastError());
                attention_bytes = true;
                if (c512_resident::launch_hook) c512_resident::launch_hook("k_vit_attention_t");
            } else if (fused) {
                // Normalize, scores, exponents and attention in one launch.
                hipLaunchKernelGGL(k_vit_attention_fused, dim3(tokens/64, 32), dim3(256), 0, c512_resident::stream,
                                   projected.data, w.scales.data, attention.data, tokens, valid);
                HIP_CHECK(hipGetLastError());
                if (c512_resident::launch_hook) c512_resident::launch_hook("k_vit_attention_fused");
            } else {
                VIT_LAUNCH(k_vit_qkv_normalize, 3*n, projected.data, w.scales.data, qkv.data, tokens);
                VIT_LAUNCH(k_vit_scores, sn, qkv.data, scores.data, tokens);
                VIT_LAUNCH(k_vit_exponents, sn, scores.data, exponents.data, int(sn));
            }
            if (fused) {
            } else if (tokens % 64 == 0) {
                VIT_LAUNCH(k_vit_attention_chunks, n, qkv.data, exponents.data, attention.data, tokens);
            } else {
                VIT_LAUNCH(k_vit_attention16_candidate, n, qkv.data, exponents.data, attention.data);
            }
            if (small) {
                VIT_LAUNCH(k_vit_residual_projection, n, attention.data, contract.data,
                           w.projection.data, w.projection_skip.data, output, tokens, 1024);
            } else {
                tiled::gemm<tiled::Split,tiled::FP8,true>(c512_resident::stream, attention.data, 1024, w.projection.data, 1024, contract.data, 1024,
                    w.projection_skip.data, output, nullptr, 1024, tokens, 1024, 1, 0, 0, 0, 4,
                    attention_bytes ? bytes(attention8) : nullptr, output8);
                HIP_CHECK(hipGetLastError());
            }
            if (!check(w.dir, "expanded", expanded.data, 4*n, verify) ||
                !check(w.dir, "hidden", hidden.data, 4*n, verify) ||
                !check(w.dir, "contract", contract.data, n, verify) ||
                !check(w.dir, "qkv_projected", projected.data, 3*n, verify) ||
                !check(w.dir, "qkv", qkv.data, 3*n, verify) ||
                !check(w.dir, "scores", scores.data, sn, verify) ||
                !check(w.dir, "exponents", exponents.data, sn, verify) ||
                !check(w.dir, "attention", attention.data, n, verify) ||
                !check(w.dir, "projection", output, n, verify)) return false;
            std::swap(input, output);
            if (small) input8 = nullptr;
            else { input8 = output8; std::swap(output8, spare8); }
        }
        result = input;
        return true;
    }

    DeviceTensor final_view() const { return result ? DeviceTensor{result, n} : DeviceTensor{}; }
};
#undef VIT_LAUNCH
} // namespace vit_resident
