#pragma once
// Candidate inverse bridge and block39; no arithmetic changes to the kernels.
#include "resident_index_map.h"
#include "decoder39_entry.hip"
#include <stdexcept>

namespace decoder39_resident {
using c512_resident::Buffer;
using c512_resident::DeviceTensor;
using c512_resident::IndexMap;
using c512_resident::traffic;

class Entry {
    static int checked_width(int width, int height, bool mapped = false) {
        if (width <= 0 || height <= 0 || (!mapped && (width*height) % 16))
            throw std::invalid_argument("resident decoder39 needs a token count that is a multiple of 16");
        return width;
    }
    int width, height, tokens;
    size_t n, skip_n;
    std::string dir;
    IndexMap inverse;
    Buffer weights, scale, main, projected, output;
    bool ready = false;

    bool check(const char* name, const float* device, size_t count, bool verify) {
        if (!verify) return true;
        ++comparisons;
        traffic.d2h_bytes += count*sizeof(float);
        return compare((dir + ": " + name).c_str(), const_cast<float*>(device), read(dir, name, count));
    }

public:
    size_t comparisons = 0;
    Entry(int w, int h, const std::string& fixture) :
        width(checked_width(w, h)), height(h), tokens(width*height), n(size_t(tokens)*1024),
        skip_n(size_t(tokens)*4*512), dir(fixture), inverse((dir+"/inverse.i32").c_str(), n),
        weights(dir, "weights", 512*1024), scale(dir, "scale", 512),
        main(n), projected(size_t(tokens)*512), output(skip_n) {}
    // Weights from the fixture, inverse map computed for this extent.
    Entry(int w, int h, const std::string& fixture, const std::vector<int32_t>& inverse_map) :
        width(checked_width(w, h, true)), height(h), tokens(width*height), n(size_t(tokens)*1024),
        skip_n(size_t(tokens)*4*512), dir(fixture), inverse(inverse_map),
        weights(dir, "weights", 512*1024), scale(dir, "scale", 512),
        main(n), projected(size_t(tokens)*512), output(skip_n) {
        if (inverse_map.size() != n) throw std::invalid_argument("decoder39 inverse map extent differs");
    }

    // Both borrowed inputs belong to the same frame and HIP default stream.
    // Caller retains ownership; no input is overwritten. Output lasts to next run.
    bool run_from_device(DeviceTensor vit, DeviceTensor skip, bool verify = false) {
        ready = false;
        if (!vit.data || vit.count != n || !skip.data || skip.count != skip_n) return false;
        if (!check("vit", vit.data, n, verify) || !check("skip", skip.data, skip_n, verify)) return false;
        hipLaunchKernelGGL(k_decoder39_inverse, dim3((n+255)/256), dim3(256), 0, c512_resident::stream,
                           vit.data, inverse.data, main.data, int(n));
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(k_decoder39_project, dim3((projected.count+255)/256), dim3(256), 0, c512_resident::stream,
                           main.data, weights.data, projected.data, tokens);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(k_decoder39_upsample_skip, dim3((skip_n+255)/256), dim3(256), 0, c512_resident::stream,
                           projected.data, skip.data, scale.data, output.data, width, height);
        HIP_CHECK(hipGetLastError());
        if (!check("main", main.data, n, verify) ||
            !check("projected", projected.data, projected.count, verify) ||
            !check("output", output.data, skip_n, verify)) return false;
        ready = true;
        return true;
    }

    DeviceTensor main_view() const { return ready ? DeviceTensor{main.data, n} : DeviceTensor{}; }
    DeviceTensor projected_view() const {
        return ready ? DeviceTensor{projected.data, projected.count} : DeviceTensor{};
    }
    DeviceTensor final_view() const { return ready ? DeviceTensor{output.data, skip_n} : DeviceTensor{}; }
};
} // namespace decoder39_resident
