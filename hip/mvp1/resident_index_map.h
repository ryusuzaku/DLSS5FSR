#pragma once
#include "split512_resident.h"
#include <cstdint>

namespace c512_resident {
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
    IndexMap(const IndexMap&) = delete;
    IndexMap& operator=(const IndexMap&) = delete;
    ~IndexMap() { HIP_CHECK(hipFree(data)); }
};
} // namespace c512_resident
