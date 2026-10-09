#pragma once
// Shared launch state of the resident components.
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <utility>
#include <vector>

namespace c512_resident {
// Stream for every resident launch and device copy; null keeps the harness's
// default-stream behaviour, an engine may select a non-blocking stream.
inline hipStream_t stream = nullptr;
// Optional profiling hook called after every resident launch (null in normal runs).
inline void (*launch_hook)(const char* kernel) = nullptr;
// Fast paths (WMMA etc.) may reorder fp32 sums inside a 32-product slice;
// RESIDENT_EXACT=1 keeps every stage byte-identical to the reference kernels.
inline bool exact_math = [] { const char* v = getenv("RESIDENT_EXACT"); return v && *v == '1'; }();
// Called with a buffer's cache keys (its Lazy object, its device pointer)
// when it is freed: the weight caches evict their copies (S336).
inline std::vector<void(*)(const void*)>& free_hooks() { static std::vector<void(*)(const void*)> h; return h; }
// A GEMM weight: a device pointer, or a lazy buffer (Buffer::Lazy) whose
// host copy can feed a weight cache without a persistent device fp32 copy.
// Converts to the device pointer (allocating and uploading a lazy one).
struct WeightRef {
    const float* ptr = nullptr;
    const void* obj = nullptr;
    const float* (*dev)(const void*) = nullptr;
    const std::vector<float>* (*hostv)(const void*) = nullptr;
    WeightRef(const float* p) : ptr(p) {}
    template<class L, class = decltype(std::declval<const L&>().host_vec())>
    WeightRef(const L& l) : obj(&l), dev([](const void* o) -> const float* { return static_cast<const L*>(o)->get(); }),
                            hostv([](const void* o) { return static_cast<const L*>(o)->host_vec(); }) {}
    const float* device() const { return ptr ? ptr : dev(obj); }
    const void* key() const { return ptr ? static_cast<const void*>(ptr) : obj; }
    const std::vector<float>* host() const { return ptr ? nullptr : hostv(obj); }
    operator const float*() const { return device(); }
};
} // namespace c512_resident
