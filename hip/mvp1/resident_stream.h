#pragma once
// Shared launch state of the resident components.
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace c512_resident {
// Stream for every resident launch and device copy; null keeps the harness's
// default-stream behaviour, an engine may select a non-blocking stream.
inline hipStream_t stream = nullptr;
// Optional profiling hook called after every resident launch (null in normal runs).
inline void (*launch_hook)(const char* kernel) = nullptr;
// Fast paths (WMMA etc.) may reorder fp32 sums inside a 32-product slice;
// RESIDENT_EXACT=1 keeps every stage byte-identical to the reference kernels.
inline bool exact_math = [] { const char* v = getenv("RESIDENT_EXACT"); return v && *v == '1'; }();
} // namespace c512_resident
