#pragma once
// Shared launch state of the resident components.
#include <hip/hip_runtime.h>

namespace c512_resident {
// Stream for every resident launch and device copy; null keeps the harness's
// default-stream behaviour, an engine may select a non-blocking stream.
inline hipStream_t stream = nullptr;
// Optional profiling hook called after every resident launch (null in normal runs).
inline void (*launch_hook)(const char* kernel) = nullptr;
} // namespace c512_resident
