// Final-picture front end (S343): the engine applied to a finished frame at
// present time, in the same frame. Independent of how the frame was made, so
// it serves any D3D12 swap chain (a present hook) and the headless harness.
//
// Per frame, on the caller's queue:
//   list A   copy the back buffer, reduce it to the working size (RGBA16F,
//            display-encoded) and copy that into a buffer shared with HIP
//   signal   `staged`
//   engine   re_sync_submit: waits for `staged` on the device, runs the
//            network, writes its change map into a second shared buffer and
//            signals `result`
//   wait     the queue waits for `result`
//   list B   change map -> texture, frame copy + map -> back buffer
// Nothing on the host waits for the device. The engine runs transfer=nvidia
// without motion (its history is read in place).
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <string>

namespace final_picture {

struct Config {
    std::wstring engineDll;      // resident_engine.dll
    std::string engineConfig;    // the engine's config file (24 lines + options)
    // The model's working size as a fraction of the frame; 0 picks the
    // largest that keeps the working frame within maxWorkingPixels.
    float workingScale = 0.0f;
    float maxWorkingPixels = 2.1e6f;  // about 1920x1080
    // NVIDIA's controls: intensity, style (0..2), local structure, local
    // tone, skin structure (-1 follows structure), automatic mask.
    float controls[6] = {1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f};
};

struct Stats {
    unsigned long long frames = 0;    // Process calls that queued a job
    unsigned long long skipped = 0;   // frames presented untouched
    unsigned int workWidth = 0, workHeight = 0;
    bool active = false;
    const char* error = "";
};

// One instance per process (the engine is one per process).
bool Init(ID3D12Device* device, ID3D12CommandQueue* queue, const Config& config);
// The back buffer in D3D12_RESOURCE_STATE_PRESENT; it is returned in that
// state. False: the frame was left untouched.
bool Process(ID3D12Resource* backBuffer);
void SetControls(const float controls[6]);
// Blocks until everything queued so far has run (resize, shutdown, tests).
void Flush();
// Before a swap chain resizes or recreates its buffers: waits as Flush does
// and drops the views of the old back buffers.
void ForgetBackBuffers();
// Where the module's log lines go (default: stderr).
void SetLog(void (*log)(const char* line));
void Shutdown();
const Stats& GetStats();

}  // namespace final_picture
