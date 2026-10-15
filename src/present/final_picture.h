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
// Nothing on the host waits for the device. The engine runs transfer=nvidia;
// without the game's motion it runs every frame on its own, as the original
// does without a motion source (no history). A caller that sees the game's
// motion vectors and depth earlier in the frame (an upscaler hook) hands
// them over through GetGuideTargets / GuidesStaged.
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
    // Sequence capture: while "<capturePath>.seq" exists (its text: a frame
    // count, default 60) that many consecutive frames are recorded as the
    // engine was handed them -- the working-size picture, the staged motion
    // and depth -- with the change it returned, into
    // "<capturePath>.pseq.<tick>\<frame>.bin" (D5INP001, D5DEP001, D5MOV001
    // as the shim's capture writes them, then D5OUT001: RGBA16F rows). The
    // frames are read back on the queue and written after the last one; the
    // trigger file is removed when they are on disk. Empty: off.
    std::wstring capturePath;
};

struct Stats {
    unsigned long long frames = 0;    // Process calls that queued a job
    unsigned long long skipped = 0;   // frames presented untouched
    unsigned long long guideFrames = 0;  // jobs that carried the game's motion
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
// The game's motion and depth for the frame about to be presented. The
// caller records, on the list the game will submit before it presents, a
// copy of each into these buffers as rows of the working size: motion as
// two floats per pixel (previous = current + v, in the units the engine's
// config names: motion=uv means fractions of the frame), depth as one float
// (reversed-Z), at the given row pitches. Then GuidesStaged; the next
// Process hands them to the engine. False until a frame has been sized.
struct GuideTargets {
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;  // null: no depth
    unsigned int width = 0, height = 0;
    unsigned int motionPitch = 0, depthPitch = 0;
    unsigned int frameWidth = 0, frameHeight = 0;  // the presented picture
};
bool GetGuideTargets(GuideTargets* out);
void GuidesStaged(bool motion, bool depth);
// How many times the engine runs the network on a frame (1..3), from the
// next frame on. False: this engine cannot change it while it runs.
bool SetPasses(int passes);
// How much of the engine's recurrence is used (the engine config's
// history=A: 1 the original, 0 every frame on its own), from the next frame
// on. False: this engine cannot change it while it runs.
bool SetHistory(float amount);
// The engine's stabilize amount (0 off .. 0.95), from the next frame on.
bool SetStabilize(float amount);
// The caller's Reset (a camera cut): the next frame runs without history.
void ResetHistory();
// Starts a sequence capture of `frames` frames at the next Process, as the
// trigger file does (Config::capturePath must be set).
void RequestCapture(int frames);
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
