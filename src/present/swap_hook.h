// The swap chain side of the final-picture front end: patches the DXGI swap
// chain class's Present / ResizeBuffers so that every presented frame of a
// D3D12 swap chain goes through final_picture::Process first. Used by the
// present hook (version.dll) and by the NGX shim's present mode.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <string>

namespace swap_hook {

struct Settings {
    bool enabled = true;
    std::wstring engineDll;
    std::string engineConfig;
    float workingScale = 0.0f, maxWorkingPixels = 2.1e6f;
    float controls[6] = {1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    int toggleKey = VK_F10;
};

// Where the hook's and the front end's log lines go.
void SetLog(void (*log)(const char* line));
// Patches the swap chain class through a throw-away swap chain on a hidden
// window. Blocks while it does; not to be called under the loader lock.
bool Install(const Settings& settings, HINSTANCE module);
// The controls the front end starts with, or takes from now on.
void SetControls(const float controls[6]);
// The queue to run on when the swap chain's own cannot be read out of the
// object (a wrapped swap chain): the queue the game submits its frame on.
void SetQueueHint(ID3D12CommandQueue* queue);

}  // namespace swap_hook
