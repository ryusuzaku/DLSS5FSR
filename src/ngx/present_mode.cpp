// Present mode (S347): the engine on the finished frame, in the same frame.
//
// The NR evaluate runs in the middle of a list the game has not submitted,
// so the engine's answer for that frame cannot be in it: the map path
// applies the previous frame's answer and carries it along the motion,
// which is where its ghosts come from. Here the evaluate changes nothing in
// the frame. It hands the game's motion vectors and depth to the
// final-picture front end (copies on the game's own list, so they are this
// frame's), and the front end, hooked into the swap chain's Present, runs
// the engine on the finished picture and adds its change before the frame
// is shown. No lag, no carry, no gate.
#include "ngx_internal.h"

#include <atomic>

#include "../present/final_picture.h"
#include "../present/swap_hook.h"

namespace ngx {
namespace {

std::atomic<int> g_state{0};          // 0 not started, 1 hooking, 2 hooked, 3 failed
std::atomic<unsigned int> g_controls{0};  // the NrControls generation handed on
swap_hook::Settings g_settings;
ID3D12Device* g_device = nullptr;  // the game's, held until the hook is in

void PresentLog(const char* line) { LOGI("present: %s", line); }

DWORD WINAPI InstallThread(void*) {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&InstallThread), &self);
    swap_hook::SetLog(PresentLog);
    const bool ok = swap_hook::Install(g_settings, self, g_device);
    if (g_device) { g_device->Release(); g_device = nullptr; }
    g_controls = 0;  // whatever the caller set meanwhile goes in again
    g_state = ok ? 2 : 3;
    return 0;
}

void Controls(float v[6]) {
    const NrControls& n = NrControlValues();
    v[0] = n.intensity; v[1] = n.style; v[2] = n.structure; v[3] = n.tone; v[4] = n.skin; v[5] = n.mask;
}

}  // namespace

void PresentModeEvaluate(ID3D12GraphicsCommandList* cl, const Config& cfg, const MotionInput* motion) {
    if (g_state == 0) {
        g_state = 1;
        g_settings.enabled = true;
        g_settings.engineDll = cfg.residentEngineDll;
        char config[1024] = {};
        WideCharToMultiByte(CP_UTF8, 0, cfg.presentEngineConfig.c_str(), -1, config, sizeof(config) - 1, nullptr, nullptr);
        g_settings.engineConfig = config;
        g_settings.workingScale = cfg.presentWorkingScale;
        g_settings.maxWorkingPixels = cfg.presentMaxWorkingPixels;
        g_settings.toggleKey = cfg.presentToggleKey;
        if (NrControlValues().generation) Controls(g_settings.controls);
        LOGI("present: mode on, engine config %s, working scale %g (0: within %.0f pixels)", config,
             cfg.presentWorkingScale, cfg.presentMaxWorkingPixels);
        // The probe swap chain is made on the game's device: a second
        // D3D12CreateDevice here would be seen by every other hook in the
        // process (OptiScaler takes each new device as its current one).
        if (FAILED(cl->GetDevice(IID_PPV_ARGS(&g_device)))) g_device = nullptr;
        HANDLE t = CreateThread(nullptr, 0, InstallThread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
        else {
            if (g_device) { g_device->Release(); g_device = nullptr; }
            g_state = 3;
        }
    }
    if (g_state != 2) return;

    const unsigned int generation = NrControlValues().generation;
    if (generation && g_controls.exchange(generation) != generation) {
        float v[6];
        Controls(v);
        swap_hook::SetControls(v);
    }

    static unsigned long long calls = 0, staged = 0, withDepth = 0;
    ++calls;
    final_picture::GuideTargets t;
    if (!motion || !motion->res || !motion->guideW || !motion->guideH || !final_picture::GetGuideTargets(&t)) return;

    // The vectors under each working pixel (nearest), scaled by NGX's MV
    // scale: fractions of the frame, as the engine's motion=uv expects. The
    // pass multiplies by target / guide size (its pixel form), cancelled here.
    bool motionOk = false, depthOk = false;
    PooledTexture* mvTex = GpuAcquireTexture(L"present motion", DXGI_FORMAT_R32G32_FLOAT, t.width, t.height);
    if (mvTex) {
        const D3D12_RESOURCE_STATES in = GpuGuessIncomingState(motion->res);
        GpuTransition(cl, motion->res, in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GpuSetState(cl, *mvTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DlssNrConstants mc;
        mc.Mode = NR_MODE_MOTION;
        mc.Width = t.width;
        mc.Height = t.height;
        mc.GuideWidth = motion->guideW;
        mc.GuideHeight = motion->guideH;
        mc.MvScaleX = motion->scaleX * (float)motion->guideW / (float)t.width;
        mc.MvScaleY = motion->scaleY * (float)motion->guideH / (float)t.height;
        const bool ok = GpuNrDispatch(cl, mc, motion->res, nullptr, nullptr, nullptr, nullptr, mvTex->res.Get(), nullptr);
        GpuTransition(cl, motion->res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, in);
        if (ok) {
            GpuSetState(cl, *mvTex, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = t.motion;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32_FLOAT, t.width, t.height, 1, t.motionPitch};
            src.pResource = mvTex->res.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            motionOk = true;
        }
    }
    PooledTexture* dTex = motionOk && t.depth && motion->depth && motion->depthW && motion->depthH
                              ? GpuAcquireTexture(L"present depth", DXGI_FORMAT_R32_FLOAT, t.width, t.height) : nullptr;
    if (dTex) {
        const D3D12_RESOURCE_STATES in = GpuGuessIncomingState(motion->depth);
        GpuTransition(cl, motion->depth, in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        GpuSetState(cl, *dTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DlssNrConstants dc;
        dc.Mode = NR_MODE_DEPTH;
        dc.Width = t.width;
        dc.Height = t.height;
        dc.GuideWidth = motion->depthW;
        dc.GuideHeight = motion->depthH;
        const bool ok = GpuNrDispatch(cl, dc, motion->depth, nullptr, nullptr, nullptr, nullptr, dTex->res.Get(), nullptr);
        GpuTransition(cl, motion->depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, in);
        if (ok) {
            GpuSetState(cl, *dTex, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = t.depth;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, t.width, t.height, 1, t.depthPitch};
            src.pResource = dTex->res.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            depthOk = true;
        }
    }
    final_picture::GuidesStaged(motionOk, depthOk);
    if (motionOk) ++staged;
    if (depthOk) ++withDepth;
    if (calls == 1 || calls == 120 || (calls % 3000) == 0 || (motionOk && staged == 1)) {
        const final_picture::Stats& st = final_picture::GetStats();
        LOGI("present: evaluate %llu: motion %ux%u -> %ux%u (scale %g,%g) staged %llu times, depth %llu; "
             "%llu frames processed, %llu with the game's motion, %llu skipped",
             calls, motion->guideW, motion->guideH, t.width, t.height, motion->scaleX, motion->scaleY, staged, withDepth,
             st.frames, st.guideFrames, st.skipped);
    }
}

}  // namespace ngx
