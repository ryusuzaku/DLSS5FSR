// See swap_hook.h.
//
// The swap chain class has one vtable for all its instances: a throw-away
// swap chain on a hidden window gives its address, and patching the slots
// there covers the game's swap chain whenever it is created. The probe is
// made through the system's own dxgi.dll, by its full path: a proxy of that
// name beside the game (OptiScaler, ReShade) would hand back its own class.
// The queue a swap chain presents on is not available through its interface
// (GetDevice answers E_NOINTERFACE for it), so it is found in the object:
// the first pointer to something with the command queue class's vtable,
// which the throw-away queue gave. D3D11 swap chains hold none and pass
// through untouched.

#include "swap_hook.h"

#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "final_picture.h"

using Microsoft::WRL::ComPtr;

namespace swap_hook {
namespace {

void (*g_log)(const char*) = nullptr;

void Log(const char* fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (g_log) g_log(line);
}


typedef HRESULT (STDMETHODCALLTYPE* PresentFn)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT (STDMETHODCALLTYPE* Present1Fn)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
typedef HRESULT (STDMETHODCALLTYPE* ResizeFn)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
typedef HRESULT (STDMETHODCALLTYPE* Resize1Fn)(IDXGISwapChain3*, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*, IUnknown* const*);
typedef HRESULT (STDMETHODCALLTYPE* ColorSpaceFn)(IDXGISwapChain3*, DXGI_COLOR_SPACE_TYPE);
PresentFn g_present = nullptr;
Present1Fn g_present1 = nullptr;
ResizeFn g_resize = nullptr;
Resize1Fn g_resize1 = nullptr;
ColorSpaceFn g_colorSpace = nullptr;
// The colour space the game last set: anything but sRGB (HDR10's PQ shares
// the 10-bit format) is left alone -- the model is shown sRGB pictures.
DXGI_COLOR_SPACE_TYPE g_space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
bool g_spaceLogged = false;

Settings g_settings;
std::mutex g_frameMutex;
ID3D12CommandQueue* g_queue = nullptr;  // the queue the front end runs on (not owned)
void* g_queueVtable = nullptr;          // the command queue class's vtable (from the probe)
IDXGISwapChain* g_seenChain = nullptr;  // the swap chain g_seenQueue was found in
ID3D12CommandQueue* g_seenQueue = nullptr;
ID3D12CommandQueue* g_hintQueue = nullptr;  // SetQueueHint (not owned)
HINSTANCE g_module = nullptr;

bool Readable(const void* p, size_t bytes) {
    MEMORY_BASIC_INFORMATION mi{};
    if (!VirtualQuery(p, &mi, sizeof(mi)) || mi.State != MEM_COMMIT) return false;
    if (mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return false;
    return reinterpret_cast<const char*>(p) + bytes <= reinterpret_cast<const char*>(mi.BaseAddress) + mi.RegionSize;
}

// The command queue a D3D12 swap chain was created on, or null.
bool g_noQueueLogged = false;
ID3D12CommandQueue* QueueOf(IDXGISwapChain* sc) {
    if (sc == g_seenChain) return g_seenQueue;
    g_seenChain = sc;
    g_seenQueue = nullptr;
    const uintptr_t* slots = reinterpret_cast<const uintptr_t*>(sc);
    for (int i = 1; i < 512 && !g_seenQueue; ++i) {
        if (!Readable(slots + i, sizeof(uintptr_t))) break;
        const uintptr_t v = slots[i];
        if (v < 0x10000 || (v & 7) || !Readable(reinterpret_cast<const void*>(v), sizeof(void*))) continue;
        if (*reinterpret_cast<void* const*>(v) != g_queueVtable) continue;
        ID3D12CommandQueue* q = reinterpret_cast<ID3D12CommandQueue*>(v);
        if (q->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
            g_seenQueue = q;
            Log("swap chain %p presents on queue %p (found at +0x%X)", (void*)sc, (void*)q, i * 8);
        }
    }
    if (!g_seenQueue && g_hintQueue) {
        g_seenQueue = g_hintQueue;
        Log("swap chain %p: its queue is not in the object (wrapped); using the game's frame queue %p", (void*)sc, (void*)g_seenQueue);
    }
    if (!g_seenQueue) {
        g_seenChain = nullptr;  // a hint may still come
        if (!g_noQueueLogged) { g_noQueueLogged = true; Log("swap chain %p: no D3D12 queue (D3D11 or wrapped); left alone", (void*)sc); }
    }
    return g_seenQueue;
}
bool g_started = false, g_failed = false, g_on = true, g_keyDown = false, g_captureDown = false, g_passesDown = false;
int g_passes = 1;
bool g_historyDown = false, g_stabilizeDown = false;
int g_stabilizeStep = -1;  // the engine config's amount until the key is first pressed (then 0.66)
int g_historyStep = -1;  // the engine config's amount until the key is first pressed (then 1)
unsigned long long g_presents = 0;

void OnPresent(IDXGISwapChain* sc) {
    std::lock_guard<std::mutex> lock(g_frameMutex);
    if (g_failed || !g_settings.enabled) return;
    if (g_space != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) {
        if (!g_spaceLogged) { g_spaceLogged = true; Log("colour space %d (HDR) is not handled yet: frames left alone", int(g_space)); }
        return;
    }
    ID3D12CommandQueue* queue = QueueOf(sc);
    if (!queue) return;  // not a D3D12 swap chain
    if (g_started && queue != g_queue) {
        // The game made a new device or queue: start over on it.
        Log("swap chain on another queue: restarting the front end");
        final_picture::Shutdown();
        g_started = false;
    }
    if (!g_started) {
        ComPtr<ID3D12Device> device;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device)))) { g_failed = true; return; }
        final_picture::Config c;
        c.engineDll = g_settings.engineDll;
        c.engineConfig = g_settings.engineConfig;
        c.workingScale = g_settings.workingScale;
        c.maxWorkingPixels = g_settings.maxWorkingPixels;
        c.capturePath = g_settings.capturePath;
        memcpy(c.controls, g_settings.controls, sizeof(c.controls));
        Log("first D3D12 present: starting the engine");
        if (!final_picture::Init(device.Get(), queue, c)) {
            Log("front end did not start: %s", final_picture::GetStats().error);
            g_failed = true;
            return;
        }
        g_queue = queue;
        g_started = true;
        Log("front end running (toggle key 0x%02X)", g_settings.toggleKey);
    }
    const bool down = g_settings.toggleKey && (GetAsyncKeyState(g_settings.toggleKey) & 0x8000) != 0;
    if (down && !g_keyDown) {
        g_on = !g_on;
        Log("toggled %s", g_on ? "on" : "off");
    }
    g_keyDown = down;
    const bool capture = g_settings.captureKey && (GetAsyncKeyState(g_settings.captureKey) & 0x8000) != 0;
    if (capture && !g_captureDown && !g_settings.capturePath.empty()) {
        Log("capture key: %d frames", g_settings.captureFrames);
        final_picture::RequestCapture(g_settings.captureFrames);
    }
    g_captureDown = capture;
    const bool passes = g_settings.passesKey && (GetAsyncKeyState(g_settings.passesKey) & 0x8000) != 0;
    if (passes && !g_passesDown) {
        const int most = g_settings.maxPasses < 1 ? 1 : g_settings.maxPasses > 3 ? 3 : g_settings.maxPasses;
        const int next = g_passes % most + 1;
        if (final_picture::SetPasses(next)) { g_passes = next; Log("passes key: %d pass%s", next, next == 1 ? "" : "es"); }
        else Log("passes key: this engine cannot change its passes while running");
    }
    g_passesDown = passes;
    const bool history = g_settings.historyKey && (GetAsyncKeyState(g_settings.historyKey) & 0x8000) != 0;
    if (history && !g_historyDown) {
        static const float amounts[4] = {1.0f, 0.5f, 0.25f, 0.0f};
        const int next = (g_historyStep + 1) % 4;
        if (final_picture::SetHistory(amounts[next])) { g_historyStep = next; Log("history key: amount %.2f", amounts[next]); }
        else Log("history key: this engine cannot change its history amount while running");
    }
    g_historyDown = history;
    const bool stab = g_settings.stabilizeKey && (GetAsyncKeyState(g_settings.stabilizeKey) & 0x8000) != 0;
    if (stab && !g_stabilizeDown) {
        // low band only; low band + still pixels; strong; off
        static const float amounts[4] = {0.66f, 0.66f, 0.8f, 0.0f}, still[4] = {0.0f, 0.7f, 0.9f, 0.0f};
        const int next = (g_stabilizeStep + 1) % 4;
        if (final_picture::SetStabilize(amounts[next], still[next])) { g_stabilizeStep = next; Log("stabilize key: amount %.2f, still pixels %.2f", amounts[next], still[next]); }
        else Log("stabilize key: this engine cannot change its stabiliser while running");
    }
    g_stabilizeDown = stab;
    if (!g_on) return;
    ComPtr<IDXGISwapChain3> sc3;
    ComPtr<ID3D12Resource> back;
    if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc3))) ||
        FAILED(sc->GetBuffer(sc3->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back))))
        return;
    final_picture::Process(back.Get());
    if ((++g_presents % 600) == 1) {
        const final_picture::Stats& st = final_picture::GetStats();
        Log("present %llu: %llu frames processed, %llu skipped, working %ux%u%s%s", g_presents, st.frames, st.skipped,
            st.workWidth, st.workHeight, st.error[0] ? "; " : "", st.error);
    }
}

HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain* sc, UINT sync, UINT flags) {
    if (!(flags & DXGI_PRESENT_TEST)) OnPresent(sc);
    return g_present(sc, sync, flags);
}

HRESULT STDMETHODCALLTYPE HookPresent1(IDXGISwapChain1* sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) {
    if (!(flags & DXGI_PRESENT_TEST)) OnPresent(sc);
    return g_present1(sc, sync, flags, p);
}

// The buffers are about to be released: nothing of ours may still use them.
void BeforeResize() {
    std::lock_guard<std::mutex> lock(g_frameMutex);
    if (g_started) final_picture::ForgetBackBuffers();
    g_seenChain = nullptr;  // look the queue up again afterwards
}

HRESULT STDMETHODCALLTYPE HookResize(IDXGISwapChain* sc, UINT count, UINT w, UINT h, DXGI_FORMAT f, UINT flags) {
    BeforeResize();
    return g_resize(sc, count, w, h, f, flags);
}

HRESULT STDMETHODCALLTYPE HookResize1(IDXGISwapChain3* sc, UINT count, UINT w, UINT h, DXGI_FORMAT f, UINT flags,
                                      const UINT* masks, IUnknown* const* queues) {
    BeforeResize();
    return g_resize1(sc, count, w, h, f, flags, masks, queues);
}

HRESULT STDMETHODCALLTYPE HookColorSpace(IDXGISwapChain3* sc, DXGI_COLOR_SPACE_TYPE space) {
    const HRESULT hr = g_colorSpace(sc, space);
    if (SUCCEEDED(hr)) {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        if (space != g_space) { g_space = space; g_spaceLogged = false; Log("swap chain colour space %d", int(space)); }
    }
    return hr;
}

template<class F> bool Patch(void** vtable, int slot, void* hook, F& original) {
    DWORD old = 0;
    if (!VirtualProtect(&vtable[slot], sizeof(void*), PAGE_EXECUTE_READWRITE, &old)) return false;
    if (vtable[slot] != hook) {
        original = reinterpret_cast<F>(vtable[slot]);
        vtable[slot] = hook;
    }
    VirtualProtect(&vtable[slot], sizeof(void*), old, &old);
    return true;
}


}  // namespace

void SetLog(void (*log)(const char*)) {
    g_log = log;
    final_picture::SetLog(log);
}

void SetControls(const float controls[6]) {
    {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        memcpy(g_settings.controls, controls, sizeof(g_settings.controls));
    }
    final_picture::SetControls(controls);
}

void SetQueueHint(ID3D12CommandQueue* queue) { g_hintQueue = queue; }

bool Install(const Settings& settings, HINSTANCE module, ID3D12Device* gameDevice) {
    {
        std::lock_guard<std::mutex> lock(g_frameMutex);
        g_settings = settings;
    }
    g_module = module;
    if (!settings.enabled) { Log("Enabled=0: not hooking"); return false; }
    if (settings.engineDll.empty() || settings.engineConfig.empty()) { Log("EngineDll / EngineConfig not set: not hooking"); return false; }

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = g_module;
    wc.lpszClassName = L"dlss5_present_probe";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_POPUP, 0, 0, 64, 64, nullptr, nullptr, g_module, nullptr);
    typedef HRESULT (WINAPI* CreateFactoryFn)(REFIID, void**);
    wchar_t dxgiPath[MAX_PATH];
    GetSystemDirectoryW(dxgiPath, MAX_PATH);
    wcscat_s(dxgiPath, L"\\dxgi.dll");
    HMODULE dxgi = LoadLibraryW(dxgiPath);
    const CreateFactoryFn createFactory = dxgi ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
    bool ok = false;
    {
        ComPtr<IDXGIFactory2> factory;
        ComPtr<ID3D12Device> device = gameDevice;
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<IDXGISwapChain1> probe;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Width = 64; sd.Height = 64;
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc = {1, 0};
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        if (hwnd && createFactory && SUCCEEDED(createFactory(IID_PPV_ARGS(&factory))) &&
            (device || SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) &&
            SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
            SUCCEEDED(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &probe))) {
            void** vtable = *reinterpret_cast<void***>(probe.Get());
            g_queueVtable = *reinterpret_cast<void**>(queue.Get());
            // IDXGISwapChain: Present 8, ResizeBuffers 13; IDXGISwapChain1: Present1 22;
            // IDXGISwapChain3: SetColorSpace1 38, ResizeBuffers1 39.
            ok = Patch(vtable, 8, (void*)&HookPresent, g_present) && Patch(vtable, 22, (void*)&HookPresent1, g_present1) &&
                 Patch(vtable, 13, (void*)&HookResize, g_resize) && Patch(vtable, 39, (void*)&HookResize1, g_resize1) &&
                 Patch(vtable, 38, (void*)&HookColorSpace, g_colorSpace);
        }
    }
    if (hwnd) DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, g_module);
    Log(ok ? "swap chain hooked" : "could not hook the swap chain");
    return ok;
}

}  // namespace swap_hook
