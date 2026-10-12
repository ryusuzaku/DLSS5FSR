// Present hook (S343): the final-picture front end inside any D3D12 game.
// Built as version.dll and placed beside the game's executable; it forwards
// the version API to the system's DLL and, on a thread of its own, patches
// the DXGI swap chain's Present / ResizeBuffers so that every presented
// frame goes through final_picture::Process first.
//
// The swap chain class has one vtable for all its instances: a throw-away
// swap chain on a hidden window gives its address, and patching the slots
// there covers the game's swap chain whenever it is created. The queue a
// swap chain presents on is not available through its interface
// (GetDevice answers E_NOINTERFACE for it), so it is found in the object:
// the first pointer to something with the command queue class's vtable,
// which the throw-away queue gave. D3D11 swap chains hold none and pass
// through untouched.
//
// Settings: dlss5_present.ini beside the DLL; log: dlss5_present.log.

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "final_picture.h"

using Microsoft::WRL::ComPtr;

namespace {

HMODULE g_self = nullptr;
std::wstring g_dir;  // this DLL's directory, with a trailing backslash
std::mutex g_logMutex;
FILE* g_logFile = nullptr;

void LogLine(const char* line) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (!g_logFile) g_logFile = _wfopen((g_dir + L"dlss5_present.log").c_str(), L"w");
    if (!g_logFile) return;
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_logFile, "[%02u:%02u:%02u.%03u] %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, line);
    fflush(g_logFile);
}

void Log(const char* fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    LogLine(line);
}

// ---------------------------------------------------------------- settings --

struct Settings {
    bool enabled = true;
    std::wstring engineDll;
    std::string engineConfig;
    float workingScale = 0.0f, maxWorkingPixels = 2.1e6f;
    float controls[6] = {1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    int toggleKey = VK_F10;
};

Settings LoadSettings() {
    Settings s;
    FILE* f = _wfopen((g_dir + L"dlss5_present.ini").c_str(), L"r");
    if (!f) { Log("no dlss5_present.ini beside the DLL"); return s; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* eq = strchr(line, '=');
        if (!eq || line[0] == ';' || line[0] == '#') continue;
        *eq = 0;
        std::string key = line, value = eq + 1;
        auto trim = [](std::string& v) {
            while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ' || v.back() == '\t')) v.pop_back();
            size_t a = 0;
            while (a < v.size() && (v[a] == ' ' || v[a] == '\t')) ++a;
            v.erase(0, a);
        };
        trim(key); trim(value);
        if (key == "Enabled") s.enabled = atoi(value.c_str()) != 0;
        else if (key == "EngineDll") { wchar_t w[MAX_PATH]; MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, w, MAX_PATH); s.engineDll = w; }
        else if (key == "EngineConfig") s.engineConfig = value;
        else if (key == "WorkingScale") s.workingScale = float(atof(value.c_str()));
        else if (key == "MaxWorkingPixels") s.maxWorkingPixels = float(atof(value.c_str()));
        else if (key == "Intensity") s.controls[0] = float(atof(value.c_str()));
        else if (key == "Style") s.controls[1] = float(atof(value.c_str()));
        else if (key == "LocalStructure") s.controls[2] = float(atof(value.c_str()));
        else if (key == "LocalTone") s.controls[3] = float(atof(value.c_str()));
        else if (key == "SkinStructure") s.controls[4] = float(atof(value.c_str()));
        else if (key == "AutoMask") s.controls[5] = float(atof(value.c_str()));
        else if (key == "ToggleKey") s.toggleKey = int(strtol(value.c_str(), nullptr, 0));
    }
    fclose(f);
    return s;
}

// -------------------------------------------------------------------- hook --

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

bool Readable(const void* p, size_t bytes) {
    MEMORY_BASIC_INFORMATION mi{};
    if (!VirtualQuery(p, &mi, sizeof(mi)) || mi.State != MEM_COMMIT) return false;
    if (mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return false;
    return reinterpret_cast<const char*>(p) + bytes <= reinterpret_cast<const char*>(mi.BaseAddress) + mi.RegionSize;
}

// The command queue a D3D12 swap chain was created on, or null.
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
    if (!g_seenQueue) Log("swap chain %p: no D3D12 queue (D3D11 or wrapped); left alone", (void*)sc);
    return g_seenQueue;
}
bool g_started = false, g_failed = false, g_on = true, g_keyDown = false;
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

DWORD WINAPI HookThread(void*) {
    // A game that never loads d3d12.dll is not ours to touch.
    for (int waited = 0; !GetModuleHandleW(L"d3d12.dll"); waited += 250) {
        if (waited > 120000) { Log("d3d12.dll was never loaded: nothing to hook"); return 0; }
        Sleep(250);
    }
    g_settings = LoadSettings();
    if (!g_settings.enabled) { Log("Enabled=0: not hooking"); return 0; }
    if (g_settings.engineDll.empty() || g_settings.engineConfig.empty()) { Log("EngineDll / EngineConfig not set: not hooking"); return 0; }
    final_picture::SetLog(LogLine);

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = g_self;
    wc.lpszClassName = L"dlss5_present_probe";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_POPUP, 0, 0, 64, 64, nullptr, nullptr, g_self, nullptr);
    bool ok = false;
    {
        ComPtr<IDXGIFactory2> factory;
        ComPtr<ID3D12Device> device;
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
        if (hwnd && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
            SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) &&
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
    UnregisterClassW(wc.lpszClassName, g_self);
    Log(ok ? "swap chain hooked" : "could not hook the swap chain");
    return 0;
}

// ----------------------------------------------------------- version.dll ---

FARPROC Real(const char* name) {
    static HMODULE real = [] {
        wchar_t path[MAX_PATH];
        GetSystemDirectoryW(path, MAX_PATH);
        wcscat_s(path, L"\\version.dll");
        return LoadLibraryW(path);
    }();
    return real ? GetProcAddress(real, name) : nullptr;
}

}  // namespace

// x64 has one calling convention and these take only integers and pointers:
// eight register / stack slots forward any of them.
typedef void* (*Forward8)(void*, void*, void*, void*, void*, void*, void*, void*);
#define PROXY(name)                                                                                          \
    extern "C" void* Proxy_##name(void* a, void* b, void* c, void* d, void* e, void* f, void* g, void* h) { \
        static const FARPROC fn = Real(#name);                                                               \
        return fn ? reinterpret_cast<Forward8>(fn)(a, b, c, d, e, f, g, h) : nullptr;                        \
    }
PROXY(GetFileVersionInfoA)
PROXY(GetFileVersionInfoByHandle)
PROXY(GetFileVersionInfoExA)
PROXY(GetFileVersionInfoExW)
PROXY(GetFileVersionInfoSizeA)
PROXY(GetFileVersionInfoSizeExA)
PROXY(GetFileVersionInfoSizeExW)
PROXY(GetFileVersionInfoSizeW)
PROXY(GetFileVersionInfoW)
PROXY(VerFindFileA)
PROXY(VerFindFileW)
PROXY(VerInstallFileA)
PROXY(VerInstallFileW)
PROXY(VerLanguageNameA)
PROXY(VerLanguageNameW)
PROXY(VerQueryValueA)
PROXY(VerQueryValueW)

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = module;
        DisableThreadLibraryCalls(module);
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(module, path, MAX_PATH);
        g_dir = path;
        g_dir.erase(g_dir.find_last_of(L"\\/") + 1);
        HANDLE t = CreateThread(nullptr, 0, HookThread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
    }
    return TRUE;
}
