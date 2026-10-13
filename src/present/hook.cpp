// Present hook (S343): the final-picture front end inside any D3D12 game.
// Built as version.dll and placed beside the game's executable; it forwards
// the version API to the system's DLL and, on a thread of its own, patches
// the DXGI swap chain's Present / ResizeBuffers so that every presented
// frame goes through final_picture::Process first.
//
// The hook itself is swap_hook.cpp.
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

#include "swap_hook.h"

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

using swap_hook::Settings;

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
        else if (key == "CaptureKey") s.captureKey = int(strtol(value.c_str(), nullptr, 0));
        else if (key == "PassesKey") s.passesKey = int(strtol(value.c_str(), nullptr, 0));
        else if (key == "CaptureFrames") s.captureFrames = atoi(value.c_str());
        else if (key == "CapturePath") { wchar_t w[MAX_PATH]; MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, w, MAX_PATH); s.capturePath = w; }
    }
    fclose(f);
    return s;
}

// -------------------------------------------------------------------- hook --

DWORD WINAPI HookThread(void*) {
    // A game that never loads d3d12.dll is not ours to touch.
    for (int waited = 0; !GetModuleHandleW(L"d3d12.dll"); waited += 250) {
        if (waited > 120000) { Log("d3d12.dll was never loaded: nothing to hook"); return 0; }
        Sleep(250);
    }
    swap_hook::SetLog(LogLine);
    swap_hook::Install(LoadSettings(), g_self);
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
