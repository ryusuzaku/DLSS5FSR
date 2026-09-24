#include "ngx_internal.h"

#include <windows.h>

namespace ngx {
namespace {
std::wstring ModuleDir(HMODULE h) {
    wchar_t path[MAX_PATH]{};
    if (!GetModuleFileNameW(h, path, MAX_PATH)) return L".";
    wchar_t* slash = wcsrchr(path, L'\\');
    if (slash) *slash = 0;
    return path;
}
}  // namespace
}  // namespace ngx

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
        ngx::SetModuleDir(ngx::ModuleDir(hModule));
    } else if (reason == DLL_PROCESS_DETACH) {
        ngx::LogShutdown();
    }
    return TRUE;
}
