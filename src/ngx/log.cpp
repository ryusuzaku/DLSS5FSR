#include "ngx_internal.h"

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <string>

namespace ngx {

namespace {
std::mutex g_logMutex;
std::wofstream g_log;
LogLevel g_level = L_INFO;
std::wstring g_logPath;

const char* LevelTag(LogLevel l) {
    switch (l) {
        case L_ERROR: return "E";
        case L_WARN:  return "W";
        case L_INFO:  return "I";
        case L_DEBUG: return "D";
    }
    return "?";
}
}  // namespace

void LogInit(const std::wstring& dir) {
    std::lock_guard<std::mutex> g(g_logMutex);
    if (g_log.is_open()) return;
    g_logPath = dir + L"\\dlssnr_shim.log";
    g_log.open(g_logPath, std::ios::out | std::ios::trunc);
    if (g_log.is_open()) {
        g_log << L"--- dlssnr_shim log ---\n";
        g_log.flush();
    }
}

void LogShutdown() {
    std::lock_guard<std::mutex> g(g_logMutex);
    if (g_log.is_open()) {
        g_log << L"--- shutdown ---\n";
        g_log.flush();
        g_log.close();
    }
}

void LogSetLevel(LogLevel l) { g_level = l; }
LogLevel LogGetLevel() { return g_level; }

void LogV(LogLevel level, const char* fmt, ...) {
    if (level > g_level) return;

    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    buf[n] = 0;

    std::lock_guard<std::mutex> g(g_logMutex);
    if (!g_log.is_open()) return;
    // Narrow -> wide. Log lines are ASCII by construction.
    std::wstring w(buf, buf + n);
    g_log << L"[" << LevelTag(level) << L"] " << w << L"\n";
    g_log.flush();
}

const char* ResultStr(NVSDK_NGX_Result r) {
    switch (r) {
        case NVSDK_NGX_Result_Success: return "Success";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "Fail_FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "Fail_PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "Fail_FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "Fail_FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "Fail_InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "Fail_ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "Fail_NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "Fail_UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "Fail_RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "Fail_MissingInput";
        default: return "Fail_Other";
    }
}

const char* FeatureStr(NVSDK_NGX_Feature f) {
    switch (f) {
        case NVSDK_NGX_Feature_Reserved0: return "Reserved0";
        case NVSDK_NGX_Feature_SuperSampling: return "SuperSampling";
        case NVSDK_NGX_Feature_InPainting: return "InPainting";
        case NVSDK_NGX_Feature_ImageSuperResolution: return "ImageSuperResolution";
        case NVSDK_NGX_Feature_SlowMotion: return "SlowMotion";
        case NVSDK_NGX_Feature_VideoSuperResolution: return "VideoSuperResolution";
        case NVSDK_NGX_Feature_ImageSignalProcessing: return "ImageSignalProcessing";
        case NVSDK_NGX_Feature_DeepResolve: return "DeepResolve";
        case NVSDK_NGX_Feature_FrameGeneration: return "FrameGeneration";
        case NVSDK_NGX_Feature_DeepDVC: return "DeepDVC";
        case NVSDK_NGX_Feature_RayReconstruction: return "RayReconstruction";
        default: break;
    }
    static char buf[32];
    snprintf(buf, sizeof(buf), "Feature(%d)", (int)f);
    return buf;
}

// ----------------------------------------------------------------- config --

namespace {
Config g_cfg;
bool g_cfgLoaded = false;

std::wstring Trim(const std::wstring& s) {
    const wchar_t* ws = L" \t\r\n";
    size_t a = s.find_first_not_of(ws);
    if (a == std::wstring::npos) return L"";
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}
}  // namespace

Config& Cfg() { return g_cfg; }

void ConfigLoad(const std::wstring& dir) {
    Config& c = g_cfg;
    std::wstring path = dir + L"\\dlssnr_shim.ini";
    std::wifstream in(path);
    if (!in.is_open()) {
        LOGI("config: no dlssnr_shim.ini, using defaults");
        g_cfgLoaded = true;
        return;
    }

    std::wstring line;
    while (std::getline(in, line)) {
        size_t hash = line.find(L'#');
        if (hash != std::wstring::npos) line = line.substr(0, hash);
        size_t semi = line.find(L';');
        if (semi != std::wstring::npos) line = line.substr(0, semi);
        size_t eq = line.find(L'=');
        if (eq == std::wstring::npos) continue;

        std::wstring key = Trim(line.substr(0, eq));
        std::wstring val = Trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        try {
            if (key == L"LogLevel")              c.logLevel = std::stoi(val);
            else if (key == L"Enabled")          c.enabled = (std::stoi(val) != 0);
            else if (key == L"TransferStrength") c.transferStrength = std::stof(val);
            else if (key == L"ColourStrength")   c.colourStrength = std::stof(val);
            else if (key == L"ModelScale")       c.modelScale = std::stof(val);
            else if (key == L"DumpFrames")       c.dumpFrames = (std::stoi(val) != 0);
            else if (key == L"DumpEvery")        c.dumpEvery = std::stoi(val);
            else if (key == L"DumpDir")          c.dumpDir = val;
            else if (key == L"DumpField")        c.dumpField = std::stoi(val);
            else if (key == L"NrPasses")         c.nrPasses = (std::stoi(val) != 0);
            else if (key == L"HipBackend")       c.hipBackend = (std::stoi(val) != 0);
            else if (key == L"HipWeightsDir")    c.hipWeightsDir = val;
            else if (key == L"HipRocwmmaInc")    c.hipRocwmmaInc = val;
            else if (key == L"HipRocInc")        c.hipRocInc = val;
            else if (key == L"HipFeBlock")       c.hipFeBlock = std::stoi(val);
            else if (key == L"HipFfnTranspose")  c.hipFfnTranspose = std::stoi(val);
            else if (key == L"HipFeLive")        c.hipFeLive = std::stoi(val);
            else if (key == L"HipFeStrength")    c.hipFeStrength = std::stof(val);
            else if (key == L"HipFeWindX")       c.hipFeWindX = std::stoi(val);
            else if (key == L"HipFeWindY")       c.hipFeWindY = std::stoi(val);
            else if (key == L"HipFeTransition")  c.hipFeTransition = std::stoi(val);
            else if (key == L"CandidatePreviewPath") c.candidatePreviewPath = val;
            else if (key == L"WhitePoint")       c.whitePoint = std::stof(val);
            else if (key == L"ProxyMode")        c.proxyMode = std::stoi(val);
            else if (key == L"MaxRatio")         c.maxRatio = std::stof(val);
            else if (key == L"Passthrough")      c.passthrough = std::stoi(val);
            else if (key == L"DebugView")        c.debugView = std::stoi(val);
            else LOGW("config: unknown key %ls", key.c_str());
        } catch (...) {
            LOGW("config: bad value for %ls = %ls", key.c_str(), val.c_str());
        }
    }

    if (c.whitePoint <= 0.0f) {
        LOGW("config: WhitePoint %.3f is not positive, using 1.0", c.whitePoint);
        c.whitePoint = 1.0f;
    }
    if (c.maxRatio < 1.0f) {
        LOGW("config: MaxRatio %.3f below 1 would darken the frame, using 1.0",
             c.maxRatio);
        c.maxRatio = 1.0f;
    }

    g_cfgLoaded = true;
    LogSetLevel((LogLevel)c.logLevel);
    LOGI("config: level=%d enabled=%d nr=%d transfer=%.3f colour=%.3f",
         c.logLevel, (int)c.enabled, (int)c.nrPasses, c.transferStrength,
         c.colourStrength);
    LOGI("config: scale=%.3f white=%.3f maxRatio=%.3f passthrough=%d debug=%d hip=%d",
         c.modelScale, c.whitePoint, c.maxRatio, c.passthrough, c.debugView,
         (int)c.hipBackend);
    // S207: the A/B knobs, echoed so a captured log identifies the ARM the run
    // actually used. The ini on disk can be switched after a launch starts, and
    // the file copy in a capture then describes a different configuration than
    // the run did -- this line cannot lie about it.
    LOGI("config: live=%d block=%d strength=%.4f transpose=%d wind=%d,%d transition=%d",
         (int)c.hipFeLive, c.hipFeBlock, c.hipFeStrength, c.hipFfnTranspose,
         c.hipFeWindX, c.hipFeWindY, (int)c.hipFeTransition);
    if (!c.candidatePreviewPath.empty())
        LOGI("config: fixed candidate preview armed at %ls (DebugView=2 required; not live inference)",
             c.candidatePreviewPath.c_str());
}

}  // namespace ngx
