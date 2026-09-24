// The exported NGX entry points.
//
// Names and signatures come straight from ext/nvngx_sdk/nvsdk_ngx.h. We are
// NOT defining NVSDK_NGX, so these are plain extern "C" definitions and the
// undecorated names come from src/ngx/exports.def.

#include "ngx_internal.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <unordered_set>

// DLSS-NR forwarder interface (see Dagherbou/OptiScaler_DLSSNR). OptiScaler's
// DlssNrFeature loads nvngx.dll_dlssnr.dll and, besides the four NGX D3D12
// entry points, requires these two data symbols and the two helpers at the
// bottom of this file. Without them it reports "the forwarder is missing its
// exports" and never calls us. All are exported through exports.def.
extern "C" {
int dlssnr_call_last_init = 0;
int dlssnr_call_last_create = 0;
}

namespace ngx {

namespace {

bool g_initialised = false;
ID3D12Device* g_device = nullptr;
std::mutex g_initMutex;

// The two blocks the SDK owns rather than the app.
ParameterImpl g_globalParams;
ParameterImpl g_capabilityParams;

std::mutex g_blocksMutex;
std::unordered_set<ParameterImpl*> g_liveBlocks;

std::wstring g_moduleDir;
std::wstring g_appDataPath;

ParameterImpl* AllocBlock() {
    auto* p = new ParameterImpl();
    std::lock_guard<std::mutex> g(g_blocksMutex);
    g_liveBlocks.insert(p);
    return p;
}

void PopulateCapabilityBlock(ParameterImpl& p) {
    // Games gate the whole DLSS path on these. Saying "available" is the
    // point of the shim; anything we cannot do yet degrades inside Evaluate.
    p.Set(NVSDK_NGX_Parameter_SuperSampling_Available, 1u);
    p.Set(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, 0u);
    p.Set(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, 0u);
    p.Set(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, 0u);

    // Streamline / UE query these binary-named slots instead.
    p.Set(NVSDK_NGX_EParameter_SuperSampling_Available, 1u);

    p.Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, 0u);
}

}  // namespace

// ------------------------------------------------------------------- init --

NVSDK_NGX_Result DoInit(unsigned long long appId, const wchar_t* appDataPath,
                        ID3D12Device* dev, const NVSDK_NGX_FeatureCommonInfo* info,
                        NVSDK_NGX_Version ver) {
    std::lock_guard<std::mutex> g(g_initMutex);

    g_appDataPath = appDataPath ? appDataPath : L"";
    LogInit(g_appDataPath.empty() ? g_moduleDir : g_appDataPath);

    LOGI("=== NVSDK_NGX_D3D12_Init ===");
    dlssnr_call_last_init = 0;
    LOGI("  appId=0x%llX appDataPath=%ls", appId,
         appDataPath ? appDataPath : L"(null)");
    LOGI("  SDK version requested = 0x%X (we speak 0x%X)", (unsigned)ver,
         (unsigned)NVSDK_NGX_Version_API);
    LOGI("  device=%p featureInfo=%p", (void*)dev, (const void*)info);

    if (info && info->PathListInfo.Path &&
        info->PathListInfo.Length < 64) {
        for (unsigned int i = 0; i < info->PathListInfo.Length; ++i)
            LOGI("  path[%u] = %ls", i, info->PathListInfo.Path[i]);
    }

    ConfigLoad(g_moduleDir);

    if (!dev) {
        LOGE("init: no device");
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }

    g_device = dev;
    if (!GpuInit(dev)) {
        LOGE("init: gpu setup failed");
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }

    // Populate both. Real NGX separates GetParameters from
    // GetCapabilityParameters, but plenty of games read the availability flags
    // off whichever block they happen to have, and a shim that answers "not
    // available" simply gets skipped.
    PopulateCapabilityBlock(g_capabilityParams);
    PopulateCapabilityBlock(g_globalParams);
    g_initialised = true;
    dlssnr_call_last_init = 1;
    LOGI("init: ok");
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DoShutdown() {
    std::lock_guard<std::mutex> g(g_initMutex);
    LOGI("=== NVSDK_NGX_D3D12_Shutdown === (features alive: %zu)",
         Registry().Count());
    Registry().Clear();
    GpuDrainDumps(0, true);  // best effort; the caller's list is done by now
    GpuShutdown();
    g_initialised = false;
    g_device = nullptr;
    // The log deliberately stays open. Games init/shutdown more than once --
    // Streamline does it on every device reset -- and truncating on the second
    // init would destroy the record of the first. It is closed at process
    // detach instead.
    return NVSDK_NGX_Result_Success;
}

// -------------------------------------------------------------- evaluate ---

namespace {

void ReadSubRect(const ParameterImpl& p, const char* baseX, const char* baseY,
                 const char* dimW, const char* dimH, unsigned int defaultW,
                 unsigned int defaultH, SubRect& out) {
    unsigned int x = 0, y = 0, w = 0, h = 0;
    p.TryGetUI(baseX, x);
    p.TryGetUI(baseY, y);
    if (!p.TryGetUI(dimW, w)) w = defaultW;
    if (!p.TryGetUI(dimH, h)) h = defaultH;
    if (w == 0) w = defaultW;
    if (h == 0) h = defaultH;
    out.x = x; out.y = y; out.w = w; out.h = h;
}

std::wstring DumpDir() {
    const Config& c = Cfg();
    if (!c.dumpDir.empty()) return c.dumpDir;
    return g_appDataPath.empty() ? g_moduleDir : g_appDataPath;
}

}  // namespace

NVSDK_NGX_Result DoEvaluateD3D12(ID3D12GraphicsCommandList* cl,
                                 const NVSDK_NGX_Handle* handle,
                                 ParameterImpl* params) {
    if (!g_initialised) return NVSDK_NGX_Result_FAIL_NotInitialized;
    if (!handle || !params) return NVSDK_NGX_Result_FAIL_InvalidParameter;

    Feature* f = Registry().Find(handle);
    if (!f) {
        LOGE("evaluate: no feature with id 0x%08X", handle ? handle->Id : 0);
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }
    if (!cl) return NVSDK_NGX_Result_FAIL_InvalidParameter;

    const uint64_t n = f->evaluateCount++;

    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* mv = nullptr;
    params->TryGetRes12(NVSDK_NGX_Parameter_Color, &color);
    params->TryGetRes12(NVSDK_NGX_Parameter_Output, &output);
    params->TryGetRes12(NVSDK_NGX_Parameter_Depth, &depth);
    params->TryGetRes12(NVSDK_NGX_Parameter_MotionVectors, &mv);

    if (n == 0 || (n % 300) == 0) {
        LOGI("evaluate #%llu %s: color=%p output=%p depth=%p mv=%p", n,
             FeatureStr(f->type), (void*)color, (void*)output, (void*)depth,
             (void*)mv);
    }
    if (n == 0) LOGD("evaluate params:\n%s", params->Dump().c_str());

    if (!color || !output) {
        LOGE("evaluate: missing %s", !color ? "Color" : "Output");
        return NVSDK_NGX_Result_FAIL_MissingInput;
    }

    SubRect colorRect{};
    ReadSubRect(*params,
                NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
                NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y,
                NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                f->width, f->height, colorRect);

    SubRect outRect{};
    ReadSubRect(*params,
                NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,
                NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y,
                nullptr, nullptr, f->targetWidth, f->targetHeight, outRect);

    const Config& c = Cfg();

    // ---- the pipeline -------------------------------------------------
    if (!c.enabled) {
        // Nothing at all: the game gets back whatever was already in Output.
        // Only useful for proving a crash is ours.
        return NVSDK_NGX_Result_Success;
    }

    // The HIP model's answer for the PREVIOUS frame, host side, before
    // anything is recorded: the resolve below binds the model texture this
    // refreshes. Inside, the frame counter proves the previous list
    // executed -- it never blocks on the game and degrades to the identity
    // model when the proof is missing.
    GpuPrepareHipModel(n, c);

    if (!GpuBlit(cl, color, colorRect, output, outRect)) {
        LOGE("evaluate: resample failed");
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }

    // The colour pipeline runs after the resample, on the full-resolution
    // frame: the model is shown the finished picture, not the one the game
    // rendered. Failure here must not fail the evaluate -- a plain resample is
    // a correct, if soft, frame.
    if (c.nrPasses && !GpuNeuralChain(cl, output, outRect, c)) {
        LOGW("evaluate: colour passes did not run, leaving the resample");
    }

    // ---- debug dumps --------------------------------------------------
    if (c.dumpFrames) {
        bool want = (n == 0) || (c.dumpEvery > 0 && (n % (uint64_t)c.dumpEvery) == 0);
        if (want) {
            wchar_t name[MAX_PATH];
            const std::wstring dir = DumpDir();
            // Color is back in whatever state it arrived in -- GpuBlit put it
            // back -- so the desc-flag guess is right for it. Output is NOT:
            // our own passes leave it in UAV, which is the state NGX hands it
            // to us in and the state the game expects it back in.
            swprintf(name, MAX_PATH, L"%ls\\nr_%04llu_in.bmp", dir.c_str(), n);
            GpuQueueDump(cl, color, colorRect, name, n,
                         GpuGuessIncomingState(color));
            swprintf(name, MAX_PATH, L"%ls\\nr_%04llu_out.bmp", dir.c_str(), n);
            GpuQueueDump(cl, output, outRect, name, n,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    // The frame counter goes LAST: the next evaluate trusts that seeing this
    // value means every shim op above -- including the resolve -- executed.
    if (c.nrPasses && c.hipBackend) GpuWriteFrameCounter(cl, n);

    GpuDrainDumps(n, false);

    return NVSDK_NGX_Result_Success;
}

// Set by dllmain so the log can be opened before Init is called.
void SetModuleDir(const std::wstring& dir) { g_moduleDir = dir; }

// ------------------------------------------------------- exported surface --

}  // namespace ngx

using namespace ngx;

extern "C" {

// ---- D3D12 ------------------------------------------------------------

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init(
    unsigned long long InApplicationId, const wchar_t* InApplicationDataPath,
    ID3D12Device* InDevice, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo,
    NVSDK_NGX_Version InSDKVersion) {
    return DoInit(InApplicationId, InApplicationDataPath, InDevice, InFeatureInfo,
                  InSDKVersion);
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init_with_ProjectID(
    const char* InProjectId, NVSDK_NGX_EngineType InEngineType,
    const char* InEngineVersion, const wchar_t* InApplicationDataPath,
    ID3D12Device* InDevice, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo,
    NVSDK_NGX_Version InSDKVersion) {
    LOGI("Init_with_ProjectID project=%s engine=%d ver=%s",
         InProjectId ? InProjectId : "(null)", (int)InEngineType,
         InEngineVersion ? InEngineVersion : "(null)");
    return DoInit(0, InApplicationDataPath, InDevice, InFeatureInfo, InSDKVersion);
}

// NGX's "ext" init. NVIDIA's D3D12 provider exports it and Streamline /
// OptiScaler call it -- without it the proxy's InitDx12 returns
// 0xBAD00000 ("the NGX core would not initialise"). No header in
// ext/nvngx_sdk declares it, and the real argument list is not public:
// observed callers pass (appId, path, device, version) with NO
// FeatureCommonInfo (reading a 4th arg as a pointer gives 0x15 and
// crashes on the PathListInfo deref). Two forms exist in the wild, so
// disambiguate the 4th argument: a version enum is tiny, a pointer is not.
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init_Ext(
    unsigned long long InApplicationId, const wchar_t* InApplicationDataPath,
    ID3D12Device* InDevice, ...) {
    va_list ap;
    va_start(ap, InDevice);
    const void* a4 = va_arg(ap, const void*);
    const NVSDK_NGX_FeatureCommonInfo* info = nullptr;
    unsigned ver = 0;
    if ((uintptr_t)a4 < 0x10000u) {          // (…, device, version)
        ver = (unsigned)(uintptr_t)a4;
    } else {                                  // (…, device, info, version)
        info = (const NVSDK_NGX_FeatureCommonInfo*)a4;
        ver = (unsigned)(uintptr_t)va_arg(ap, const void*);
    }
    va_end(ap);
    LOGI("Init_Ext appId=0x%llX ver=0x%X info=%p", InApplicationId, ver,
         (const void*)info);
    return DoInit(InApplicationId, InApplicationDataPath, InDevice, info,
                  (NVSDK_NGX_Version)ver);
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown(void) { return DoShutdown(); }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*) {
    return DoShutdown();
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter** OutParameters) {
    if (!OutParameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *OutParameters = &g_globalParams;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters) {
    if (!OutParameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *OutParameters = AllocBlock();
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters) {
    if (!OutParameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    PopulateCapabilityBlock(g_capabilityParams);
    *OutParameters = &g_capabilityParams;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters) {
    if (!InParameters) return NVSDK_NGX_Result_Success;
    std::lock_guard<std::mutex> g(g_blocksMutex);
    auto* impl = static_cast<ParameterImpl*>(InParameters);
    auto it = g_liveBlocks.find(impl);
    if (it == g_liveBlocks.end()) {
        // Not ours: the SDK-owned blocks are handed out by GetParameters and
        // GetCapabilityParameters. Destroying them is a no-op, not an error.
        return NVSDK_NGX_Result_Success;
    }
    g_liveBlocks.erase(it);
    delete impl;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetScratchBufferSize(
    NVSDK_NGX_Feature InFeatureId, const NVSDK_NGX_Parameter* InParameters,
    size_t* OutSizeInBytes) {
    LOGI("GetScratchBufferSize(%s)", FeatureStr(InFeatureId));
    (void)InParameters;
    if (OutSizeInBytes) *OutSizeInBytes = 0;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(
    ID3D12GraphicsCommandList* InCmdList, NVSDK_NGX_Feature InFeatureID,
    NVSDK_NGX_Parameter* InParameters, NVSDK_NGX_Handle** OutHandle) {
    if (!OutHandle) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (!g_initialised) return NVSDK_NGX_Result_FAIL_NotInitialized;
    if (!InParameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;

    auto* impl = static_cast<ParameterImpl*>(InParameters);
    Feature* f = Registry().Create(InFeatureID, *impl);
    dlssnr_call_last_create = f ? 1 : 0;
    if (!f) return NVSDK_NGX_Result_FAIL_FeatureAlreadyExists;

    *OutHandle = &f->handle;
    (void)InCmdList;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* InHandle) {
    if (!InHandle) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return Registry().Release(InHandle) ? NVSDK_NGX_Result_Success
                                        : NVSDK_NGX_Result_FAIL_FeatureNotFound;
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetFeatureRequirements(
    IDXGIAdapter* Adapter, const NVSDK_NGX_FeatureDiscoveryInfo* Info,
    NVSDK_NGX_FeatureRequirement* OutSupported) {
    LOGI("GetFeatureRequirements adapter=%p feature=%s", (void*)Adapter,
         Info ? FeatureStr(Info->FeatureID) : "?");
    if (!OutSupported) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    OutSupported->FeatureSupported = NVSDK_NGX_FeatureSupportResult_Supported;
    OutSupported->MinHWArchitecture = 0;
    memset(OutSupported->MinOSVersion, 0, sizeof(OutSupported->MinOSVersion));
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature(
    ID3D12GraphicsCommandList* InCmdList, const NVSDK_NGX_Handle* InFeatureHandle,
    NVSDK_NGX_Parameter* InParameters, PFN_NVSDK_NGX_ProgressCallback InCallback) {
    (void)InCallback;
    return DoEvaluateD3D12(InCmdList, InFeatureHandle,
                           static_cast<ParameterImpl*>(InParameters));
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature_C(
    ID3D12GraphicsCommandList* InCmdList, const NVSDK_NGX_Handle* InFeatureHandle,
    const NVSDK_NGX_Parameter* InParameters,
    PFN_NVSDK_NGX_ProgressCallback_C InCallback) {
    (void)InCallback;
    return DoEvaluateD3D12(InCmdList, InFeatureHandle,
                           const_cast<ParameterImpl*>(
                               static_cast<const ParameterImpl*>(InParameters)));
}

// ---- D3D11 / CUDA: present so a game that probes every API still links ----

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init(unsigned long long, const wchar_t*,
                                                  ID3D11Device*,
                                                  const NVSDK_NGX_FeatureCommonInfo*,
                                                  NVSDK_NGX_Version) {
    LOGE("D3D11_Init called: only D3D12 is implemented");
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
// Export parity with NVIDIA's providers (see D3D12_Init_Ext above). D3D11
// is not implemented, but the symbol must exist so probes find it.
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init_Ext(
    unsigned long long, const wchar_t*, ID3D11Device*,
    const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version) {
    LOGE("D3D11_Init_Ext called: only D3D12 is implemented");
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init_with_ProjectID(
    const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*, ID3D11Device*,
    const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown(void) {
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown1(ID3D11Device*) {
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetParameters(NVSDK_NGX_Parameter** p) {
    if (p) *p = &g_globalParams;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_AllocateParameters(NVSDK_NGX_Parameter** p) {
    if (!p) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *p = AllocBlock();
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_GetCapabilityParameters(NVSDK_NGX_Parameter** p) {
    if (!p) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    PopulateCapabilityBlock(g_capabilityParams);
    *p = &g_capabilityParams;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_DestroyParameters(NVSDK_NGX_Parameter* p) {
    return NVSDK_NGX_D3D12_DestroyParameters(p);
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetScratchBufferSize(
    NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t* s) {
    if (s) *s = 0;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext*, NVSDK_NGX_Feature,
                              NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**) {
    LOGE("D3D11_CreateFeature: only D3D12 is implemented");
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle* h) {
    return Registry().Release(h) ? NVSDK_NGX_Result_Success
                                 : NVSDK_NGX_Result_FAIL_FeatureNotFound;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetFeatureRequirements(
    IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*,
    NVSDK_NGX_FeatureRequirement* o) {
    if (!o) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    o->FeatureSupported = NVSDK_NGX_FeatureSupportResult_AdapterUnsupported;
    o->MinHWArchitecture = 0;
    memset(o->MinOSVersion, 0, sizeof(o->MinOSVersion));
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_EvaluateFeature(ID3D11DeviceContext*, const NVSDK_NGX_Handle*,
                                NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D11_EvaluateFeature_C(ID3D11DeviceContext*, const NVSDK_NGX_Handle*,
                                  const NVSDK_NGX_Parameter*,
                                  PFN_NVSDK_NGX_ProgressCallback_C) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_Init(unsigned long long, const wchar_t*,
                                                const NVSDK_NGX_FeatureCommonInfo*,
                                                NVSDK_NGX_Version) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_Init_with_ProjectID(
    const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*,
    const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_Shutdown(void) {
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_GetParameters(NVSDK_NGX_Parameter** p) {
    if (p) *p = &g_globalParams;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_CUDA_AllocateParameters(NVSDK_NGX_Parameter** p) {
    if (!p) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *p = AllocBlock();
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_CUDA_GetCapabilityParameters(NVSDK_NGX_Parameter** p) {
    if (!p) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *p = &g_capabilityParams;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_CUDA_DestroyParameters(NVSDK_NGX_Parameter* p) {
    return NVSDK_NGX_D3D12_DestroyParameters(p);
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_GetScratchBufferSize(
    NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t* s) {
    if (s) *s = 0;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature,
                                                         NVSDK_NGX_Parameter*,
                                                         NVSDK_NGX_Handle**) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_ReleaseFeature(NVSDK_NGX_Handle* h) {
    return Registry().Release(h) ? NVSDK_NGX_Result_Success
                                 : NVSDK_NGX_Result_FAIL_FeatureNotFound;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_CUDA_EvaluateFeature(const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*,
                               PFN_NVSDK_NGX_ProgressCallback) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}
NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_CUDA_EvaluateFeature_C(const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
                                 PFN_NVSDK_NGX_ProgressCallback_C) {
    return NVSDK_NGX_Result_FAIL_NotImplemented;
}

NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_UpdateFeature(const NVSDK_NGX_Application_Identifier* ApplicationId,
                        NVSDK_NGX_Feature FeatureID) {
    LOGI("UpdateFeature feature=%s", FeatureStr(FeatureID));
    (void)ApplicationId;
    return NVSDK_NGX_Result_Success;
}

const wchar_t* NVSDK_CONV GetNGXResultAsString(NVSDK_NGX_Result InNGXResult) {
    static wchar_t buf[64];
    const char* s = ResultStr(InNGXResult);
    for (int i = 0; s[i] && i < 63; ++i) buf[i] = (wchar_t)s[i];
    buf[63] = 0;
    return buf;
}

// ---- C entry points for the typed parameter helpers -------------------

#define NGX_PARAM_C_SET(NAME, TYPE, FIELD)                                    \
    void NVSDK_CONV NVSDK_NGX_Parameter_Set##NAME(NVSDK_NGX_Parameter* p,      \
                                                  const char* n, TYPE v) {     \
        if (p) p->Set(n, v);                                                   \
    }
#define NGX_PARAM_C_GET(NAME, TYPE)                                           \
    NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_Parameter_Get##NAME(                 \
        NVSDK_NGX_Parameter* p, const char* n, TYPE* v) {                      \
        return p ? p->Get(n, v) : NVSDK_NGX_Result_FAIL_InvalidParameter;      \
    }

NGX_PARAM_C_SET(ULL, unsigned long long, ull)
NGX_PARAM_C_SET(F, float, f)
NGX_PARAM_C_SET(D, double, d)
NGX_PARAM_C_SET(UI, unsigned int, ui)
NGX_PARAM_C_SET(I, int, i)
NGX_PARAM_C_SET(D3d11Resource, ID3D11Resource*, p11)
NGX_PARAM_C_SET(D3d12Resource, ID3D12Resource*, p12)
NGX_PARAM_C_SET(VoidPointer, void*, vp)

NGX_PARAM_C_GET(ULL, unsigned long long)
NGX_PARAM_C_GET(F, float)
NGX_PARAM_C_GET(D, double)
NGX_PARAM_C_GET(UI, unsigned int)
NGX_PARAM_C_GET(I, int)
NGX_PARAM_C_GET(D3d11Resource, ID3D11Resource*)
NGX_PARAM_C_GET(D3d12Resource, ID3D12Resource*)
NGX_PARAM_C_GET(VoidPointer, void*)

#undef NGX_PARAM_C_SET
#undef NGX_PARAM_C_GET

// ---- DLSS-NR forwarder interface (see Dagherbou/OptiScaler_DLSSNR) --------
// OptiScaler's DlssNrFeature drives nvngx.dll_dlssnr.dll through these, on top
// of the four NGX D3D12 entry points, and refuses to run at all if any is
// missing ("the forwarder is missing its exports"). Our parameter block is an
// NVSDK_NGX_Parameter, so its vtable is the NGX header order (slot 1 is the
// float setter); probe_float writes through whichever slot the caller is
// testing, which is exactly the calibration it performs.
static int g_nrFloatSlot = 1;

void dlssnr_call_set_float_slot(int slot) {
    if (slot >= 0 && slot < 8) g_nrFloatSlot = slot;
    LOGI("dlssnr: set_float_slot %d", slot);
}

void dlssnr_call_probe_float(void* params, const char* name, float value, int slot) {
    if (!params || !name || slot < 0 || slot > 15) return;
    void** vt = *reinterpret_cast<void***>(params);
    reinterpret_cast<void (*)(void*, const char*, float)>(vt[slot])(params, name, value);
}

// ---- DLSS-NR core calls (Dagherbou/OptiScaler_DLSSNR) --------------------
// OptiScaler drives the NR model through these four, passing the frame's
// resources directly rather than by NGX parameter name. Ours routes them into
// the shim's own NR pipeline, so the pass runs on HIP. The signature argument
// lists are copied verbatim from the mod's forwarder (dlssnr_forwarder.cpp).
void* dlssnr_call_create(const wchar_t* /*snippetPath*/, const wchar_t* dataPath,
                         ID3D12Device* device, ID3D12GraphicsCommandList* /*cmd*/,
                         void* capabilityParams, unsigned int width,
                         unsigned int height, int preset, float intensity, int style,
                         float localStructure, float localTone, float skinStructure,
                         int useAutoMask, int uiCorrection) {
    auto* p = static_cast<ParameterImpl*>(capabilityParams);
    if (!p) {
        dlssnr_call_last_create = 0;
        return nullptr;
    }
    // The NR provider is loaded from its own path (nvngx.dll_dlssnr.dll), so it
    // is a separate module instance from the game-facing nvngx.dll and has not
    // seen Init. Bring ourselves up on the device the caller handed us, or
    // DoEvaluateD3D12 answers FAIL_NotInitialized and OptiScaler disables the
    // pass for the session.
    if (!g_initialised && device)
        DoInit(0x24480451ull, dataPath, device, nullptr, (NVSDK_NGX_Version)0x15);
    // Create latches Width/Height/OutWidth/OutHeight; the tuning is read once,
    // when the feature is built, so it has to be set here and not at evaluate.
    p->Set(NVSDK_NGX_Parameter_Width, width);
    p->Set(NVSDK_NGX_Parameter_Height, height);
    p->Set(NVSDK_NGX_Parameter_OutWidth, width);
    p->Set(NVSDK_NGX_Parameter_OutHeight, height);
    p->Set("DLSSNR.Hint.Render.Preset", (unsigned int)preset);
    p->Set("DLSSNR.Intensity", intensity);
    p->Set("DLSSNR.Style", (unsigned int)style);
    p->Set("DLSSNR.LocalStructureStrength", localStructure);
    p->Set("DLSSNR.LocalToneStrength", localTone);
    p->Set("DLSSNR.SkinStructureStrength", skinStructure);
    p->Set("DLSSNR.UseAutoMask", (unsigned int)useAutoMask);
    p->Set("DLSSNR.UICorrection", (unsigned int)uiCorrection);
    dlssnr_call_last_init = 1;  // loaded == initialised for this shim
    Feature* f = Registry().Create((NVSDK_NGX_Feature)18, *p);
    dlssnr_call_last_create = f ? 1 : 0;
    if (!f) return nullptr;
    LOGI("dlssnr: create %ux%u preset=%d intensity=%.2f id=0x%08X", width, height,
         preset, intensity, f->handle.Id);
    return &f->handle;
}

int dlssnr_call_evaluate(ID3D12GraphicsCommandList* cmd, void* feature,
                         void* capabilityParams, ID3D12Resource* color,
                         ID3D12Resource* depth, ID3D12Resource* motion,
                         ID3D12Resource* output, unsigned int width,
                         unsigned int height, unsigned int guideWidth,
                         unsigned int guideHeight, int depthInverted, int reset,
                         float intensity, int style, float localStructure,
                         float localTone, float skinStructure, int useAutoMask,
                         float mvScaleX, float mvScaleY) {
    auto* p = static_cast<ParameterImpl*>(capabilityParams);
    if (!p || !feature) return (int)NVSDK_NGX_Result_FAIL_InvalidParameter;
    // Map the direct arguments onto the NGX names DoEvaluateD3D12 reads.
    p->Set(NVSDK_NGX_Parameter_Color, color);
    p->Set(NVSDK_NGX_Parameter_Output, output);
    p->Set(NVSDK_NGX_Parameter_Depth, depth);
    p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, width);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, height);
    (void)guideWidth; (void)guideHeight; (void)depthInverted; (void)reset;
    (void)intensity; (void)style; (void)localStructure; (void)localTone;
    (void)skinStructure; (void)useAutoMask; (void)mvScaleX; (void)mvScaleY;
    return (int)DoEvaluateD3D12(cmd, (const NVSDK_NGX_Handle*)feature, p);
}

void dlssnr_call_set_extras(void* capabilityParams, float /*globalTone*/,
                            ID3D12Resource* ui, ID3D12Resource* uiAlpha,
                            ID3D12Resource* backbuffer, unsigned int uiWidth,
                            unsigned int uiHeight, unsigned int /*bbWidth*/,
                            unsigned int /*bbHeight*/) {
    auto* p = static_cast<ParameterImpl*>(capabilityParams);
    if (!p) return;
    p->Set("DLSSNR.UI", ui);
    p->Set("DLSSNR.UIAlpha", uiAlpha);
    p->Set("DLSSNR.Backbuffer", backbuffer);
    p->Set("DLSSNR.UISubrectWidth", uiWidth);
    p->Set("DLSSNR.UISubrectHeight", uiHeight);
}

void dlssnr_call_release(void* feature) {
    if (feature) Registry().Release((const NVSDK_NGX_Handle*)feature);
}

}  // extern "C"
