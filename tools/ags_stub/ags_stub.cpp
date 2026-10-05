// A stand-in for AMD's amd_ags_x64.dll (AGS 5.4) for running Cyberpunk 2077 on
// vkd3d-proton under Windows (the DLSSNR-AMD Vulkan route). The real AGS hands
// the device to AMD's native D3D12 driver extensions, which do not exist on a
// vkd3d-proton device (it crashes in amdxc64.dll). This one reports no AMD
// extensions and creates a plain D3D12 device, like Proton's replacement on
// Linux. Only the six functions the game imports exist.
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <string.h>

extern "C" {

struct AGSContext { int dummy; };

struct AGSGPUInfo {  // AGS 5.4
    int agsVersionMajor, agsVersionMinor, agsVersionPatch, isWACKCompliant;
    const char* driverVersion;
    const char* radeonSoftwareVersion;
    int numDevices;
    void* devices;
};

struct AGSDX12DeviceCreationParams {
    IDXGIAdapter* pAdapter;
    IID iid;
    D3D_FEATURE_LEVEL FeatureLevel;
};

struct AGSDX12ReturnedParams {
    ID3D12Device* pDevice;
    unsigned int extensionsSupported;  // a bitfield struct in 5.4: none
};

enum { AGS_SUCCESS = 0, AGS_FAILURE = 1, AGS_INVALID_ARGS = 2 };

static AGSContext g_context;

__declspec(dllexport) int agsInit(AGSContext** context, const void* /*config*/, AGSGPUInfo* gpuInfo) {
    if (!context) return AGS_INVALID_ARGS;
    *context = &g_context;
    if (gpuInfo) {
        memset(gpuInfo, 0, sizeof(*gpuInfo));
        gpuInfo->agsVersionMajor = 5;
        gpuInfo->agsVersionMinor = 4;
        gpuInfo->driverVersion = "";
        gpuInfo->radeonSoftwareVersion = "";
    }
    return AGS_SUCCESS;
}

__declspec(dllexport) int agsDeInit(AGSContext* /*context*/) { return AGS_SUCCESS; }

__declspec(dllexport) int agsDriverExtensionsDX12_CreateDevice(AGSContext* /*context*/,
                                                                const AGSDX12DeviceCreationParams* creation,
                                                                const void* /*extensionParams*/,
                                                                AGSDX12ReturnedParams* returned) {
    if (!creation || !returned) return AGS_INVALID_ARGS;
    HMODULE d3d12 = GetModuleHandleA("d3d12.dll");
    if (!d3d12) d3d12 = LoadLibraryA("d3d12.dll");
    auto create = d3d12 ? (PFN_D3D12_CREATE_DEVICE)GetProcAddress(d3d12, "D3D12CreateDevice") : nullptr;
    if (!create) return AGS_FAILURE;
    void* device = nullptr;
    HRESULT hr = create(creation->pAdapter, creation->FeatureLevel, creation->iid, &device);
    if (FAILED(hr) || !device) return AGS_FAILURE;
    returned->pDevice = (ID3D12Device*)device;
    returned->extensionsSupported = 0;
    return AGS_SUCCESS;
}

__declspec(dllexport) int agsDriverExtensionsDX12_DestroyDevice(AGSContext* /*context*/, ID3D12Device* device,
                                                                 unsigned int* deviceReferences) {
    if (!device) return AGS_INVALID_ARGS;
    const ULONG left = device->Release();
    if (deviceReferences) *deviceReferences = left;
    return AGS_SUCCESS;
}

__declspec(dllexport) int agsDriverExtensionsDX12_PushMarker(AGSContext*, ID3D12GraphicsCommandList*, const char*) {
    return AGS_SUCCESS;
}

__declspec(dllexport) int agsDriverExtensionsDX12_PopMarker(AGSContext*, ID3D12GraphicsCommandList*) {
    return AGS_SUCCESS;
}

}  // extern "C"
