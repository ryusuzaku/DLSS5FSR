// Internal declarations for the nvngx.dll shim.
//
// ABI note: everything the game sees comes from ext/nvngx_sdk/nvsdk_ngx*.h,
// which is NVIDIA's public NGX SDK header. The vtable layout of
// NVSDK_NGX_Parameter and the exact signatures of the NVSDK_NGX_D3D12_*
// entry points are taken from there verbatim; do not "tidy" them.
//
// We build WITHOUT defining NVSDK_NGX, so the header declares plain
// extern "C" functions and exports come from src/ngx/exports.def. That keeps
// the exported names undecorated, which is what a game's nvngx.lib imports.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

// d3dx12.h is not in the Windows SDK; it ships with the DirectX samples (MIT)
// and lives in ext/d3dx12. It must come before nvsdk_ngx.h, which only
// forward-declares CD3DX12_HEAP_PROPERTIES.
#include "d3dx12.h"

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"

using Microsoft::WRL::ComPtr;

namespace ngx {

// ---------------------------------------------------------------- logging --

enum LogLevel { L_ERROR = 0, L_WARN = 1, L_INFO = 2, L_DEBUG = 3 };

void LogInit(const std::wstring& dir);
void LogShutdown();
void LogSetLevel(LogLevel l);
LogLevel LogGetLevel();
void LogV(LogLevel level, const char* fmt, ...);

#define LOGE(...) ngx::LogV(ngx::L_ERROR, __VA_ARGS__)
#define LOGW(...) ngx::LogV(ngx::L_WARN, __VA_ARGS__)
#define LOGI(...) ngx::LogV(ngx::L_INFO, __VA_ARGS__)
#define LOGD(...) ngx::LogV(ngx::L_DEBUG, __VA_ARGS__)

const char* ResultStr(NVSDK_NGX_Result r);
const char* FeatureStr(NVSDK_NGX_Feature f);

// ----------------------------------------------------------------- config --

struct Config {
    int logLevel = L_INFO;
    bool enabled = true;

    // Neural-render strength. 0 must be a bit-exact passthrough.
    float transferStrength = 0.0f;
    float colourStrength = 0.0f;

    // Fraction of display resolution the model runs at. The reference mod
    // exposes this as the main performance lever; keep it from day one.
    float modelScale = 1.0f;

    // 0 = resample only. 1 = encode -> model -> resolve. Kept separate from
    // `enabled` so the colour pipeline can be switched off without taking the
    // whole feature down, which is how a strength-0 regression gets bisected.
    bool nrPasses = true;

    // Run the model pass through HIP on the ROCm runtime. Degrades to the
    // identity model when ROCm cannot be loaded, so the DLL still works on
    // machines without it.
    bool hipBackend = true;

    // Self-test only: directory holding the extracted DLSSNR tensors, the
    // rocWMMA include dir, and ROCm's own include dir (hiprtc's implicit
    // path lacks it). When all three are set, the first successful model
    // run also pushes the real QKV weights (3072x1024 e4m3) through the
    // rocWMMA FP8 GEMM with a one-hot input -- bit-exact, so the checksum
    // the log prints proves the whole weights-to-mma chain. Dev-machine keys.
    std::wstring hipWeightsDir;
    std::wstring hipRocwmmaInc;
    std::wstring hipRocInc;

    // 0 off, 1 one-shot frontend-block proof (HANDOFF §34), 2 = 1 plus the
    // 64x32 block answer delivered into sharedOut on every model run
    // (debug view; moves the strength-0 bit-identical check while set --
    // never enable it in a proof run), 3 = staged-window variant of the
    // proof on staged proxy bytes (8x8 window at HipFeWindX/Y, RGBA8-only
    // gate; logs the window bytes + block checksum for the offline oracle
    // recompute; manual-ini-only, moves d7 by design). Needs
    // HipWeightsDir + HipRocInc.
    int hipFeBlock = 0;

    // S148: the FFN weights are stored OUT-major [out][in] (every mma in every
    // cubin is m16n8k32.row.col, whose B operand is an (N,K) row-major matrix).
    // Default 1 = read them that way. Set 0 to fall back to the old in-major
    // [in][out] read, which is what every golden before S148 encoded. The
    // harness FFN checks only pin the DEFAULT reading; with 0 set they log a
    // skip rather than fail, since the golden compiled in is the default one.
    // This exists so the in-game check can A/B the two readings directly.
    // S229b/S232: default 2 = TINLAYOUT in OUR bit order. The blob IS in a
    // bit-level tile order -- S116 derived ours from PTX arithmetic, their
    // derive_native_ffn_layout.py derived theirs from recovered connectivity --
    // and the dense reading (1) is neither. Measured through pass 8: the tiled
    // arms sit at mean|d| 0.104 (ours) and 0.111 (theirs) against the untouched
    // frame, dense at 0.243/0.244.
    //
    // Mode 4 was briefly the default on the argument that its map is
    // validation-backed. S232 withdraws that: the two maps are not two answers to
    // one question, they are built for DIFFERENT DESTINATION GRIDS. Theirs targets
    // [C][4C] -- 4C = heads*H1P, their recovery asserts hidden_count = 4C -- which
    // is a head-block-diagonal expansion of our compact [W*heads][H1P]. Their
    // bit maps fed through our stride do not tile our region: w2's destinations
    // reach 33663 in a 32768-byte region at C=256, so 896 bytes are never written
    // and 896 spill into w3 (128/32 bytes at C=128/64). Ours writes dst[n*K + k]
    // over n<N, k<K, which covers the region exactly by construction, and
    // tools/wload_map.py check reports BIJECTION+ROUNDTRIP OK. The field proxy
    // already preferred ours, 0.104 against 0.111.
    int hipFfnTranspose = 2;

    // S176: THE WIRING SWITCH. Until now the production model path was
    // RunModel()'s copyKernel -- a byte copy, so NR changed nothing no matter
    // what the blocks computed. With this set, the verified chain runs on the
    // REAL staged proxy pixels EVERY FRAME and its answer is written into the
    // output staging buffer, so NR finally has a visible effect.
    //
    // Gated off by default: the chain currently implements the C=64 connected
    // block only, out of 71 blocks, so this renders a 32x32 patch of real
    // network output inside an otherwise untouched frame. It proves the
    // production path can carry network compute -- it is not a correct
    // renderer, and it must not be mistaken for one.
    int hipFeLive = 0;

    // S191: how far the live write-back blends the frame toward the model's
    // refined samples, 0..1. 0 leaves the frame bit-identical, so it is safe to
    // leave the path wired in. This is the first knob that changes the IMAGE
    // rather than a diagnostic overlay.
    float hipFeStrength = 0.3f;

    // Level-3 window origin in staged-proxy pixels (default top-left;
    // out-of-range logs a skip instead of running). Manual runs only.
    int hipFeWindX = 0, hipFeWindY = 0;

    // S209: run block 4 -- the C=32 stage with block 4's weights followed by the
    // width projection (S208) -- so the live field is [64][64] instead of
    // [64][32], i.e. the first multi-width activation on the live path. 0 falls
    // back to the C=32 block (§198), which is what the image has shown so far.
    int hipFeTransition = 1;

    // Diagnostic only: replay one validated 256x256 candidate frame through
    // the model texture with DebugView=2. This is fixed-image output, not live
    // network inference; empty keeps the normal model path.
    std::wstring candidatePreviewPath;

    // Diagnostic one-shot capture of the completed staged proxy. This is the
    // input boundary for a future full-frame candidate, not a model output.
    std::wstring candidateInputCapturePath;

    // Scene value the game's tonemapper calls white. Everything the encode
    // does is relative to it, and getting it wrong is not a subtle error: the
    // proxy is either crushed or clipped and the model is shown a picture that
    // does not exist.
    float whitePoint = 1.0f;

    // Which proxy curve the encode uses: 0 the old soft knee, 1 the hybrid,
    // 2 scale-and-encode only. See the encode block in dlssnr.hlsl and
    // DlssNrConstants::ProxyMode.
    int proxyMode = 2;

    // Ceiling on how far a pixel may be brightened. The transfer is a ratio
    // and a ratio against a near-black pixel is unbounded without one.
    float maxRatio = 4.0f;

    // 1 = the game's buffer is already display-encoded, so the encode is a
    // copy. Some games hand DLSS a frame that has been through their own
    // tonemapper; converting one that needs no conversion is pure damage.
    int passthrough = 0;

    // 0 off, 1 the proxy, 2 the model's answer, 3 the difference amplified.
    int debugView = 0;

    bool dumpFrames = false;
    int dumpEvery = 0;  // 0 = only frame 0; n>0 = every n-th evaluate
    std::wstring dumpDir;
    // S223d: 1 = the dump of an f16 buffer is a window on the GAIN FIELD the
    // live write-back produces (128 = 1.0, [0.8, 1.2] -> [0, 255]) instead of
    // the display fold, which clips a field that lives near 1.0 to white.
    int dumpField = 0;
};

Config& Cfg();
void ConfigLoad(const std::wstring& dir);

// -------------------------------------------------------------- parameter --

// The real NGX block stores a value per (name, type). We keep one tagged
// value per name and coerce on Get, which is what makes Set/Get round-trips
// work regardless of which overload the game happened to use.
struct ParamValue {
    enum Type { ULL, F, D, UI, I, RES11, RES12, VOIDP, EMPTY };
    Type type = EMPTY;
    union {
        unsigned long long ull;
        float f;
        double d;
        unsigned int ui;
        int i;
        ID3D11Resource* p11;
        ID3D12Resource* p12;
        void* vp;
    } v{};

    ParamValue() { v.ull = 0; }
    static ParamValue FromULL(unsigned long long x) { ParamValue p; p.type = ULL; p.v.ull = x; return p; }
    static ParamValue FromF(float x)                { ParamValue p; p.type = F;   p.v.f = x;   return p; }
    static ParamValue FromD(double x)               { ParamValue p; p.type = D;   p.v.d = x;   return p; }
    static ParamValue FromUI(unsigned int x)        { ParamValue p; p.type = UI;  p.v.ui = x;  return p; }
    static ParamValue FromI(int x)                  { ParamValue p; p.type = I;   p.v.i = x;   return p; }
    static ParamValue FromRes11(ID3D11Resource* x)  { ParamValue p; p.type = RES11; p.v.p11 = x; return p; }
    static ParamValue FromRes12(ID3D12Resource* x)  { ParamValue p; p.type = RES12; p.v.p12 = x; return p; }
    static ParamValue FromVoid(void* x)             { ParamValue p; p.type = VOIDP; p.v.vp = x; return p; }

    std::string Describe() const;
};

// Public SDK vtable. Method order must match ext/nvngx_sdk/nvsdk_ngx_params.h.
class ParameterImpl final : public NVSDK_NGX_Parameter {
public:
    void Set(const char* name, unsigned long long value) override;
    void Set(const char* name, float value) override;
    void Set(const char* name, double value) override;
    void Set(const char* name, unsigned int value) override;
    void Set(const char* name, int value) override;
    void Set(const char* name, ID3D11Resource* value) override;
    void Set(const char* name, ID3D12Resource* value) override;
    void Set(const char* name, void* value) override;

    NVSDK_NGX_Result Get(const char* name, unsigned long long* out) const override;
    NVSDK_NGX_Result Get(const char* name, float* out) const override;
    NVSDK_NGX_Result Get(const char* name, double* out) const override;
    NVSDK_NGX_Result Get(const char* name, unsigned int* out) const override;
    NVSDK_NGX_Result Get(const char* name, int* out) const override;
    NVSDK_NGX_Result Get(const char* name, ID3D11Resource** out) const override;
    NVSDK_NGX_Result Get(const char* name, ID3D12Resource** out) const override;
    NVSDK_NGX_Result Get(const char* name, void** out) const override;

    void Reset() override;

    // Non-virtual helpers used by the shim itself.
    const ParamValue* Find(const char* name) const;
    bool TryGetUI(const char* name, unsigned int& out) const;
    bool TryGetULL(const char* name, unsigned long long& out) const;
    bool TryGetF(const char* name, float& out) const;
    bool TryGetRes12(const char* name, ID3D12Resource** out) const;
    bool TryGetVoid(const char* name, void** out) const;
    void CopyFrom(const ParameterImpl& other);
    std::vector<std::pair<std::string, ParamValue>> Snapshot() const;
    std::string Dump() const;

private:
    mutable std::mutex m_;
    std::unordered_map<std::string, ParamValue> map_;
};

// --------------------------------------------------------------- features --

struct SubRect { unsigned int x = 0, y = 0, w = 0, h = 0; };

struct Feature {
    NVSDK_NGX_Handle handle{};
    NVSDK_NGX_Feature type = NVSDK_NGX_Feature_Reserved0;

    // Latched at create time. The reference is emphatic: writing these only at
    // evaluate time does nothing, so we snapshot them here.
    unsigned int width = 0, height = 0;          // render / input
    unsigned int targetWidth = 0, targetHeight = 0;  // display / output
    unsigned int perfQuality = 0;
    int createFlags = 0;
    bool enableOutputSubrects = false;

    std::vector<std::pair<std::string, ParamValue>> createParams;

    uint64_t evaluateCount = 0;
};

class FeatureRegistry {
public:
    Feature* Create(NVSDK_NGX_Feature type, const ParameterImpl& createParams);
    Feature* Find(const NVSDK_NGX_Handle* h);
    bool Release(const NVSDK_NGX_Handle* h);
    size_t Count() const;
    void Clear();

private:
    mutable std::mutex m_;
    unsigned int nextId_ = 0x444C5301;  // "DLS"
    std::unordered_map<unsigned int, std::unique_ptr<Feature>> byId_;
};

FeatureRegistry& Registry();

// --------------------------------------------------------------- gpu side --

enum DlssNrMode : unsigned int {
    NR_MODE_ENCODE = 0,
    NR_MODE_RESOLVE = 1,
    NR_MODE_DOWNSAMPLE = 2,
};

// The CPU mirror of cbuffer Params in shaders/dlssnr.hlsl.
//
// These are root constants, not a CBV. The reference needs one constant buffer
// per descriptor heap because three dispatches recorded on one command list
// all map and overwrite a single upload buffer before any of them runs --
// encode and downsample end up reading the resolve's parameters. Root
// constants are copied into the command stream, so there is no buffer to
// fight over and no 256-byte alignment rule to get wrong.
//
// The size still has to match the shader's CB0[5]: 20 dwords.
struct DlssNrConstants {
    unsigned int Mode = NR_MODE_ENCODE;
    float WhitePoint = 1.0f;
    unsigned int Width = 0, Height = 0;
    float TransferStrength = 0.0f;
    float ColourStrength = 0.0f;
    unsigned int DebugView = 0;
    float MaxRatio = 4.0f;
    unsigned int Passthrough = 0;
    float MvScaleX = 0.0f, MvScaleY = 0.0f;
    unsigned int GuideWidth = 0, GuideHeight = 0;
    unsigned int CompareMode = 0;
    float CompareSplit = 0.5f, CompareZoom = 1.0f;
    unsigned int CompareSwap = 0;
    // 0 the old soft knee, 1 the hybrid (identity through the midtones, a
    // reversible roll above), 2 scale-and-encode only -- the default, and the
    // one the model should be shown. See the encode block in dlssnr.hlsl.
    unsigned int ProxyMode = 2;
    unsigned int Pad1 = 0, Pad2 = 0;
};
// Must match the shader's `dcl_constantbuffer CB0[5]`. A mismatch here is
// silent: the constants land in the wrong registers and the frame is wrong in
// a way that looks like a colour bug rather than a layout bug.
static_assert(sizeof(DlssNrConstants) == 80, "DlssNrConstants must be 20 dwords");

// A texture the shim owns: the proxy the model is shown, the untouched copy of
// the frame, and the model's own output. State is tracked here because the
// resource is ours from creation to death and the game never sees it.
struct PooledTexture {
    std::wstring tag;
    ComPtr<ID3D12Resource> res;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    unsigned int w = 0, h = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct GpuContext {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;        // our own, for internal work
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> scratchList;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;

    ComPtr<ID3D12DescriptorHeap> heapCbvSrvUav;  // shader visible
    UINT cbvSrvUavStride = 0;
    UINT nextDescriptor = 0;

    ComPtr<ID3D12RootSignature> blitRoot;
    ComPtr<ID3D12PipelineState> blitPso;
    ComPtr<ID3D12DescriptorHeap> samplerHeap;  // static linear sampler

    ComPtr<ID3D12RootSignature> nrRoot;
    ComPtr<ID3D12PipelineState> nrPso;

    // The frame counter: a 16-byte UAV buffer the shim writes as the LAST
    // op it records on the game's command list each evaluate. Being last, a
    // read of k proves list k executed in full. The next evaluate reads it
    // through our own queue and accepts n-2: a game that records a frame
    // ahead of submission (Cyberpunk does, permanently) can only ever expose
    // that much. Cleared to 0 at init -- a fresh DEFAULT-heap allocation is
    // not guaranteed to read as zero, and 0 is the floor the first evaluates
    // rely on (the gate clamps its want to 0 until n >= 2).
    ComPtr<ID3D12RootSignature> counterRoot;
    ComPtr<ID3D12PipelineState> counterPso;
    ComPtr<ID3D12Resource> counterBuf;

    // The one-frame-late HIP model. hipFrameReady says the model texture
    // holds the previous frame's answer and the resolve may bind it; the
    // format/size record what was staged, so a mid-stream resize degrades
    // to the identity model for one frame instead of mismatching.
    bool hipFrameReady = false;
    DXGI_FORMAT hipFmt = DXGI_FORMAT_UNKNOWN;
    unsigned int hipW = 0, hipH = 0;

    // Which of the two ping-ponged model textures prepare #n wrote (n & 1).
    // Frame n's resolve binds this slot; prepare #(n+1) writes the other, so
    // the list that is still in flight never reads a texture we rewrite. The
    // gate's n-2 tolerance depends on this -- see GpuPrepareHipModel.
    unsigned int hipModelParity = 0;

    // Held by unique_ptr so a pointer to a pooled texture stays valid when the
    // pool grows. GpuNeuralChain holds `proxy` across the acquisition of
    // `keep`, and a vector<PooledTexture> reallocates on that push_back --
    // which handed the encode a dangling PooledTexture and made the whole
    // chain fail with no error anywhere.
    std::vector<std::unique_ptr<PooledTexture>> pool;

    bool valid = false;
};

GpuContext& Gpu();

bool GpuInit(ID3D12Device* dev);
void GpuShutdown();

// Transition helper. Uses the caller's command list so ordering is the
// game's problem, not ours.
void GpuTransition(ID3D12GraphicsCommandList* cl, ID3D12Resource* res,
                   D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

// Scaled copy src(sub) -> dst(sub) via a compute pass. dst must be UAV-capable.
// This is the "identity upscaler" that makes the shim installable before the
// neural core exists; the RenoDX resolve will reuse the same scaffolding.
bool GpuBlit(ID3D12GraphicsCommandList* cl, ID3D12Resource* src, SubRect srcRect,
             ID3D12Resource* dst, SubRect dstRect);

// ----------------------------------------------------------- colour passes --

// A typeless resource cannot be viewed, and the buffer the upscaler writes is
// occasionally declared that way, so the typed member of the same family is
// substituted.
DXGI_FORMAT TypedFormat(DXGI_FORMAT f);

// The best guess available for a resource the game owns and we have not
// touched: derived from the desc flags, because NGX hands us resources with no
// state and there is no way to query one. WRONG for anything our own passes
// have since transitioned -- call it only on a resource in the state it
// arrived in.
D3D12_RESOURCE_STATES GpuGuessIncomingState(ID3D12Resource* res);

// A texture the shim owns, recreated only when the size or format changes.
// Never null on success. The tag is what makes the lookup stable across
// resizes: "proxy", "keep", "model".
PooledTexture* GpuAcquireTexture(const wchar_t* tag, DXGI_FORMAT fmt,
                                 unsigned int w, unsigned int h);

// Transition a pooled texture, tracking its state so the next pass does not
// have to guess.
void GpuSetState(ID3D12GraphicsCommandList* cl, PooledTexture& t,
                 D3D12_RESOURCE_STATES want);

// One dispatch of the colour shader. Slots a mode does not read may be null;
// a stand-in is bound in their place, because an unbound descriptor is not an
// empty read, it is a read from nothing.
bool GpuNrDispatch(ID3D12GraphicsCommandList* cl, const DlssNrConstants& c,
                   ID3D12Resource* source, ID3D12Resource* model,
                   ID3D12Resource* original, ID3D12Resource* motion,
                   ID3D12Resource* prevEdit, ID3D12Resource* target,
                   ID3D12Resource* keep);

// encode -> [model] -> resolve on the caller's command list, over a frame that
// has already been resampled into `output`.
//
// `output` arrives as the resampled frame and leaves as the finished one, so
// it is the only resource here the game also owns; every barrier on it is
// against the state NGX says it is handed to us in.
//
// The model is the identity unless the HIP backend has an answer ready:
// `proxy` is bound as both the model's input and its answer, so the colour
// pipeline runs and changes nothing. When the backend is live, the proxy is
// additionally staged into the shared input buffer for the next evaluate.
bool GpuNeuralChain(ID3D12GraphicsCommandList* cl, ID3D12Resource* output,
                    SubRect outRect, const Config& cfg);

// ------------------------------------------------------------- HIP backend --

// hip_backend.cpp: everything ROCm. All entry points resolve the runtime
// dynamically (amdhip64_7.dll / hiprtc0715.dll) and degrade to "not
// available" when ROCm is absent, so the DLL works on any machine.
//
//   HipStartup        load runtime, pick device 0, compile the identity kernel
//   HipShutdown       drop imports and handles (call before the device dies)
//   HipEnsureStaging  (re)create + import the shared in/out buffers for
//                     w x h x bpp bytes (row pitch 256-aligned, exposed by
//                     HipStagingRowPitch)
//   HipRunModel       launch the model kernel on the staged input and
//                     synchronise; sharedOut then holds the answer
//   HipStagingIn/Out  the D3D12 resources, for the copies around the launch
//
// The model kernel is an identity byte copy for now; TensorBackend's real
// network replaces exactly that.
bool HipStartup();
void HipShutdown();
bool HipEnsureStaging(unsigned int w, unsigned int h, unsigned int bpp);
bool HipRunModel();
bool HipSelfTest();  // one-shot FP8 GEMM with real weights; see Config
bool HipChainTest();  // one-shot Stage-1 v4 chain on real weights (§27)
bool HipFeBlockTest();  // one-shot frontend+chain debug block (§34)
bool HipB2BlockTest();  // one-shot block0->block1 chain on real weights (§35)
bool HipB3BlockTest();  // one-shot block3 on staged X3e, real weights (§37)
bool HipB67BlockTest();  // one-shot block67 on staged X67e, real weights (§38)
bool HipB68BlockTest();  // one-shot block68 on staged X68e, real weights (§39)
bool HipB69BlockTest();  // one-shot block69 on staged X69e, real weights (§40)
bool HipTailRun();  // unscored chained 67->68->69 device run, INFO only (§41)
bool HipBlock2Test();  // one-shot block2 on staged Xblock2, real weights (§42; NOT b2-block, which is block1)
bool HipB4BlockTest();  // one-shot block4 prefix on staged X4e, real weights (§45, Test-14)
bool HipB4DsBlockTest();  // one-shot block4 prefix + 64x32 ds proj, real weights (§45, Test-15)
bool HipC64sBlockTest();  // one-shot C=64 2-head temp+bias scores, real weights (§52, Test-16)
bool HipC64eBlockTest();  // one-shot C=64 trick-exp episode on staged S (§52, Test-17)
bool HipC64oBlockTest();  // one-shot C=64 Pq x V context on staged Pq/V (§78, Test-18)
bool HipC64pBlockTest();  // one-shot C=64 proj+gate2 on staged Ocat/y, real Wproj (§78, Test-19)
bool HipC64fBlockTest();  // one-shot C=64 FFN on staged x, real w1/w2/gate1 (§78, Test-20)
bool HipC64qBlockTest();  // one-shot C=64 QKV GEMM on staged X, real Wqkv (§90, Test-21)
bool HipC64cBlockTest();  // one-shot C=64 FFN contract on staged h, real W2 (§111, Test-22)
bool HipC64xBlockTest();  // one-shot C=64 FFN expand on staged x, real w1 (§113, Test-23)
bool HipC64f2BlockTest();
bool HipC128f2BlockTest();  // one-shot CONNECTED C=128 FFN (Phase A)
bool HipC256f2BlockTest();  // one-shot CONNECTED C=256 FFN (Phase A)
bool HipVitFfn2BlockTest();  // one-shot ViT block31 FFN (Phase C)
bool HipVitQkvBlockTest();   // one-shot ViT block31 QKV (Phase C)
bool HipVitProjBlockTest();  // one-shot ViT block31 projection (Phase C)  // one-shot CONNECTED C=64 FFN, expand->act->contract (§116, Test-24)
bool HipC64blkBlockTest();
bool HipC128blkBlockTest(); // connected C=128 block (S160)
bool HipC256blkBlockTest(); // connected C=256 block (S169)
bool HipC32blkBlockTest();  // connected C=32 block (S170)  // one-shot CONNECTED C=64 BLOCK, FFN->QKV->attn->proj (§116, Test-25)
void HipFeBlockView();  // level-2 cached-block view into sharedOut (§34)
bool HipFeBlockStaged();  // level-3 staged-window run, real pixels (§34)
bool HipCandidatePreview();  // fixed-image model-texture bridge, DebugView=2 only
bool HipCandidateInputCapture();  // one-shot staged-proxy capture
UINT64 HipStagingRowPitch();
UINT64 HipStagingBytes();
ID3D12Resource* HipStagingIn();
ID3D12Resource* HipStagingOut();
bool HipUsable();

// The one-frame-late choreography, driven from DoEvaluateD3D12:
//
//   GpuPrepareHipModel(n, cfg)   HOST side, before anything is recorded:
//                                verifies the frame counter reads n-1 (proof
//                                the previous list's shim ops executed), runs
//                                the HIP model on the staged input, and
//                                refreshes the pooled model texture from the
//                                shared output on our own queue. No-op when
//                                the backend is off; never blocks on the game.
//   GpuHipFrameReady()           whether the model texture currently holds a
//                                HIP answer the resolve may bind.
//   GpuWriteFrameCounter(cl, n)  RECORD side, as the LAST op of the evaluate:
//                                the 1x1 counter dispatch that the next
//                                GpuPrepareHipModel waits to see.
bool GpuPrepareHipModel(uint64_t evaluateIndex, const Config& cfg);
bool GpuHipFrameReady();
void GpuWriteFrameCounter(ID3D12GraphicsCommandList* cl, uint64_t n);

// -------------------------------------------------------------- constants ---

constexpr unsigned int kNrConstantDwords = sizeof(DlssNrConstants) / 4;
static_assert(sizeof(DlssNrConstants) == kNrConstantDwords * 4,
              "DlssNrConstants must be a whole number of dwords");

// Debug dumps.
//
// The copy has to be recorded on the CALLER's command list, not our own queue:
// our queue has no ordering relationship with the game's, so copying from our
// queue reads the texture before the frame's work has run (measured: an
// all-black dump). So: queue the copy here, then read the buffer back a couple
// of evaluates later, by which time the game has certainly submitted.
//
// `stateIn` is the state the resource is actually in right now. Our own passes
// leave Output in UAV, which is not what the desc-flag guess would say, and a
// transition with the wrong StateBefore is a validation error that only shows
// up under a debug layer or a stricter driver.
bool GpuQueueDump(ID3D12GraphicsCommandList* cl, ID3D12Resource* res, SubRect rect,
                 const std::wstring& path, uint64_t nowEvaluate,
                 D3D12_RESOURCE_STATES stateIn);
// Writes out any dump whose data is ready. Call once per evaluate and at
// shutdown.
void GpuDrainDumps(uint64_t nowEvaluate, bool drainAll);

// ------------------------------------------------------------- API surface --

// Called from dllmain so the log and config can be located before Init.
void SetModuleDir(const std::wstring& dir);

// Implemented in ngx_api.cpp.
NVSDK_NGX_Result DoInit(unsigned long long appId, const wchar_t* appDataPath,
                        ID3D12Device* dev, const NVSDK_NGX_FeatureCommonInfo* info,
                        NVSDK_NGX_Version ver);
NVSDK_NGX_Result DoShutdown();
NVSDK_NGX_Result DoEvaluateD3D12(ID3D12GraphicsCommandList* cl,
                                 const NVSDK_NGX_Handle* handle,
                                 ParameterImpl* params);

}  // namespace ngx
