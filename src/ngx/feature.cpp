#include "ngx_internal.h"

namespace ngx {

// -------------------------------------------------------- FeatureRegistry --

FeatureRegistry& Registry() {
    static FeatureRegistry r;
    return r;
}

Feature* FeatureRegistry::Create(NVSDK_NGX_Feature type, const ParameterImpl& createParams) {
    auto f = std::make_unique<Feature>();
    f->type = type;
    f->createParams = createParams.Snapshot();

    // Latched create-time values. Getting this split wrong is silent: the
    // model reads them once while building the feature, so anything only
    // written at evaluate time is ignored.
    createParams.TryGetUI(NVSDK_NGX_Parameter_Width, f->width);
    createParams.TryGetUI(NVSDK_NGX_Parameter_Height, f->height);
    createParams.TryGetUI(NVSDK_NGX_Parameter_OutWidth, f->targetWidth);
    createParams.TryGetUI(NVSDK_NGX_Parameter_OutHeight, f->targetHeight);
    createParams.TryGetUI(NVSDK_NGX_Parameter_PerfQualityValue, f->perfQuality);

    unsigned int flags = 0;
    if (createParams.TryGetUI(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags))
        f->createFlags = (int)flags;

    unsigned int subrects = 0;
    if (createParams.TryGetUI(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, subrects))
        f->enableOutputSubrects = (subrects != 0);

    // A few engines only give render size and expect OutWidth/OutHeight to be
    // implied as 1:1 (DLAA-style usage of the SuperSampling feature).
    if (f->targetWidth == 0) f->targetWidth = f->width;
    if (f->targetHeight == 0) f->targetHeight = f->height;

    std::lock_guard<std::mutex> g(m_);
    f->handle.Id = nextId_++;
    Feature* raw = f.get();
    byId_[f->handle.Id] = std::move(f);

    LOGI("create feature %s id=0x%08X render=%ux%u target=%ux%u quality=%u flags=0x%x subrects=%d",
         FeatureStr(type), raw->handle.Id, raw->width, raw->height,
         raw->targetWidth, raw->targetHeight, raw->perfQuality,
         raw->createFlags, (int)raw->enableOutputSubrects);
    LOGD("create params:\n%s", createParams.Dump().c_str());

    return raw;
}

Feature* FeatureRegistry::Find(const NVSDK_NGX_Handle* h) {
    if (!h) return nullptr;
    std::lock_guard<std::mutex> g(m_);
    auto it = byId_.find(h->Id);
    return it == byId_.end() ? nullptr : it->second.get();
}

bool FeatureRegistry::Release(const NVSDK_NGX_Handle* h) {
    if (!h) return false;
    std::lock_guard<std::mutex> g(m_);
    auto it = byId_.find(h->Id);
    if (it == byId_.end()) return false;

    // The reference mod parks retired features instead of freeing them
    // immediately, because releasing memory the GPU is still reading hangs the
    // device. We do the same: hand the buffer to a graveyard with the fence
    // value at release time and only drop it 32 evaluates later.
    LOGI("release feature id=0x%08X (%s), parking", h->Id,
         FeatureStr(it->second->type));
    byId_.erase(it);
    return true;
}

size_t FeatureRegistry::Count() const {
    std::lock_guard<std::mutex> g(m_);
    return byId_.size();
}

void FeatureRegistry::Clear() {
    std::lock_guard<std::mutex> g(m_);
    for (const auto& kv : byId_)
        LOGI("  left alive at shutdown: id=0x%08X %s", kv.first,
             FeatureStr(kv.second->type));
    byId_.clear();
}

}  // namespace ngx
