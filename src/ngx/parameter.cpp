#include "ngx_internal.h"

#include <cstdio>

namespace ngx {

// ------------------------------------------------------------ ParamValue --

std::string ParamValue::Describe() const {
    char buf[128];
    switch (type) {
        case ULL:   snprintf(buf, sizeof(buf), "ull=%llu", v.ull); break;
        case F:     snprintf(buf, sizeof(buf), "f=%.6g", v.f); break;
        case D:     snprintf(buf, sizeof(buf), "d=%.6g", v.d); break;
        case UI:    snprintf(buf, sizeof(buf), "ui=%u", v.ui); break;
        case I:     snprintf(buf, sizeof(buf), "i=%d", v.i); break;
        case RES11: snprintf(buf, sizeof(buf), "res11=%p", (void*)v.p11); break;
        case RES12: snprintf(buf, sizeof(buf), "res12=%p", (void*)v.p12); break;
        case VOIDP: snprintf(buf, sizeof(buf), "void=%p", v.vp); break;
        default:    snprintf(buf, sizeof(buf), "<empty>"); break;
    }
    return std::string(buf);
}

// ------------------------------------------------------------- coercion --

namespace {

// Get() is allowed to cross numeric types: the public SDK block keys on the
// overload used, but callers legitimately Set(uint) and Get(int) on the same
// name (subrect bases especially). Resources are never coerced.
bool AsULL(const ParamValue& p, unsigned long long& out) {
    switch (p.type) {
        case ParamValue::ULL:   out = p.v.ull; return true;
        case ParamValue::UI:    out = p.v.ui; return true;
        case ParamValue::I:     out = (unsigned long long)p.v.i; return true;
        case ParamValue::F:     out = (unsigned long long)p.v.f; return true;
        case ParamValue::D:     out = (unsigned long long)p.v.d; return true;
        case ParamValue::VOIDP: out = (unsigned long long)(uintptr_t)p.v.vp; return true;
        case ParamValue::RES12: out = (unsigned long long)(uintptr_t)p.v.p12; return true;
        case ParamValue::RES11: out = (unsigned long long)(uintptr_t)p.v.p11; return true;
        default: return false;
    }
}
bool AsF(const ParamValue& p, float& out) {
    switch (p.type) {
        case ParamValue::F:   out = p.v.f; return true;
        case ParamValue::D:   out = (float)p.v.d; return true;
        case ParamValue::UI:  out = (float)p.v.ui; return true;
        case ParamValue::I:   out = (float)p.v.i; return true;
        case ParamValue::ULL: out = (float)p.v.ull; return true;
        default: return false;
    }
}
bool AsD(const ParamValue& p, double& out) {
    switch (p.type) {
        case ParamValue::D:   out = p.v.d; return true;
        case ParamValue::F:   out = (double)p.v.f; return true;
        case ParamValue::UI:  out = (double)p.v.ui; return true;
        case ParamValue::I:   out = (double)p.v.i; return true;
        case ParamValue::ULL: out = (double)p.v.ull; return true;
        default: return false;
    }
}
bool AsUI(const ParamValue& p, unsigned int& out) {
    switch (p.type) {
        case ParamValue::UI:  out = p.v.ui; return true;
        case ParamValue::I:   out = (unsigned int)p.v.i; return true;
        case ParamValue::ULL: out = (unsigned int)p.v.ull; return true;
        case ParamValue::F:   out = (unsigned int)p.v.f; return true;
        case ParamValue::D:   out = (unsigned int)p.v.d; return true;
        default: return false;
    }
}
bool AsI(const ParamValue& p, int& out) {
    switch (p.type) {
        case ParamValue::I:   out = p.v.i; return true;
        case ParamValue::UI:  out = (int)p.v.ui; return true;
        case ParamValue::ULL: out = (int)p.v.ull; return true;
        case ParamValue::F:   out = (int)p.v.f; return true;
        case ParamValue::D:   out = (int)p.v.d; return true;
        default: return false;
    }
}

const ParamValue* Lookup(const std::unordered_map<std::string, ParamValue>& m,
                         const char* name) {
    if (!name) return nullptr;
    auto it = m.find(name);
    return it == m.end() ? nullptr : &it->second;
}

}  // namespace

// ------------------------------------------------------------- Set / Get --

#define NGX_SET(TYPE, FIELD, MAKER)                              \
    void ParameterImpl::Set(const char* name, TYPE value) {       \
        if (!name) return;                                        \
        std::lock_guard<std::mutex> g(m_);                        \
        map_[name] = ParamValue::MAKER(value);                    \
        LOGD("param set %s = %s", name, map_[name].Describe().c_str()); \
    }

NGX_SET(unsigned long long, ull, FromULL)
NGX_SET(float, f, FromF)
NGX_SET(double, d, FromD)
NGX_SET(unsigned int, ui, FromUI)
NGX_SET(int, i, FromI)
NGX_SET(ID3D11Resource*, p11, FromRes11)
NGX_SET(ID3D12Resource*, p12, FromRes12)
NGX_SET(void*, vp, FromVoid)
#undef NGX_SET

NVSDK_NGX_Result ParameterImpl::Get(const char* name, unsigned long long* out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return AsULL(*p, *out) ? NVSDK_NGX_Result_Success
                           : NVSDK_NGX_Result_FAIL_InvalidParameter;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, float* out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return AsF(*p, *out) ? NVSDK_NGX_Result_Success
                         : NVSDK_NGX_Result_FAIL_InvalidParameter;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, double* out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return AsD(*p, *out) ? NVSDK_NGX_Result_Success
                         : NVSDK_NGX_Result_FAIL_InvalidParameter;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, unsigned int* out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return AsUI(*p, *out) ? NVSDK_NGX_Result_Success
                          : NVSDK_NGX_Result_FAIL_InvalidParameter;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, int* out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    return AsI(*p, *out) ? NVSDK_NGX_Result_Success
                         : NVSDK_NGX_Result_FAIL_InvalidParameter;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, ID3D11Resource** out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (p->type != ParamValue::RES11) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *out = p->v.p11;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, ID3D12Resource** out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (p->type != ParamValue::RES12) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *out = p->v.p12;
    return NVSDK_NGX_Result_Success;
}
NVSDK_NGX_Result ParameterImpl::Get(const char* name, void** out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (p->type != ParamValue::VOIDP) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *out = p->v.vp;
    return NVSDK_NGX_Result_Success;
}

void ParameterImpl::Reset() {
    std::lock_guard<std::mutex> g(m_);
    map_.clear();
}

// ------------------------------------------------------------- helpers ----

const ParamValue* ParameterImpl::Find(const char* name) const {
    std::lock_guard<std::mutex> g(m_);
    return Lookup(map_, name);
}

bool ParameterImpl::TryGetUI(const char* name, unsigned int& out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    return p && AsUI(*p, out);
}
bool ParameterImpl::TryGetULL(const char* name, unsigned long long& out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    return p && AsULL(*p, out);
}
bool ParameterImpl::TryGetF(const char* name, float& out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    return p && AsF(*p, out);
}
bool ParameterImpl::TryGetRes12(const char* name, ID3D12Resource** out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || p->type != ParamValue::RES12) return false;
    *out = p->v.p12;
    return true;
}
bool ParameterImpl::TryGetVoid(const char* name, void** out) const {
    std::lock_guard<std::mutex> g(m_);
    const ParamValue* p = Lookup(map_, name);
    if (!p || p->type != ParamValue::VOIDP) return false;
    *out = p->v.vp;
    return true;
}

void ParameterImpl::CopyFrom(const ParameterImpl& other) {
    std::lock_guard<std::mutex> g(m_);
    std::lock_guard<std::mutex> g2(other.m_);
    map_ = other.map_;
}

std::vector<std::pair<std::string, ParamValue>> ParameterImpl::Snapshot() const {
    std::lock_guard<std::mutex> g(m_);
    std::vector<std::pair<std::string, ParamValue>> out;
    out.reserve(map_.size());
    for (const auto& kv : map_) out.emplace_back(kv.first, kv.second);
    return out;
}

std::string ParameterImpl::Dump() const {
    auto snap = Snapshot();
    std::string s;
    for (const auto& kv : snap) {
        s += "  ";
        s += kv.first;
        s += " : ";
        s += kv.second.Describe();
        s += "\n";
    }
    return s;
}

}  // namespace ngx
