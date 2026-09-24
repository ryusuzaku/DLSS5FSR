#!/usr/bin/env python3
"""tools/extract_chain.py -- regenerate hip/mvp1/swin_1h_chain.hip from
hip/mvp1/swin_1h.hip §2 (HANDOFF §26).

The chain file is a verbatim extraction (modulo documented linkage
differences); never hand-edit drift into it -- change swin_1h.hip and
re-run this. Verifies kernel-body identity after writing.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "hip", "mvp1", "swin_1h.hip")
DST = os.path.join(ROOT, "hip", "mvp1", "swin_1h_chain.hip")
#
# Phase A2 (HANDOFF 135): the stage dimensions are measured constants EXCEPT
# the channel width, so one source emits a chain per width by overriding
# exactly two macros. W (32) and H1P (128) do NOT change with width -- 135
# measured them constant and 136 verified the layout against real tensors.
WIDTH = 64
if "--width" in sys.argv:
    WIDTH = int(sys.argv[sys.argv.index("--width") + 1])
# S145: the ViT FFN is THIS SAME kernel with ONE pass, i.e. W == C, and H1 = 4096.
# The ViT block's layer0 is [4096,1024] and layer1 [1024,4096] with a 2048-byte
# fp16 residue = exactly 1024 gates = our gate1[W]. So W and H1P must be
# overridable as well; the width-only path stays byte-identical.
PASS_W = 32
if "--w" in sys.argv:
    PASS_W = int(sys.argv[sys.argv.index("--w") + 1])
H1P = 128
if "--h1" in sys.argv:
    H1P = int(sys.argv[sys.argv.index("--h1") + 1])
_STEM = "swin_1h_chain"
if (WIDTH, PASS_W, H1P) != (64, 32, 128):
    _STEM += "_c%d" % WIDTH
    if PASS_W != 32:
        _STEM += "w%d" % PASS_W
    if H1P != 128:
        _STEM += "h%d" % H1P
DST = os.path.join(ROOT, "hip", "mvp1", _STEM + ".hip")
SUF = "" if WIDTH == 64 else "C%d" % WIDTH
if PASS_W != 32:
    SUF += "W%d" % PASS_W
if H1P != 128:
    SUF += "H%d" % H1P

HOST_MIRROR_HDR = """// ---------------------------------------------------------------------------
// 1. Host mirror (pure C++; identical math to the kernels, f32)
// ---------------------------------------------------------------------------"""


def section(src, start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[i:j].rstrip() + "\n"


import re

# hiprtc nests uintN_t in __hip_internal:: (version-coupled); fundamental
# spellings are valid under both hiprtc and hipcc with identical widths on
# every platform we target (8/16/32-bit char/short/int, LP64 and LLP64).
TYPEMAP = (("uint8_t", "unsigned char"), ("uint16_t", "unsigned short"),
           ("uint32_t", "unsigned int"))


def detype(s):
    for a, b in TYPEMAP:
        s = re.sub(r"\b%s\b" % a, b, s)
    return s


def main():
    src = open(SRC).read()
    dims = section(src, "#define SW_D_IN 16", "static uint32_t sw_lcg_next")
    assert HOST_MIRROR_HDR in dims
    dims = dims.replace(HOST_MIRROR_HDR, "").rstrip() + "\n"
    assert "Host mirror" not in dims
    f32_to_f16 = section(src, "HD inline uint16_t sw_f32_to_f16_bits",
                         "// f16 bits -> f32 (exact).")
    f16_to_f32 = section(src, "static float sw_f16_to_f32",
                         "// Single-rounding f16 FMA")
    h2e4m3 = section(src, "HD inline uint8_t sw_h2e4m3",
                     "// e4m3 -> f32 dequant")
    e4m3_dec = section(src, "HD inline float sw_e4m3_decode",
                       "// Y[m][o] = sum_k")
    expdefs = section(src, "// Fast-exp softmax constants",
                      "// QK-norm rows")
    mpcubic_half = section(src, "__device__ inline __half sw_mpcubic_half",
                           "// patch GEMM: e4m3 X[M][16]")
    kernels_raw = section(src, "__global__ void k_patch_gemm",
                          "// primitive-proof kernels")
    # kernel bodies verbatim modulo linkage+types (checked, not trusted)
    i = src.index("__global__ void k_patch_gemm")
    j = src.index("// primitive-proof kernels", i)
    assert (detype(kernels_raw) ==
            detype(src[i:j].rstrip() + "\n")), "kernel drift detected"
    kernels = kernels_raw

    out = []
    out.append(
        "// hip/mvp1/swin_1h_chain.hip -- production kernel source for the\n"
        "// Stage-1 v4 chain (HANDOFF §26). Device-only, self-contained: the\n"
        "// HIP backend (src/ngx/hip_backend.cpp) feeds this file to hiprtc;\n"
        "// hipcc test mains prove the same kernels via hip/mvp1/swin_1h.hip\n"
        "// §2.\n"
        "//\n"
        "// KEEP IN SYNC with swin_1h.hip §2 -- regenerate with\n"
        "// tools/extract_chain.py, never hand-edit. The 21/21 + 15/15\n"
        "// suites pin swin_1h.hip's copies; the backend's blockout check\n"
        "// pins this file's copies at every real-weights run.\n"
        "// Differences from the §2 copies: extern \"C\" (hipModuleGetFunction\n"
        "// lookup), __device__ helpers, fundamental type spellings\n"
        "// (hiprtc nests uintN_t in __hip_internal::), no host/test code.\n"
        "// Use: hiprtc compiles this file standalone (built-in device vars,\n"
        "// like the proven kSelfTestSource); hipcc test mains must include\n"
        "// <hip/hip_runtime.h> BEFORE including this file.\n"
        "#include <hip/hip_fp16.h>\n"
        "// NOTE: no <stdint.h> -- it collides with hiprtc's builtin runtime\n"
        "// header (LP64 vs LLP64 typedefs); uintN_t come from hiprtc\n"
        "// implicitly, and from <hip/hip_runtime.h> under hipcc (which must\n"
        "// be included first there).\n")
    # The pre-block parameter block, emitted INLINE. The backend feeds this
    # file to hiprtc, which compiles it in a temp dir with no include path into
    # hip/mvp1 -- an #include of sw_fe_params.inc here cannot resolve (found the
    # hard way: "fatal error: 'sw_fe_params.inc' file not found", HANDOFF §120.6).
    out.append(open(os.path.join(ROOT, "hip", "mvp1",
                                 "sw_fe_params.inc")).read())

    parts = [dims.replace("HD inline", "__device__ inline"),
             f32_to_f16.replace("HD inline", "__device__ inline"),
             f16_to_f32.replace("static float",
                                "__device__ inline float"),
             h2e4m3.replace("HD inline", "__device__ inline"),
             e4m3_dec.replace("HD inline", "__device__ inline"),
             expdefs, mpcubic_half,
             kernels.replace("__global__ void k_",
                             'extern "C" __global__ void k_')]
    out.extend(detype(p) for p in parts)
    body = "\n".join(out)

    n_k = body.count('extern "C" __global__')
    # 13 chain kernels + k_frontend (§34: the §13.3 front-end, RGBA8 proxy ->
    # X e4m3 [64][16]) + k_decode_e4m3 (§41: tail-run bridges, exact)
    # + k_c64scores (§52/Test-16: C=64 2-head temp+bias scores)
    # + k_c64ctx (§78/Test-18: C=64 Pq x V context)
    # + k_c64proj (§78/Test-19: C=64 proj + gate2 residual)
    # + k_c64ffn_act/k_c64ffn2 (§78/Test-20: C=64 FFN expand+act,
    #   contract + gate1 residual)
    # + k_c64qkv (§90/Test-21: C=64 QKV GEMM, split-K halves).
    # + k_c64contract (§111/Test-22: C=64 FFN contract, K=128).
    # + k_c64expand (§113/Test-23: C=64 FFN expand, K=64, N=128).
    # + k_c64ffn2 (§116/Test-24: CONNECTED C=64 FFN, expand->act->contract).
    # + k_c64qkv_split / k_c64cat_q (§116/Test-25: CONNECTED C=64 BLOCK glue).
    # Bump this when a kernel joins the chain on purpose.
    # 25 fused-stage kernels + k_vit_qkv (S147, the ViT QKV projection)
    assert n_k == 28, n_k   # + k_live_patch (S181), k_live_refine (S191)
    for sym in ("SW_EPS_NORM", "SW_EXP_MAGIC", "SW_H1 128", "blockIdx",
                "rsqrtf", "__half", "unsigned short", "unsigned int",
                "unsigned char"):
        assert sym in body, sym
    for sym in ("printf", "std::vector", "#include <vector>",
                "static uint32_t sw_lcg_next", "__int128", "HD inline",
                "uint8_t", "uint16_t", "uint32_t"):
        assert sym not in body, sym

    # CRLF-stable: the committed chain files are CRLF (Windows runs);
    # bare "w" would emit LF on Linux and dirty every byte. Writing
    # with newline="\r\n" makes extraction platform-byte-identical.
    if (WIDTH, PASS_W, H1P) != (64, 32, 128):
        import re as _re
        for _pat, _val, _nm in (
                (chr(35) + "define SW_C64_C ", WIDTH, "SW_C64_C"),
                (chr(35) + "define SW_C64_HEADS ", WIDTH // 32, "SW_C64_HEADS"),
                (chr(35) + "define SW_C64_W ", PASS_W, "SW_C64_W"),
                (chr(35) + "define SW_C64_H1P ", H1P, "SW_C64_H1P")):
            _body_new, _n = _re.subn(_re.escape(_pat) + "[0-9]+",
                                     _pat + str(_val), body)
            assert _n == 1, "width override missed " + _nm
            body = _body_new
    open(DST, "w", newline="\r\n").write(body)

    # Backend embedding: the shim compiles this exact source via hiprtc, so
    # ship it as a string literal the .cpp can #include (avoids a third
    # runtime file next to the weights).
    delim = "HIPSWINCHAIN"
    assert (")" + delim + '"') not in body, "delimiter collision"
    # MSVC caps a single string literal at 16380 bytes (C2026); the chain
    # outgrew that (§34 front-end). Emit the source as adjacent raw-string
    # chunks -- the compiler concatenates them back into one literal.
    CHUNK = 8192
    parts = [body[i:i + CHUNK] for i in range(0, len(body), CHUNK)] or [""]
    inc = ("// Generated by tools/extract_chain.py -- DO NOT HAND-EDIT.\n"
           "// Emitted as adjacent raw-string chunks: MSVC rejects a single\n"
           "// literal > 16380 bytes (C2026).\n"
           "static const char kChainSource%s[] =\n" % SUF)
    # NOTE: no newline padding around the chunks -- a split token must
    # rejoin exactly, so the raw strings are emitted back-to-back.
    inc += "\n".join('    R"' + delim + "(" + c + ")" + delim + '"'
                     for c in parts)
    inc += ";\n"
    dst_inc = os.path.join(ROOT, "hip", "mvp1", _STEM + ".inc")
    if False:
        dst_inc = dst_inc.replace("swin_1h_chain",
                                  "swin_1h_chain_c%d" % WIDTH)
    open(dst_inc, "w", newline="\r\n").write(inc)
    print("wrote %s (%d bytes, %d kernels)" % (DST, len(body), n_k))
    print("wrote %s (%d bytes)" % (dst_inc, len(inc)))


if __name__ == "__main__":
    sys.exit(main())
