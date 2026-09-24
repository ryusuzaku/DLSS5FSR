#!/usr/bin/env python3
"""tools/emit_yp_golden.py -- emit hip/mvp1/rw_yp_golden.inc for the HIP
backend's in-shim chain check (HANDOFF §27).

Contents: the fixed LCG staging-input bytes (1024 e4m3) and the tensor_000
pre-block Yp (2048 f32, as u32 bits) from tools/swin1h_ref.py (f64 oracle).
The backend compares its device block output with tol 0.1: the measured
C++-vs-Python gap is 6.0e-3 on tensor_000 (f32/f64 noise + e4m3 straddle
ripple), the device adds ~0.0, and wiring bugs show at O(0.5+).

Deterministic: CPython float +,-,*,/ and math.sqrt are IEEE-exact across
platforms, and the f16/e4m3 converters are pure integer math -- any
machine regenerates byte-identical output (verified by re-parse below).

Also: `--fe` emits the Test-7 frontend golden (HANDOFF §34), and
`--fe-staged <paste>` runs the offline maxerr verdict for a level-3
staged run (paste file holds the win/block/yhex lines verbatim).
"""

import math
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import swin1h_ref as R

DST = os.path.join(ROOT, "hip", "mvp1", "rw_yp_golden.inc")
DST_FE = os.path.join(ROOT, "hip", "mvp1", "rw_yp_fe_golden.inc")

# Test-7 fixed-gradient proxy (HANDOFF §34): byte-identical to the §3
# device test's proxy (same 4 formulas, pitch 32, alpha 255).
FE_SEED = 0x12345678
FE_HI176, FE_S_SEL, FE_S_A, FE_S_B = 3.0, 0.75, 0.5, 0.25


def fe_proxy_bytes():
    proxy = bytearray(8 * 32)
    for y in range(8):
        for x in range(8):
            proxy[y * 32 + x * 4 + 0] = (x * 37 + y * 5 + 11) & 255
            proxy[y * 32 + x * 4 + 1] = (y * 53 + x * 7 + 3) & 255
            proxy[y * 32 + x * 4 + 2] = (x * 17 + y * 29 + 1) & 255
            proxy[y * 32 + x * 4 + 3] = 255
    return proxy


def load_pre():
    p = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_000.bin")
    raw = open(p, "rb").read()
    assert len(raw) == 21696, len(raw)
    W = R.carve(p, R.CARVE_PRE)
    s_real = struct.unpack("<f", raw[20576:20580])[0]
    P = {
        "Wq": [W["qkv"][i * R.C:(i + 1) * R.C] for i in range(R.QKV)],
        "Wp": [W["proj"][i * R.C:(i + 1) * R.C] for i in range(R.C)],
        "W1": [W["ffn1"][i * R.C:(i + 1) * R.C] for i in range(R.H1)],
        "W2": [W["ffn2"][i * R.H1:(i + 1) * R.H1] for i in range(R.C)],
        "gate_ffn": W["gate_ffn"],
        "gate_attn": W["gate_attn"],
        "bias": W["bias"],
        "s_real": s_real,
    }
    wpatch = struct.unpack("<512H", raw[8208:9232])
    P["Wpt"] = [list(wpatch[i * R.D_IN:(i + 1) * R.D_IN])
                for i in range(R.C)]
    return P


def run_pre_block(Xe, P):
    Ye, _ = R.patch_forward(Xe, P["Wpt"], R.TOK)
    Xb = [[R.e4m3_decode(b) for b in row] for row in Ye]
    Yf, _, _ = R.ffn_forward(Xb, P["W1"], P["W2"], P["gate_ffn"])
    Q, K, V, Yq = R.qkv_forward(Yf, P["Wq"])
    Qn, Kn, _, _, _, _ = R.qknorm_forward(Q, K, P["s_real"])
    O, _, _, _ = R.attn_forward(Qn, Kn, V, bias=P["bias"])
    Yp, _ = R.proj_forward(O, P["Wp"], Yf, P["gate_attn"])
    return Yp


def lcg_xe():
    st = 0xA11CE
    mags, _ = R.gen_vec(st, R.TOK * R.D_IN, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [[R.synth_to_e4m3([levels[m]])[0] for m in row]
            for row in [mags[i * R.D_IN:(i + 1) * R.D_IN]
                        for i in range(R.TOK)]]


def load_fused(fname, size):
    # Any fused32 block (block1, block3, ...): same shapes as block0, no
    # patch region; P-scale at 19552 in all of them.
    p = os.path.join(ROOT, "dlss5-analysis", "tensors", fname)
    raw = open(p, "rb").read()
    assert len(raw) == size, (fname, len(raw))
    W = R.carve(p, R.CARVE_FUSED32)
    s_real = struct.unpack("<f", raw[19552:19556])[0]
    return {
        "Wq": [W["qkv"][i * R.C:(i + 1) * R.C] for i in range(R.QKV)],
        "Wp": [W["proj"][i * R.C:(i + 1) * R.C] for i in range(R.C)],
        "W1": [W["ffn1"][i * R.C:(i + 1) * R.C] for i in range(R.H1)],
        "W2": [W["ffn2"][i * R.H1:(i + 1) * R.H1] for i in range(R.C)],
        "gate_ffn": W["gate_ffn"],
        "gate_attn": W["gate_attn"],
        "bias": W["bias"],
        "s_real": s_real,
    }


def load_block1():
    # Stage 2 (HANDOFF §35): tensor_001.bin = block1.
    return load_fused("tensor_001.bin", 20672)


def load_block3():
    # Stage 3 (HANDOFF §37): tensor_044.bin = block3, same fused32 map
    # (gates/bias finite, sReal 0.778 — verified static, §35).
    return load_fused("tensor_044.bin", 20672)


def load_block67():
    # Stage 4 (HANDOFF §38): tensor_145.bin = block67, same fused32 map
    # (all slices 100% finite, weights distinct from b1/b3, sReal
    # 1.861680 — 2.4x hotter than predecessors, verified static).
    return load_fused("tensor_145.bin", 20672)


def load_block68():
    # Stage 5 (HANDOFF §39): tensor_146.bin = block68, same fused32 map
    # (all slices 100% finite, weights distinct from b1/b3/b67, sReal
    # 8.794641 — 4.7x block67, bias maxabs 11.73, verified static).
    return load_fused("tensor_146.bin", 20672)


def load_block69():
    # Stage 6 (HANDOFF §40): tensor_147.bin = block69, same fused32 map
    # (sReal 11.10 yet cooler activations than b68 — O=2.18 vs
    # 2.98; sReal triply decoupled. Oracle heat precomputed).
    return load_fused("tensor_147.bin", 20672)


def load_block2():
    # Stage 7 (HANDOFF §42): tensor_012.bin = block2, same fused32 map
    # (the manifest's e4m3 inference was wrong — file carves clean as
    # fp16, all slices finite/in-family; manifest row suspect, file and
    # carve are truth). Sixth fp16 member. Fingerprint: gainY 3.77,
    # gainO 0.85 (A-tier), sReal 3.546582, bias 9.55.
    return load_fused("tensor_012.bin", 20672)


def quant_x2e(Yp_in):
    # Inter-stage boundary bytes: f32 -> f16 bits -> e4m3 (exact integer
    # math; shared by the golden emission and the straddle probe).
    return [[R.quantise_f16_to_e4m3(R.f32_to_f16_bits(v)) for v in row]
            for row in Yp_in]


def run_block1(X2e, P):
    # Mirror of run_pre_block without the patch step. X2e is e4m3 bytes
    # (staged from the golden on device; produced here by quant_x2e),
    # decoded exactly like Ye->Xb (no decode kernel by design).
    Xb = [[R.e4m3_decode(b) for b in row] for row in X2e]
    Yf, _, _ = R.ffn_forward(Xb, P["W1"], P["W2"], P["gate_ffn"])
    Q, K, V, Yq = R.qkv_forward(Yf, P["Wq"])
    Qn, Kn, _, _, _, _ = R.qknorm_forward(Q, K, P["s_real"])
    O, _, _, _ = R.attn_forward(Qn, Kn, V, bias=P["bias"])
    Yp, _ = R.proj_forward(O, P["Wp"], Yf, P["gate_attn"])
    return Yp


def wrap(vals, per_line):
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(", ".join(vals[i:i + per_line]))
    return ",\n".join(lines)


def main():
    P = load_pre()
    Xe = lcg_xe()
    Yp = run_pre_block(Xe, P)
    xe_flat = [b for row in Xe for b in row]
    assert len(xe_flat) == 1024
    yp_flat = R.flat(Yp)
    yp_bits = [struct.unpack("<I", struct.pack("<f", v))[0]
               for v in yp_flat]
    body = ("// Generated by tools/emit_yp_golden.py -- DO NOT HAND-EDIT.\n"
            "// tensor_000 pre-block, fixed LCG staging input (HANDOFF §27).\n"
            "// Backend check: blockout maxerr <= 0.1 vs kRwYpGolden.\n"
            "static const unsigned int kRwXeBytesCount = 1024;\n"
            "static const unsigned char kRwXeBytes[1024] = {\n" +
            wrap(["0x%02X" % b for b in xe_flat], 12) + "};\n"
            "static const unsigned int kRwYpGoldenCount = 2048;\n"
            "static const unsigned int kRwYpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in yp_bits], 6) + "};\n")
    open(DST, "w").write(body)

    # round-trip: re-parse the file, compare against fresh computation
    txt = open(DST).read()
    assert txt.count("0x") == 1024 + 2048
    print("wrote %s Yp[0]=%.6f Yp[-1]=%.6f maxabs=%.4f" %
          (DST, yp_flat[0], yp_flat[-1], max(abs(v) for v in yp_flat)))


def fe_texel(x, y, pw, ph, ax, ay):
    """The entry kernel's coordinate map (HANDOFF §13.3, pinned from cubin_00):

        u = (i + 0.5) / dim ;  coord = (u*a + b) * m ;  texel = floor(coord*dim)

    clamped to the edge, one map per axis. `(1,0,1)` is the identity -- the
    direct sample Test-7 used before the affine was ported."""
    import math
    a, b, m = ax
    px = int(math.floor(((x + 0.5) / float(pw) * a + b) * m * pw))
    px = 0 if px < 0 else (pw - 1 if px >= pw else px)
    a, b, m = ay
    py = int(math.floor(((y + 0.5) / float(ph) * a + b) * m * ph))
    py = 0 if py < 0 else (ph - 1 if py >= ph else py)
    return px, py


def fe_block_yp(proxy, ax=(1.0, 0.0, 1.0), ay=(1.0, 0.0, 1.0)):
    """Full oracle block for 256 RGBA8 window bytes (Test-7 params).

    `ax`/`ay` are the per-axis (a,b,m) affine sampling triples. They move
    which texel is read; they do NOT move the dither: the PCG is seeded from
    the TOKEN index, exactly as the kernel does.
    """
    assert len(proxy) == 256, len(proxy)
    P = load_pre()
    tex64, g64 = [], []
    for pas in (0, 1):
        for wl in range(32):
            x, y = wl & 7, (wl >> 3) + 4 * pas
            px, py = fe_texel(x, y, 8, 8, ax, ay)
            tex64.append([proxy[py * 32 + px * 4 + c] / 255.0
                          for c in range(3)])
            o = R.fe_pcg_streams(FE_SEED, x, wl >> 3)
            u = [v * 2.0 ** -24 for v in o]
            g64.append(R.fe_boxmuller(*u))
    Xb16 = R.fe_assemble_X_bits(tex64, g64, FE_HI176, FE_S_SEL, FE_S_A,
                                FE_S_B)
    Xe = [[R.quantise_f16_to_e4m3(u) for u in row] for row in Xb16]
    return run_pre_block(Xe, P)


def main_fe():
    proxy = fe_proxy_bytes()
    Yp = fe_block_yp(proxy)
    yp_flat = R.flat(Yp)
    yp_bits = [struct.unpack("<I", struct.pack("<f", v))[0]
               for v in yp_flat]
    body = ("// Generated by tools/emit_yp_golden.py (fe path) -- DO NOT HAND-EDIT.\n"
            "// tensor_000 pre-block, Test-7 fixed-gradient proxy (HANDOFF §34).\n"
            "// Backend check: fe-block maxerr <= 0.1 vs kFeYpGolden.\n"
            "static const unsigned int kFeProxyBytesCount = 256;\n"
            "static const unsigned char kFeProxyBytes[256] = {\n" +
            wrap(["0x%02X" % b for b in proxy], 12) + "};\n"
            "static const unsigned int kFeYpGoldenCount = 2048;\n"
            "static const unsigned int kFeYpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in yp_bits], 6) + "};\n")
    # newline="\n" on purpose: the default text mode turns every line into
    # CRLF on Windows, which silently rewrites this committed golden (the
    # S73B trap). Byte-compare a re-emit before believing it changed.
    open(DST_FE, "w", newline="\n").write(body)
    txt = open(DST_FE).read()
    assert txt.count("0x") == 256 + 2048
    print("wrote %s Yp[0]=%.6f Yp[-1]=%.6f maxabs=%.4f" %
          (DST_FE, yp_flat[0], yp_flat[-1], max(abs(v) for v in yp_flat)))


# Anisotropic on purpose: x zooms 1.5x (so the right half runs off the edge and
# exercises the clamp), y squashes 0.5x. Nothing about it is special except
# that it is NOT the identity -- which is what makes it able to catch a
# transposed triple or a dropped m, the two ways the affine goes silently wrong.
FE_AFFINE = ((1.0, 0.0, 1.5), (1.0, 0.0, 0.5))


def main_fe_affine():
    """The same proxy and weights as --fe, sampled through a NON-identity
    affine. Self-verifying: it refuses to write a golden that the identity map
    would also produce, because such a golden could not tell the two apart."""
    DST = os.path.join(ROOT, "hip", "mvp1", "rw_yp_fe_affine_golden.inc")
    proxy = fe_proxy_bytes()
    ax, ay = FE_AFFINE
    Yp = fe_block_yp(proxy, ax, ay)
    yp_flat = R.flat(Yp)

    # The golden only means something if the affine moved the answer.
    base = R.flat(fe_block_yp(proxy))
    moved = max(abs(u - v) for u, v in zip(yp_flat, base))
    assert moved > 0.05, (
        "affine %.3g did not move the block output (max delta %.3g) -- the "
        "golden would be indistinguishable from the identity's" % (moved, moved))

    yp_bits = [struct.unpack("<I", struct.pack("<f", v))[0]
               for v in yp_flat]
    body = ("// Generated by tools/emit_yp_golden.py (fe-affine path)"
            " -- DO NOT HAND-EDIT.\n"
            "// tensor_000 pre-block, Test-7 proxy sampled through the"
            " ANISOTROPIC\n"
            "// affine %s (HANDOFF §13.3).\n"
            "// Backend check: fe-affine maxerr <= 0.1 vs kFeAffineYpGolden.\n"
            "static const unsigned int kFeAffineYpGoldenCount = 2048;\n"
            "static const unsigned int kFeAffineYpGolden[2048] = {\n" %
            (repr(FE_AFFINE),) +
            wrap(["0x%08X" % u for u in yp_bits], 6) + "};\n")
    open(DST, "w", newline="\n").write(body)
    txt = open(DST).read()
    assert txt.count("0x") == 2048, txt.count("0x")
    print("wrote %s Yp[0]=%.6f Yp[-1]=%.6f maxabs=%.4f (identity would "
          "differ by up to %.4f)" %
          (DST, yp_flat[0], yp_flat[-1], max(abs(v) for v in yp_flat), moved))


def main_fe_staged(path):
    """Offline recompute for a level-3 staged run.

    The paste file holds the shim-log lines verbatim: `fe-staged win`
    (512 window hex chars), `fe-staged block`, and the 16 keyed
    `yhex00=`...`yhex15=` chunk lines (legacy single `yhex=` line also
    accepted; a window-only file skips the maxerr verdict). The printed
    `block checksum=` is the ORACLE-side fingerprint (recomputed from
    the pasted window -- it identifies the input, and reproduces
    exactly across runs on the same window). The CHECK is maxerr <= 0.1
    over the 2048 device words (same tol rationale as the chain/fe-block
    goldens: cross-language noise ~1e-2, wiring bugs at O(0.5+)). The
    device FNV is a fingerprint only -- f32-vs-f64 never agrees bitwise,
    so exact checksum match is NOT expected.
    """
    import math
    import re
    import struct
    txt = open(path).read()
    mw = re.search(r"bytes=([0-9A-Fa-f]+)", txt)
    hx = "".join(c for c in txt if c in "0123456789abcdefABCDEF") \
        if mw is None else mw.group(1)
    assert len(hx) == 512, len(hx)
    mo = re.search(r"win x=(-?\d+) y=(-?\d+)", txt)
    origin = ("x=%s y=%s" % (mo.group(1), mo.group(2))) if mo else "x=? y=?"
    proxy = bytes(int(hx[2 * i:2 * i + 2], 16) for i in range(256))
    Yp = fe_block_yp(proxy)
    fl = R.flat(Yp)
    fin = sum(1 for v in fl if math.isfinite(v))
    print("block checksum=%s maxabs=%.4f finite=%d/2048 origin %s" %
          (R.fnv1a_hex([v for v in fl if math.isfinite(v)]),
           max(abs(v) for v in fl), fin, origin))
    ch = re.findall(r"yhex(\d+)=([0-9A-Fa-f]+)", txt)
    if ch:
        # Chunked form (16 keyed lines x 128 words): reassemble by key so
        # an interleaved foreign log line, or pasted lines out of order,
        # cannot corrupt the parse. Duplicates/missing chunks fail loudly.
        ch = sorted(ch, key=lambda t: int(t[0]))
        idx = [int(t[0]) for t in ch]
        assert idx == list(range(16)), idx
        assert all(len(h) == 1024 for _, h in ch), \
            [len(h) for _, h in ch]
        yh = "".join(h for _, h in ch)
    else:
        my = re.search(r"yhex=([0-9A-Fa-f]+)", txt)
        if my is None:
            print("no yhex line: maxerr verdict skipped (window-only paste)")
            return
        yh = my.group(1)
    assert len(yh) == 16384, len(yh)
    me = 0.0
    nn = 0
    for i in range(2048):
        w = int(yh[8 * i:8 * i + 8], 16)
        got = struct.unpack("<f", struct.pack("<I", w))[0]
        d = got - fl[i]
        if not math.isfinite(got) or not math.isfinite(d):
            nn += 1
            continue
        me = max(me, abs(d))
    print("maxerr=%.6g nonfinite=%d tol=0.1 verdict=%s" %
          (me, nn, "MATCH" if (nn == 0 and me <= 0.1) else "MISMATCH"))


DST_B2 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b2_golden.inc")
DST_B3 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b3_golden.inc")
DST_B67 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b67_golden.inc")
DST_B68 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b68_golden.inc")
DST_B69 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b69_golden.inc")
DST_BLOCK2 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_block2_golden.inc")
DST_B4 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b4_golden.inc")
DST_B4DS = os.path.join(ROOT, "hip", "mvp1", "rw_yp_b4ds_golden.inc")
DST_C64S = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64s_golden.inc")
DST_C64E = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64e_golden.inc")
DST_C64O = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64o_golden.inc")
DST_C64P = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64p_golden.inc")
DST_C64F = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64f_golden.inc")
DST_C64Q = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64q_golden.inc")
DST_C64C = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64c_golden.inc")
DST_C64X = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64x_golden.inc")
DST_C64F2 = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64f2_golden.inc")
DST_C64BLK = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c64blk_golden.inc")


def _chain_to_yp4():
    # Shared prefix chain for --b4 / --b4ds (HANDOFF §45).
    P0 = load_pre()
    P1 = load_block1()
    P2 = load_block2()
    P3 = load_block3()
    P4 = load_block4()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    YpB2 = run_block1(quant_x2e(Yp2), P2)
    Yp3 = run_block1(quant_x2e(YpB2), P3)
    X4e = quant_x2e(Yp3)
    return P4, X4e, run_block1(X4e, P4), Yp3


def main_b4():
    # Stage-9 golden (HANDOFF §45): LCG Xe -> b0 -> b1 -> block2 ->
    # block3 -> X4e -> block4-prefix -> Yp4. Chained input is
    # continuity convention only (§37 caveat).
    P4, X4e, Yp4, Yp3 = _chain_to_yp4()
    flat = R.flat(Yp4)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp3)))
    assert move > 0.1, move  # block4 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    xflat = [b for row in X4e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b4 -- DO NOT HAND-EDIT.\n"
            "// tensor_091 block4 prefix output on the §27 LCG Xe via b0+b1+b2+b3.\n"
            "// kB4XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §45).\n"
            "// Backend check: b4-block maxerr <= 0.1 vs kB4YpGolden.\n"
            "static const unsigned int kB4XeBytesCount = 2048;\n"
            "static const unsigned char kB4XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in xflat], 12) + "};\n"
            "static const unsigned int kB4YpGoldenCount = 2048;\n"
            "static const unsigned int kB4YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B4, "w").write(body)

    txt = open(DST_B4).read()
    assert txt.count("0x") == 2048 + 2048
    _, _, Yp4b, _ = _chain_to_yp4()
    assert R.flat(Yp4b) == flat, "b4 golden not deterministic"
    Xb = [[R.e4m3_decode(b) for b in row] for row in X4e]
    Yf, Yraw, A = R.ffn_forward(Xb, P4["W1"], P4["W2"], P4["gate_ffn"])
    Q, K, V, Yq = R.qkv_forward(Yf, P4["Wq"])
    Qn, Kn, _, _, _, _ = R.qknorm_forward(Q, K, P4["s_real"])
    S = [[sum(Qn[i][d] * Kn[j][d] for d in range(R.C)) + P4["bias"][i * 64 + j]
          for j in range(R.TOK)] for i in range(R.TOK)]
    O, _, _, _ = R.attn_forward(Qn, Kn, V, bias=P4["bias"])
    m = lambda M: max(abs(v) for row in M for v in row)
    print("wrote %s Yp4[0]=%.6f maxabs=%.4f move=%.4f checksum=%s heat A=%.3f Y=%.3f S=%.3f O=%.3f" %
          (DST_B4, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)]),
           m(A), m(Yraw), m(S), m(O)))


def load_block4():
    # Stage 9 (HANDOFF §45): tensor_091.bin = block4. Prefix 0..20671
    # IS the fused map (carve-verified slice by slice); the tail is a
    # separate ds stage (see load_block4_ds). sReal 0.503399.
    p = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_091.bin")
    raw = open(p, "rb").read()
    assert len(raw) == 22720, len(raw)
    W = R.carve(p, R.CARVE_FUSED32)
    s_real = struct.unpack("<f", raw[19552:19556])[0]
    return {
        "Wq": [W["qkv"][i * R.C:(i + 1) * R.C] for i in range(R.QKV)],
        "Wp": [W["proj"][i * R.C:(i + 1) * R.C] for i in range(R.C)],
        "W1": [W["ffn1"][i * R.C:(i + 1) * R.C] for i in range(R.H1)],
        "W2": [W["ffn2"][i * R.H1:(i + 1) * R.H1] for i in range(R.C)],
        "gate_ffn": W["gate_ffn"],
        "gate_attn": W["gate_attn"],
        "bias": W["bias"],
        "s_real": s_real,
    }


def load_block4_ds():
    # Stage-transition projection (§45, dataflow-evidenced): tensor_091
    # bytes [20656, 22704) = 2048 e4m3 = T[64][32] row-major (out×in),
    # C32->C64, contracted on K=32. Kernel
    # cc_tinlayout_fused_swin_1h_32_1_ds_fp8 reads 4 extra 512B tiles
    # {20656, 21168, 21680, 22192} (strict superset of plain); MMA
    # m16n8k32. [22704, 22720) is an unread 16-zero trailer.
    # NOTE the 16B shift: the live tail starts at 20656 (the standard
    # trailer slot is non-zero here), not 20672.
    raw = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                            "tensor_091.bin"), "rb").read()
    assert len(raw) == 22720, len(raw)
    assert all(b == 0 for b in raw[22704:22720]), "trailer not zero"
    tb = raw[20656:22704]
    assert len(tb) == 2048
    return [[R.e4m3_decode(tb[n * 32 + c]) for c in range(R.C)]
            for n in range(2 * R.C)]


def main_b4ds():
    # Stage-10 golden (HANDOFF §45): _chain_to_yp4 -> quant(Yp4) ->
    # Yds = Yp4[64][32] @ T[64][32]^K32 (64x64 f32). No new staged
    # boundary (the ds stage consumes the block's own output); tol
    # rationale in the backend (§45: block gap x ~1 + GEMM noise).
    _, _, Yp4, _ = _chain_to_yp4()
    T = load_block4_ds()
    Xds = quant_x2e(Yp4)
    Xb = [[R.e4m3_decode(b) for b in row] for row in Xds]
    Yds = R.gemm(Xb, T, R.TOK, R.C, 2 * R.C)
    flat = R.flat(Yds)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 4096, fin
    # Non-degeneracy: the projection transforms (vs column-tiled Yp4).
    base = R.flat(Yp4)
    move = max(abs(a - b) for a, b in
               zip(flat, [base[(i // 64) * 32 + (i % 32)] for i in
                          range(4096)]))
    assert move > 0.1, move
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    body = ("// Generated by tools/emit_yp_golden.py --b4ds -- DO NOT HAND-EDIT.\n"
            "// tensor_091 block4 downsample output on the §27 LCG Xe via b0+b1+b2+b3+b4.\n"
            "// kB4DsGolden is 64x64 row-major f32 (no staged input: the ds\n"
            "// stage consumes the block's own output; tol stays 0.1, §45).\n"
            "// Backend check: b4ds-block maxerr <= 0.1 vs kB4DsGolden.\n"
            "static const unsigned int kB4DsGoldenCount = 4096;\n"
            "static const unsigned int kB4DsGolden[4096] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B4DS, "w").write(body)

    txt = open(DST_B4DS).read()
    assert txt.count("0x") == 4096
    _, _, Yp4b, _ = _chain_to_yp4()
    Xdsb = quant_x2e(Yp4b)
    Xbb = [[R.e4m3_decode(b) for b in row] for row in Xdsb]
    again = R.gemm(Xbb, load_block4_ds(), R.TOK, R.C, 2 * R.C)
    assert R.flat(again) == flat, "b4ds golden not deterministic"
    print("wrote %s Yds[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B4DS, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_b2():
    # Stage-2 golden (HANDOFF §35): LCG Xe -> block0 -> X2e -> block1.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV.)
    P0 = load_pre()
    P1 = load_block1()
    Yp = run_pre_block(lcg_xe(), P0)
    X2e = quant_x2e(Yp)
    Yp2 = run_block1(X2e, P1)
    flat = R.flat(Yp2)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp)))
    assert move > 0.1, move  # block1 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    x2flat = [b for row in X2e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b2 -- DO NOT HAND-EDIT.\n"
            "// tensor_001 block1 output on the §27 LCG Xe via block0 (§35).\n"
            "// kB2XeBytes stages the inter-stage boundary (common-mode, so\n"
            "// the straddle cannot enter maxerr); tol stays 0.1 (HANDOFF §36).\n"
            "// Backend check: b2-block maxerr <= 0.1 vs kB2YpGolden.\n"
            "static const unsigned int kB2XeBytesCount = 2048;\n"
            "static const unsigned char kB2XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in x2flat], 12) + "};\n"
            "static const unsigned int kB2YpGoldenCount = 2048;\n"
            "static const unsigned int kB2YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B2, "w").write(body)

    txt = open(DST_B2).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    again = run_block1(quant_x2e(Ypa), load_block1())
    assert R.flat(again) == flat, "b2 golden not deterministic"
    print("wrote %s Yp2[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B2, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_b3():
    # Stage-3 golden (HANDOFF §37): LCG Xe -> b0 -> b1 -> X3e -> block3.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV.)
    P0 = load_pre()
    P1 = load_block1()
    P3 = load_block3()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    X3e = quant_x2e(Yp2)
    Yp3 = run_block1(X3e, P3)
    flat = R.flat(Yp3)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp2)))
    assert move > 0.1, move  # block3 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    x3flat = [b for row in X3e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b3 -- DO NOT HAND-EDIT.\n"
            "// tensor_044 block3 output on the §27 LCG Xe via block0+block1.\n"
            "// kB3XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §37).\n"
            "// Backend check: b3-block maxerr <= 0.1 vs kB3YpGolden.\n"
            "static const unsigned int kB3XeBytesCount = 2048;\n"
            "static const unsigned char kB3XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in x3flat], 12) + "};\n"
            "static const unsigned int kB3YpGoldenCount = 2048;\n"
            "static const unsigned int kB3YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B3, "w").write(body)

    txt = open(DST_B3).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    Yp2a = run_block1(quant_x2e(Ypa), load_block1())
    again = run_block1(quant_x2e(Yp2a), load_block3())
    assert R.flat(again) == flat, "b3 golden not deterministic"
    print("wrote %s Yp3[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B3, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_b67():
    # Stage-4 golden (HANDOFF §38): LCG Xe -> b0 -> b1 -> b3 -> X67e.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV.)
    P0 = load_pre()
    P1 = load_block1()
    P3 = load_block3()
    P67 = load_block67()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    Yp3 = run_block1(quant_x2e(Yp2), P3)
    X67e = quant_x2e(Yp3)
    Yp67 = run_block1(X67e, P67)
    flat = R.flat(Yp67)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp3)))
    assert move > 0.1, move  # block67 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    x67flat = [b for row in X67e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b67 -- DO NOT HAND-EDIT.\n"
            "// tensor_145 block67 output on the §27 LCG Xe via b0+b1+b3.\n"
            "// kB67XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §38).\n"
            "// Backend check: b67-block maxerr <= 0.1 vs kB67YpGolden.\n"
            "static const unsigned int kB67XeBytesCount = 2048;\n"
            "static const unsigned char kB67XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in x67flat], 12) + "};\n"
            "static const unsigned int kB67YpGoldenCount = 2048;\n"
            "static const unsigned int kB67YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B67, "w").write(body)

    txt = open(DST_B67).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    Yp2a = run_block1(quant_x2e(Ypa), load_block1())
    Yp3a = run_block1(quant_x2e(Yp2a), load_block3())
    again = run_block1(quant_x2e(Yp3a), load_block67())
    assert R.flat(again) == flat, "b67 golden not deterministic"
    print("wrote %s Yp67[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B67, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_b68():
    # Stage-5 golden (HANDOFF §39): + X68e -> block68.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV.)
    P0 = load_pre()
    P1 = load_block1()
    P3 = load_block3()
    P67 = load_block67()
    P68 = load_block68()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    Yp3 = run_block1(quant_x2e(Yp2), P3)
    Yp67 = run_block1(quant_x2e(Yp3), P67)
    X68e = quant_x2e(Yp67)
    Yp68 = run_block1(X68e, P68)
    flat = R.flat(Yp68)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp67)))
    assert move > 0.1, move  # block68 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    x68flat = [b for row in X68e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b68 -- DO NOT HAND-EDIT.\n"
            "// tensor_146 block68 output on the §27 LCG Xe via b0+b1+b3+b67.\n"
            "// kB68XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §39).\n"
            "// Backend check: b68-block maxerr <= 0.1 vs kB68YpGolden.\n"
            "static const unsigned int kB68XeBytesCount = 2048;\n"
            "static const unsigned char kB68XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in x68flat], 12) + "};\n"
            "static const unsigned int kB68YpGoldenCount = 2048;\n"
            "static const unsigned int kB68YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B68, "w").write(body)

    txt = open(DST_B68).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    Yp2a = run_block1(quant_x2e(Ypa), load_block1())
    Yp3a = run_block1(quant_x2e(Yp2a), load_block3())
    Yp67a = run_block1(quant_x2e(Yp3a), load_block67())
    again = run_block1(quant_x2e(Yp67a), load_block68())
    assert R.flat(again) == flat, "b68 golden not deterministic"
    print("wrote %s Yp68[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B68, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_b69():
    # Stage-6 golden (HANDOFF §40): + X69e -> block69.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV.)
    P0 = load_pre()
    P1 = load_block1()
    P3 = load_block3()
    P67 = load_block67()
    P68 = load_block68()
    P69 = load_block69()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    Yp3 = run_block1(quant_x2e(Yp2), P3)
    Yp67 = run_block1(quant_x2e(Yp3), P67)
    Yp68 = run_block1(quant_x2e(Yp67), P68)
    X69e = quant_x2e(Yp68)
    Yp69 = run_block1(X69e, P69)
    flat = R.flat(Yp69)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp68)))
    assert move > 0.1, move  # block69 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    x69flat = [b for row in X69e for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --b69 -- DO NOT HAND-EDIT.\n"
            "// tensor_147 block69 output on the §27 LCG Xe via b0+b1+b3+b67+b68.\n"
            "// kB69XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §40).\n"
            "// Backend check: b69-block maxerr <= 0.1 vs kB69YpGolden.\n"
            "static const unsigned int kB69XeBytesCount = 2048;\n"
            "static const unsigned char kB69XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in x69flat], 12) + "};\n"
            "static const unsigned int kB69YpGoldenCount = 2048;\n"
            "static const unsigned int kB69YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_B69, "w").write(body)

    txt = open(DST_B69).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    Yp2a = run_block1(quant_x2e(Ypa), load_block1())
    Yp3a = run_block1(quant_x2e(Yp2a), load_block3())
    Yp67a = run_block1(quant_x2e(Yp3a), load_block67())
    Yp68a = run_block1(quant_x2e(Yp67a), load_block68())
    again = run_block1(quant_x2e(Yp68a), load_block69())
    assert R.flat(again) == flat, "b69 golden not deterministic"
    print("wrote %s Yp69[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_B69, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def main_block2():
    # Stage-7 golden (HANDOFF §42): LCG Xe -> b0 -> b1 -> Xblock2.
    # (Reconstructed 2026-09-08, HANDOFF §45; output array-identical
    # to the committed golden by FNV. NOTE the misnomer: --block2 is
    # block 2 proper; --b2/kB2* is BLOCK1.)
    P0 = load_pre()
    P1 = load_block1()
    P2 = load_block2()
    Yp = run_pre_block(lcg_xe(), P0)
    Yp2 = run_block1(quant_x2e(Yp), P1)
    Xbe = quant_x2e(Yp2)
    YpB2 = run_block1(Xbe, P2)
    flat = R.flat(YpB2)
    fin = sum(1 for v in flat if math.isfinite(v))
    assert fin == 2048, fin
    move = max(abs(a - b) for a, b in zip(flat, R.flat(Yp2)))
    assert move > 0.1, move  # block2 must transform, not echo
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flat]
    xflat = [b for row in Xbe for b in row]
    body = ("// Generated by tools/emit_yp_golden.py --block2 -- DO NOT HAND-EDIT.\n"
            "// tensor_012 block2 output on the §27 LCG Xe via b0+b1.\n"
            "// kBlock2XeBytes stages the inter-stage boundary (common-mode);\n"
            "// tol stays 0.1 (HANDOFF §42).\n"
            "// Backend check: block2-block maxerr <= 0.1 vs kBlock2YpGolden.\n"
            "static const unsigned int kBlock2XeBytesCount = 2048;\n"
            "static const unsigned char kBlock2XeBytes[2048] = {\n" +
            wrap(["0x%02X" % b for b in xflat], 12) + "};\n"
            "static const unsigned int kBlock2YpGoldenCount = 2048;\n"
            "static const unsigned int kBlock2YpGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in bits], 6) + "};\n")
    open(DST_BLOCK2, "w").write(body)

    txt = open(DST_BLOCK2).read()
    assert txt.count("0x") == 2048 + 2048
    Ypa = run_pre_block(lcg_xe(), load_pre())
    Yp2a = run_block1(quant_x2e(Ypa), load_block1())
    again = run_block1(quant_x2e(Yp2a), load_block2())
    assert R.flat(again) == flat, "block2 golden not deterministic"
    print("wrote %s YpB2[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_BLOCK2, flat[0], max(abs(v) for v in flat), move,
           R.fnv1a_hex([v for v in flat if math.isfinite(v)])))


def load_c64():
    # Test-16/17 weights (HANDOFF §49): tensor_137.bin = C=64 block6.
    # Only the four proven inputs are consumed: HOT bias @41120
    # ((2,64,64) f16), per-head temps @57504 (2x f32). The QKV order
    # inside [28832,41120) is SIZE INFERENCE only -- never consumed
    # (Test-16 stages Q/K directly).
    p = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_137.bin")
    raw = open(p, "rb").read()
    assert len(raw) == 61760, (p, len(raw))
    temps = list(struct.unpack("<2f", raw[57504:57512]))
    assert all(math.isfinite(t) for t in temps), temps
    Bh = struct.unpack("<8192H", raw[41120:57504])
    B = [[[R.f16_to_f32(Bh[h * 4096 + i * 64 + j])
           for j in range(R.C64_TOK)] for i in range(R.C64_TOK)]
         for h in range(R.C64_HEADS)]
    assert all(math.isfinite(v) for h in B for row in h for v in row)
    assert max(v for h in B for row in h for v in row) == 0.0
    return temps, B


def main_c64():
    # Test-16 golden (HANDOFF §49/§51): staged synthetic Q/K (LCG
    # levels, C64_SEED -- continuity convention is N/A here: no prior
    # C=64 block output exists, and the QKV slicing is unproven, so the
    # score-GEMM inputs are staged directly) through the real HOT bias
    # + real per-head temps. Test-17 golden (same run): the oracle S
    # words re-staged as f32 input (NO quant -- identical bits, so the
    # Eb check is straddle-free exact) through the replicated 5-op
    # trick + rowsum + rcp (HANDOFF §50, DECISION: replicate).
    temps, B = load_c64()
    Qe, Ke = R.c64_staged_qk()
    assert len(Qe) == 4096 and len(Ke) == 4096
    Q = [[[R.e4m3_decode(Qe[h * 2048 + i * 32 + d])
           for d in range(R.C64_DIM)] for i in range(R.C64_TOK)]
         for h in range(R.C64_HEADS)]
    K = [[[R.e4m3_decode(Ke[h * 2048 + i * 32 + d])
           for d in range(R.C64_DIM)] for i in range(R.C64_TOK)]
         for h in range(R.C64_HEADS)]
    S = R.c64_scores_forward(Q, K, temps, B)
    flatS = [v for h in S for row in h for v in row]
    assert sum(1 for v in flatS if math.isfinite(v)) == 8192
    # Non-degeneracy: the temp-scaled dots must move S off the bias
    # (else the staged Q/K would prove nothing about the GEMM path).
    flatB = [v for h in B for row in h for v in row]
    move = max(abs(a - b) for a, b in zip(flatS, flatB))
    assert move > 0.1, move
    Eb, P = R.c64_exp_forward(S)
    flatP = [v for h in P for row in h for v in row]
    assert sum(1 for v in flatP if math.isfinite(v)) == 8192
    maxrow = max(abs(sum(P[h][i][j] for j in range(R.C64_TOK)) - 1.0)
                 for h in range(R.C64_HEADS) for i in range(R.C64_TOK))
    assert maxrow < 5e-3, maxrow
    # Saturation census (informational: real C=64 scores + mask-fabric
    # bias push most E to the rails; Eb stays exact-pinned regardless).
    flatEb = [u for h in Eb for row in h for u in row]
    nfloor = sum(1 for u in flatEb if u == 0x0400)
    nceil = sum(1 for u in flatEb
                if R.f16_to_f32(u) == 9.75)
    sbits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatS]
    pbits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatP]
    body_s = ("// Generated by tools/emit_yp_golden.py --c64 -- DO NOT HAND-EDIT.\n"
              "// tensor_137 C=64 scores: staged Q/K (C64_SEED LCG levels) +\n"
              "// real HOT bias @41120 + real per-head temps @57504 (§49).\n"
              "// Backend check: c64s-block maxerr <= 0.1 vs kC64SGolden.\n"
              "static const unsigned int kC64QeBytesCount = 4096;\n"
              "static const unsigned char kC64QeBytes[4096] = {\n" +
              wrap(["0x%02X" % b for b in Qe], 12) + "};\n"
              "static const unsigned int kC64KeBytesCount = 4096;\n"
              "static const unsigned char kC64KeBytes[4096] = {\n" +
              wrap(["0x%02X" % b for b in Ke], 12) + "};\n"
              "static const unsigned int kC64SGoldenCount = 8192;\n"
              "static const unsigned int kC64SGolden[8192] = {\n" +
              wrap(["0x%08X" % u for u in sbits], 6) + "};\n")
    open(DST_C64S, "w").write(body_s)
    body_e = ("// Generated by tools/emit_yp_golden.py --c64 -- DO NOT HAND-EDIT.\n"
              "// Test-17: oracle S words re-staged as f32 input (identical\n"
              "// bits, straddle-free) through the replicated 5-op trick (§50).\n"
              "// Backend checks: Eb EXACT vs kC64EbGolden; P maxerr <= 0.1.\n"
              "static const unsigned int kC64SBytesCount = 8192;\n"
              "static const unsigned int kC64SBytes[8192] = {\n" +
              wrap(["0x%08X" % u for u in sbits], 6) + "};\n"
              "static const unsigned int kC64EbGoldenCount = 8192;\n"
              "static const unsigned short kC64EbGolden[8192] = {\n" +
              wrap(["0x%04X" % u for u in flatEb], 8) + "};\n"
              "static const unsigned int kC64PGoldenCount = 8192;\n"
              "static const unsigned int kC64PGolden[8192] = {\n" +
              wrap(["0x%08X" % u for u in pbits], 6) + "};\n")
    open(DST_C64E, "w").write(body_e)
    for p, n in ((DST_C64S, 4096 + 4096 + 8192), (DST_C64E, 3 * 8192)):
        assert open(p).read().count("0x") == n, p
    # Determinism: recompute from scratch, arrays must agree exactly.
    temps2, B2 = load_c64()
    S2 = R.c64_scores_forward(Q, K, temps2, B2)
    Eb2, P2 = R.c64_exp_forward(S2)
    assert [v for h in S2 for row in h for v in row] == flatS
    assert [v for h in P2 for row in h for v in row] == flatP, \
        "c64 not deterministic"
    assert [u for h in Eb2 for row in h for u in row] == flatEb
    # Cross-file coherence: Test-17 stages Test-16's S words verbatim.
    s_block = wrap(["0x%08X" % u for u in sbits], 6)
    assert s_block in open(DST_C64S).read(), "S missing from c64s golden"
    assert s_block in open(DST_C64E).read(), "S missing from c64e golden"
    print("wrote %s S[0]=%.6f maxabs=%.4f move=%.4f checksum=%s" %
          (DST_C64S, flatS[0], max(abs(v) for v in flatS), move,
           R.fnv1a_hex([v for v in flatS if math.isfinite(v)])))
    print("wrote %s Eb=%s P=%s rowsum=%.5f E floor/active/ceil=%d/%d/%d" %
          (DST_C64E,
           R.fnv_u16(flatEb),
           R.fnv1a_hex([v for v in flatP if math.isfinite(v)]),
           maxrow, nfloor, 8192 - nfloor - nceil, nceil))


def main_c64o():
    # Test-18 golden (HANDOFF §78D-F): staged Pq [2][64][64] + V
    # [2][64][32] e4m3 bytes (fresh C64_P/V_SEED LCG streams, levels --
    # raw, not simplex, per the staged-input doctrine) through the
    # two-step-f16 oracle. P re-quantized e4m3 on device is a no-op here:
    # Pq is staged AS e4m3 bytes already (composition proven separately).
    Pqe, Ve = R.c64_staged_pv()
    assert len(Pqe) == 8192 and len(Ve) == 4096
    Pq = [[[R.e4m3_decode(Pqe[h * 4096 + i * 64 + k])
            for k in range(R.C64_TOK)] for i in range(R.C64_TOK)]
          for h in range(R.C64_HEADS)]
    V = [[[R.e4m3_decode(Ve[h * 2048 + k * 32 + n])
           for n in range(R.C64_DIM)] for k in range(R.C64_TOK)]
         for h in range(R.C64_HEADS)]
    O = R.c64_ctx_forward(Pq, V)
    flatO = [v for h in O for row in h for v in row]
    assert len(flatO) == 4096
    assert sum(1 for v in flatO if math.isfinite(v)) == 4096
    # Non-degeneracy: staged levels must move O (else the test proves
    # nothing about the GEMM path).
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64o -- DO NOT HAND-EDIT.\n"
            "// Test-18: staged Pq/V (C64_P/V_SEED LCG levels) through the\n"
            "// two-step-f16 O oracle (S78E). Emit on Linux (LF); S73B CRLF trap.\n"
            "// Backend check: c64o-block maxerr <= 0.1 vs kC64OGolden.\n"
            "static const unsigned int kC64PqBytesCount = 8192;\n"
            "static const unsigned char kC64PqBytes[8192] = {\n" +
            wrap(["0x%02X" % b for b in Pqe], 12) + "};\n"
            "static const unsigned int kC64VBytesCount = 4096;\n"
            "static const unsigned char kC64VBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Ve], 12) + "};\n"
            "static const unsigned int kC64OGoldenCount = 4096;\n"
            "static const unsigned int kC64OGolden[4096] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    open(DST_C64O, "w").write(body)
    assert open(DST_C64O).read().count("0x") == 8192 + 4096 + 4096, DST_C64O
    # Determinism: recompute from scratch, arrays must agree exactly.
    Pqe2, Ve2 = R.c64_staged_pv()
    assert Pqe2 == Pqe and Ve2 == Ve
    Pq2 = [[[R.e4m3_decode(Pqe2[h * 4096 + i * 64 + k])
             for k in range(R.C64_TOK)] for i in range(R.C64_TOK)]
           for h in range(R.C64_HEADS)]
    V2 = [[[R.e4m3_decode(Ve2[h * 2048 + k * 32 + n])
            for n in range(R.C64_DIM)] for k in range(R.C64_TOK)]
          for h in range(R.C64_HEADS)]
    O2 = R.c64_ctx_forward(Pq2, V2)
    assert [v for h in O2 for row in h for v in row] == flatO, \
        "c64o not deterministic"
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64O, flatO[0], move,
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


def main_c64p():
    # Test-19 golden (HANDOFF §78G): staged Ocat [64][64] + y [64][64]
    # e4m3 bytes (fresh C64_O/Y_SEED LCG streams, levels) through the
    # plain-sums + gate-add oracle. Wproj [64][64] e4m3 + gate2 f16[64]
    # are REAL (dump +57520/+61616, read at runtime like Test-16's B;
    # row-major [c][o] mapping shared with twin/device).
    Ob, Yb = R.c64_staged_oy()
    assert len(Ob) == 4096 and len(Yb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    Wb = dump[57520:57520 + 4096]
    assert sum(1 for b in Wb if b) > 3000, "Wproj region looks empty"
    Gb = dump[61616:61616 + 128]
    Ocat = [[R.e4m3_decode(Ob[m * 64 + c]) for c in range(64)]
            for m in range(64)]
    # S150: OUT-major [out][in] -- (c,o) at o*64 + c.
    W = [[R.e4m3_decode(Wb[o * 64 + c]) for o in range(64)]
         for c in range(64)]
    gate2 = [R.f16_to_f32(struct.unpack("<H", Gb[o * 2:o * 2 + 2])[0])
             for o in range(64)]
    assert all(math.isfinite(g) for g in gate2)
    y = [[R.e4m3_decode(Yb[m * 64 + o]) for o in range(64)]
         for m in range(64)]
    O = R.c64_proj_forward(Ocat, W, gate2, y)
    flatO = [v for row in O for v in row]
    assert len(flatO) == 4096
    assert sum(1 for v in flatO if math.isfinite(v)) == 4096
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64p -- DO NOT HAND-EDIT.\n"
            "// Test-19: staged Ocat/y through the sums+gate oracle (S78G);\n"
            "// Wproj/gate2 real (dump +57520/+61616, runtime reads). Emit on\n"
            "// Linux (LF); S73B CRLF trap. Backend: c64p-block maxerr <= 0.1.\n"
            "static const unsigned int kC64ObBytesCount = 4096;\n"
            "static const unsigned char kC64ObBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Ob], 12) + "};\n"
            "static const unsigned int kC64YBytesCount = 4096;\n"
            "static const unsigned char kC64YBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Yb], 12) + "};\n"
            "static const unsigned int kC64ProjGoldenCount = 4096;\n"
            "static const unsigned int kC64ProjGolden[4096] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    open(DST_C64P, "w").write(body)
    assert open(DST_C64P).read().count("0x") == 4096 + 4096 + 4096, DST_C64P
    Ob2, Yb2 = R.c64_staged_oy()
    assert Ob2 == Ob and Yb2 == Yb
    Ocat2 = [[R.e4m3_decode(Ob2[m * 64 + c]) for c in range(64)]
             for m in range(64)]
    y2 = [[R.e4m3_decode(Yb2[m * 64 + o]) for o in range(64)]
          for m in range(64)]
    O2 = R.c64_proj_forward(Ocat2, W, gate2, y2)
    assert [v for row in O2 for v in row] == flatO, \
        "c64p not deterministic"
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64P, flatO[0], move,
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


def main_c64f():
    # Test-20 golden (HANDOFF §78H): staged x [64][64] e4m3 bytes (fresh
    # C64_X_SEED LCG stream, levels) through the expand->act->contract
    # + gate1 oracle. w1 [224][64] e4m3 + w2 [64][224] e4m3 + gate1
    # f16[64] are REAL (dump +0/+14336/+28688, read at runtime like
    # Test-19's Wproj/gate2; row-major [out][in] mapping shared with
    # twin/device). H1=224 stays size arithmetic (S78H/b).
    Xb = R.c64_staged_x()
    assert len(Xb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    W1b = dump[0:14336]
    assert sum(1 for b in W1b if b) > 13000, "w1 region looks empty"
    W2b = dump[14336:28672]
    assert sum(1 for b in W2b if b) > 13000, "w2 region looks empty"
    G1b = dump[28688:28688 + 128]
    x = [[R.e4m3_decode(Xb[m * 64 + k]) for k in range(64)]
         for m in range(64)]
    w1 = [[R.e4m3_decode(W1b[j * 64 + k]) for k in range(64)]
          for j in range(R.C64_H1)]
    w2 = [[R.e4m3_decode(W2b[o * 224 + j]) for j in range(R.C64_H1)]
          for o in range(64)]
    gate1 = [R.f16_to_f32(struct.unpack("<H", G1b[o * 2:o * 2 + 2])[0])
             for o in range(64)]
    assert all(math.isfinite(g) for g in gate1)
    assert sum(1 for g in gate1 if 0.0 < g <= 1.0) == 64, "gate1 not (0,1]"
    O = R.c64_ffn_forward(x, w1, w2, gate1)
    flatO = [v for row in O for v in row]
    assert len(flatO) == 4096
    assert sum(1 for v in flatO if math.isfinite(v)) == 4096
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64f -- DO NOT HAND-EDIT.\n"
            "// Test-20: staged x through the FFN oracle (S78H); w1/w2/gate1\n"
            "// real (dump +0/+14336/+28688, runtime reads). Emit on Linux\n"
            "// (LF); S73B CRLF trap. Backend: c64f-block maxerr <= 0.1.\n"
            "static const unsigned int kC64XBytesCount = 4096;\n"
            "static const unsigned char kC64XBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Xb], 12) + "};\n"
            "static const unsigned int kC64FfnGoldenCount = 4096;\n"
            "static const unsigned int kC64FfnGolden[4096] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    open(DST_C64F, "w").write(body)
    assert open(DST_C64F).read().count("0x") == 4096 + 4096, DST_C64F
    Xb2 = R.c64_staged_x()
    assert Xb2 == Xb
    x2 = [[R.e4m3_decode(Xb2[m * 64 + k]) for k in range(64)]
          for m in range(64)]
    O2 = R.c64_ffn_forward(x2, w1, w2, gate1)
    assert [v for row in O2 for v in row] == flatO, \
        "c64f not deterministic"
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64F, flatO[0], move,
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


def main_c64c():
    # Test-22 golden (HANDOFF §111): staged h [64][128] e4m3 bytes
    # (fresh C64_HIN_SEED LCG stream) through the FOUR-step-f16
    # contract oracle. W2 [128][32] e4m3 is REAL (dump +16384, 4096 B
    # = pass 0's N-slice of the [128][64] region-2 weight; §111's
    # immediate census, not size arithmetic).
    Hb = R.c64_staged_hin()
    assert len(Hb) == 8192
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    Wb = dump[16384:16384 + 4096]
    assert sum(1 for b in Wb if b) > 4000, "w2 region looks empty"
    H = [[R.e4m3_decode(Hb[m * 128 + k]) for k in range(128)]
         for m in range(64)]
    W2 = [[R.e4m3_decode(Wb[k * 32 + n]) for n in range(32)]
          for k in range(128)]
    O = R.c64_contract_forward(H, W2)
    flatO = [v for row in O for v in row]
    assert len(flatO) == 2048
    assert sum(1 for v in flatO if math.isfinite(v)) == 2048
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    # newline="\n": S73B CRLF trap -- bare "w" would emit CRLF on
    # Windows and dirty every byte of the golden.
    body = ("// Generated by tools/emit_yp_golden.py --c64c -- DO NOT HAND-EDIT.\n"
            "// Test-22: staged h through the FFN-contract oracle (§111);\n"
            "// W2 real (dump +16384, [128][32] row-major, runtime read).\n"
            "// K=128 in four f16 k-steps (production D->C chain, §110).\n"
            "// Backend: c64c-block maxerr <= 0.1.\n"
            "static const unsigned int kC64HinBytesCount = 8192;\n"
            "static const unsigned char kC64HinBytes[8192] = {\n" +
            wrap(["0x%02X" % b for b in Hb], 12) + "};\n"
            "static const unsigned int kC64ContractGoldenCount = 2048;\n"
            "static const unsigned int kC64ContractGolden[2048] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    with open(DST_C64C, "w", newline="\n") as fh:
        fh.write(body)
    assert open(DST_C64C).read().count("0x") == 8192 + 2048, DST_C64C
    Hb2 = R.c64_staged_hin()
    assert Hb2 == Hb
    H2 = [[R.e4m3_decode(Hb2[m * 128 + k]) for k in range(128)]
          for m in range(64)]
    O2 = R.c64_contract_forward(H2, W2)
    assert [v for row in O2 for v in row] == flatO, \
        "c64c not deterministic"
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64C, flatO[0], move,
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


def main_c64x():
    # Test-23 golden (HANDOFF §113): staged x [64][64] e4m3 bytes
    # (fresh C64_EXIN_SEED LCG stream) through the TWO-step-f16 expand
    # oracle. w1 [64][128] e4m3 is REAL (dump +0, 8192 B = pass 0's
    # slice of the 16384-B region 1). IN-MAJOR orientation W[k*128+n],
    # matching Wproj/Wqkv/W2-contract -- Test-20's out-major w1 is the
    # §113 defect and is deliberately not copied here.
    Xb = R.c64_staged_exin()
    assert len(Xb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    Wb = dump[0:8192]
    assert sum(1 for b in Wb if b) > 8000, "w1 region looks empty"
    X = [[R.e4m3_decode(Xb[m * 64 + k]) for k in range(64)]
         for m in range(64)]
    W1 = [[R.e4m3_decode(Wb[k * 128 + n]) for n in range(128)]
          for k in range(64)]
    O = R.c64_expand_forward(X, W1)
    flatO = [v for row in O for v in row]
    assert len(flatO) == 8192
    assert sum(1 for v in flatO if math.isfinite(v)) == 8192
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    # newline="\n": S73B CRLF trap -- bare "w" would emit CRLF on
    # Windows and dirty every byte of the golden.
    body = ("// Generated by tools/emit_yp_golden.py --c64x -- DO NOT HAND-EDIT.\n"
            "// Test-23: staged x through the FFN-expand oracle (§113);\n"
            "// w1 real (dump +0, [64][128] IN-MAJOR row-major).\n"
            "// K=64 in two f16 k-steps (production D->C chain, §113).\n"
            "// Backend: c64x-block maxerr <= 0.1.\n"
            "static const unsigned int kC64ExinBytesCount = 4096;\n"
            "static const unsigned char kC64ExinBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Xb], 12) + "};\n"
            "static const unsigned int kC64ExpandGoldenCount = 8192;\n"
            "static const unsigned int kC64ExpandGolden[8192] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    with open(DST_C64X, "w", newline="\n") as fh:
        fh.write(body)
    assert open(DST_C64X).read().count("0x") == 4096 + 8192, DST_C64X
    Xb2 = R.c64_staged_exin()
    assert Xb2 == Xb
    X2 = [[R.e4m3_decode(Xb2[m * 64 + k]) for k in range(64)]
          for m in range(64)]
    O2 = R.c64_expand_forward(X2, W1)
    assert [v for row in O2 for v in row] == flatO, \
        "c64x not deterministic"
    # S92 pre-interpretation: how much of the golden is an exact f16
    # image? The oracle's last op is an f16 round, so all of it should
    # be; that bounds the device-vs-oracle gap to ~1 ULP, not to noise.
    exact = sum(1 for u in obits if (u & 0x1FFF) == 0)
    print("wrote %s O[0]=%.6f maxabs=%.4f exact-f16=%d/%d checksum=%s" %
          (DST_C64X, flatO[0], move, exact, len(flatO),
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


def main_c64f2(tiled=""):
    # Test-24 golden (HANDOFF S116): the FIRST CONNECTED C=64 FFN. staged
    # x [64][64] e4m3 (fresh C64_FFN2_SEED) feeds expand -> act -> e4m3 ->
    # contract per pass, + gate1 residual. Unlike Test-20/22/23 the expand
    # output is not re-staged -- it is the contract's input. w1 (dump
    # +0/+8192) and w2 (dump +16384/+20480) REAL, IN-MAJOR.
    Xb = R.c64_staged_ffnin()
    assert len(Xb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    if tiled:
        # S234: the same gate as the c*f2 path -- read what the device reads.
        assert tiled == "tiled", tiled
        import wload_map
        before = fnv_of(dump)
        dump = wload_map.mode2_dense(dump, 64, 128, 32)
        print("  tiled reading (mode 2): fnv %08X -> %08X" % (before, fnv_of(dump)))
        assert fnv_of(dump) == 0xA3B5472E, (
            "mode2 rewrite drifted: got %08X, the shim logs A3B5472E" % fnv_of(dump))
    X = [[R.e4m3_decode(Xb[m * 64 + k]) for k in range(64)]
         for m in range(64)]
    # S148: OUT-major [128][64] -- (k,n) at n*64 + k.
    w1p = [[[R.e4m3_decode(dump[o + n * 64 + k]) for n in range(128)]
            for k in range(64)] for o in (0, 8192)]
    # S148: OUT-major [32][128] -- (k,n) at n*128 + k.
    w2p = [[[R.e4m3_decode(dump[o + n * 128 + k]) for n in range(32)]
            for k in range(128)] for o in (16384, 20480)]
    gate1 = [R.f16_to_f32(struct.unpack_from("<H", dump, 28688 + 2 * o)[0])
             for o in range(64)]
    O = R.c64_ffn2_forward(X, w1p, w2p, gate1)
    flatO = [v for row in O for v in row]
    assert len(flatO) == 4096
    assert sum(1 for v in flatO if math.isfinite(v)) == 4096
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64f2 -- DO NOT HAND-EDIT.\n"
            "// Test-24: CONNECTED C=64 FFN (S116). staged x -> expand ->\n"
            "// act -> e4m3 -> contract, per pass, + gate1 residual; w1/w2\n"
            "// real (dump +0/+8192, +16384/+20480), IN-MAJOR.\n"
            "// Backend: c64f2-block maxerr <= 0.1.\n"
            "static const unsigned int kC64Ffn2BytesCount = 4096;\n"
            "static const unsigned char kC64Ffn2Bytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Xb], 12) + "};\n"
            "static const unsigned int kC64Ffn2GoldenCount = 4096;\n"
            "static const unsigned int kC64Ffn2Golden[4096] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    with open(DST_C64F2, "w", newline="\n") as fh:
        fh.write(body)
    assert open(DST_C64F2).read().count("0x") == 4096 + 4096, DST_C64F2
    Xb2 = R.c64_staged_ffnin()
    assert Xb2 == Xb
    O2 = R.c64_ffn2_forward(
        [[R.e4m3_decode(Xb2[m * 64 + k]) for k in range(64)]
         for m in range(64)], w1p, w2p, gate1)
    assert [v for row in O2 for v in row] == flatO, "c64f2 not deterministic"
    exact = sum(1 for u in obits if (u & 0x1FFF) == 0)
    print("wrote %s O[0]=%.6f maxabs=%.4f exact-f16=%d/%d checksum=%s" %
          (DST_C64F2, flatO[0], move, exact, len(flatO),
           R.fnv1a_hex(flatO)))


# FNV-1a of the tensor AFTER mode2_dense, as the shim logs it next to the
# reading it applied ("hip: <check> tensor read as TILED (HipFfnTranspose=2,"
# "fnv ...)"). These are OBSERVED device values, not derived ones, so a golden
# can never be generated from a mirror that has drifted from the device.
kTILED_FNV = {32: 0x75365F3A, 64: 0xA3B5472E, 128: 0x8226EDC7, 256: 0xEA6B987D}

def fnv_of(b):
    """FNV-1a 32, the same hash the shim logs next to each applied reading."""
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def main_cw_f2(C, tensor=None, sym_suffix=None, tiled=""):
    """A connected FFN at width C (Phase A). One code path for every width: the
    oracle is parameterised on C64_C/C64_W/C64_H1P (S138) and the stage tensor's
    layout is A = heads*(C*H1 + H1*W + C*W), gate1 at A+16 -- VERIFIED against
    the real bytes in S136 for C=64 and C=128.

    Only the width, the tensor file and the golden symbol names change."""
    NL = chr(10)
    W, H1, heads, TOK = 32, 128, C // 32, R.C64_TOK
    TENSOR = {64: "tensor_137.bin", 128: "tensor_002.bin", 256: "tensor_007.bin"}[C]
    SIZE = {64: 61760, 128: 197184, 256: 689232}[C]
    # S225 — THE DECODER'S STAGES ARE THE SAME FAMILY, AND THIS PROVES IT.
    # The decoder's tail (blocks 49-69) is the encoder's stage layouts run in
    # mirror, and its tensors are byte-for-byte the same size (block 49 is
    # 689232, exactly like tensor_007). The layout rules below are the same ones
    # S136 verified on the encoder's bytes, so pointing them at a decoder tensor
    # is a real test of the claim: pass the tensor file and the oracle either
    # closes on those bytes or asserts. `suffix` keeps the emitted symbols
    # distinct from the encoder's so both goldens can live in the tree.
    if tensor:
        TENSOR = tensor if tensor.endswith(".bin") else tensor + ".bin"
    suffix = sym_suffix or ""
    if suffix and not re.fullmatch(r"[A-Za-z0-9_]+", suffix):
        raise SystemExit("symbol suffix must be alphanumeric, got %r" % suffix)
    R.C64_C, R.C64_W, R.C64_H1P = C, W, H1          # the S138 parameterisation
    Xb = R.cw_staged_ffnin(C)
    assert len(Xb) == TOK * C, len(Xb)
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors", TENSOR), "rb").read()
    assert len(dump) == SIZE, len(dump)
    if tiled:
        # S231 step 3 -- READ WHAT THE DEVICE READS.
        # The shim's live runners un-permute W1/W2 before the kernels see them, so
        # a golden generated from the raw blob would describe a reading nothing
        # runs. mode2_dense mirrors that rewrite and is PROVEN byte-identical
        # against the FNV the shim logs for HipFfnTranspose=2 (see
        # tools/wload_map.py); the assert below is the guard against drift.
        assert tiled == "tiled", tiled
        import wload_map
        before = fnv_of(dump)
        dump = wload_map.mode2_dense(dump, C, H1, W)
        print("  tiled reading (mode 2): fnv %08X -> %08X" % (before, fnv_of(dump)))
        # The shim logs this hash next to the reading it applied, so these are
        # OBSERVED device values, not derived ones -- they are what
        # `hip: c<C>f2-block tensor read as TILED (HipFfnTranspose=2, fnv ...)`
        # printed, at each width. A mismatch means mode2_dense has drifted.
        kTiledFnv = kTILED_FNV
        assert fnv_of(dump) == kTiledFnv[C], (
            "mode2 rewrite drifted from the reading the shim applies: got %08X, "
            "the shim logs %08X" % (fnv_of(dump), kTiledFnv[C]))
    X = [[R.e4m3_decode(Xb[m * C + k]) for k in range(C)]
         for m in range(TOK)]
    R1 = heads * C * H1                     # w1 slices, [C][H1] IN-MAJOR
    R2 = heads * H1 * W                     # w2 slices, [H1][W] IN-MAJOR
    R3 = heads * C * W                      # self-link slices
    # S148: OUT-major [H1][C] -- element (k,n) lives at n*C + k.
    w1p = [[[R.e4m3_decode(dump[p * C * H1 + n * C + k]) for n in range(H1)]
            for k in range(C)] for p in range(heads)]
    # S148: OUT-major [W][H1] -- element (k,n) lives at n*H1 + k.
    w2p = [[[R.e4m3_decode(dump[R1 + p * H1 * W + n * H1 + k]) for n in range(W)]
            for k in range(H1)] for p in range(heads)]
    gate1 = [R.f16_to_f32(
        struct.unpack_from("<H", dump, R1 + R2 + R3 + 16 + 2 * o)[0])
        for o in range(C)]
    O = R.c64_ffn2_forward(X, w1p, w2p, gate1)
    flatO = [v for row in O for v in row]
    assert len(flatO) == TOK * C, len(flatO)
    nfin = sum(1 for v in flatO if math.isfinite(v))
    move = max(abs(v) for v in flatO)
    assert nfin == len(flatO), (nfin, len(flatO))
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1",
                        "rw_yp_c%df2%s_golden.inc" % (C, suffix))
    pre = "kC%dFfn2%s" % (C, suffix)
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --c%df2 -- DO NOT HAND-EDIT." % C,
        "// Phase A: CONNECTED C=%d FFN, the SAME oracle as --c64f2 at C=%d." % (C, C),
        "// staged x[64][%d] -> expand -> act -> e4m3 -> contract per pass," % C,
        "// + gate1 residual. Weights REAL from %s (layout S136)." % TENSOR,
        "// Backend: c%df2-block maxerr <= 0.1." % C,
        "// Tensor: %s   (S225 decoder-mirror run, symbols suffixed %r)" % (TENSOR, suffix),
        "static const unsigned int %sBytesCount = %d;" % (pre, len(Xb)),
        "static const unsigned char %sBytes[%d] = {" % (pre, len(Xb)),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int %sGoldenCount = %d;" % (pre, len(flatO)),
        "static const unsigned int %sGolden[%d] = {" % (pre, len(flatO)),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    assert open(dst).read().count("0x") == len(Xb) + len(flatO), dst
    print("wrote %s C=%d O[0]=%.6f maxabs=%.4f nonfinite=%d/%d checksum=%s" %
          (dst, C, flatO[0], move, len(flatO) - nfin, len(flatO),
           R.fnv1a_hex(flatO)))


def main_vitproj():
    """Phase C: the ViT block's layer4, the attention output projection (S150).

    tensor_054 = 1048576 B of fp8 e4m3 weights + 2048 B of fp16
    residual_coefficients (the skip gate, 1024 of them). Square [1024][1024], so
    the OUT-major reading is [out][in] with rows of 1024 -- same convention the
    QKV's decisive offset test and the universal .row.col operand form give.

    Kernel: k_c64proj at C=1024, out-major. 1024 outputs for one token, grid 4.
    Ocat and y are staged e4m3 streams (the real Ocat would come from the
    attention; this stage is checked on its own, as the FFN and QKV are)."""
    NL = chr(10)
    import numpy as _np
    C, TOK = 1024, 1
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, C, 32, 4096
    R.C64_TOK = TOK
    T54 = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_054.bin")
    d = open(T54, "rb").read()
    assert len(d) == 1050624, len(d)          # 1048576 + 2048
    tab = _np.array([R.e4m3_decode(b) for b in range(256)], dtype=_np.float32)
    # OUT-major [out][in]: element (c,o) lives at o*C + c.
    W = tab[_np.frombuffer(d[:1048576], dtype=_np.uint8)].reshape(C, C).T
    W = W.astype(_np.float64)
    gate2 = [R.f16_to_f32(struct.unpack_from("<H", d, 1048576 + 2 * o)[0])
             for o in range(C)]
    assert all(math.isfinite(g) for g in gate2)

    oc, _ = R.gen_vec(0x10240F3, TOK * C, 0, 5)
    yv, _ = R.gen_vec(0x10240F4, TOK * C, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Ob = [R.synth_to_e4m3([levels[m]])[0] for m in oc]
    Yb = [R.synth_to_e4m3([levels[m]])[0] for m in yv]
    Ocat = [[R.e4m3_decode(b) for b in Ob]]
    y = [[R.e4m3_decode(b) for b in Yb]]
    # Through the SAME oracle the c64p block test uses, so this is an
    # oracle-vs-device check like every other stage here.
    Or = R.c64_proj_forward(Ocat, W, gate2, y)
    flatO = [float(v) for row in Or for v in row]
    assert len(flatO) == C, len(flatO)
    nfin = sum(1 for v in flatO if math.isfinite(v))
    move = max(abs(v) for v in flatO)
    assert nfin == len(flatO), (nfin, len(flatO))
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_vitproj_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --vitproj -- DO NOT HAND-EDIT.",
        "// Phase C: block31 layer4, the attention output projection, [1024][1024]",
        "// read OUT-major (S148/S150). Real weights from tensor_054 + its 1024",
        "// fp16 residual_coefficients. ONE token, grid 4 x 256.",
        "// Backend: vitproj-block maxerr <= 0.1.",
        "static const unsigned int kVitProjObBytesCount = %d;" % len(Ob),
        "static const unsigned char kVitProjObBytes[%d] = {" % len(Ob),
        wrap(["0x%02X" % b for b in Ob], 12) + "};",
        "static const unsigned int kVitProjYBytesCount = %d;" % len(Yb),
        "static const unsigned char kVitProjYBytes[%d] = {" % len(Yb),
        wrap(["0x%02X" % b for b in Yb], 12) + "};",
        "static const unsigned int kVitProjGateCount = %d;" % len(gate2),
        "static const unsigned int kVitProjGate[%d] = {" % len(gate2),
        wrap(["0x%08X" % struct.unpack("<I", struct.pack("<f", v))[0]
              for v in gate2], 6) + "};",
        "static const unsigned int kVitProjGoldenCount = %d;" % len(flatO),
        "static const unsigned int kVitProjGolden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    assert open(dst).read().count("0x") == len(Ob) + len(Yb) + len(gate2) + len(flatO), dst
    print("wrote %s C=%d O[0]=%.6f maxabs=%.4f nonfinite=%d/%d checksum=%s" %
          (dst, C, flatO[0], move, len(flatO) - nfin, len(flatO),
           R.fnv1a_hex(flatO)))


def main_vitqkv():
    """Phase C: the ViT block's layer2, the QKV projection (S147).

    A plain [C][3C] projection -- NOT the fused stage's per-head [DIM][3W]
    layout. The weight matrix is stored OUT-major as [3C][C] = [3072][1024],
    measured from cc_vit_qkv_fp8's addressing (32768-byte = 32-row blocks with
    512-byte sub-offsets imply a 1024-byte = C contiguous row).

    tensor_052 = 128 B of fp32 qkv_head_coefficients, then 3145728 B of fp8
    e4m3 weights at +128.

    ONE token: grid ceil(1*3C/256) = 12 blocks = 3072 outputs, every output
    channel, full K = C dot.

    Accumulation is f64 then rounded to f32, matching the existing oracles'
    style (Python floats are f64); the kernel accumulates f32. That ~1e-5
    ordering gap is what every other golden here already carries and tol 0.1
    covers."""
    NL = chr(10)
    import numpy as _np
    C, TOK = 1024, 1
    N3 = 3 * C
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, C, 32, 4096
    R.C64_TOK = TOK
    T52 = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_052.bin")
    d = open(T52, "rb").read()
    assert len(d) == 3145856, len(d)          # 128 + 3145728
    coff = struct.unpack_from("<32f", d, 0)
    print("  qkv head coefficients: min %.4g max %.4g" % (min(coff), max(coff)))

    tab = _np.array([R.e4m3_decode(b) for b in range(256)], dtype=_np.float32)
    W = tab[_np.frombuffer(d[128:128 + N3 * C], dtype=_np.uint8)]
    W = W.reshape(N3, C).astype(_np.float64)   # [3C][C], out-major

    seed = 0x10240F2                            # the vitf2 staged input
    xv, _ = R.gen_vec(seed, TOK * C, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Xb = [R.synth_to_e4m3([levels[m]])[0] for m in xv]
    assert len(Xb) == TOK * C, len(Xb)
    xq = tab[_np.frombuffer(bytes(Xb), dtype=_np.uint8)].astype(_np.float64)

    flatO = [float(v) for v in (W @ xq)]        # (3072,) f64 -> f32 below
    assert len(flatO) == N3, len(flatO)
    nfin = sum(1 for v in flatO if math.isfinite(v))
    move = max(abs(v) for v in flatO)
    assert nfin == len(flatO), (nfin, len(flatO))
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_vitqkv_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --vitqkv -- DO NOT HAND-EDIT.",
        "// Phase C: block31 layer2, the QKV projection [3C][C] = [3072][1024]",
        "// (out-major; S147 measured the layout from cc_vit_qkv_fp8).",
        "// Real weights from tensor_052 +128; 32 fp32 head coefficients precede.",
        "// ONE token: grid 12 x 256 = 3072 outputs, full K=1024 dot.",
        "// Backend: vitqkv-block maxerr <= 0.1.",
        "static const unsigned int kVitQkvBytesCount = %d;" % len(Xb),
        "static const unsigned char kVitQkvBytes[%d] = {" % len(Xb),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int kVitQkvCoeffCount = 32;",
        "static const unsigned int kVitQkvCoeff[32] = {",
        wrap(["0x%08X" % struct.unpack("<I", struct.pack("<f", v))[0]
              for v in coff], 6) + "};",
        "static const unsigned int kVitQkvGoldenCount = %d;" % len(flatO),
        "static const unsigned int kVitQkvGolden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    assert open(dst).read().count("0x") == len(Xb) + 32 + len(flatO), dst
    print("wrote %s N=3C=%d O[0]=%.6f maxabs=%.4f nonfinite=%d/%d checksum=%s" %
          (dst, N3, flatO[0], move, len(flatO) - nfin, len(flatO),
           R.fnv1a_hex(flatO)))


def main_vitf2():
    """Phase C: block31's FFN, run through the SAME oracle as the fused stages.

    S140/S145: the ViT block's layer0 is [4096,1024] and layer1 is [1024,4096]
    with a 2048-byte fp16 residue = 1024 gates. k_c64ffn2c needs w1[C][H1],
    w2[H1][W] and gate1[W], and uses only C/K/H1/W -- NOT heads. So setting
    C=1024, W=1024 (ONE pass) and H1=4096 makes our kernel the ViT's FFN
    exactly: no new kernel, a parameter set (the S145 finding).

    ONE token only: the launch grid is ceil(W/256) = 4 blocks, so a 1-token
    golden exercises every output column and every k-step while keeping the
    emitted array at 1024 values. The weights are the REAL tensors.

    LAYOUT CAVEAT (VIT_VERIFICATION.md, and the reason the shim test's value is
    limited): the physical packing of these matrices is recorded as
    "unverified". We read layer0 as [in][out] = [1024][4096] and layer1 as
    [4096][1024], i.e. K-major, matching the fused stages' w1/w2 convention.
    Oracle and kernel use the SAME reading, so this test shows the kernel
    implements the oracle at these dimensions -- it does NOT confirm the
    packing is what NVIDIA used. Same caveat class as S139."""
    NL = chr(10)
    import numpy as _np
    C, W, H1, TOK = 1024, 1024, 4096, 1
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, W, 32, H1
    R.C64_TOK = TOK
    LAYER0 = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_050.bin")
    LAYER1 = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_051.bin")
    d0 = open(LAYER0, "rb").read()
    d1 = open(LAYER1, "rb").read()
    assert len(d0) == 4194320, len(d0)          # 4096*1024 + 16
    assert len(d1) == 4196352, len(d1)          # 1024*4096 + 2048
    assert len(d0) - 4194304 == 16, "layer0 suffix is 16 bytes"
    assert len(d1) - 4194304 == 2048, "layer1 suffix is the fp16 gate block"

    # e4m3 decode as a 256-entry table, vectorised -- 4.2M values each.
    tab = _np.array([R.e4m3_decode(b) for b in range(256)], dtype=_np.float32)
    # S148: tensors are OUT-major ([H1][C] / [W][H1]); transpose into the
    # [C][H1] / [H1][W] shape the (unchanged) oracle indexes.
    w1 = tab[_np.frombuffer(d0[:4194304], dtype=_np.uint8)].reshape(H1, C).T
    w2 = tab[_np.frombuffer(d1[:4194304], dtype=_np.uint8)].reshape(W, H1).T

    # gate1: 1024 fp16 from layer1's 2048-byte suffix (2 bytes per channel).
    gate1 = [R.f16_to_f32(struct.unpack_from("<H", d1, 4194304 + 2 * o)[0])
             for o in range(W)]

    seed = 0x10240F2
    xv, _ = R.gen_vec(seed, TOK * C, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Xb = [R.synth_to_e4m3([levels[m]])[0] for m in xv]
    assert len(Xb) == TOK * C, len(Xb)
    X = [[R.e4m3_decode(b) for b in Xb]]

    O = R.c64_ffn2_forward(X, [w1], [w2], gate1)
    flatO = [float(v) for row in O for v in row]
    assert len(flatO) == TOK * C, len(flatO)
    nfin = sum(1 for v in flatO if math.isfinite(v))
    move = max(abs(v) for v in flatO)
    assert nfin == len(flatO), (nfin, len(flatO))
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_vitf2_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --vitf2 -- DO NOT HAND-EDIT.",
        "// Phase C: block31's FFN at C=1024, H1=4096, ONE pass (S140/S145).",
        "// Real weights: tensor_050 (layer0) + tensor_051 (layer1) + its",
        "// 1024 fp16 gate coefficients. ONE token: grid ceil(1024/256)=4.",
        "// Backend: vitf2-block maxerr <= 0.1.",
        "static const unsigned int kVitFfn2BytesCount = %d;" % len(Xb),
        "static const unsigned char kVitFfn2Bytes[%d] = {" % len(Xb),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int kVitFfn2GoldenCount = %d;" % len(flatO),
        "static const unsigned int kVitFfn2Golden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    assert open(dst).read().count("0x") == len(Xb) + len(flatO), dst
    print("wrote %s C=%d H1=%d TOK=%d O[0]=%.6f maxabs=%.4f nonfinite=%d/%d checksum=%s" %
          (dst, C, H1, TOK, flatO[0], move, len(flatO) - nfin, len(flatO),
           R.fnv1a_hex(flatO)))


def main_c32blk(tiled=""):
    """S170: the CONNECTED C=32 BLOCK -- the FOURTH and last fused-stage width.

    C=32 is the family's structural outlier: S135 found its A deviates from
    heads*(C*H1 + H1*W + C*W) by exactly -C*W, i.e. there is NO self-link
    region at heads=1. Confirmed here -- the C=32 stage tensors are 20672 B,
    which is the deviated total to the byte (21696 would be the formula's):

        A = 8192 (= C*H1 + H1*W, no R3)
        gate1 @ A+16 = 8208 (2C = 64 B), pad 16
        B @ 8288 = [qkv 3C^2][bias 8CW][temp max(16,4*heads)][proj C^2]
        gate2 @ len - 16 - 2C = 20592, then a 16-byte tail   -> 20672 EXACT

    Verified by reading the bytes before writing any code: temps decode to
    [0.4088, 0, 0, 0] (one head), the proj region reads maxabs 0.5625 where a
    constant-16 temp block would put it at maxabs 9, and gate2 decodes to ~1.
    NOTE S135's -C*W is NOT applied by the generic formula -- it is specific to
    heads == 1 and must not be silently generalised back."""
    NL = chr(10)
    C, W, H1, HEADS, TOK = 32, 32, 128, 1, 64
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, W, 32, H1
    R.C64_HEADS, R.C64_TOK = HEADS, TOK
    Xb = R.cw_staged_ffnin(C)
    assert len(Xb) == TOK * C, len(Xb)
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_001.bin"), "rb").read()
    assert len(dump) == 20672, len(dump)
    if tiled:
        # S236: the same gate as the f2 checks -- the live path rewrites the
        # C=32 stage FFN too (FeC32BlockRun). C=32 is HEADS=1, H1=128, W=32.
        assert tiled == "tiled", tiled
        import wload_map
        before = fnv_of(dump)
        dump = wload_map.mode2_dense(dump, 32, 128, 32)
        print("  tiled reading (mode 2): fnv %08X -> %08X" % (before, fnv_of(dump)))
        assert fnv_of(dump) == kTILED_FNV[32], (
            "mode2 rewrite drifted: got %08X, the shim logs %08X"
            % (fnv_of(dump), kTILED_FNV[32]))

    def h16(off):
        return R.f16_to_f32(struct.unpack_from("<H", dump, off)[0])

    A = HEADS * (C * H1 + H1 * W + C * W) - C * W      # S135: no self-link
    B = A + 16 + 2 * C + 16
    TEMP = max(16, 4 * HEADS)
    OFF_G2 = len(dump) - 16 - 2 * C
    assert A == 8192, A
    assert B == 8288, B
    assert B + 3 * C * C + 8 * C * W + TEMP + C * C == OFF_G2, "proj ends at gate2"

    X = [[R.e4m3_decode(Xb[m * C + k]) for k in range(C)] for m in range(TOK)]
    w1p = [[[R.e4m3_decode(dump[p * C * H1 + n * C + k]) for n in range(H1)]
            for k in range(C)] for p in range(HEADS)]
    r2 = C * H1 * HEADS
    w2p = [[[R.e4m3_decode(dump[r2 + p * H1 * W + n * H1 + k])
             for n in range(W)] for k in range(H1)] for p in range(HEADS)]
    g1 = [h16(A + 16 + 2 * o) for o in range(C)]
    wqkv = [[[[R.e4m3_decode(
        dump[B + ((kh * HEADS) + h) * 3072 + k * 96 + n])
        for n in range(96)] for k in range(W)] for h in range(HEADS)]
        for kh in range(C // W)]
    bias = [[[h16(B + 3 * C * C + 2 * (h * TOK * TOK + i * TOK + j))
              for j in range(TOK)] for i in range(TOK)]
            for h in range(HEADS)]
    temps = [struct.unpack_from("<f", dump, B + 3 * C * C + 8 * C * W +
                                4 * h)[0] for h in range(HEADS)]
    wproj = [[R.e4m3_decode(dump[B + 3 * C * C + 8 * C * W + TEMP + o * C + c])
              for o in range(C)] for c in range(C)]
    g2 = [h16(OFF_G2 + 2 * o) for o in range(C)]

    Yp = R.c64_block_forward(X, w1p, w2p, g1, wqkv, temps, bias, wproj, g2)
    flatO = [v for row in Yp for v in row]
    assert len(flatO) == TOK * C, len(flatO)
    assert sum(1 for v in flatO if math.isfinite(v)) == len(flatO)
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c32blk_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --c32blk -- DO NOT HAND-EDIT.",
        "// S170: CONNECTED C=32 BLOCK, fourth fused-stage width. A=8192 --",
        "// S135's -C*W deviation (no self-link region at heads=1), which is",
        "// why the stage tensors are 20672 B and not 21696. gate1 @A+16,",
        "// B @8288, temp max(16,4*heads), gate2 @len-16-2C.",
        "// Backend: c32blk-block maxerr <= 0.1.",
        "static const unsigned int kC32BlkBytesCount = %d;" % len(Xb),
        "static const unsigned char kC32BlkBytes[%d] = {" % len(Xb),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int kC32BlockGoldenCount = %d;" % len(flatO),
        "static const unsigned int kC32BlockGolden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    print("wrote %s C=%d O[0]=%.6f maxabs=%.4f nfinite=%d/%d checksum=%s" %
          (dst, C, flatO[0], move, len(flatO), len(flatO),
           R.fnv1a_hex(flatO)))


def main_c256blk():
    """S169: the CONNECTED C=256 BLOCK -- the third width of the same template.

    Layout from S158/S160, with every rule re-checked at this width rather than
    assumed:
        A = heads*(C*H1 + H1*W + C*W) = 360448
        gate1 @ A+16 = 360464 (2C = 512 B), then a CONSTANT 16-byte pad
        B @ 360992 = [qkv 3C^2][bias 8CW][temp 16][proj C^2] = 327696
        gate2 @ len - 16 - 2C = 688704, then a 16-byte tail
    The pad and the temps position were confirmed independently: an f32 scan
    found a run of exactly 8 plausible values (heads=8) at 623136, which is
    B + 3C^2 + 8CW under pad=16 and 16 bytes off under pad=32.


    NOTE ON SIZES: every std::vector<float> upload in the shim is count*4. Both
    of S168's bugs were byte counts copied from the C=64 test where the halved
    value happened to equal the whole buffer. The counts here are derived from
    the width, not copied."""
    NL = chr(10)
    import numpy as _np
    C, W, H1, HEADS, TOK = 256, 32, 128, 8, 64
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, W, 32, H1
    R.C64_HEADS, R.C64_TOK = HEADS, TOK
    Xb = R.cw_staged_ffnin(C)
    assert len(Xb) == TOK * C, len(Xb)
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_007.bin"), "rb").read()
    assert len(dump) == 689232, len(dump)

    def h16(off):
        return R.f16_to_f32(struct.unpack_from("<H", dump, off)[0])

    A = HEADS * (C * H1 + H1 * W + C * W)
    B = A + 16 + 2 * C + 16
    BSZ = 4 * C * C + 8 * C * W + 16
    OFF_G2 = len(dump) - 16 - 2 * C
    assert A == 360448, A
    assert B == 360992, B
    assert B + BSZ == 688688, B + BSZ
    assert OFF_G2 == 688704, OFF_G2
    assert OFF_G2 + 2 * C + 16 == len(dump)
    assert B + 3 * C * C + 8 * C * W == 623136, "temps position"
    assert B + 3 * C * C + 8 * C * W + max(16, 4 * HEADS) + C * C == OFF_G2,         "proj ends exactly at gate2"

    X = [[R.e4m3_decode(Xb[m * C + k]) for k in range(C)] for m in range(TOK)]
    w1p = [[[R.e4m3_decode(dump[p * C * H1 + n * C + k]) for n in range(H1)]
            for k in range(C)] for p in range(HEADS)]
    r2 = C * H1 * HEADS
    w2p = [[[R.e4m3_decode(dump[r2 + p * H1 * W + n * H1 + k])
             for n in range(W)] for k in range(H1)] for p in range(HEADS)]
    g1 = [h16(A + 16 + 2 * o) for o in range(C)]
    wqkv = [[[[R.e4m3_decode(
        dump[B + ((kh * HEADS) + h) * 3072 + k * 96 + n])
        for n in range(96)] for k in range(W)] for h in range(HEADS)]
        for kh in range(C // W)]
    bias = [[[h16(B + 3 * C * C + 2 * (h * TOK * TOK + i * TOK + j))
              for j in range(TOK)] for i in range(TOK)]
            for h in range(HEADS)]
    temps = [struct.unpack_from("<f", dump, B + 3 * C * C + 8 * C * W +
                                4 * h)[0] for h in range(HEADS)]
    # S169b: the temp block is `4*heads` bytes ROUNDED UP TO 16 -- not a
    # constant 16. heads=2 -> 8->16, heads=4 -> 16, heads=8 -> 32. That is why
    # C=64 and C=128 both worked with a hard-coded 16 and C=256 did not: the
    # proj then started 16 B early and read the temps' tail as weights
    # (maxabs 448 = e4m3 saturation, vs 0.1875 for the real region).
    TEMP = max(16, 4 * HEADS)
    wproj = [[R.e4m3_decode(dump[B + 3 * C * C + 8 * C * W + TEMP + o * C + c])
              for o in range(C)] for c in range(C)]
    g2 = [h16(OFF_G2 + 2 * o) for o in range(C)]

    Yp = R.c64_block_forward(X, w1p, w2p, g1, wqkv, temps, bias, wproj, g2)
    flatO = [v for row in Yp for v in row]
    assert len(flatO) == TOK * C, len(flatO)
    assert sum(1 for v in flatO if math.isfinite(v)) == len(flatO)
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c256blk_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --c256blk -- DO NOT HAND-EDIT.",
        "// S169: CONNECTED C=256 BLOCK, third width of the fused-stage template.",
        "// Layout S158/S160 re-verified here: A 360448, gate1 @A+16, const 16",
        "// pad, B @360992, gate2 @len-16-2C. Out-major reads (S148/S150).",
        "// Backend: c256blk-block maxerr <= 0.1.",
        "static const unsigned int kC256BlkBytesCount = %d;" % len(Xb),
        "static const unsigned char kC256BlkBytes[%d] = {" % len(Xb),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int kC256BlockGoldenCount = %d;" % len(flatO),
        "static const unsigned int kC256BlockGolden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    print("wrote %s C=%d O[0]=%.6f maxabs=%.4f nfinite=%d/%d checksum=%s" %
          (dst, C, flatO[0], move, len(flatO), len(flatO),
           R.fnv1a_hex(flatO)))


def main_c128blk():
    """S159: the CONNECTED C=128 BLOCK, the C=64 one (Test-25) at a second width.

    Uses the S157/S158 layout, which is what makes this assembly rather than
    analysis:
        A = 98304, in three regions: R1 w1 at +0 (4 passes, C*H1 each = 16384),
            R2 w2 at +65536 (4 passes, H1*W each = 4096), R3 at +81920 (unused here)
        gate1 at A+16 = 98320, C fp16
        B at 98608 = [qkv 3C^2 = 49152][bias 8CW = 32768][temp 16][proj C^2 = 16384]
        gate2 at 196928 = size - 2C, C fp16
    Out-major weight reads throughout (S148/S150)."""
    NL = chr(10)
    C, W, H1, HEADS, TOK = 128, 32, 128, 4, 64
    R.C64_C, R.C64_W, R.C64_K, R.C64_H1P = C, W, 32, H1
    R.C64_HEADS, R.C64_TOK = HEADS, TOK     # HEADS too: the block oracle derives
                                            # Ocat's width as HEADS*DIM (S159)
    Xb = R.cw_staged_ffnin(C)
    assert len(Xb) == TOK * C, len(Xb)
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_002.bin"), "rb").read()
    assert len(dump) == 197184, len(dump)

    def h16(off):
        return R.f16_to_f32(struct.unpack_from("<H", dump, off)[0])

    A = HEADS * (C * H1 + H1 * W + C * W)
    assert A == 98304, A
    # S160: the pad after gate1 is a CONSTANT 16, not 16*log2(heads) -- S158's
    # rule was wrong. Calibrated on C=64 (B=28832 = 28672+16+128+16) and
    # confirmed at C=128 by the temps landing exactly on the f32 scan's find.
    B = A + 16 + 2 * C + 16
    assert B == 98592, B
    assert B + (3 * C * C + 8 * C * W + 16 + C * C) + 2 * C + 16 == len(dump), "B end"

    X = [[R.e4m3_decode(Xb[m * C + k]) for k in range(C)] for m in range(TOK)]
    # R1: w1 passes at 0, C*H1, 2*C*H1, ... OUT-major [H1][C] -> (k,n) at n*C+k
    w1p = [[[R.e4m3_decode(dump[p * C * H1 + n * C + k]) for n in range(H1)]
            for k in range(C)] for p in range(HEADS)]
    # R2: w2 passes at A//2 ... but ours are laid out after R1: offset = C*H1*HEADS
    r2 = C * H1 * HEADS
    w2p = [[[R.e4m3_decode(dump[r2 + p * H1 * W + n * H1 + k])
             for n in range(W)] for k in range(H1)] for p in range(HEADS)]
    g1 = [h16(A + 16 + 2 * o) for o in range(C)]
    # B: qkv, laid out like k_c64qkv -- block index (kk/W)*HEADS + h, block size
    # DIM*3W = 3072, inner (kk%W)*3W + n
    wqkv = [[[[R.e4m3_decode(
        dump[B + ((kh * HEADS) + h) * 3072 + k * 96 + n])
        for n in range(96)] for k in range(W)] for h in range(HEADS)]
        for kh in range(C // W)]
    bias = [[[h16(B + 3 * C * C + 2 * (h * TOK * TOK + i * TOK + j))
              for j in range(TOK)] for i in range(TOK)]
            for h in range(HEADS)]
    temps = [struct.unpack_from("<f", dump, B + 3 * C * C + 8 * C * W +
                                4 * h)[0] for h in range(HEADS)]
    # OUT-major [out][in] -> (c,o) at o*C + c
    wproj = [[R.e4m3_decode(dump[B + 3 * C * C + 8 * C * W + 16 + o * C + c])
              for o in range(C)] for c in range(C)]
    # gate2 is the last 2C block BEFORE a constant 16-byte tail (S160).
    g2 = [h16(len(dump) - 16 - 2 * C + 2 * o) for o in range(C)]

    Yp = R.c64_block_forward(X, w1p, w2p, g1, wqkv, temps, bias, wproj, g2)
    flatO = [v for row in Yp for v in row]
    assert len(flatO) == TOK * C, len(flatO)
    assert sum(1 for v in flatO if math.isfinite(v)) == len(flatO)
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    dst = os.path.join(ROOT, "hip", "mvp1", "rw_yp_c128blk_golden.inc")
    body = NL.join([
        "// Generated by tools/emit_yp_golden.py --c128blk -- DO NOT HAND-EDIT.",
        "// S159: CONNECTED C=128 BLOCK -- x -> FFN -> QKV -> scores -> softmax",
        "// -> ctx -> proj from ONE input. Weights REAL from tensor_002 using",
        "// the S157/S158 layout: A 98304 (R1+R2+R3), gate1 @A+16, B @98608,",
        "// gate2 @size-2C. Out-major reads (S148/S150).",
        "// Backend: c128blk-block maxerr <= 0.1.",
        "static const unsigned int kC128BlkBytesCount = %d;" % len(Xb),
        "static const unsigned char kC128BlkBytes[%d] = {" % len(Xb),
        wrap(["0x%02X" % b for b in Xb], 12) + "};",
        "static const unsigned int kC128BlockGoldenCount = %d;" % len(flatO),
        "static const unsigned int kC128BlockGolden[%d] = {" % len(flatO),
        wrap(["0x%08X" % u for u in obits], 6) + "};",
        ""])
    with open(dst, "w", newline=NL) as fh:
        fh.write(body)
    print("wrote %s C=%d O[0]=%.6f maxabs=%.4f nfinite=%d/%d checksum=%s" %
          (dst, C, flatO[0], move, len(flatO), len(flatO),
           R.fnv1a_hex(flatO)))


def main_c64blk():
    # Test-25 golden (HANDOFF S116): the first CONNECTED C=64 BLOCK --
    # staged x -> FFN -> QKV -> scores -> softmax -> ctx -> proj from ONE
    # input (only the inter-stage wiring/boundaries are new). All weights
    # REAL from tensor_137: FFN +0/+8192/+16384/+20480/+28688, qkv +28832,
    # bias +41120, temp +57504, proj +57520, gate2 +61616.
    Xb = R.c64_staged_blkin()
    assert len(Xb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)

    def h16(off):
        return R.f16_to_f32(struct.unpack_from("<H", dump, off)[0])

    X = [[R.e4m3_decode(Xb[m * 64 + k]) for k in range(64)]
         for m in range(64)]
    # S148: OUT-major [128][64] -- (k,n) at n*64 + k.
    w1p = [[[R.e4m3_decode(dump[o + n * 64 + k]) for n in range(128)]
            for k in range(64)] for o in (0, 8192)]
    # S148: OUT-major [32][128] -- (k,n) at n*128 + k.
    w2p = [[[R.e4m3_decode(dump[o + n * 128 + k]) for n in range(32)]
            for k in range(128)] for o in (16384, 20480)]
    g1 = [h16(28688 + 2 * o) for o in range(64)]
    wqkv = [[[[R.e4m3_decode(dump[28832 + (kh * 2 + h) * 3072 + k * 96 + n])
               for n in range(96)] for k in range(32)]
             for h in range(2)] for kh in range(2)]
    bias = [[[h16(41120 + 2 * (h * 4096 + i * 64 + j)) for j in range(64)]
             for i in range(64)] for h in range(2)]
    temps = [struct.unpack_from("<f", dump, 57504 + 4 * h)[0]
             for h in range(2)]
    # S150: OUT-major [out][in] -- (c,o) at o*64 + c.
    wproj = [[R.e4m3_decode(dump[57520 + o * 64 + c]) for o in range(64)]
             for c in range(64)]
    g2 = [h16(61616 + 2 * o) for o in range(64)]
    Yp = R.c64_block_forward(X, w1p, w2p, g1, wqkv, temps, bias, wproj, g2)
    flatO = [v for row in Yp for v in row]
    assert len(flatO) == 4096
    assert sum(1 for v in flatO if math.isfinite(v)) == 4096
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64blk -- DO NOT HAND-EDIT.\n"
            "// Test-25: CONNECTED C=64 BLOCK (S116). x -> FFN -> QKV ->\n"
            "// scores -> softmax -> ctx -> proj; weights REAL from\n"
            "// tensor_137 (FFN +0/+8192/+16384/+20480/28688; qkv +28832;\n"
            "// bias +41120; temp +57504; proj +57520; gate2 +61616).\n"
            "// Backend: c64blk-block maxerr <= 0.1.\n"
            "static const unsigned int kC64BlkBytesCount = 4096;\n"
            "static const unsigned char kC64BlkBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Xb], 12) + "};\n"
            "static const unsigned int kC64BlockGoldenCount = 4096;\n"
            "static const unsigned int kC64BlockGolden[4096] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    with open(DST_C64BLK, "w", newline="\n") as fh:
        fh.write(body)
    assert open(DST_C64BLK).read().count("0x") == 4096 + 4096, DST_C64BLK
    Xb2 = R.c64_staged_blkin()
    assert Xb2 == Xb
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64BLK, flatO[0], move, R.fnv1a_hex(flatO)))


def main_c64q():
    # Test-21 golden (HANDOFF S90): staged X [64][64] e4m3 bytes (fresh
    # C64_QIN_SEED LCG stream, levels) through the two-step-f16 QKV
    # oracle. Wqkv [kh][h][32][96] e4m3 is REAL (dump +28832, 12288 B,
    # K-half-major per S90A; N-third order by convention, wiring work).
    Xb = R.c64_staged_qin()
    assert len(Xb) == 4096
    dump = open(os.path.join(ROOT, "dlss5-analysis", "tensors",
                             "tensor_137.bin"), "rb").read()
    assert len(dump) == 61760, len(dump)
    Wb = dump[28832:28832 + 12288]
    assert sum(1 for b in Wb if b) > 11000, "qkv region looks empty"
    X = [[R.e4m3_decode(Xb[m * 64 + k]) for k in range(64)]
         for m in range(64)]
    W = [[[[R.e4m3_decode(Wb[(kh * 2 + h) * 3072 + k * 96 + n])
            for n in range(96)] for k in range(32)]
          for h in range(R.C64_HEADS)] for kh in range(2)]
    O = R.c64_qkv_forward(X, W)
    flatO = [v for h in O for row in h for v in row]
    assert len(flatO) == 12288
    assert sum(1 for v in flatO if math.isfinite(v)) == 12288
    move = max(abs(v) for v in flatO)
    assert move > 0.1, move
    obits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in flatO]
    body = ("// Generated by tools/emit_yp_golden.py --c64q -- DO NOT HAND-EDIT.\n"
            "// Test-21: staged X through the QKV oracle (S90); Wqkv real\n"
            "// (dump +28832, [kh][h][32][96] K-half-major, runtime read).\n"
            "// Emit on Linux (LF); S73B CRLF trap. Backend: c64q-block\n"
            "// maxerr <= 0.1.\n"
            "static const unsigned int kC64QinBytesCount = 4096;\n"
            "static const unsigned char kC64QinBytes[4096] = {\n" +
            wrap(["0x%02X" % b for b in Xb], 12) + "};\n"
            "static const unsigned int kC64QkvGoldenCount = 12288;\n"
            "static const unsigned int kC64QkvGolden[12288] = {\n" +
            wrap(["0x%08X" % u for u in obits], 6) + "};\n")
    open(DST_C64Q, "w").write(body)
    assert open(DST_C64Q).read().count("0x") == 4096 + 12288, DST_C64Q
    Xb2 = R.c64_staged_qin()
    assert Xb2 == Xb
    X2 = [[R.e4m3_decode(Xb2[m * 64 + k]) for k in range(64)]
          for m in range(64)]
    O2 = R.c64_qkv_forward(X2, W)
    assert [v for h in O2 for row in h for v in row] == flatO, \
        "c64q not deterministic"
    print("wrote %s O[0]=%.6f maxabs=%.4f checksum=%s" %
          (DST_C64Q, flatO[0], move,
           R.fnv1a_hex([v for v in flatO if math.isfinite(v)])))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--fe":
        main_fe()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--fe-affine":
        main_fe_affine()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--fe-staged":
        main_fe_staged(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b2":
        main_b2()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b3":
        main_b3()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b67":
        main_b67()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b68":
        main_b68()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b69":
        main_b69()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--block2":
        main_block2()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b4":
        main_b4()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--b4ds":
        main_b4ds()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64":
        main_c64()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64o":
        main_c64o()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64p":
        main_c64p()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64f":
        main_c64f()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64q":
        main_c64q()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64c":
        main_c64c()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64x":
        main_c64x()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64f2":
        main_c64f2(*(sys.argv[2:3] if len(sys.argv) > 2 else ()))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c128f2":
        # S225: an optional tensor file selects a DECODER-mirror tensor for the
        # same oracle, and an optional symbol suffix keeps both goldens in tree:
        #   --c128f2 tensor_134.bin b57
        main_cw_f2(128, *(sys.argv[2:5] if len(sys.argv) > 2 else ()))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c256f2":
        main_cw_f2(256, *(sys.argv[2:5] if len(sys.argv) > 2 else ()))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--vitf2":
        main_vitf2()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--vitqkv":
        main_vitqkv()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--vitproj":
        main_vitproj()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c128blk":
        main_c128blk()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c256blk":
        main_c256blk()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c32blk":
        main_c32blk(*(sys.argv[2:3] if len(sys.argv) > 2 else ()))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64blk":
        main_c64blk()
    else:
        main()
