#!/usr/bin/env python3
"""tools/o_carve.py -- 16B carve-shift audit + real-weights re-proof (gate S).

Closes HANDOFF 33.12(5): the entry-kernel B-window is [8208,9232) but
the tables carved patch as [8192,9216) + A-residue16 @9216. Content
evidence (this gate) shows [8192,8208) is 8/8 f16 zeros (pad16,
mirroring fused pad16 @8272) and [9216,9232) is in-range f16 weights
(patch tail): patch = [8208,9232), residue label dissolved.

S1. Tables: CARVE_PRE patch == ("patch", 8208, 512, "f16"), no
    residue entry, gate_ffn @9232; fused unchanged; carve_audit().
S2. Pads: pre [8192,8208) and fused [8272,8288) all f16 zeros.
S3. Patch purity: [8208,9232) 512/512 finite, |x| < 1.0.
S4. Kernel-window pin: B-window bases 8208/8720 (lane*16) == table
    patch window exactly.
S5. Real-weights re-proof: tensor_000 full chain on the corrected
    carve (independent driver over swin1h_ref forward fns);
    finite=2048/2048, checksum pinned.
S6. Verdict CARVE-OK.

Stdlib-only, GPU-free, byte-deterministic stdout. Needs the tensor
files (in-repo) + kernel_lines; ~1 min; runbook 19.
"""

import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import swin1h_ref as S  # noqa: E402
from bias_lanemap import kernel_lines  # noqa: E402
from o_halves import lane16_base  # noqa: E402

TDIR = os.path.join(ROOT, "dlss5-analysis", "tensors")

# Pinned by the S5 re-proof run (real chain change, not drift:
# pre-audit checksum was 71408DC0 with patch @8192 + residue).
RW_CHECKSUM = "46AAC2A1"


def check_S1():
    pre = {nm: (off, n, kind) for nm, off, n, kind in S.CARVE_PRE}
    assert pre["patch"] == (8208, 32 * 16, "f16"), pre["patch"]
    assert pre["gate_ffn"] == (9232, 32, "f16"), pre["gate_ffn"]
    assert pre["qkv"] == (9312, 96 * 32, "e4m3"), pre["qkv"]
    assert not any("residu" in nm for nm in pre), sorted(pre)
    fuz = {nm: (off, n, kind) for nm, off, n, kind in S.CARVE_FUSED32}
    assert fuz["gate_ffn"] == (8208, 32, "f16"), fuz["gate_ffn"]
    assert fuz["qkv"] == (8288, 96 * 32, "e4m3"), fuz["qkv"]
    assert S.carve_audit(), "carve_audit"
    print("S1 tables: pre patch=[8208,9232) no-residue; fused "
          "unchanged; audit clean")


def raw_of(fname, size):
    p = os.path.join(TDIR, fname)
    raw = open(p, "rb").read()
    assert len(raw) == size, (fname, len(raw), size)
    return raw


def f16s(buf, lo, hi):
    return struct.unpack("<%dH" % ((hi - lo) // 2), buf[lo:hi])


def check_S2(raw0, raw1):
    for buf, lo, tag in ((raw0, 8192, "pre-pad16"),
                         (raw1, 8272, "fused-pad16")):
        u = f16s(buf, lo, lo + 16)
        assert all(x == 0 for x in u), (tag, ["%04X" % x for x in u])
    print("S2 pads: pre [8192,8208) + fused [8272,8288) all f16 zeros")


def check_S3(raw0):
    u = f16s(raw0, 8208, 9232)
    assert len(u) == 512, len(u)
    vals = [S.f16_to_f32(x) for x in u]
    import math
    assert all(math.isfinite(v) for v in vals), "nonfinite patch"
    mx = max(abs(v) for v in vals)
    assert mx < 1.0, mx
    nz = sum(1 for v in vals if v == 0.0)
    print("S3 patch: [8208,9232) 512/512 finite maxabs=%.4f zeros=%d "
          "min=%.4f max=%.4f" % (mx, nz, min(vals), max(vals)))


def check_S4(raw0):
    ls = kernel_lines()
    assert lane16_base(ls, 974) == 8208, show(ls, 974)
    assert lane16_base(ls, 983) == 8720, show(ls, 983)
    print("S4 kernel-window: B bases 8208/8720 == table patch "
          "[8208,9232) exactly")


def show(ls, ln):
    return ls[ln - 1].strip()[:60]


def check_S5(raw0):
    # Independent driver over swin1h_ref forward fns (mirrors
    # cmd_realweights tensor_000 branch): synth staging-like inputs
    # through REAL corrected-carve weights.
    W = S.carve(os.path.join(TDIR, "tensor_000.bin"), S.CARVE_PRE)
    st = 0xA11CE
    mags, st = S.gen_vec(st, S.TOK * S.D_IN, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Xe = [[S.synth_to_e4m3([levels[m]])[0] for m in row]
          for row in [mags[i * S.D_IN:(i + 1) * S.D_IN]
                      for i in range(S.TOK)]]
    s_real = struct.unpack("<f", raw0[20576:20580])[0]
    Wq = [W["qkv"][i * S.C:(i + 1) * S.C] for i in range(S.QKV)]
    Wp = [W["proj"][i * S.C:(i + 1) * S.C] for i in range(S.C)]
    W1 = [W["ffn1"][i * S.C:(i + 1) * S.C] for i in range(S.H1)]
    W2 = [W["ffn2"][i * S.H1:(i + 1) * S.H1] for i in range(S.C)]
    B = W["bias"]
    wpatch = struct.unpack("<512H", raw0[8208:9232])
    Wpt = [list(wpatch[i * S.D_IN:(i + 1) * S.D_IN])
           for i in range(S.C)]
    Ye, Yf = S.patch_forward(Xe, Wpt, S.TOK)
    import math
    pmx = max(abs(v) for v in S.flat(Yf))
    pnf = sum(1 for v in S.flat(Yf) if not math.isfinite(v))
    assert pnf == 0, pnf
    Xb = [[S.e4m3_decode(b) for b in row] for row in Ye]
    Yf2, _, _ = S.ffn_forward(Xb, W1, W2, W["gate_ffn"])
    Q, K, V, Yq = S.qkv_forward(Yf2, Wq)
    Qn, Kn, _, _, _, _ = S.qknorm_forward(Q, K, s_real)
    O, Sc, _, _ = S.attn_forward(Qn, Kn, V, bias=B)
    Yp, _ = S.proj_forward(O, Wp, Yf2, W["gate_attn"])

    def mx(M):
        return max(abs(v) for v in S.flat(M))

    fin = sum(1 for v in S.flat(Yp) if math.isfinite(v))
    assert fin == S.TOK * S.C, fin
    cks = S.fnv1a_hex([v for v in S.flat(Yp) if math.isfinite(v)])
    print("S5 re-proof: patch max|out|=%.3f finite; stages "
          "ffn=%.2f qkv=%.2f scores=%.2f attn=%.2f proj=%.2f "
          "finite=%d/%d checksum=%s" %
          (pmx, mx(Yf2), mx(Yq), mx(Sc), mx(O), mx(Yp),
           fin, S.TOK * S.C, cks))
    if RW_CHECKSUM is not None:
        assert cks == RW_CHECKSUM, (cks, RW_CHECKSUM)


def main():
    raw0 = raw_of("tensor_000.bin", 21696)
    raw1 = raw_of("tensor_001.bin", 20672)
    print("S0 sizes: tensor_000 21696 OK; tensor_001 20672 OK")
    check_S1()
    check_S2(raw0, raw1)
    check_S3(raw0)
    check_S4(raw0)
    check_S5(raw0)
    print("CARVE-OK")


if __name__ == "__main__":
    main()
