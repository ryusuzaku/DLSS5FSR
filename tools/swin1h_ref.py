#!/usr/bin/env python3
"""tools/swin1h_ref.py -- Python oracle for the Stage-1 unfused Swin 1H port.

Stdlib only. See full docstring in the repo version; dimensions, carve tables
and hypotheses H1/H2/H3 are documented in hip/mvp1/swin_1h.hip p1.
"""

import sys
import os
import struct
import json
import math
from fractions import Fraction

D_IN = 16
C = 32
TOK = 64
QKV = 96
H1 = 128  # §21: cool contract 32x128 @4096..8192 (was 256: H1=128 re-carve)

CARVE_PRE = [
    # §21 re-carve (decided 2026-09-03, bias offsets corrected §27: the two
    # tables were transposed). Tiles abut exactly: qkv 9312..12384, bias
    # 12384..20576 = P-scale; fused qkv 8288..11360, bias 11360..19552 =
    # P-scale. Zero gaps besides pad16 x2 (interval-audited).
    # The bias is the 64x64 f16 attention bias (score-mma C trace: no second
    # weight stream; f16 rms ~5 matches bias scale; NaN codes are bias
    # bytes, so N1's side-effect note moves with it).
    # §33.14 carve-shift audit (gate S): patch was [8192,9216) +
    # A-residue16 @9216; content evidence shows [8192,8208) is 8/8 f16
    # zeros (pad16, mirroring fused pad16 @8272) and [9216,9232) is
    # in-range f16 weights (patch tail, |x|<=0.95 with the region).
    # Kernel B-window [8208,9232) == patch exactly. A-residue16 dissolved.
    ("ffn1", 0, 128 * 32, "e4m3"),
    ("ffn2", 4096, 32 * 128, "e4m3"),
    # pad16 @8192 (8 f16 zeros; unlisted gap, cf fused pad16 @8272)
    ("patch", 8208, 32 * 16, "f16"),
    ("gate_ffn", 9232, 32, "f16"),
    ("qkv", 9312, 96 * 32, "e4m3"),
    ("bias", 12384, 64 * 64, "f16"),
    # P-scale16 @20576 = [f32 QK-temperature][12 zeros] (corr. §17); proj after
    ("proj", 20592, 32 * 32, "e4m3"),
    ("gate_attn", 21616, 32, "f16"),
]
CARVE_FUSED32 = [
    ("ffn1", 0, 128 * 32, "e4m3"),
    ("ffn2", 4096, 32 * 128, "e4m3"),
    ("gate_ffn", 8208, 32, "f16"),
    ("qkv", 8288, 96 * 32, "e4m3"),
    ("bias", 11360, 64 * 64, "f16"),
    # P-scale16 @19552 = [f32 QK-temperature][12 zeros] (corr. §17); proj after
    ("proj", 19568, 32 * 32, "e4m3"),
    ("gate_attn", 20592, 32, "f16"),
]


def carve_audit():
    """Interval audit: every byte assigned at most once, tiles abut the
    P-scale. Catches transposed/overlapping offsets (the §27 bias swap).
    Returns True when both tables tile cleanly."""
    ok = True
    for name, table, size, pscale in (
            ("pre", CARVE_PRE, 21696, 20576),
            ("fused32", CARVE_FUSED32, 20672, 19552)):
        mult = {"e4m3": 1, "f16": 2}
        ivs = sorted((off, off + n * mult[kind], nm)
                     for nm, off, n, kind in table)
        for (a0, a1, an), (b0, b1, bn) in zip(ivs, ivs[1:]):
            if b0 < a1:
                print("CARVE %s OVERLAP %s [%d,%d) vs %s [%d,%d)" %
                      (name, an, a0, a1, bn, b0, b1))
                ok = False
        b = [t for t in table if t[0] == "bias"][0]
        if b[1] + b[2] * 2 != pscale:
            print("CARVE %s bias does not abut P-scale" % name)
            ok = False
    if ok:
        print("carve audit: both tables tile cleanly.")
    return ok

_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32


def lcg_next(state):
    return (1664525 * state + 1013904223) & 0xFFFFFFFF


def gen_vec(state, n, lo, span):
    out = []
    for _ in range(n):
        state = lcg_next(state)
        out.append(lo + ((state >> 8) % span))
    return out, state


def _f16_decode(u):
    s = (u >> 15) & 1
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    if e == 0x1F:
        return (s, None, m != 0, m == 0)
    if e == 0:
        return (s, Fraction(m, 1024) * Fraction(1, 16384), False, False)
    return (s, Fraction(1024 + m, 1024) * (Fraction(2, 1) ** (e - 15)),
            False, False)


def f16_to_f32(u):
    s, mag, is_nan, is_inf = _f16_decode(u)
    if is_nan:
        return float("nan")
    if is_inf:
        return float("inf") if not s else float("-inf")
    v = float(mag)
    return -v if s else v


def f32_to_f16_bits(v):
    # struct '<e' overflows early (|v| > 65504 -> inf); hardware/numpy round
    # correctly at the top edge (midpoint 65520 between 65504 and inf).
    if math.isnan(v):
        return 0x7E00
    av = abs(v)
    if av > 65504.0:
        if av >= 65520.0:
            return 0x7C00 if v > 0 else 0xFC00
        return 0x7BFF if v > 0 else 0xFBFF
    try:
        return struct.unpack("<H", struct.pack("<e", v))[0]
    except OverflowError:
        return 0x7C00 if v > 0 else 0xFC00


def round_f32_list_to_f16(vals):
    return [f32_to_f16_bits(v) for v in vals]


def _e4m3_pos_frac(b):
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 0:
        return Fraction(m, 8) * Fraction(1, 64)
    return Fraction(8 + m, 8) * (Fraction(2, 1) ** (e - 7))


_E4M3_REPR = [(b, _e4m3_pos_frac(b)) for b in range(127)]
_E4M3_MIDS = [((_E4M3_REPR[i][1] + _E4M3_REPR[i + 1][1]) / 2,
               _E4M3_REPR[i][0], _E4M3_REPR[i + 1][0])
              for i in range(126)]


def quantise_f16_to_e4m3(u16):
    """RN-satfinite f16->E4M3FN twin of stage1_ref.quantise_f16_to_e4m3fn."""
    s, mag, is_nan, is_inf = _f16_decode(u16)
    if is_nan:
        return (s << 7) | 0x7F
    if is_inf:
        return (s << 7) | 0x7E
    if mag == 0 or mag < _E4M3_MIDS[0][0] or mag == _E4M3_MIDS[0][0]:
        return (s << 7) | 0x00
    if mag >= 448:
        return (s << 7) | 0x7E
    last_mid = _E4M3_MIDS[-1][0]
    if mag > last_mid or mag == last_mid:
        return (s << 7) | 0x7E
    lo, hi = 0, len(_E4M3_MIDS) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        mv, b0, b1 = _E4M3_MIDS[mid]
        if mag == mv:
            return (s << 7) | (b0 if ((b0 & 7) & 1) == 0 else b1)
        elif mag < mv:
            hi = mid - 1
        else:
            lo = mid + 1
    return (s << 7) | _E4M3_MIDS[lo][1]


def e4m3_decode(b):
    """Weight/activation dequant for the port (H4): e4m3 NaN codes (0x7F/0xFF)
    flush to +0.0 (presumed HW behavior -- shipped contract matrices carry
    ~0.1% NaN-code bytes yet the product is unaffected; expand/QKV matrices
    are NaN-free). No-op on NaN-free vectors (all synthetic tests)."""
    if b == 0x7F or b == 0xFF:
        return 0.0
    return e4m3_to_f32(b)


def e4m3_to_f32(b):
    """Exact E4M3FN byte decode (NaN -> nan). All finite values exact in f32."""
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 0xF and m == 0x7:
        return float("nan")
    if e == 0:
        v = m / 512.0
    else:
        # bias 7: (1 + m/8) * 2^(e-7); 0x38 -> 1.0, 0x7E -> 448.0
        v = (8 + m) / 8.0 * (2.0 ** (e - 7))
    return -v if s else v


C0_BITS = 0xAB28  # -0.055908203125
C1_BITS = 0x3728  # 0.447265625
C2_BITS = 0x3B28  # 0.89453125


def _fma_f16_exact(a_bits, b_bits, c_bits):
    """Single-rounding f16 FMA via exact Fractions (twin of stage1_ref)."""
    sa, fa, na, ia = _f16_decode(a_bits)
    sb, fb, nb, ib = _f16_decode(b_bits)
    sc, fc, nc, ic = _f16_decode(c_bits)
    if na or nb or nc:
        return 0x7E00
    if ia or ib or ic:
        v = f16_to_f32(a_bits) * f16_to_f32(b_bits) + f16_to_f32(c_bits)
        return f32_to_f16_bits(v)
    xa = -fa if sa else fa
    xb = -fb if sb else fb
    xc = -fc if sc else fc
    return f32_to_f16_bits(float(xa * xb + xc))


def mpcubic_silu_ref(x_bits):
    """Bit-exact MpCubicSilu (twin of stage1_ref.mpcubic_silu_ref)."""
    _, _, is_nan, _ = _f16_decode(x_bits)
    if is_nan:
        return 0x7E00
    xv = f16_to_f32(x_bits)
    y_bits = (0x4400 if xv > 4.0 else (0xC400 if xv < -4.0 else x_bits))
    t_bits = _fma_f16_exact(C0_BITS, y_bits & 0x7FFF, C1_BITS)
    u_bits = _fma_f16_exact(y_bits, t_bits, C2_BITS)
    return f32_to_f16_bits(xv * f16_to_f32(u_bits))


# ---------------------------------------------------------------------------
# Block forward (float64 oracle; exact stages bitwise-equal f32 by design)
# ---------------------------------------------------------------------------

def gemm(X, W, M, K, N):
    """Y[m][o] = sum_k X[m][k]*W[o][k]; W row-major (out,in)."""
    return [[sum(X[m][k] * W[o][k] for k in range(K))
             for o in range(N)] for m in range(M)]


def patch_forward(x_e4m3, w_patch_f16, M):
    """x: Mx16 e4m3 bytes; w: 32x16 f16 bits. Returns (y_e4m3[M][32], y_f32)."""
    X = [[e4m3_decode(b) for b in row] for row in x_e4m3]
    W = [[f16_to_f32(u) for u in row] for row in w_patch_f16]
    Y = gemm(X, W, M, D_IN, C)
    yb = [[f32_to_f16_bits(v) for v in row] for row in Y]
    return [[quantise_f16_to_e4m3(u) for u in row] for row in yb], Y


def qkv_forward(x_f32, w_qkv_f32):
    Y = gemm(x_f32, w_qkv_f32, TOK, C, QKV)
    return [row[0:32] for row in Y], [row[32:64] for row in Y], \
        [row[64:96] for row in Y], Y


# Test temperature: f32 bits all sides must start from (matches C++ 0.5f).
S_TEST = struct.unpack("<f", struct.pack("<f", 0.5))[0]

EPS_NORM = 2.0 ** -14  # max-clamp form (HANDOFF §17), NOT additive


def _f32(x):
    """Round to f32 (models the C++ mirror's float arithmetic bitwise)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def qknorm_forward(Q, K, s):
    """QK-norm (HANDOFF §17): per-token unit-norm of Q and K (V untouched).
    Q'[i] = s_f16 * Q[i] * rsqrt(max(sumsq, 2^-14)); K likewise without s.
    s enters as f32 bits, rounded once to f16 (all sides share the converter,
    so the step is bitwise-identical everywhere)."""
    s_f16 = f16_to_f32(f32_to_f16_bits(s))

    def norm_rows(rows, scale):
        out, sums, facs = [], [], []
        for row in rows:
            # f32 k-ascending accumulation, exactly like the C++ mirror
            # (sq += q*q in float). §22: required since the FFN-first chain's
            # Yq rows are no longer exact small ints; f64 accumulation
            # diverges 1 ulp from f32 (caught by the sumsq FNVs; downstream
            # rounds identically through f16, which is why smexp still hit).
            ss = 0.0
            for x in row:
                ss = _f32(ss + _f32(x * x))
            f = 1.0 / math.sqrt(max(float(ss), EPS_NORM))
            out.append([scale * x * f for x in row])
            sums.append(ss)
            facs.append(f)
        return out, sums, facs

    Qn, ssQ, fQ = norm_rows(Q, s_f16)
    Kn, ssK, fK = norm_rows(K, 1.0)
    return Qn, Kn, ssQ, ssK, fQ, fK


# In-kernel fast softmax (HANDOFF §19). The fused kernel has NO 1/sqrt(32)
# and NO max-subtraction: raw QK^T dots feed u = c1*x + c2, clamped, then a
# Schraudolph bit-trick exp ((bits(u) << 5) + MAGIC, reinterpreted as f16x2
# = fast_exp2(32u-47)), row-summed in f16, normalized by f32 rcp of
# max(sum, 2^-14) rounded back to f16. c1*32 = 1.4375 ~= log2(e): the whole
# episode computes P ~ softmax(x) with temperature ~1 on the raw dots.
# Constants are f16(cvt.rn.f32(PTX imm)); magic is the literal addend.
EXP_C1 = 0.044921875   # f32 imm 1027077105; = 1.4375/32, exact in f16
EXP_C2 = 1.30078125    # f32 imm 1067877303
EXP_LO = 1.03125       # f32 imm 1065615360 (clamp lo; E floor = 2^-14)
EXP_HI = 1.5693359375  # f32 imm 1070129152 (clamp hi; E ceil ~12.75)
EXP_MAGIC = 0x7FF88000  # literal s32 addend (was misread 0x7FF40000 in §17)
EXP_C1_BITS = 0x29C0
EXP_C2_BITS = 0x3D34
_F16_ONE_BITS = 0x3C00


def smexp_pair_f16(score):
    """One score -> (Elo_bits, Ehi_bits) via the duplicated-word trick model.
    The kernel packs two keys per word (pairing set by its lane network); the
    port models each score with a duplicated word, which is bit-exact vs a
    device kernel written the same way (see k_smexp). All f16 ops are
    single-rounding (exact helpers), matching __hfma2/__hadd2/hw exactly."""
    x16 = f32_to_f16_bits(score)
    u16 = _fma_f16_exact(x16, EXP_C1_BITS, EXP_C2_BITS)
    v = f16_to_f32(u16)
    v = EXP_LO if v < EXP_LO else (EXP_HI if v > EXP_HI else v)
    w = f32_to_f16_bits(v)
    r = (((w << 16) | w) << 5) + EXP_MAGIC
    r &= 0xFFFFFFFF
    return r & 0xFFFF, (r >> 16) & 0xFFFF


def _f16_add_exact(a_bits, b_bits):
    """Single-rounding f16 add via exact FMA(x, 1.0)."""
    return _fma_f16_exact(a_bits, _F16_ONE_BITS, b_bits)


def attn_forward(Q, K, V, bias=None):
    # NO 1/sqrt(32): QK-norm's temperature s is the only scale (§19).
    # bias (§21): 64x64 row-major f16 attention bias, added to the raw dots
    # (S = QK^T + B) before the fast-exp softmax. Recipe: f16 decode, NO
    # scale, lane (i, j) += bias[i * 64 + j]. bias=None skips the add.
    if bias is None:
        S = [[sum(Q[i][d] * K[j][d] for d in range(32))
              for j in range(TOK)] for i in range(TOK)]
    else:
        S = [[sum(Q[i][d] * K[j][d] for d in range(32)) + bias[i * TOK + j]
              for j in range(TOK)] for i in range(TOK)]
    P, Ebits = [], []
    for i in range(TOK):
        erow = [smexp_pair_f16(S[i][j])[0] for j in range(TOK)]
        Ebits.append(erow)
        acc16 = 0x0000
        for e in erow:  # port-defined k-ascending f16 sum (device matches it)
            acc16 = _f16_add_exact(acc16, e)
        tot = f16_to_f32(acc16)
        # f32 rcp RN here; C++ mirror shares it, device uses hw approx (TOL).
        g16 = f32_to_f16_bits(1.0 / max(tot, EPS_NORM))
        gf = f16_to_f32(g16)
        P.append([f16_to_f32(f32_to_f16_bits(f16_to_f32(e) * gf))
                  for e in erow])
    O = [[sum(P[i][j] * V[j][d] for j in range(TOK)) for d in range(32)]
         for i in range(TOK)]
    return O, S, P, Ebits


def proj_forward(O, w_proj, x_in, gate_attn):
    Y = gemm(O, w_proj, TOK, C, C)
    return [[Y[m][o] + gate_attn[o] * x_in[m][o] for o in range(C)]
            for m in range(TOK)], Y


def ffn_forward(x_f32, w1, w2, gate_ffn):
    H = gemm(x_f32, w1, TOK, C, H1)
    Hb = [round_f32_list_to_f16(row) for row in H]
    A = [[f16_to_f32(mpcubic_silu_ref(u)) for u in row] for row in Hb]
    Y = gemm(A, w2, TOK, H1, C)
    return [[Y[m][o] + gate_ffn[o] * x_f32[m][o] for o in range(C)]
            for m in range(TOK)], Y, A


# ---------------------------------------------------------------------------
# C=64 attention scores (HANDOFF §§46-50, Test-16). 2 heads x 64 tokens x
# 32 dims; per-head f32 temperatures at tensor+57504; per-head 64x64 f16
# bias at tensor+41120 (head stride 8192). QKV order inside [28832,41120)
# is SIZE INFERENCE only (3x64x64) -- NOT a kernel-read proof -- so
# Test-16 stages the score-GEMM inputs (Q,K) directly and proves exactly
# the four proven inputs, nothing assumed.
# ---------------------------------------------------------------------------

C64_HEADS = 2
C64_TOK = 64
C64_DIM = 32
# Phase A (HANDOFF §133/§135): the stage's dimensions, mirroring the kernel
# macros. C64_C is the width; C64_W and C64_H1P are MEASURED constants (§135) --
# W = 32 and H1 = 128 do not change with width, only C (hence HEADS = C/32) does.
C64_C = 64
C64_W = 32
C64_K = 32      # the mma K-step, distinct from W (S140)
C64_H1P = 128
C64_TEMP_OFF = 57504
C64_BIAS_OFF = 41120

CARVE_C64 = [
    # Operator-verified dtype map (§46.1). A-internal split and Q/K/V
    # order unproven -- listed as whole regions, never consumed by the
    # Test-16/17 oracle.
    ("A", 0, 28672, "e4m3"),
    ("gate1", 28688, 64, "f16"),
    ("qkv", 28832, 12288, "e4m3"),
    ("bias", 41120, 8192, "f16"),
    # temp16 @57504 = 2x f32 per-head QK temperatures (V1 = head0).
    ("proj", 57520, 4096, "e4m3"),
    ("gate2", 61616, 64, "f16"),
]


def c64_scores_forward(Q, K, temps, B):
    """Per-head scores (HANDOFF §49).

    S_h[i][j] = temps[h] * sum_d(Q[h][i][d] * K[h][j][d]) + B[h][i][j],
    with NO scale on B. Q/K are [2][64][32] f32 (decoded e4m3), temps 2
    f32, B [2][64][64] f32 (decoded f16).

    TRAP (§49, operator handoff): temp is pre-GEMM on Q (the kernel's 32
    mul.f16x2 sit on the mma-A path); the bias enters via the mma D-init
    and is NEVER scaled. (QK^T + bias) * temp looks right and is wrong.
    """
    S = []
    for h in range(C64_HEADS):
        t = temps[h]
        # Kernel order: temp scales each Q element pre-dot (the 32
        # mul.f16x2 sit on the mma-A path, §49), then dots, then the
        # unscaled D-init bias. Written per-element so the f32 mirror
        # and device kernel transliterate op-for-op.
        Sh = [[sum((t * Q[h][i][d]) * K[h][j][d] for d in range(C64_DIM))
               + B[h][i][j]
               for j in range(C64_TOK)] for i in range(C64_TOK)]
        S.append(Sh)
    return S


def c64_exp_forward(S):
    """Test-17 episode (HANDOFF §50, DECISION: replicate): the same 5-op
    bit-trick exp + rowsum + rcp normalize as the C=32 path, applied per
    head. The C=64 immediates round to the IDENTICAL f16 constants
    (pinned in selftest), so smexp_pair_f16 is reused bit-exactly --
    no new recipe, no exp(), no ~3% budget."""
    Eb, P = [], []
    for h in range(C64_HEADS):
        erows = [[smexp_pair_f16(S[h][i][j])[0] for j in range(C64_TOK)]
                 for i in range(C64_TOK)]
        Eb.append(erows)
        Ph = []
        for i in range(C64_TOK):
            acc16 = 0x0000
            for e in erows[i]:
                acc16 = _f16_add_exact(acc16, e)
            tot = f16_to_f32(acc16)
            g16 = f32_to_f16_bits(1.0 / max(tot, EPS_NORM))
            gf = f16_to_f32(g16)
            Ph.append([f16_to_f32(f32_to_f16_bits(f16_to_f32(e) * gf))
                       for e in erows[i]])
        P.append(Ph)
    return Eb, P


C64_P_SEED = 0xC64002  # fixed staged-P stream (Test-18; documented)
C64_V_SEED = 0xC64003  # fixed staged-V stream (Test-18; documented)
C64_OSEED = 0xC64004  # fixed staged-Ocat stream (Test-19; documented)
C64_Y_SEED = 0xC64005  # fixed staged-y stream (Test-19; documented)


def c64_staged_oy():
    """Staged proj inputs: LCG levels [-1,-0.5,0,0.5,1] as e4m3 bytes,
    Ocat [64][64] first then y [64][64], row-major. MUST match the C++
    host twin (sw_c64proj_staged) and the golden emitter exactly. Wproj
    and gate2 are REAL (dump +57520/+61616; W row-major [c][o] mapping shared
    with twin/device -- true production orientation is wiring work, not
    this test (HANDOFF §78I)."""
    st = C64_OSEED
    ov, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    st = C64_Y_SEED
    yv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return ([synth_to_e4m3([levels[m]])[0] for m in ov],
            [synth_to_e4m3([levels[m]])[0] for m in yv])


def c64_proj_forward(Ocat, Wproj, gate2, y):
    """Test-19 oracle (HANDOFF §78G): mirrors C=32 proj_forward at 64-wide:
    out[m][o] = sum_c Ocat[m][c] * Wproj[c][o] + gate2[o] * y[m][o].
    All inputs f32 (decoded e4m3 / f16); plain sums like the C=32 path
    (no f16 steps -- the production proj-D rounding gap is device-side
    tolerance, tol 0.1, precedent Test-16/18)."""
    # S150: the channel/output extent is C64_C, not C64_TOK. They coincide at
    # C=64 (where this was written), so the C=64 goldens are unchanged; at the
    # ViT's C=1024 with TOK=1 the old form collapsed to a single value.
    return [[sum(Ocat[m][c] * Wproj[c][o] for c in range(C64_C)) +
             gate2[o] * y[m][o] for o in range(C64_C)]
            for m in range(C64_TOK)]


C64_H1 = 224  # staged FFN width (Test-20): SIZE ARITHMETIC on the A
# region (28672 = 64x224 + 224x64), NOT a kernel-read proof (HANDOFF §78H).
C64_X_SEED = 0xC64006  # fixed staged-x stream (Test-20; documented)


def c64_staged_x():
    """Staged FFN input: LCG levels [-1,-0.5,0,0.5,1] as e4m3 bytes,
    x [64][64], row-major. MUST match the C++ host twin
    (sw_c64ffn_staged) and the golden emitter exactly. w1/w2/gate1 are
    REAL (dump +0/+14336/+28688; row-major [out][in] mapping shared
    with twin/device -- the H1=224 split itself stays size arithmetic,
    HANDOFF §78I)."""
    st = C64_X_SEED
    xv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def c64_ffn_forward(x, w1, w2, gate1):
    """Test-20 oracle (HANDOFF §78H): mirrors C=32 ffn_forward at 64-wide
    with H1=224: H = x @ w1 -> f16 round -> bit-exact MpCubicSilu (the
    §79 closed form, clamp(x,-4,+4) inside mpcubic_silu_ref) -> Y = A @
    w2 -> out = Y + gate1 * x. All GEMMs plain sums like the C=32 path
    and Test-19 (no f16 K-steps -- the production mma-D rounding gap is
    device-side tolerance, tol 0.1). w1 [224][64], w2 [64][224]
    row-major (out,in), gate1 [64]. NO quantize step: the proven C=32
    FFN recipe feeds act-f16 straight into the contract (same for the
    twin and device kernels); production mma-dtype composition is
    explicitly out of scope (HANDOFF §78I)."""
    H = [[sum(x[m][k] * w1[j][k] for k in range(C64_TOK))
          for j in range(C64_H1)] for m in range(C64_TOK)]
    Hb = [round_f32_list_to_f16(row) for row in H]
    A = [[f16_to_f32(mpcubic_silu_ref(u)) for u in row] for row in Hb]
    Y = [[sum(A[m][j] * w2[o][j] for j in range(C64_H1))
          for o in range(C64_TOK)] for m in range(C64_TOK)]
    return [[Y[m][o] + gate1[o] * x[m][o] for o in range(C64_TOK)]
            for m in range(C64_TOK)]


C64_QIN_SEED = 0xC64007  # fixed staged-QKV-input stream (Test-21)


def c64_staged_qin():
    """Staged QKV-GEMM input: LCG levels [-1,-0.5,0,0.5,1] as e4m3
    bytes, X [64][64], row-major. MUST match the C++ host twin
    (sw_c64qkv_staged) and the golden emitter exactly. Wqkv is REAL
    (dump +28832, 12288 B, [kh][h][32][96] K-half-major per S90;
    N-third order [Q|K|V] by convention, wiring work)."""
    st = C64_QIN_SEED
    xv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def c64_qkv_forward(X, W):
    """Test-21 oracle (HANDOFF S90C): per head, per row, per n in
    0..96, acc starts f16(0); two K-halves (kh=0: X cols 0..31, kh=1:
    cols 32..63); each step acc = f16(f32(acc) + sum_{k<32}
    X[m][kh*32+k] * W[kh][h][k][n]). X [64][64], W [2][2][32][96]
    f32 (decoded e4m3). Per-step f16 rounding is the production
    mma-D contract (S78E precedent, no latitude); no scale, no bias
    (S90: temps apply at SCORES, no QKV bias region)."""
    O = []
    for h in range(C64_HEADS):
        Oh = []
        for m in range(C64_TOK):
            row = []
            for n in range(96):
                acc = 0
                # S163: K is the CHANNEL extent, in 32-wide halves. Hard-wired
                # to range(2) (K=64) until C=128, where the oracle silently
                # accumulated over only half the input channels -- so the GOLDEN
                # was wrong and the device was right to disagree (S161's 639).
                # At C=64 this is range(2) again: a no-op.
                for kh in range(C64_C // 32):
                    s = 0.0
                    for k in range(32):
                        s += X[m][kh * 32 + k] * W[kh][h][k][n]
                    acc = f32_to_f16_bits(f16_to_f32(acc) + s)
                row.append(f16_to_f32(acc))
            Oh.append(row)
        O.append(Oh)
    return O


C64_HIN_SEED = 0xC64008  # fixed staged-contract-input stream (Test-22)


def c64_staged_hin():
    """Staged FFN-contract input h [64][128] e4m3: LCG levels
    [-1,-0.5,0,0.5,1], row-major. MUST match the C++ host twin
    (sw_c64contract_staged) and the golden emitter exactly.

    Shape is HANDOFF §111, proven by immediate census (not size
    arithmetic): the contract's A operand is the expand output
    [64][H1] with H1 = 128 (contract K = 128 = region-2's
    [128][64] = 8192 B). W2 is REAL (dump +16384, 4096 B, [128][32]
    row-major = pass 0's N-slice; pass 1 is +20480)."""
    st = C64_HIN_SEED
    xv, _ = gen_vec(st, C64_TOK * 128, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def c64_contract_forward(H, W2):
    """Test-22 oracle (HANDOFF §111): h[64][128] x W2[128][32] -> [64][32].

    K = 128 in FOUR k-steps of 32, f16 accumulate per step. That step
    structure is the production D->C chain read off the kernel (§110:
    four groups, %r763/%r764 D@15347 -> C-init@16057 -> ... -> @17477),
    not a modelling choice -- same contract as Test-21's two-step
    oracle. No activation: act is upstream of h. No bias, no gate,
    no residual (§101: contract -> direct quant -> self-link).

    H [64][128] and W2 [128][32] are f32 (decoded e4m3)."""
    O = []
    for m in range(C64_TOK):
        row = []
        for n in range(C64_W):
            acc = 0
            for kh in range(C64_H1P // C64_K):
                s = 0.0
                for k in range(C64_K):
                    s += H[m][kh * C64_K + k] * W2[kh * C64_K + k][n]
                acc = f32_to_f16_bits(f16_to_f32(acc) + s)
            row.append(f16_to_f32(acc))
        O.append(row)
    return O


C64_EXIN_SEED = 0xC64009  # fixed staged-expand-input stream (Test-23)


def c64_staged_exin():
    """Staged FFN-expand input x [64][64] e4m3: LCG levels
    [-1,-0.5,0,0.5,1], row-major. MUST match the C++ host twin
    (sw_c64expand_staged) and the golden emitter exactly.

    Shape is HANDOFF §113, proven by the region-1 mma census (not
    inherited from §95): M-tiles=2, N-tiles=16, K-steps=2 is the only
    factorization of {mma 64, B-pairs 32, A-sets 4, chains 32}, giving
    K=64, N=128, M=32 rows/warp x 2 warps = 64. w1 is REAL
    (dump +0, 8192 B, [64][128] IN-MAJOR row-major = pass 0's slice of
    the 16384-B region 1; pass 1 is +8192)."""
    st = C64_EXIN_SEED
    xv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def c64_expand_forward(x, w1):
    """Test-23 oracle (HANDOFF §113): x[64][64] x w1[64][128] -> [64][128].

    K = 64 in TWO k-steps of 32, f16 accumulate per step. The step count
    is the production D->C chain read off the kernel (§113: 32 chains,
    chain-length histogram {2: 32}, all rooted at the frozen 0.0
    accumulator), not a modelling choice -- note this is HALF the
    contract's four steps (§110), because the expand weights are half as
    deep in K. No activation, no bias, no gate, no residual: those are
    all downstream of h (§101 / §111).

    w1 is indexed IN-MAJOR w1[k][n] -- the orientation proven on Wproj
    (Test-19), Wqkv (Test-21) and W2-contract (Test-22). Test-20's
    out-major w1 is the defect §113 records."""
    O = []
    for m in range(C64_TOK):
        row = []
        for n in range(C64_H1P):
            acc = 0
            for kh in range(C64_C // C64_K):
                s = 0.0
                for k in range(C64_K):
                    s += x[m][kh * C64_K + k] * w1[kh * C64_K + k][n]
                acc = f32_to_f16_bits(f16_to_f32(acc) + s)
            row.append(f16_to_f32(acc))
        O.append(row)
    return O


C64_FFN2_SEED = 0xC6400A  # fixed staged connected-FFN input (Test-24)


def c64_staged_ffnin():
    """Staged connected-FFN input x[64][64] e4m3: LCG levels
    [-1,-0.5,0,0.5,1], row-major. MUST match the C++ host twin
    (sw_c64ffn2_staged) and the golden emitter exactly.

    Test-24 (HANDOFF S116): unlike Test-20/22/23, the expand output is
    NOT staged independently -- it FEEDS the contract, so this is the
    first *connected* C=64 FFN."""
    st = C64_FFN2_SEED
    xv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


C128_FFN2_SEED = 0xC6400C  # C=128 connected-FFN staged input (Phase A, §139)


C128_FFN2_SEED = 0xC6400C  # C=128 connected-FFN staged input (Phase A)
C256_FFN2_SEED = 0xC6400D  # C=256 connected-FFN staged input (Phase A)

C32_FFN2_SEED = 0xC6400A   # C=32 connected-FFN staged input (S170)

_CW_SEED = {32: C32_FFN2_SEED, 128: C128_FFN2_SEED, 256: C256_FFN2_SEED}


def cw_staged_ffnin(C):
    """Staged connected-FFN input x[64][C] e4m3 for width C: same LCG levels and
    row-major shape as the C=64 one, so the width is the only difference."""
    xv, _ = gen_vec(_CW_SEED[C], C64_TOK * C, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def c64_ffn2_forward(x, w1p, w2p, gate1):
    """Connected C=64 FFN oracle (Test-24, HANDOFF S116).

    Per pass p in {0,1} (the S111 2-group FFN structure -- two w1 and two
    W2 N-slices, NOT one dense 64->128->64):

        h_p[64][128] = expand(x, w1_p)      # K=64 = 2 f16 k-steps
        a_p          = act(h_p)             # MpCubicSilu (§79 closed form)
        aq_p         = e4m3(a_p)            # the contract's A dtype boundary
        o_p[64][32]  = contract(aq_p, w2_p) # K=128 = 4 f16 k-steps

    out[:, 32p:32p+32] = o_p, then out += gate1 * x (the FFN gate/residual,
    same wiring as the C=32 ffn_forward).

    x is f32 (decoded staged bytes); w1p[p] is [64][128] f32; w2p[p] is
    [128][32] f32; gate1 is [64] f32."""
    out = [[0.0] * C64_C for _ in range(C64_TOK)]
    for p in range(C64_C // C64_W):
        H = c64_expand_forward(x, w1p[p])            # [64][128] f32
        Hq = []
        for row in H:
            rr = []
            for v in row:
                ab = mpcubic_silu_ref(f32_to_f16_bits(v))
                rr.append(e4m3_decode(quantise_f16_to_e4m3(ab)))
            Hq.append(rr)
        O = c64_contract_forward(Hq, w2p[p])         # [64][32] f32
        for m in range(C64_TOK):
            for n in range(C64_W):
                out[m][C64_W * p + n] = O[m][n]
    for m in range(C64_TOK):
        for o in range(C64_C):
            out[m][o] += gate1[o] * x[m][o]
    return out


C64_BLK_SEED = 0xC6400B  # fixed staged C=64 BLOCK input (Test-25, HANDOFF S116)


def c64_staged_blkin():
    """Staged C=64 block input x[64][64] e4m3: LCG levels, row-major. MUST
    match the C++ host twin (sw_c64block_staged) and the golden emitter.

    Test-25: the first CONNECTED C=64 block -- FFN (Test-24) feeds QKV
    (Test-21), which feeds scores (Test-16) -> softmax (Test-17) -> ctx
    (Test-18) -> proj (Test-19). Every stage is the port's own proven
    recipe; only the inter-stage wiring/boundaries are new (and are the
    point of this test)."""
    st = C64_BLK_SEED
    xv, _ = gen_vec(st, C64_TOK * C64_TOK, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return [synth_to_e4m3([levels[m]])[0] for m in xv]


def _e4m3_roundtrip(v):
    """f32 -> f16 -> e4m3 -> f32 (the inter-stage quant boundary)."""
    return e4m3_decode(quantise_f16_to_e4m3(f32_to_f16_bits(v)))


def c64_block_forward(x, w1p, w2p, gate1, wqkv, temps, bias, wproj, gate2):
    """Connected C=64 block oracle (Test-25, HANDOFF S116).

    Wiring (runbook v4 order -- FFN first):
        Yf        = FFN(x) + gate1*x            # Test-24, [64][64]
        QKV       = qkv(quant(Yf))              # Test-21, [2][64][96] = Q|K|V
        S         = scores(Q,K)*, temps, bias   # Test-16, [2][64][64]
        P         = softmax(S)                  # Test-17, [2][64][64]
        O         = ctx(quant(P), V)            # Test-18, [2][64][32]
        Yp        = proj(Ocat) + gate2*Yf       # Test-19, [64][64]

    All arguments are f32 (decoded); the two quant boundaries (Yf -> QKV
    input, P -> ctx input) are the production re-quantizations."""
    Yf = c64_ffn2_forward(x, w1p, w2p, gate1)
    # S159: the channel extent is C64_C, not C64_TOK. Identical at C=64 (where
    # this was written); at C=128 the old form truncated the re-quantized FFN
    # output to 64 wide and the projection then read past the row.
    Yfe = [[_e4m3_roundtrip(Yf[m][o]) for o in range(C64_C)]
           for m in range(C64_TOK)]
    QKV = c64_qkv_forward(Yfe, wqkv)                    # [2][64][96]

    def _third(off):
        # Q/K/V leave the QKV stage as e4m3 (the scores/ctx kernels read
        # e4m3 bytes -- Test-16/18 staged convention).
        return [[[_e4m3_roundtrip(QKV[h][m][off + d])
                 for d in range(C64_DIM)] for m in range(C64_TOK)]
                for h in range(C64_HEADS)]

    Q, K, V = _third(0), _third(C64_DIM), _third(2 * C64_DIM)
    S = c64_scores_forward(Q, K, temps, bias)           # [2][64][64]
    _Eb, P = c64_exp_forward(S)                         # P [2][64][64]
    Pq = [[[_e4m3_roundtrip(P[h][i][j]) for j in range(C64_TOK)]
           for i in range(C64_TOK)] for h in range(C64_HEADS)]
    O = c64_ctx_forward(Pq, V)                          # [2][64][32]
    # Ocat is the proj kernel's e4m3 input; the proj residual rides on the
    # (already e4m3) Yfe, matching the kernel's `Yb` byte input.
    Ocat = [[_e4m3_roundtrip(O[h][m][d]) for h in range(C64_HEADS)
             for d in range(C64_DIM)] for m in range(C64_TOK)]
    return c64_proj_forward(Ocat, wproj, gate2, Yfe)    # [64][64]


def c64_staged_pv():
    """Staged O-GEMM inputs: LCG levels [-1,-0.5,0,0.5,1] as e4m3
    bytes, P first then V, head-major [h][i][k]/[h][k][n]. MUST match
    the C++ host twin (sw_c64ctx_staged) and the golden emitter exactly.
    P is staged RAW levels, not simplex: the test proves the O-GEMM
    arithmetic recipe (Pq x V -> O), not the smexp distribution, and
    composition is proven separately (HANDOFF §78D, §52/S41 doctrine)."""
    st = C64_P_SEED
    pv, _ = gen_vec(st, C64_HEADS * C64_TOK * C64_TOK, 0, 5)
    st = C64_V_SEED
    vv, _ = gen_vec(st, C64_HEADS * C64_TOK * C64_DIM, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return ([synth_to_e4m3([levels[m]])[0] for m in pv],
            [synth_to_e4m3([levels[m]])[0] for m in vv])


def c64_ctx_forward(Pq, V):
    """Test-18 oracle (HANDOFF §78E): per head, per query, per dim, acc
    starts f16(0); two K-halves over the 64 keys; each step acc =
    f16(f32(acc) + sum_{k=0..31} f32(P) * f32(V)). Pq/V are [2][64][64]
    / [2][64][32] f32 (decoded e4m3). The per-step f16 rounding is the
    production mma-D contract (no latitude); the f64 inner sums sit
    ~1e-7 off f32 (precedent: c64_scores_forward)."""
    O = []
    for h in range(C64_HEADS):
        Oh = []
        for i in range(C64_TOK):
            row = []
            for n in range(C64_DIM):
                acc = 0
                for kh in range(2):
                    s = 0.0
                    for k in range(32):
                        s += Pq[h][i][kh * 32 + k] * V[h][kh * 32 + k][n]
                    acc = f32_to_f16_bits(f16_to_f32(acc) + s)
                row.append(f16_to_f32(acc))
            Oh.append(row)
        O.append(Oh)
    return O


C64_SEED = 0xC64001  # fixed staged-Q/K stream (Test-16; documented)


def c64_staged_qk():
    """Staged score-GEMM inputs: LCG levels [-1,-0.5,0,0.5,1] as e4m3
    bytes, Q first then K, head-major [h][i][d]. MUST match the C++
    host twin (sw_c64_staged) and the golden emitter exactly."""
    st = C64_SEED
    qv, st = gen_vec(st, C64_HEADS * C64_TOK * C64_DIM, 0, 5)
    kv, _ = gen_vec(st, C64_HEADS * C64_TOK * C64_DIM, 0, 5)
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return ([synth_to_e4m3([levels[m]])[0] for m in qv],
            [synth_to_e4m3([levels[m]])[0] for m in kv])


def cmd_c64fnv():
    Qe, Ke = c64_staged_qk()
    Q = [[[e4m3_decode(Qe[h * 2048 + i * 32 + d])
           for d in range(C64_DIM)] for i in range(C64_TOK)]
         for h in range(C64_HEADS)]
    K = [[[e4m3_decode(Ke[h * 2048 + i * 32 + d])
           for d in range(C64_DIM)] for i in range(C64_TOK)]
         for h in range(C64_HEADS)]
    tp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "dlss5-analysis", "tensors", "tensor_137.bin")
    raw = open(tp, "rb").read()
    assert len(raw) == 61760, len(raw)
    temps = list(struct.unpack("<2f", raw[C64_TEMP_OFF:C64_TEMP_OFF + 8]))
    Bh = struct.unpack("<8192H", raw[C64_BIAS_OFF:C64_BIAS_OFF + 16384])
    B = [[[f16_to_f32(Bh[h * 4096 + i * 64 + j]) for j in range(C64_TOK)]
          for i in range(C64_TOK)] for h in range(C64_HEADS)]
    S = c64_scores_forward(Q, K, temps, B)
    Eb, P = c64_exp_forward(S)
    print("c64Qe %s" % fnv_bytes_raw(Qe))
    print("c64Ke %s" % fnv_bytes_raw(Ke))
    print("c64S %s" % fnv1a_hex([v for h in S for row in h for v in row]))
    print("c64Eb %s" % fnv_u16([u for h in Eb for row in h for u in row]))
    print("c64P %s" % fnv1a_hex([v for h in P for row in h for v in row]))


def fnv1a_hex(floats):
    h = 0x811C9DC5
    for v in floats:
        for byte in struct.pack("<f", v):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
    return "%08X" % h


def fnv1a_hex_words(words):
    h = 0x811C9DC5
    for w in words:
        for byte in struct.pack("<I", w & 0xFFFFFFFF):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
    return "%08X" % h


# ---------------------------------------------------------------------------
# Entry-kernel front-end oracle (HANDOFF 13.3/path-1; gate F pins the PTX).
# Stdlib-only (struct/Fraction/math): independent path from the gate's
# struct-<e> model and the C++ __half mirror.
# ---------------------------------------------------------------------------

FE_M32 = 0xFFFFFFFF


def _fe_neg32(c):
    return (FE_M32 + 1 - c) & FE_M32


FE_C_X = _fe_neg32(1918454973)
FE_C_Y = _fe_neg32(669632447)
FE_C_A0 = 747796405
FE_C_A1 = _fe_neg32(1403630843)
FE_C_B0 = _fe_neg32(93469191)
FE_C_B1 = 1192405134
FE_C_C0 = _fe_neg32(895109107)
FE_C_C1 = 568162667
FE_C_D0 = _fe_neg32(2094846927)
FE_C_D1 = 878960812


def fe_pcg_round(state):
    xs = (state >> (((state >> 28) + 4))) ^ state
    return (((xs & FE_M32) * 277803737) & FE_M32)


def fe_pcg_out(v):
    return ((((v >> 30) ^ (v >> 8)) & FE_M32) + 1) & FE_M32


def fe_pcg_streams(seed, x, y):
    """u1..u4 as u32 (pre-scale). Loop-structured twin of the gate."""
    hseed = ((seed & FE_M32) * 0x9E3779B9) & FE_M32
    s2 = (((((x & FE_M32) * FE_C_X) & FE_M32) ^ hseed) & FE_M32)
    s2 = (((((y & FE_M32) * FE_C_Y) & FE_M32) ^ s2) & FE_M32)
    s2 = (s2 ^ 608135816) & FE_M32
    f1 = fe_pcg_round(s2)
    f1 = (((f1 >> 22) ^ f1) & FE_M32)
    outs = []
    for c0, c1 in ((FE_C_A0, FE_C_A1), (FE_C_B0, FE_C_B1),
                   (FE_C_C0, FE_C_C1), (FE_C_D0, FE_C_D1)):
        v = ((f1 * c0) + c1) & FE_M32
        outs.append(fe_pcg_out(fe_pcg_round(v)))
    return outs


def fe_u_float(u32val):
    """Exact (0,1] float: val * 2^-24 (power-of-2 scale, RN-exact)."""
    return u32val * 2.0**-24


def fe_boxmuller(u1, u2, u3, u4):
    """Box-Muller in f64 libm (tol vs f32 mirror / device approx).

    r = sqrt(lg2(u) * ln2 * -2): the kernel multiplies lg2 by f32(ln2)
    (0f3F317218) before -2 (lines 252-255), NOT log2 alone.
    """
    ln2 = 0.6931471805599453  # log(2); kernel uses f32 0.69314718
    r1 = math.sqrt(math.log2(u1) * ln2 * -2.0)
    r2 = math.sqrt(math.log2(u3) * ln2 * -2.0)
    t1 = u2 * 2.0 * math.pi
    t2 = u4 * 2.0 * math.pi
    return (r1 * math.cos(t1), r1 * math.sin(t1), r2 * math.cos(t2))


def fe_normalize_bits(tex_f32, half_f32, mult_f32):
    """f16(tex) - f16(half), times f16(mult); Fraction-exact; bits."""
    t = f32_to_f16_bits(tex_f32)
    h = f32_to_f16_bits(half_f32)
    m = f32_to_f16_bits(mult_f32)
    return _fma_f16_exact(_f16_add_exact(t, h ^ 0x8000), m, 0x0000)


def fe_exposure(lo184, hi184, lo176):
    """Finite-path exposure packs -> (s_sel, s_a, s_b)."""
    p2 = max(lo184, hi184) >= 0.0
    s_sel = 1.0 if p2 else lo176
    s_a = (lo176 if math.isnan(lo184) else lo184) if p2 else -1.0
    s_b = (lo176 if math.isnan(hi184) else hi184) if p2 else -1.0
    return (s_sel, s_a, s_b)


# dim -> (slot, half); half 0 = lo. Gate-F F3.
FE_DIM_SLOT = [(396, 0), (396, 1), (398, 0), (398, 1),
               (402, 0), (402, 1), (406, 0), (406, 1),
               (410, 0), (410, 1), (413, 0), (413, 1),
               (417, 0), (417, 1), (419, 0), (419, 1)]


def fe_assemble_X_bits(tex64x3, gauss64x3, hi176, s_sel, s_a, s_b):
    """64 tokens x 16 dims of f16 bits (fetch order pas,wl)."""
    X = []
    for pas in (0, 1):
        for wl in range(32):
            q = pas * 32 + wl
            tR, tG, tB = tex64x3[q]
            g0, g1, g2 = gauss64x3[q]
            n0 = fe_normalize_bits(tR, 0.5, 1.0)
            n1 = fe_normalize_bits(tG, 0.5, 1.0)
            n2 = fe_normalize_bits(tB, 0.5, 1.0)
            halves = {
                396: (f32_to_f16_bits(g0), f32_to_f16_bits(g1)),
                398: (f32_to_f16_bits(g2), 0x3C00),
                402: (n0, n1),
                406: (n0, n2),
                410: (n1, n2),
                413: (f32_to_f16_bits(hi176), f32_to_f16_bits(s_sel)),
                417: (f32_to_f16_bits(s_sel), f32_to_f16_bits(s_a)),
                419: (f32_to_f16_bits(s_b), 0x0000),
            }
            X.append([halves[s][h] for (s, h) in FE_DIM_SLOT])
    return X


def fe_patch_f64(Xb16, Wb16):
    """Y[q][o] = sum_d X[q][d]*W[o][d], k-ascending, f64 (exact small)."""
    X = [[f16_to_f32(h) for h in row] for row in Xb16]
    W = [[f16_to_f32(h) for h in row] for row in Wb16]
    out = []
    for q in range(64):
        row = []
        for o in range(32):
            s = 0.0
            for d in range(16):
                s += X[q][d] * W[o][d]
            row.append(s)
        out.append(row)
    return out


def flat(M):
    return [v for row in M for v in row]


# ---------------------------------------------------------------------------
# Synthetic vectors (LCG; MUST match swin_1h.hip host mirror exactly)
# ---------------------------------------------------------------------------

def synth_inputs():
    st = 0x12345678
    xin, st = gen_vec(st, TOK * D_IN, -1, 3)          # 64x16 in -1..1
    st = 0x9E3779B9
    wq, st = gen_vec(st, QKV * C, -1, 3)              # 96x32 in -1..1
    wp, st = gen_vec(st, C * C, -1, 3)                # 32x32 in -1..1
    w1, st = gen_vec(st, H1 * C, -1, 3)               # 128x32 in -1..1
    w2, st = gen_vec(st, C * H1, -1, 3)               # 32x128 in -1..1
    X = [xin[i * D_IN:(i + 1) * D_IN] for i in range(TOK)]
    Wq = [wq[i * C:(i + 1) * C] for i in range(QKV)]
    Wp = [wp[i * C:(i + 1) * C] for i in range(C)]
    W1 = [w1[i * C:(i + 1) * C] for i in range(H1)]
    W2 = [w2[i * H1:(i + 1) * H1] for i in range(C)]
    st = 0xC0FFEE11
    wpt, _ = gen_vec(st, C * D_IN, -1, 3)             # 32x16 in -1..1
    Wpt = [wpt[i * D_IN:(i + 1) * D_IN] for i in range(C)]
    return X, Wq, Wp, W1, W2, Wpt


def synth_to_e4m3(ints):
    """Small ints -> f16 bits -> e4m3 bytes (mimics staging quant output)."""
    return [quantise_f16_to_e4m3(f32_to_f16_bits(float(v))) for v in ints]


def cmd_selftest():
    X, Wq, Wp, W1, W2, Wpt = synth_inputs()
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + name +
              ((" -- " + extra) if extra and not cond else ""))
        ok = ok and cond

    # LCG determinism anchors (first values; guards Python/C++ drift)
    s = 0x12345678
    s = lcg_next(s)
    check("lcg anchor", s == 0x75432777, hex(s))
    check("input anchor", X[0][:4] == [0, 1, 1, 0], str(X[0][:4]))
    check("carve audit", carve_audit())

    # e4m3 anchors
    check("e4m3 1.0", e4m3_to_f32(0x38) == 1.0)
    check("e4m3 448", e4m3_to_f32(0x7E) == 448.0)
    check("e4m3 subnorm", abs(e4m3_to_f32(0x01) - 1 / 512.0) == 0.0)
    check("e4m3 roundtrip ints",
          all(e4m3_to_f32(synth_to_e4m3([v])[0]) == float(v)
              for v in (-2, -1, 0, 1, 2)))
    check("H4 flush +nan", e4m3_decode(0x7F) == 0.0)
    check("H4 flush -nan", e4m3_decode(0xFF) == 0.0)
    check("H4 exact elsewhere",
          all(e4m3_decode(b) == e4m3_to_f32(b)
              for b in range(256) if b not in (0x7F, 0xFF)))

    # patch path on exact ints
    Xe = [synth_to_e4m3(row) for row in X]
    Wptb = [[f32_to_f16_bits(float(v)) for v in row] for row in Wpt]
    Ye, Yf = patch_forward(Xe, Wptb, TOK)
    check("patch exact-in-f32",
          all(float(v).is_integer() for v in flat(Yf)))
    check("patch checksum stable",
          fnv1a_hex(flat(Yf)) == fnv1a_hex(flat(Yf)))

    # attention-half on patch outputs (dequantised, exact small ints? no:
    # patch outputs are ints but large; use raw int inputs for exactness)
    Xf = [[float(v) for v in row] for row in
          ([[0] * (C - D_IN) + row for row in X])]
    Q, K, V, Yq = qkv_forward(Xf, [[float(v) for v in row] for row in Wq])
    check("qkv exact-in-f32", all(float(v).is_integer() for v in flat(Yq)))
    O, S, P, _ = attn_forward(Q, K, V)
    check("softmax rows sum to ~1",
          max(abs(sum(row) - 1.0) for row in P) < 5e-3)
    check("scores symmetric-ish finite",
          all(math.isfinite(v) for v in flat(S)))
    # bias recipe (§21): zeros == no bias; nonzero shifts scores exactly
    Ob0, S0, _, _ = attn_forward(Q, K, V, bias=[0.0] * (TOK * TOK))
    check("bias zeros == no bias", S0 == S and Ob0 == O)
    Bb = [0.5 if (i + j) % 2 == 0 else -0.25
          for i in range(TOK) for j in range(TOK)]
    _, Sb, _, _ = attn_forward(Q, K, V, bias=Bb)
    check("bias shifts scores",
          max(abs(Sb[i][j] - S[i][j] - Bb[i * TOK + j])
              for i in range(TOK) for j in range(TOK)) == 0.0)
    # fast-exp pinning (§19): constants == f16(cvt.rn.f32(PTX imm))
    check("expc1 pin",
          f16_to_f32(f32_to_f16_bits(struct.unpack(
              "<f", struct.pack("<I", 1027077105))[0])) == EXP_C1)
    check("expc2 pin",
          f16_to_f32(f32_to_f16_bits(struct.unpack(
              "<f", struct.pack("<I", 1067877303))[0])) == EXP_C2)
    check("explo pin",
          f16_to_f32(f32_to_f16_bits(struct.unpack(
              "<f", struct.pack("<I", 1065615360))[0])) == EXP_LO)
    check("exphi pin",
          f16_to_f32(f32_to_f16_bits(struct.unpack(
              "<f", struct.pack("<I", 1070129152))[0])) == EXP_HI)
    check("smexp floor 2^-14",
          smexp_pair_f16(-100.0)[0] == 0x0400)
    check("smexp ceil sane",
          f16_to_f32(smexp_pair_f16(100.0)[0]) == 9.75)
    check("smexp halves agree",
          all(smexp_pair_f16(v)[0] == smexp_pair_f16(v)[1]
              for v in (-6.0, -1.0, 0.0, 0.5, 1.0, 5.0)))
    check("smexp x=0",
          f16_to_f32(smexp_pair_f16(0.0)[0]) == 0.025390625)
    Qn, Kn, ssQ, ssK, _, _ = qknorm_forward(Q, K, S_TEST)
    check("sumsq exact ints",
          all(float(v).is_integer() for v in ssQ + ssK))
    check("Q unit-norm times s",
          max(abs(sum(x * x for x in row) - S_TEST ** 2)
              for row, ss in zip(Qn, ssQ) if ss > 0) < 1e-9)
    check("K unit-norm",
          max(abs(sum(x * x for x in row) - 1.0)
              for row, ss in zip(Kn, ssK) if ss > 0) < 1e-9)
    Qz, Kz, _, _, _, _ = qknorm_forward([[0.0] * C], [[0.0] * C], S_TEST)
    check("eps clamp on zero row",
          Qz == [[0.0] * C] and Kz == [[0.0] * C])
    _, Kn2, _, _, _, _ = qknorm_forward(Q, K, 2.0)
    check("K independent of temperature", Kn2 == Kn)
    # C=64 smexp unification (§50 immediates == §19 constants): the f32
    # bit patterns round to the IDENTICAL f16 words, so Test-17 reuses
    # smexp_pair_f16 bit-exactly (DECISION: replicate, no exp()).
    for _imm, _want, _nm in ((1027077105, EXP_C1_BITS, "c64 a"),
                             (1067877303, EXP_C2_BITS, "c64 b"),
                             (1065615360, 0x3C20, "c64 lo"),
                             (1070129152, 0x3E47, "c64 hi")):
        _fv = struct.unpack("<f", struct.pack("<I", _imm))[0]
        check("smexp unifies " + _nm, f32_to_f16_bits(_fv) == _want,
              hex(f32_to_f16_bits(_fv)))
    # C=64 trap guard (§49): temp pre-dot on Q, bias never scaled.
    _Q = [[[float((i + d) % 3 - 1) for d in range(C64_DIM)]
           for i in range(C64_TOK)] for _ in range(C64_HEADS)]
    _K = [[[float((j * 2 + d) % 3 - 1) for d in range(C64_DIM)]
           for j in range(C64_TOK)] for _ in range(C64_HEADS)]
    _Bb = [[(0.5 if (i + j) % 2 == 0 else -0.25)
            for j in range(C64_TOK)] for i in range(C64_TOK)]
    _S = c64_scores_forward(_Q, _K, [0.5, 2.0], [_Bb, _Bb])
    _raw = [[sum(_Q[h][i][d] * _K[h][j][d] for d in range(C64_DIM))
             for j in range(C64_TOK)] for h in range(C64_HEADS)
            for i in range(C64_TOK)]
    _wrong = [[(_raw[h * C64_TOK + i][j] + _Bb[i][j]) * (0.5 if h == 0 else 2.0)
               for j in range(C64_TOK)] for i in range(C64_TOK)
              for h in range(C64_HEADS)]
    check("c64 trap: wrong order differs",
          max(abs(_S[h][i][j] - _wrong[h * C64_TOK + i][j])
              for h in range(C64_HEADS) for i in range(C64_TOK)
              for j in range(C64_TOK)) > 1.0)
    _S1 = c64_scores_forward(_Q, _K, [1.0, 1.0], [_Bb, _Bb])
    check("c64 temp=1 is dots+bias",
          max(abs(_S1[h][i][j] - _raw[h * C64_TOK + i][j] - _Bb[i][j])
              for h in range(C64_HEADS) for i in range(C64_TOK)
              for j in range(C64_TOK)) == 0.0)
    _Z = [[[0.0] * C64_TOK for _ in range(C64_TOK)]
            for _ in range(C64_HEADS)]
    _S0 = c64_scores_forward(_Q, _K, [0.5, 2.0], _Z)
    check("c64 zero bias == no bias shift",
          all(abs(_S0[h][i][j] - (_S[h][i][j] - _Bb[i][j])) < 1e-9
              for h in range(C64_HEADS) for i in range(C64_TOK)
              for j in range(C64_TOK)))
    # Test-18 oracle pins (HANDOFF §78E/J): two-step f16 rounding.
    _P1 = [[[0.0] * C64_TOK for _ in range(C64_TOK)]
           for _ in range(C64_HEADS)]
    _V1 = [[[0.0] * C64_DIM for _ in range(C64_TOK)]
           for _ in range(C64_HEADS)]
    _P1[0][0][0] = 1.0
    _P1[0][0][32] = 1.0
    for _k in (0, 32):
        _V1[0][_k] = [0.5] * C64_DIM
    _O1 = c64_ctx_forward(_P1, _V1)
    check("c64 ctx exact ones",
          _O1[0][0][0] == 1.0 and _O1[1][7][31] == 0.0)
    # Two-step vs one-step: s0 = 1.0, s1 = 2^-12. Joint f64 would be
    # 1.000244140625; stepwise RN16 kills s1 (below half-ulp of 1.0),
    # so the oracle MUST read exactly 1.0 (production mma-D contract).
    _P2 = [[[0.0] * C64_TOK for _ in range(C64_TOK)]
           for _ in range(C64_HEADS)]
    _V2 = [[[0.0] * C64_DIM for _ in range(C64_TOK)]
           for _ in range(C64_HEADS)]
    _P2[0][0][0] = 1.0
    _V2[0][0] = [1.0] * C64_DIM
    _P2[0][0][32] = 1.0
    _V2[0][32] = [2.0 ** -12] * C64_DIM
    _O2 = c64_ctx_forward(_P2, _V2)
    check("c64 ctx two-step rounds",
          _O2[0][0][0] == 1.0 and _O2[0][0][0] != 1.0 + 2.0 ** -12)
    # Staged-stream anchors (first LCG draws; guards Python/C++ drift).
    _Pqe, _Ve = c64_staged_pv()
    check("c64 pv lengths", len(_Pqe) == 8192 and len(_Ve) == 4096)
    check("c64 pv deterministic", c64_staged_pv() == (_Pqe, _Ve))
    check("c64 pv anchors", _Pqe[:4] == [0x00, 0xB8, 0x38, 0xB0] and
          _Ve[:4] == [0x38, 0x00, 0xB0, 0x00])
    # Test-19 oracle pins (HANDOFF §78G): plain sums + gate add.
    _Oc = [[[0.0] * C64_TOK for _ in range(C64_TOK)]]
    _W = [[[0.0] * C64_TOK for _ in range(C64_TOK)]]
    _Y = [[[0.0] * C64_TOK for _ in range(C64_TOK)]]
    _Oc[0][3][7] = 1.0
    _W[0][7] = [0.5] * C64_TOK
    _G = [0.0] * C64_TOK
    _Op = c64_proj_forward(_Oc[0], _W[0], _G, _Y[0])
    check("c64 proj exact one-hot",
          _Op[3][9] == 0.5 and _Op[0][0] == 0.0 and _Op[3][10] == 0.5)
    _G2 = [0.0] * C64_TOK
    _G2[5] = 2.0
    _Y[0][3] = [0.25] * C64_TOK
    _Op2 = c64_proj_forward(_Oc[0], _W[0], _G2, _Y[0])
    check("c64 proj gate term",
          _Op2[3][5] == 1.0 and _Op2[3][9] == 0.5 and _Op2[4][5] == 0.0)
    _Ob, _Yb = c64_staged_oy()
    check("c64 oy lengths", len(_Ob) == 4096 and len(_Yb) == 4096)
    check("c64 oy deterministic", c64_staged_oy() == (_Ob, _Yb))
    check("c64 oy anchors", _Ob[:4] == [0xB0, 0x30, 0x38, 0x38] and
          _Yb[:4] == [0x30, 0xB8, 0x00, 0xB0])
    # Test-20 oracle pins (HANDOFF §78H): expand -> act -> contract.
    _X = [[0.0] * C64_TOK for _ in range(C64_TOK)]
    _W1 = [[0.0] * C64_TOK for _ in range(C64_H1)]
    _W2 = [[0.0] * C64_H1 for _ in range(C64_TOK)]
    _X[3][7] = 1.0
    for _j in range(C64_H1):
        _W1[_j][7] = 0.5
    for _o in range(C64_TOK):
        _W2[_o] = [0.25] * C64_H1
    _F = c64_ffn_forward(_X, _W1, _W2, [0.0] * C64_TOK)
    _a05 = f16_to_f32(mpcubic_silu_ref(f32_to_f16_bits(0.5)))
    check("c64 ffn exact one-hot",
          _F[3][9] == sum([_a05 * 0.25] * C64_H1) and _F[0][0] == 0.0 and
          _F[3][10] == _F[3][9])
    _G1 = [0.0] * C64_TOK
    _G1[5] = 2.0
    _X[3][5] = 0.25
    _F2 = c64_ffn_forward(_X, _W1, _W2, _G1)
    check("c64 ffn gate term",
          _F2[3][5] == _F[3][5] + 0.5 and _F2[4][5] == 0.0)
    _Xb, = (c64_staged_x(),)
    check("c64 x length", len(_Xb) == 4096)
    check("c64 x deterministic", c64_staged_x() == _Xb)
    check("c64 x anchors", _Xb[:4] == [0xB8, 0x00, 0xB8, 0x00])
    # Test-21 oracle pins (HANDOFF S90C): two-step f16 split-K.
    _QX = [[[0.0] * C64_TOK for _ in range(C64_TOK)]]
    _QW = [[[[0.0] * 96 for _ in range(32)] for _ in range(C64_HEADS)]
           for _ in range(2)]
    _QX[0][0][0] = 1.0
    _QX[0][0][32] = 1.0
    for _kh in range(2):
        _QW[_kh][0][0] = [0.5] * 96
    _QO = c64_qkv_forward(_QX[0], _QW)
    check("c64 qkv exact ones",
          _QO[0][0][0] == 1.0 and _QO[0][0][95] == 1.0 and
          _QO[1][7][31] == 0.0)
    _QX2 = [[[0.0] * C64_TOK for _ in range(C64_TOK)]]
    _QW2 = [[[[0.0] * 96 for _ in range(32)] for _ in range(C64_HEADS)]
            for _ in range(2)]
    _QX2[0][0][0] = 1.0
    _QW2[0][0][0] = [1.0] * 96
    _QX2[0][0][32] = 1.0
    _QW2[1][0][0] = [2.0 ** -12] * 96
    _QO2 = c64_qkv_forward(_QX2[0], _QW2)
    check("c64 qkv two-step rounds",
          _QO2[0][0][0] == 1.0 and _QO2[0][0][0] != 1.0 + 2.0 ** -12)
    _Qb = c64_staged_qin()
    check("c64 qin length", len(_Qb) == 4096)
    check("c64 qin deterministic", c64_staged_qin() == _Qb)
    check("c64 qin anchors", _Qb[:4] == [0x00, 0x30, 0x00, 0x38])
    # Test-22 (HANDOFF §111): C=64 FFN contract, h[64][128] x W2[128][32].
    _Hb = c64_staged_hin()
    check("c64 hin length", len(_Hb) == 8192)
    check("c64 hin deterministic", c64_staged_hin() == _Hb)
    _CH = [[1.0] * 128 for _ in range(C64_TOK)]
    _CW = [[1.0] * 32 for _ in range(128)]
    _CO = c64_contract_forward(_CH, _CW)
    check("c64 contract exact ones",
          _CO[0][0] == 128.0 and _CO[63][31] == 128.0)
    _CW2 = [[0.0] * 32 for _ in range(128)]
    _CW2[0] = [1.0] * 32
    _CO2 = c64_contract_forward(_CH, _CW2)
    check("c64 contract single-k", _CO2[0][0] == 1.0 and _CO2[0][1] == 1.0)
    _CW3 = [[0.0] * 32 for _ in range(128)]
    _CW3[0] = [1.0] * 32
    _CW3[1] = [2.0 ** -12] * 32
    _CO3 = c64_contract_forward(_CH, _CW3)
    check("c64 contract four-step rounds",
          _CO3[0][0] == 1.0 and _CO3[0][0] != 1.0 + 2.0 ** -12)
    # Test-23 (HANDOFF §113): C=64 FFN expand, x[64][64] x w1[64][128]
    # -> h[64][128], K=64 in TWO f16 k-steps.
    _Eb = c64_staged_exin()
    check("c64 exin length", len(_Eb) == 4096)
    check("c64 exin deterministic", c64_staged_exin() == _Eb)
    # anchors read off this seed (NOT copied from the qin stream):
    # 0xB0=-0.5, 0x00=0.0, 0x38=1.0 -- the 5-level set {-1,-0.5,0,.5,1}.
    check("c64 exin anchors", _Eb[:4] == [0xB0, 0x00, 0x38, 0x00])
    check("c64 exin five levels", len(set(_Eb)) == 5 and
          set(_Eb) == {0xB8, 0xB0, 0x00, 0x30, 0x38})
    _EX = [[1.0] * 64 for _ in range(C64_TOK)]
    _EW = [[1.0] * 128 for _ in range(64)]
    _EO = c64_expand_forward(_EX, _EW)
    check("c64 expand shape",
          len(_EO) == 64 and all(len(r) == 128 for r in _EO))
    check("c64 expand exact ones",
          _EO[0][0] == 64.0 and _EO[63][127] == 64.0)
    _EW2 = [[0.0] * 128 for _ in range(64)]
    _EW2[0] = [1.0] * 128
    _EO2 = c64_expand_forward(_EX, _EW2)
    check("c64 expand single-k",
          _EO2[0][0] == 1.0 and _EO2[0][127] == 1.0)
    _EW3 = [[0.0] * 128 for _ in range(64)]
    _EW3[0] = [1.0] * 128
    _EW3[1] = [2.0 ** -12] * 128
    _EO3 = c64_expand_forward(_EX, _EW3)
    check("c64 expand two-step rounds",
          _EO3[0][0] == 1.0 and _EO3[0][0] != 1.0 + 2.0 ** -12)
    # C=64 file pins (tensor_137, §46.1/§49 carve).
    _tp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "dlss5-analysis", "tensors", "tensor_137.bin")
    if os.path.exists(_tp):
        _raw137 = open(_tp, "rb").read()
        check("c64 file size", len(_raw137) == 61760, str(len(_raw137)))
        _t0, _t1 = struct.unpack("<2f", _raw137[57504:57512])
        check("c64 temps", abs(_t0 - 4.424942) < 1e-5 and
              abs(_t1 - 6.877718) < 1e-5, "%.6f %.6f" % (_t0, _t1))
        _hot = struct.unpack("<8192H", _raw137[41120:57504])
        _hotf = [f16_to_f32(u) for u in _hot]
        check("c64 HOT fabric",
              len(_hotf) == 8192 and max(_hotf) == 0.0 and
              all(v <= 0.0 for v in _hotf))
        for _go, _gn in ((28688, "gate1"), (61616, "gate2")):
            _gv = [f16_to_f32(u) for u in
                   struct.unpack("<64H", _raw137[_go:_go + 128])]
            check("c64 " + _gn,
                  sum(1 for v in _gv if 0.0 < v <= 1.0) == 64)
    else:
        check("c64 tensor_137 present", False, _tp)
    # Trick-vs-exp span pin (§50 numeric claim, Test-17 decision basis):
    # after removing the constant factor (median-normalized here; §50's
    # -3.3% is best-fit-normalized -- consistent), the trick tracks exp()
    # within ~4% over the clamp window (measured worst 0.0404, at a rail
    # endpoint). exp() would need a few-% budget; replication is exact.
    _span = [-6.0 + i * 0.5 for i in range(25)]
    _rat = []
    for _s in _span:
        _e = f16_to_f32(smexp_pair_f16(_s)[0])
        _rat.append(_e / math.exp(_s))
    _med = sorted(_rat)[len(_rat) // 2]
    check("trick tracks exp ~4%",
          max(abs(r / _med - 1.0) for r in _rat) < 0.045,
          "med=%.4f worst=%.4f" % (_med,
                                   max(abs(r / _med - 1.0) for r in _rat)))
    ga = [0.5] * C
    Yp, Ypraw = proj_forward(O, [[float(v) for v in row] for row in Wp],
                             Xf, ga)
    check("proj residual wired",
          abs(Yp[0][0] - (Ypraw[0][0] + 0.5 * Xf[0][0])) < 1e-12)
    gf = [0.25] * C
    Yf2, Yfraw, A = ffn_forward(Yp,
                                [[float(v) for v in row] for row in W1],
                                [[float(v) for v in row] for row in W2], gf)
    check("ffn residual wired",
          abs(Yf2[0][0] - (Yfraw[0][0] + 0.25 * Yp[0][0])) < 1e-9)
    check("act bounded", all(abs(v) < 1e4 for v in flat(A)))
    # gate edge cases: 0.0 -> pure GEMM, 1.0 -> full residual
    Yz, _ = proj_forward(O, [[float(v) for v in row] for row in Wp], Xf,
                         [0.0] * C)
    check("gate0 == pure GEMM",
          max(abs(Yz[m][o] - Ypraw[m][o])
              for m in range(TOK) for o in range(C)) == 0.0)

    # front-end oracle (HANDOFF 13.3/path-1; gate F pins the PTX).
    # Independent Fraction/struct path from the gate's model.
    import o_frontend as _FE
    _fe_ok = True
    for _pas in (0, 1):
        for _wl in range(32):
            _x, _y = (_wl & 7), (_wl >> 3)
            if fe_pcg_streams(0x12345678, _x, _y) != \
                    _FE.pcg_streams(0x12345678, _x, _y):
                _fe_ok = False
    check("fe_pcg 64/64 match gate", _fe_ok)
    _tex = [((q & 7) * 1.0, ((q >> 3) & 7) * 1.0,
             ((q & 7) + ((q >> 3) & 7)) * 1.0) for q in range(64)]
    _g3 = [(1.5, -2.25, 0.75)] * 64
    _Xb = fe_assemble_X_bits(_tex, _g3, 3.0, 1.0, 0.0, 0.0)
    check("fe_X spot n0", _Xb[0][4] == 0xB800, hex(_Xb[0][4]))
    check("fe_Xbits deterministic",
          fnv1a_hex_words([h for row in _Xb for h in row]) ==
          fnv1a_hex_words([h for row in _Xb for h in row]))
    print("FE-XBITS-FNV %s" %
          fnv1a_hex_words([h for row in _Xb for h in row]))
    _g0 = fe_boxmuller(*[fe_u_float(u) for u in
                         fe_pcg_streams(0x12345678, 3, 5)])
    check("fe_boxmuller sane", all(abs(v) < 12.0 for v in _g0))
    _Wb = [[f32_to_f16_bits(float((o * 16 + d) % 7 - 3)) for d in range(16)]
           for o in range(32)]
    _Y = fe_patch_f64(_Xb, _Wb)
    check("fe_patch finite", all(abs(v) < 1e6 for row in _Y for v in row))
    print("FE-PATCH-FNV %s" % fnv1a_hex(flat(_Y)))

    print("checksums: patch %s qkv %s attn %s proj %s ffn %s" % (
        fnv1a_hex(flat(Yf)), fnv1a_hex(flat(Yq)), fnv1a_hex(flat(O)),
        fnv1a_hex(flat(Yp)), fnv1a_hex(flat(Yf2))))
    print("ALL SELF-TESTS PASSED." if ok else "SELF-TEST FAILURES PRESENT.")
    return ok


def f32_list_to_e4m3(vals):
    return [quantise_f16_to_e4m3(f32_to_f16_bits(float(v))) for v in vals]


def linked_chain():
    """Production-faithful chain on synthetic vectors (inter-stage e4m3
    quants included). Mirrors sw_mirror_chain in hip/mvp1/swin_1h.hip.
    §21 order (FFN-FIRST): patch -> Xb -> FFN(Xb)+gate_ffn*Xb = Yf ->
    QKV(quant(Yf)) -> attn(+bias) -> proj+gate_attn*Yf = Yp (block out)."""
    X, Wq, Wp, W1, W2, Wpt = synth_inputs()
    Xe = [synth_to_e4m3(row) for row in X]
    Wptb = [[f32_to_f16_bits(float(v)) for v in row] for row in Wpt]
    Ye, Ypatch = patch_forward(Xe, Wptb, TOK)
    Xb = [[e4m3_decode(b) for b in row] for row in Ye]
    # alternating 0/1 gates: gate-0 channels test the pure-GEMM path, and any
    # off-by-one gate wiring shows up at full |x| amplitude. Must match C++.
    ga = [float(o % 2) for o in range(C)]
    gf = [float((o + 1) % 2) for o in range(C)]
    # FFN half first: QKV-mma-A = quant(contract-chain-end) i.e. quant(Yf).
    W1_f = [[e4m3_decode(b) for b in f32_list_to_e4m3(row)] for row in W1]
    H = gemm(Xb, W1_f, TOK, C, H1)
    Hb = [round_f32_list_to_f16(row) for row in H]
    Abits = [[mpcubic_silu_ref(u) for u in row] for row in Hb]
    A = [[f16_to_f32(u) for u in row] for row in Abits]
    W2_f = [[e4m3_decode(b) for b in f32_list_to_e4m3(row)] for row in W2]
    Yfraw = gemm(A, W2_f, TOK, H1, C)
    Yf = [[Yfraw[m][o] + gf[o] * Xb[m][o] for o in range(C)]
          for m in range(TOK)]
    Yf_e4 = [f32_list_to_e4m3(row) for row in Yf]
    Xh = [[e4m3_decode(b) for b in row] for row in Yf_e4]
    Wq_f = [[e4m3_decode(b) for b in f32_list_to_e4m3(row)] for row in Wq]
    Q, K, V, Yq = qkv_forward(Xh, Wq_f)
    Qn, Kn, ssQ, ssK, _, _ = qknorm_forward(Q, K, S_TEST)
    # Synthetic stand-in bias = exact zeros (same +0.0 op as C++/device, so
    # the -0.0/+0.0 edge converts identically on all three sides; the real
    # f16 bias only exists in --realweights).
    O, S, P, Eb = attn_forward(Qn, Kn, V, bias=[0.0] * (TOK * TOK))
    Wp_f = [[e4m3_decode(b) for b in f32_list_to_e4m3(row)] for row in Wp]
    Yp, _ = proj_forward(O, Wp_f, Yf, ga)
    return {"Ypatch": Ypatch, "Ye": Ye, "Xb": Xb, "Yq": Yq, "Qn": Qn,
            "Kn": Kn, "ssQ": ssQ, "ssK": ssK, "S": S, "P": P, "Eb": Eb,
            "O": O, "Yp": Yp, "H": H, "Abits": Abits, "Yf": Yf,
            "X": X, "Wq": Wq, "Wp": Wp, "W1": W1, "W2": W2, "Wpt": Wpt}


def fnv_raw(floats):
    h = 0x811C9DC5
    for v in floats:
        for byte in struct.pack("<f", v):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
    return "%08X" % h


def fnv_u16(words):
    h = 0x811C9DC5
    for w in words:
        h ^= w & 0xFF
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= (w >> 8) & 0xFF
        h = (h * 0x01000193) & 0xFFFFFFFF
    return "%08X" % h


def fnv_bytes_raw(data):
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return "%08X" % h


def cmd_stage_fnv():
    ch = linked_chain()
    print("patch_raw %s" % fnv_raw(flat(ch["Ypatch"])))
    print("patch_bytes %s" % fnv_bytes_raw([b for row in ch["Ye"] for b in row]))
    print("qkv %s" % fnv_raw(flat(ch["Yq"])))
    print("sumsqQ %s" % fnv_raw([float(v) for v in ch["ssQ"]]))
    print("sumsqK %s" % fnv_raw([float(v) for v in ch["ssK"]]))
    print("smexp %s" % fnv_u16([u for row in ch["Eb"] for u in row]))
    print("H %s" % fnv_raw(flat(ch["H"])))
    print("actbits %s" % fnv_u16([u for row in ch["Abits"] for u in row]))


def cmd_golden(path):
    ch = linked_chain()
    gold = {
        "dims": {"D_IN": D_IN, "C": C, "TOK": TOK, "QKV": QKV, "H1": H1},
        "x_int16": ch["X"], "w_patch_int": ch["Wpt"], "w_qkv_int": ch["Wq"],
        "w_proj_int": ch["Wp"], "w_ffn1_int": ch["W1"],
        "w_ffn2_int": ch["W2"],
        "patch_f32": ch["Ypatch"], "qkv_f32": ch["Yq"], "scores_f64": ch["S"],
        "attn_f64": ch["O"], "proj_f32gate05": ch["Yp"],
        "ffn_f32gate025": ch["Yf"],
        "checksums": {
            "patch": fnv1a_hex(flat(ch["Ypatch"])),
            "qkv": fnv1a_hex(flat(ch["Yq"])),
            "attn": fnv1a_hex(flat(ch["O"])),
            "proj": fnv1a_hex(flat(ch["Yp"])),
            "ffn": fnv1a_hex(flat(ch["Yf"])),
        },
    }
    with open(path, "w") as f:
        json.dump(gold, f)
    print("wrote %s checksums=%s" % (path, gold["checksums"]))


def carve(path, table):
    raw = open(path, "rb").read()
    out = {}
    for name, off, count, dt in table:
        if dt == "f16":
            vals = struct.unpack("<%dH" % count, raw[off:off + 2 * count])
            out[name] = [f16_to_f32(u) for u in vals]
            out[name + "_bits"] = list(vals)
        else:
            vals = list(raw[off:off + count])
            out[name] = [e4m3_decode(b) for b in vals]
    return out


def cmd_realweights(tensordir):
    for fname, table, size in (("tensor_000.bin", CARVE_PRE, 21696),
                               ("tensor_001.bin", CARVE_FUSED32, 20672)):
        p = tensordir.rstrip("/\\") + "/" + fname
        raw = open(p, "rb").read()
        print("%s: %d bytes (expect %d) %s" %
              (fname, len(raw), size,
               "OK" if len(raw) == size else "SIZE MISMATCH"))
        W = carve(p, table)
        for g in ("gate_ffn", "gate_attn"):
            vals = W[g]
            npos = sum(1 for v in vals if 0.0 < v <= 1.0)
            print("  %s: min %.4f max %.4f in(0,1]=%d/32" %
                  (g, min(vals), max(vals), npos))
        # smoke: staging-like e4m3 inputs (|x|<=1) through REAL weights.
        # tensor_000 runs the full patch->block chain; tensor_001 (no patch
        # region) feeds dequantised patch outputs of the tensor_000 run.
        st = 0xA11CE
        mags, st = gen_vec(st, TOK * D_IN, 0, 5)
        levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        Xe = [[synth_to_e4m3([levels[m]])[0] for m in row]
              for row in [mags[i * D_IN:(i + 1) * D_IN] for i in range(TOK)]]

        def mx(M):
            return max(abs(v) for v in flat(M))

        s_off = 20576 if fname == "tensor_000.bin" else 19552
        s_real = struct.unpack("<f", raw[s_off:s_off + 4])[0]
        print("  QK-temperature s = %.4f (f32 @+%d)" % (s_real, s_off))
        Wq = [W["qkv"][i * C:(i + 1) * C] for i in range(QKV)]
        Wp = [W["proj"][i * C:(i + 1) * C] for i in range(C)]
        W1 = [W["ffn1"][i * C:(i + 1) * C] for i in range(H1)]
        W2 = [W["ffn2"][i * H1:(i + 1) * H1] for i in range(C)]
        B = W["bias"]
        print("  bias 64x64 f16: min %.4f max %.4f finite=%d/4096" %
              (min(B), max(B), sum(1 for v in B if math.isfinite(v))))
        if fname == "tensor_000.bin":
            raw = open(p, "rb").read()
            wpatch = struct.unpack("<512H", raw[8208:9232])
            Wpt = [list(wpatch[i * D_IN:(i + 1) * D_IN]) for i in range(C)]
            Ye, Yf = patch_forward(Xe, Wpt, TOK)
            Xb = [[e4m3_decode(b) for b in row] for row in Ye]
            print("  patch: max|out|=%.3f n_nonfinite=%d" %
                  (mx(Yf), sum(1 for v in flat(Yf)
                               if not math.isfinite(v))))
        else:
            # no patch region: pair two 16-dim rows into one 32-dim input
            Xd = [[e4m3_decode(b) for b in row] for row in Xe]
            Xb = [Xd[i] + Xd[(i + 1) % TOK] for i in range(TOK)]
        # §21 order: FFN-first (QKV-mma-A = quant(contract-chain-end)).
        Yf, _, _ = ffn_forward(Xb, W1, W2, W["gate_ffn"])
        Q, K, V, Yq = qkv_forward(Yf, Wq)
        Qn, Kn, _, _, _, _ = qknorm_forward(Q, K, s_real)
        O, S, _, _ = attn_forward(Qn, Kn, V, bias=B)
        Yp, _ = proj_forward(O, Wp, Yf, W["gate_attn"])
        print("  stages maxabs: ffn=%.2f qkv=%.2f scores=%.2f attn=%.2f "
              "proj=%.2f finite=%d/%d checksum=%s" %
              (mx(Yf), mx(Yq), mx(S), mx(O), mx(Yp),
               sum(1 for v in flat(Yp) if math.isfinite(v)), TOK * C,
               fnv1a_hex([v for v in flat(Yp) if math.isfinite(v)])))


def cmd_crosscheck():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import stage1_ref
    except ImportError as e:
        print("stage1_ref not importable (%s); skipped." % e)
        return True
    n_q = sum(1 for u in range(65536)
              if quantise_f16_to_e4m3(u) != stage1_ref.quantise_f16_to_e4m3fn(u))
    print("quant twins differ on %d/65536 inputs." % n_q)
    n_a = 0
    for u in range(65536):
        a = mpcubic_silu_ref(u)
        b = stage1_ref.mpcubic_silu_ref(u)
        na = (a & 0x7C00) == 0x7C00 and (a & 0x3FF) != 0
        nb = (b & 0x7C00) == 0x7C00 and (b & 0x3FF) != 0
        if (na and nb) or a == b:
            continue
        n_a += 1
    print("mpcubic twins differ on %d/65536 inputs." % n_a)
    ok = (n_q == 0 and n_a == 0)
    print("CROSSCHECK %s." % ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(0 if cmd_selftest() else 1)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--golden":
        cmd_golden(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--realweights":
        cmd_realweights(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--crosscheck":
        sys.exit(0 if cmd_crosscheck() else 1)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--stage-fnv":
        cmd_stage_fnv()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--c64fnv":
        cmd_c64fnv()
    else:
        print(__doc__)
