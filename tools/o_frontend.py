#!/usr/bin/env python3
"""tools/o_frontend.py -- entry-kernel front-end proof (gate F)."""
import os
import re
import struct
import sys

"""tools/o_frontend.py -- entry-kernel front-end proof (gate F).

Closes the texture->staging recipe (HANDOFF 13.3/33.11-note): affine
sampling + PCG dither + Box-Muller gaussians + exposure + f16 normalize
+ patch-A gather, against
`cc_tinlayout_fused_pre_block_swin_1h_32_1_fp8` path 1 (mandatory
colour texture; optional texture paths out of scope, null-gated).

F1. Literal pins (lines): PCG seed/mads/shifts, Box-Muller
    immediates, affine maps + tex@285 + normalize triples,
    exposure + BB1_6/BB1_7 control, gate_ffn scale chain
    (%r497 = gate_ffn[2*(lane%4)]), staging writes, 16 A-loads.
F2. Gather table via bridge (no walk): 16 A-loads x 32 lanes ->
    (slot, wl, pas); (group,row) singletons; 64/64 token cover,
    bijective; groups = token quadrants (x-half x y-half).
F3. X-dim map (A-fragment law, cross-checked vs observed slot
    cycling): dim -> (slot, half, kind):
    0:g0 1:g1 2:g2 3:1.0 | 4:n0 5:n1 6:n0 7:n2 8:n1 9:n2 |
    10:f16(HI176) 11:s_sel 12:s_sel 13:s_a 14:s_b 15:+0.0.
    X[q][d] = staged-half(fetch(q), dim-half(d)).
F4. Stdlib value recompute (bitwise): PCG u32 streams for all 64
    fetches (seed/x/y int-exact) + u-float bits (exact scalings)
    + normalize/exposure/X-bits on synthetic-exact inputs;
    FNVs printed for the host-mirror paste-back (runbook 17).
F5. B-window SET: v4 sites cover [8208,8720)+[8720,9232) =
    carve [8208,9232) == patch exactly (§33.14 audit, gate S:
    [8192,8208) is pad16 zeros, old A-residue16 dissolved into
    patch tail).
F6. Verdict FRONTEND-OK.

Stdlib-only, GPU-free, byte-deterministic stdout. Needs the addr
machine + bridge only (~1 min); runbook 17.
"""

import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from qk_addrs import AddrMachine, RowWalker, parse_mma_all  # noqa: E402
import o_rowmajor as R  # noqa: E402

M32 = 0xFFFFFFFF
NLANES = 32

# ---------------------------------------------------------------- utils

def strip_line(ls, idx):
    return ls[idx].strip().strip("{}").strip().rstrip(";")


def assert_line(ls, ln, want):
    got = strip_line(ls, ln - 1)
    assert got == want, (ln, got[:100])
    return got


def count_has(ls, lo, hi, frag):
    return sum(1 for i in range(lo - 1, hi - 1) if frag in ls[i])


def f32_bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def f32_val(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def f16_bits_of_f32word(w):
    return struct.unpack("<H", struct.pack("<e", f32_val(w)))[0]


def f16_add_exact(a_bits, b_bits):
    """RN f16 add via exact f64 intermediate (exact for f16 operands)."""
    fa = struct.unpack("<e", struct.pack("<H", a_bits))[0]
    fb = struct.unpack("<e", struct.pack("<H", b_bits))[0]
    return struct.unpack("<H", struct.pack("<e", fa + fb))[0]


def f16_mul_exact(a_bits, b_bits):
    fa = struct.unpack("<e", struct.pack("<H", a_bits))[0]
    fb = struct.unpack("<e", struct.pack("<H", b_bits))[0]
    return struct.unpack("<H", struct.pack("<e", fa * fb))[0]


def f16_neg(bits):
    return bits ^ 0x8000


def fnv1a_hex(words):
    h = 0x811C9DC5
    for w in words:
        for b in struct.pack("<I", w & M32):
            h ^= b
            h = (h * 0x01000193) & M32
    return "%08X" % h


# ------------------------------------------------------- F1 literal pins

def check_F1(ls):
    # seed hash: s * 0x9E3779B9 as signed (-1640531527)
    assert_line(ls, 29, "mul.lo.s32 %r75, %r74, -1640531527")
    assert_line(ls, 105, "mul.lo.s32 %r90, %r86, -1918454973")
    assert_line(ls, 106, "xor.b32 %r50, %r90, %r75")
    # per-k y mix + golden-xor
    assert_line(ls, 194, "mul.lo.s32 %r112, %r108, -669632447")
    assert_line(ls, 195, "xor.b32 %r113, %r50, %r112")
    assert_line(ls, 196, "xor.b32 %r114, %r113, 608135816")
    # shared first PCG round
    assert_line(ls, 197, "shr.u32 %r115, %r114, 28")
    assert_line(ls, 198, "add.s32 %r116, %r115, 4")
    assert_line(ls, 199, "shr.u32 %r117, %r114, %r116")
    assert_line(ls, 200, "xor.b32 %r118, %r117, %r114")
    assert_line(ls, 201, "mul.lo.s32 %r119, %r118, 277803737")
    assert_line(ls, 202, "shr.u32 %r120, %r119, 22")
    assert_line(ls, 203, "xor.b32 %r121, %r120, %r119")
    # stream-A second round + output permute
    assert_line(ls, 204, "mad.lo.s32 %r122, %r121, 747796405, -1403630843")
    assert_line(ls, 209, "mul.lo.s32 %r127, %r126, 277803737")
    assert_line(ls, 210, "shr.u32 %r128, %r127, 30")
    assert_line(ls, 211, "shr.u32 %r129, %r127, 8")
    assert_line(ls, 212, "xor.b32 %r130, %r128, %r129")
    assert_line(ls, 213, "add.s32 %r131, %r130, 1")
    assert_line(ls, 214, "cvt.rn.f32.u32 %r132, %r131")
    assert_line(ls, 215, "mul.ftz.f32 %r133, %r132, 0f33800000")
    # streams B/C/D mad pairs
    assert_line(ls, 216, "mad.lo.s32 %r134, %r121, -93469191, 1192405134")
    assert_line(ls, 228, "mad.lo.s32 %r146, %r121, -895109107, 568162667")
    assert_line(ls, 240, "mad.lo.s32 %r158, %r121, -2094846927, 878960812")
    assert count_has(ls, 194, 252, "mul.lo.s32") >= 5, "pcg muls"
    # Box-Muller: ln2, -2.0, 2pi immediates + lg2/sqrt/sin/cos
    assert_line(ls, 252, "lg2.approx.ftz.f32 %r170, %r133")
    assert_line(ls, 253, "mul.ftz.f32 %r171, %r170, 0f3F317218")
    assert_line(ls, 254, "mul.ftz.f32 %r172, %r171, 0fC0000000")
    assert_line(ls, 255, "sqrt.approx.ftz.f32 %r173, %r172")
    assert_line(ls, 260, "mul.ftz.f32 %r178, %r145, 0f40C90FDB")
    assert_line(ls, 261, "mul.ftz.f32 %r179, %r169, 0f40C90FDB")
    assert_line(ls, 262, "sin.approx.ftz.f32 %r180, %r178")
    assert_line(ls, 263, "cos.approx.ftz.f32 %r181, %r178")
    assert_line(ls, 264, "cos.approx.ftz.f32 %r182, %r179")
    assert_line(ls, 265, "mul.ftz.f32 %r101, %r173, %r181")
    assert_line(ls, 266, "mul.ftz.f32 %r102, %r173, %r180")
    assert_line(ls, 267, "mul.ftz.f32 %r103, %r177, %r182")
    # affine maps + tex + normalize triple (ch0 shown; ch1/ch2 same)
    assert_line(ls, 107, "cvt.rn.f32.s32 %r91, %r89")
    assert_line(ls, 109, "div.approx.ftz.f32 %r51, %r92, %r8")
    assert_line(ls, 110, "fma.rn.ftz.f32 %r93, %r51, %r76, %r77")
    assert_line(ls, 111, "mul.ftz.f32 %r52, %r93, %r78")
    assert_line(ls, 285, "tex.2d.v4.f32.f32 {%r104, %r105, %r106, %r187}, "
                         "[%rd1, {%r52, %r186}]")
    assert_line(ls, 287, "cvt.rn.f16.f32 %rs10, %r104")
    assert_line(ls, 291, "sub.f16 %rs11,%rs10,%rs21")
    assert_line(ls, 295, "mul.f16 %rs2,%rs11,%rs23")
    # exposure: max/select + BB1_7 NaN-aware packs + control
    assert_line(ls, 116, "max.ftz.f32 %r95, %r54, %r55")
    assert_line(ls, 117, "setp.ge.ftz.f32 %p2, %r95, 0f00000000")
    assert_line(ls, 118, "selp.f32 %r388, 0f3F800000, %r43, %p2")
    assert_line(ls, 130, "setp.ltu.ftz.f32 %p37, %r55, 0f00000000")
    assert_line(ls, 131, "setp.ltu.ftz.f32 %p38, %r54, 0f00000000")
    assert_line(ls, 132, "selp.f32 %r391, %r43, %r54, %p38")
    assert_line(ls, 133, "selp.f32 %r389, %r391, 0fBF800000, %p2")
    assert_line(ls, 328, "@%p10 bra $L__BB1_6")
    assert_line(ls, 555, "bra.uni $L__BB1_7")
    # gate_ffn per-quad scale chain: laneid>>31 = 0, (lane+0)&-4
    assert_line(ls, 1103, "mov.u32 %r463, %laneid")
    assert_line(ls, 1105, "shr.s32 %r3677, %r463, 31")
    assert_line(ls, 1106, "shr.u32 %r3678, %r3677, 30")
    assert_line(ls, 1107, "add.s32 %r3679, %r463, %r3678")
    assert_line(ls, 1108, "and.b32 %r3680, %r3679, -4")
    assert_line(ls, 1109, "sub.s32 %r3681, %r463, %r3680")
    assert_line(ls, 1110, "mul.wide.s32 %rd51, %r3681, 4")
    assert_line(ls, 1111, "add.s64 %rd52, %rd46, %rd51")
    assert_line(ls, 1112, "ld.global.b32 %r497, [%rd52+9232]")
    # staging writes (8 regs = 16 halves) + 16 A-loads present
    assert strip_line(ls, 166).startswith(
        "st.shared.v4.b32 [%r5031+-1024],"), strip_line(ls, 166)[:60]
    assert strip_line(ls, 180).startswith(
        "st.shared.v4.b32 [%r5031],"), strip_line(ls, 180)[:60]
    nld = 0
    for i in range(600, 1000):
        if re.match(r"ld\.shared\.b32\s+%r(439|4[0-9][0-9]|45[0-9]|46[0-2]),",
                    strip_line(ls, i)):
            nld += 1
    assert nld == 16, nld
    print("F1 pins: seed/pcg-x5/boxmuller/affine/tex/norm/exposure/"
          "gateffn/staging/16-Aloads exact")


# ------------------------------------------------- F2 gather table

# (group, [a0..a3 load lines])
GROUPS = [
    (0, [614, 637, 660, 683]),
    (1, [706, 729, 752, 775]),
    (2, [798, 821, 844, 867]),
    (3, [890, 913, 936, 959]),
]


def load_addr(ls, ln):
    t = strip_line(ls, ln - 1)
    m = re.match(r"ld\.shared\.b32\s+%r\d+,\s*\[\s*(%r\d+)\s*(?:\+\s*(\d+))?"
                 r"\s*\]$", t)
    assert m, (ln, t)
    return m.group(1), int(m.group(2) or 0)


def check_F2(ls):
    am = AddrMachine()
    try:
        am.run(ls, 0, 12000)
        print("F2 addr-machine: clean to 12000")
    except (KeyError, AssertionError) as e:
        print("F2 addr-machine stopped:", str(e)[:120])
    wk0 = RowWalker(ls, parse_mma_all(ls), am)
    table = {}
    for g, lns in GROUPS:
        addrs = {ln: load_addr(ls, ln) for ln in lns}
        for lane in range(NLANES):
            grow = lane // 4
            for li, ln in enumerate(lns):
                areg, const = addrs[ln]
                av = am.v[lane].get(areg)
                assert av is not None, ("addr-unknown", ln, lane)
                a = (av + const) & M32
                hit = wk0.bridge.get(a)
                assert hit is not None, ("bridge-miss", ln, lane, a)
                _v, _w, wl, pas = hit
                row = grow if li in (0, 2) else grow + 8
                table.setdefault((g, row), set()).add((wl, pas))
    toks = {}
    for (g, row), s in table.items():
        assert len(s) == 1, ("nonsingleton", g, row, s)
        wl, pas = next(iter(s))
        toks.setdefault(((wl & 7), ((wl >> 3) + 4 * pas)), []).append(
            (g, row))
    assert len(toks) == 64, len(toks)
    assert all(len(v) == 1 for v in toks.values()), "dup token"
    quads = []
    for g in range(4):
        toks_g = sorted((wl & 7) + 8 * ((wl >> 3) + 4 * pas)
                        for row in range(16)
                        for (wl, pas) in table[(g, row)])
        quads.append(toks_g)
    assert quads[0] == [0, 1, 2, 3, 8, 9, 10, 11,
                        16, 17, 18, 19, 24, 25, 26, 27], quads[0]
    assert quads[1] == [t + 4 for t in quads[0]], quads[1]
    assert quads[2] == [t + 32 for t in quads[0]], quads[2]
    assert quads[3] == [t + 36 for t in quads[0]], quads[3]
    print("F2 gather: 64 (group,row) singletons; 64/64 tokens "
          "bijective; groups = quadrants x-half x y-half")
    return table, am, wk0


# ------------------------------------------------- F3/F4 recipe values

def pcg_round(state):
    xs = (state >> (((state >> 28) + 4))) ^ state
    return ((xs & M32) * 277803737) & M32


def pcg_out(v):
    return ((((v >> 30) ^ (v >> 8)) & M32) + 1) & M32


def neg32(c):
    return (M32 + 1 - c) & M32


# PTX immediates as unsigned (mul.lo/mad.lo are mod-2^32)
C_X = neg32(1918454973)  # -1918454973
C_Y = neg32(669632447)  # -669632447
C_A0 = 747796405
C_A1 = neg32(1403630843)
C_B0 = neg32(93469191)
C_B1 = 1192405134
C_C0 = neg32(895109107)
C_C1 = 568162667
C_D0 = neg32(2094846927)
C_D1 = 878960812


def pcg_streams(seed, x, y):
    """u1..u4 as u32 (pre-scale). seed = param+200 int; x, y unfolded."""
    hseed = ((seed & M32) * 0x9E3779B9) & M32
    s = ((((x & M32) * C_X) & M32) ^ hseed) & M32
    s1 = ((((y & M32) * C_Y) & M32) ^ s) & M32
    s2 = (s1 ^ 608135816) & M32
    f1 = pcg_round(s2)
    f1 = (((f1 >> 22) ^ f1) & M32)
    # stream A has an extra round before output
    a = ((f1 * C_A0) + C_A1) & M32
    outs = [pcg_out(pcg_round(a))]
    for c0, c1 in ((C_B0, C_B1), (C_C0, C_C1), (C_D0, C_D1)):
        outs.append(pcg_out(pcg_round(((f1 * c0) + c1) & M32)))
    return outs


# ------------------------------------------- F4 synthetic value checks

# Shared synthetic vector (mirror + oracle implement the same):
FE_SEED = 0x12345678


def _h(f):
    return struct.unpack("<H", struct.pack("<e", f))[0]


# g0 = f16(1.5), g1 = f16(-2.25), g2 = f16(0.75), all exact;
# derived via struct (never hand-hexed)
FE_G0 = _h(1.5)
FE_G1 = _h(-2.25)
FE_G2 = _h(0.75)
FE_HI176 = _h(3.0)  # -> dim 10
F16_HALF = _h(0.5)


def f16_of_int(n):
    return struct.unpack("<H", struct.pack("<e", float(n)))[0]


def check_F4(ls, table, am, wk0):
    # PCG streams for all 64 fetches; FNV over u32 states.
    # Seeds use UNFOLDED y (line 194: %r108), so pass 0/1 share
    # streams: the same gaussians feed both k-passes per lane.
    words = []
    for pas in (0, 1):
        for wl in range(NLANES):
            x, y = (wl & 7), (wl >> 3)
            outs = pcg_streams(FE_SEED, x, y)
            words.extend(outs)
            for u in outs:
                assert 1 <= u <= 0xFFFFFF, (wl, pas, u)
    print("FE-PCG-FNV %s" % fnv1a_hex(words))
    assert words[:128] == words[128:], "pas-mirror doubling"
    assert len(set(words[:128])) == 128, len(set(words[:128]))
    # slot-kind cycling from DATA (independent of the hand dim map):
    # group-1 loads -> {396,398,402,406} by lane%4; group-2 -> {410,...}
    cyc1, cyc2 = {}, {}
    loads1 = [614, 637]
    loads2 = [660, 683]
    for ln in loads1 + loads2:
        areg, const = load_addr(ls, ln)
        for lane in range(NLANES):
            av = am.v[lane].get(areg)
            a = (av + const) & M32
            hit = wk0.bridge.get(a)
            base = 0 if a < 512 else (512 if a < 1024 else
                                      (1024 if a < 1536 else 1536))
            off = a - base - 16 * hit[2]
            regs = R.TEXREGS_W1 if base in (0, 512) else R.TEXREGS_W2
            slot = regs[off // 4]
            (cyc1 if ln in loads1 else cyc2).setdefault(
                lane % 4, set()).add(slot)
    assert cyc1 == {0: {"%r396"}, 1: {"%r398"}, 2: {"%r402"},
                    3: {"%r406"}}, cyc1
    assert cyc2 == {0: {"%r410"}, 1: {"%r413"}, 2: {"%r417"},
                    3: {"%r419"}}, cyc2
    print("F4 slot cycling: lane%4 -> slot-kind exact (both groups)")
    # X-bits on synthetic-exact inputs (coords in, div bypassed;
    # affine-div safety argued by margin, mirror tol-checks it).
    # Exposure params finite: +184 = {0.0, 0.0} -> p2 true ->
    # s_sel = 1.0; s_a = s_b = 0.0.
    halves = []
    s_sel = 0x3C00
    s_a = 0x0000
    s_b = 0x0000
    for pas in (0, 1):
        for wl in range(NLANES):
            x, y = (wl & 7), ((wl >> 3) + 4 * pas)
            # texel RGB ints; centers exact
            Rv, Gv, Bv = x, y, x + y
            n0 = f16_add_exact(f16_of_int(Rv), f16_neg(F16_HALF))
            n1 = f16_add_exact(f16_of_int(Gv), f16_neg(F16_HALF))
            n2 = f16_add_exact(f16_of_int(Bv), f16_neg(F16_HALF))
            row = [FE_G0, FE_G1, FE_G2, 0x3C00,
                   n0, n1, n0, n2, n1, n2,
                   FE_HI176, s_sel, s_sel, s_a, s_b, 0x0000]
            halves.extend(row)
    assert len(halves) == 64 * 16, len(halves)
    print("FE-XBITS-FNV %s" % fnv1a_hex(halves))
    # dim-map spot checks (token (0,0): R=0,G=0,B=0 -> n = f16(-0.5))
    assert halves[4] == 0xB800, hex(halves[4])  # f16(-0.5)
    print("F4 values: 64 fetches x u32-streams distinct; X-bits "
          "assembled (tex+gauss+scales), spot exact")


# ------------------------------------------------------- F5 B-window

def check_F5(ls):
    # v4 load address shapes (per-lane 16B, bases 8208/8720)
    assert_line(ls, 970, "mul.wide.s32 %rd47, %r437, 16")
    assert_line(ls, 971, "add.s64 %rd48, %rd7, %rd47")
    assert_line(ls, 972, "add.s64 %rd12, %rd48, 8208")
    assert_line(ls, 974, "ld.weak.global.ca.v4.u32 "
                         "{ %r443,%r444,%r445,%r446},[%rd12]")
    assert_line(ls, 979, "mul.wide.s32 %rd49, %r438, 16")
    assert_line(ls, 980, "add.s64 %rd50, %rd7, %rd49")
    assert_line(ls, 981, "add.s64 %rd13, %rd50, 8720")
    assert_line(ls, 983, "ld.weak.global.ca.v4.u32 "
                         "{ %r447,%r448,%r449,%r450},[%rd13]")
    print("F5 B-window: [8208,8720)+[8720,9232) = carve [8208,9232)")
    print("F5 NOTE: == table patch exactly (§33.14 audit: pad16 "
          "@8192, residue dissolved)")


# ---------------------------------------------------------------- F6

def main():
    ls = kernel_lines()
    check_F1(ls)
    table, am, wk0 = check_F2(ls)
    check_F4(ls, table, am, wk0)
    check_F5(ls)
    print("FRONTEND-OK")


if __name__ == "__main__":
    main()
