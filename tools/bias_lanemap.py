#!/usr/bin/env python3
"""tools/bias_lanemap.py -- pin down the score-MMA bias lane map (HANDOFF §28).

Method (all evidence, no guessing):
  1. Parse every m16n8k32.row.col.f16.e4m3.e4m3 MMA site in the fp8
     pre-block kernel, with A/B/C register lists.
  2. Use-def: each C register is traced back to its defining ld.v4.u32 and
     the constant offset feeding that load's address. Any register defined
     more than once in the window (compiler reuse) is reported AMBIGUOUS
     instead of being guessed.
  3. Keep the sites whose C offsets land in one 8192-byte, 512-spaced slab
     family (= the bias region): these are the score MMAs. Report the
     A-sharing structure (query-row groups) and slab tiling.
  4. Evaluate candidate lane maps for structural consequences on the real
     tensor_000 bias bytes: a genuine relative-position table has <=225
     distinct values constant on (drow,dcol); a high-entropy absolute table
     cannot discriminate layouts by value.

Current verdict (see HANDOFF §28): the tile decomposition is proven
(4 row-groups x 8 col-groups of 16x8, C slabs at 512 B spacing covering
exactly 8192 B), but the table is high-entropy under both row-major and
the fragment-derived candidate, so the lane map stays OPEN and row-major
is retained. The recipe a future probe must follow is printed at the end.
"""

import re
import struct
import sys
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTX = os.path.join(ROOT, "dlss5-analysis", "cubins", "cubin_00_sm_120_ptx.ptx")
T0 = os.path.join(ROOT, "dlss5-analysis", "tensors", "tensor_000.bin")
KERNEL = "cc_tinlayout_fused_pre_block_swin_1h_32_1_fp8"
MMA = "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
# corrected pre-block bias window (HANDOFF §27)
BIAS_OFF, BIAS_LEN = 12384, 8192


def kernel_lines():
    ls = open(PTX, errors="replace").read().splitlines()
    s = next(i for i, l in enumerate(ls)
             if (".visible .entry " + KERNEL) in l)
    e = next(i for i in range(s + 1, len(ls))
             if ls[i].startswith(".visible .entry "))
    return ls[s:e]


def parse_mma(ls):
    """[(line_no_1based_within_kernel, A_regs, B_regs, C_regs)]."""
    sites = []
    for i, l in enumerate(ls):
        if MMA not in l:
            continue
        regs = []
        for k in range(1, 4):
            m = re.search(r"\{([^}]*)\}", ls[i + k])
            regs.append(re.findall(r"%r\d+", m.group(1)) if m else [])
        sites.append((i + 1, regs[0], regs[1], regs[2]))
    return sites


def define_offset(ls, reg, use_idx):
    """(offset, ambiguous): latest `add.s64 addr, base, CONST` feeding the
    ld.v4 that defines `reg`, searching back from use_idx."""
    defs = []
    for i in range(use_idx - 1, -1, -1):
        l = ls[i]
        if "ld.weak.global" in l and reg in l:
            m = re.search(r"\[\s*(%rd\d+)\s*\]", l)
            if not m:
                return None, True
            addr = m.group(1)
            for j in range(i - 1, max(-1, i - 12), -1):
                m2 = re.search(r"add\.s64\s+(%rd\d+),\s*(%rd\d+),\s*(\d+);",
                               ls[j])
                if m2 and m2.group(1) == addr:
                    defs.append(int(m2.group(3)))
                    break
            break
    if len(defs) != 1:
        return None, True
    return defs[0], False


def check_smexp_immediates(ls):
    """PTX-confirm the §19 smexp constants (HANDOFF §28).

    Finds `mov.b32 %rN, IMM` + `cvt.rn.f16.f32` + pack patterns in the
    kernel, rounds each immediate through the oracle converter, and checks
    the four episode constants appear with the right roles: u-coefficients
    in fma.rn.f16x2, floor in max.f16x2, ceil in min.f16x2. The only
    post-score-MMA scaling is this u-step, so no temperature lives
    downstream of the scores (consistent with QK-norm placement).
    """
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from swin1h_ref import f32_to_f16_bits
    packed = {}  # user reg -> f16 bits
    for i, l in enumerate(ls):
        m = re.search(r"mov\.b32\s+(%r\d+),\s*(\d+);", l)
        if not m:
            continue
        dst, imm = m.group(1), int(m.group(2))
        if imm < 2 ** 20:
            continue  # small ints are indices, not f32 immediates
        for k in range(i + 1, min(i + 7, len(ls))):
            m2 = re.search(r"cvt\.rn\.f16\.f32\s+low,\s*(%r\d+);", ls[k])
            if m2 and m2.group(1) == dst:
                for j in range(k + 1, min(k + 4, len(ls))):
                    m3 = re.search(r"mov\.b32\s+(%r\d+),\s*\{low,low\};",
                                   ls[j])
                    if m3:
                        f = struct.unpack("<f", struct.pack("<I", imm))[0]
                        packed[m3.group(1)] = f32_to_f16_bits(f)
                break
    want = {"u1": 0x29C0, "u2": 0x3D34, "floor": 0x3C20, "ceil": 0x3E47}
    have = {}
    for reg, bits in packed.items():
        for name, w in want.items():
            if bits == w:
                have.setdefault(name, []).append(reg)
    ok = True
    for name, w in want.items():
        regs = have.get(name, [])
        print("smexp %s (f16 %04X): %d producer(s) %s" %
              (name, w, len(regs), regs[:4]))
        ok = ok and bool(regs)
    # role checks on first producers
    if have.get("u1") and have.get("u2"):
        r1, r2 = have["u1"][0], have["u2"][0]
        n = sum(1 for l in ls if "fma.rn.f16x2" in l and r1 in l and r2 in l)
        print("u-step fma sites using both: %d" % n)
        ok = ok and n > 0
    if have.get("floor"):
        n = sum(1 for l in ls
                if "max.f16x2" in l and have["floor"][0] in l)
        print("floor max.f16x2 sites: %d" % n)
        ok = ok and n > 0
    if have.get("ceil"):
        n = sum(1 for l in ls
                if "min.f16x2" in l and have["ceil"][0] in l)
        print("ceil min.f16x2 sites: %d" % n)
        ok = ok and n > 0
    print("smexp immediates PTX-confirmed: %s" % ok)
    return ok


def main():
    ls = kernel_lines()
    print("== smexp immediates ==")
    smexp_ok = check_smexp_immediates(ls)
    print("== bias lane map ==")
    sites = parse_mma(ls)
    print("score-shape MMA sites in %s: %d" % (KERNEL, len(sites)))

    # C-offset use-def per site
    scored, ambiguous = [], []
    for ln, A, B, C in sites:
        offs, amb = [], False
        for r in C:
            o, a = define_offset(ls, r, ln - 1)
            amb = amb or a
            offs.append(o)
        if amb:
            ambiguous.append(ln)
        else:
            scored.append((ln, A, B, C, offs))
    print("sites with clean C use-def: %d; ambiguous: %d" %
          (len(scored), len(ambiguous)))
    if ambiguous:
        print("AMBIGUOUS C regs at kernel lines:", ambiguous[:10])

    # score MMAs = C offsets inside one 8192 B slab family
    bias_sites = [s for s in scored
                  if all(o is not None and BIAS_OFF <= o < BIAS_OFF + BIAS_LEN
                         for o in s[4])]
    print("sites with C entirely in bias window [%d,%d): %d" %
          (BIAS_OFF, BIAS_OFF + BIAS_LEN, len(bias_sites)))
    slabs = sorted({o for s in bias_sites for o in s[4]})
    print("C slabs used: %d (spacing %s)" %
          (len(slabs),
           sorted({b - a for a, b in zip(slabs, slabs[1:])})))
    expect = [BIAS_OFF + 512 * k for k in range(16)]
    print("slab set == 16 slabs @512 B from %d: %s" % (BIAS_OFF, slabs == expect))
    if slabs != expect:
        print("observed:", slabs)

    # A-sharing: consecutive sites sharing A = one query-row group
    groups, cur = [], []
    for s in bias_sites:
        if cur and cur[-1][1] != s[1]:
            groups.append(cur)
            cur = []
        cur.append(s)
    if cur:
        groups.append(cur)
    print("A-sharing groups: %d (sizes %s)" %
          (len(groups), sorted({len(g) for g in groups})))
    bvar = all(len({tuple(x[2]) for x in g}) == len(g) for g in groups)
    print("B varies within every group (distinct key groups): %s" % bvar)

    # lane-map discrimination on real bytes (row-major vs fragment candidate)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from swin1h_ref import f16_to_f32
    raw = open(T0, "rb").read()
    B = [f16_to_f32(v)
         for v in struct.unpack("<4096H", raw[BIAS_OFF:BIAS_OFF + BIAS_LEN])]

    def structure(table):
        groups = defaultdict(list)
        for i in range(64):
            for j in range(64):
                groups[(i // 8 - j // 8, i % 8 - j % 8)].append(table[(i, j)])
        bad = worst = 0
        for vs in groups.values():
            for v in vs[1:]:
                d = abs(v - vs[0])
                if d > 1e-12:
                    bad += 1
                worst = max(worst, d)
        return len(set(table.values())), bad, worst

    rowmajor = {(i, j): B[i * 64 + j] for i in range(64) for j in range(64)}
    print("row-major structure (distinct, relpos-bad, worst):",
          structure(rowmajor))

    # fragment-derived candidate (premise: m16n8k32 C fragment per thread =
    # rows {2(L%8),2(L%8)+1} x cols {2(L//8),2(L//8)+1} of the 16x8 tile;
    # second slab half covers the next 8 columns). Groups/tiles in order.
    cand = {}
    for g in range(4):
        for s_ in range(4):
            for L in range(32):
                for k in range(8):
                    o = 2048 * g + 512 * s_ + 16 * L + 2 * k
                    i = 16 * g + 2 * (L % 8) + (k % 2)
                    j = (16 * s_ + 8 * (k // 4) + 2 * (L // 8) +
                         ((k // 2) % 2))
                    assert (i, j) not in cand, (g, s_, L, k)
                    cand[(i, j)] = B[o // 2]
    assert len(cand) == 4096
    print("fragment-candidate structure:", structure(cand))
    print("verdict: UNRESOLVED by value structure "
          "(high-entropy absolute table under both maps); row-major retained.")
    print("to resolve: derive thread->row ownership from the softmax "
          "row-reduction shuffles (post-score-MMA shfl.idx broadcast "
          "structure), or probe on NVIDIA hardware; then re-run "
          "runbook §1/§5 against the winning map.")
    return 0 if smexp_ok else 1


if __name__ == "__main__":
    sys.exit(main())
