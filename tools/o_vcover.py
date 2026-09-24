#!/usr/bin/env python3
"""tools/o_vcover.py -- movmatrix V-cover bijectivity (gate V).

The 32 `movmatrix.sync.trans.aligned.m8n8.b16` sites transpose the
computed V tile (V-proj MMA-D) feeding O-MMA B (r4690+). qk_addrs.py
holds a CANDIDATE octet-transpose peer rule for these sites; this gate
validates it and proves the cover bijective at trace granularity.

V1. Census: exactly 32 movmatrix sites, all
    movmatrix.sync.trans.aligned.m8n8.b16, unpredicated (full warp),
    dests r2848-2879 consecutive, srcs r2816-2847 consecutive.
V2. Peer-rule combinatorics (pure, no trace): every consumer lane reads
    exactly 2 DISTINCT source lanes in range; every source lane is read
    by exactly 2 consumers (uniform 2-cover: no row dropped or
    triple-read); consumer octet g gathers one pair from each source
    octet at intra-octet offset 2g (transpose-shaped gather/scatter).
    Lane-level necessity for bijection; intra-lane element order stays
    with the frag-row micro-geometry open item.
V3. Source cover: all 32 srcs def to 16 distinct m16n8k32 MMA-D sites
    x exactly 2 regs, each D brace FULLY consumed (fed == Dregs, width
    2, no leftovers).
V4. Dest cover: all 32 dests quantize 1:1 through
    cvt.rn.satfinite.e4m3x2.f16x2 into %rs280-311 consecutive, packed by
    16 mov.b32 into the 16 wave-1 O-site B regs (sites 12270-12319).
V5. Consumers + boundary: O-site groups sharing the tile-0 B set
    (12270/12326/14082 groups); each B reg's nearest preceding def is
    its pack (def_walk, register-reuse-safe); no def-shape redef of the
    tile-0 B regs before the last consumer group. Other O-group B tiles
    (r3322+ ld.global, r4289+ QK-era, r4757+ ld.global) are disjoint
    from tile 0 and movmatrix-fed nowhere: enumerated, not chased.

Stdlib-only, GPU-free, byte-deterministic stdout.
"""

import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import parse_mma_full, nearest_def  # noqa: E402
from o_fanin import def_walk  # noqa: E402

NLANES = 32
MM_OP = "movmatrix.sync.trans.aligned.m8n8.b16"
CVT_OP = "cvt.rn.satfinite.e4m3x2.f16x2"


def mm_peers(lane):
    """Candidate octet-transpose peer rule (same formula as qk_addrs.py:
    consumer octet g=L//8, intra-octet i=L%8 reads source pair
    8*(i//2)+2*g, +1)."""
    g, i = divmod(lane, 8)
    b = 8 * (i // 2) + 2 * g
    return (b, b + 1)


def main():
    ls = kernel_lines()

    # ---- V1: census ----
    mm = []  # (ln, dest, src)
    for i, l in enumerate(ls):
        s = l.strip()
        if "movmatrix" not in s:
            continue
        assert s.startswith("movmatrix"), ("predicated/other", i + 1, s[:60])
        m = re.match(r"movmatrix\.sync\.trans\.aligned\.m8n8\.b16\s+"
                     r"(%r\d+),\s*(%r\d+);?$", s)
        assert m, ("shape", i + 1, s[:80])
        mm.append((i + 1, m.group(1), m.group(2)))
    assert len(mm) == 32, len(mm)
    dests = sorted((d for _, d, _ in mm), key=lambda r: int(r[2:]))
    srcs = sorted((s for _, _, s in mm), key=lambda r: int(r[2:]))
    assert [int(r[2:]) for r in dests] == list(range(2848, 2880)), dests
    assert [int(r[2:]) for r in srcs] == list(range(2816, 2848)), srcs
    print("V1 census: 32x %s unpredicated span@%d-@%d "
          "dest=r2848-2879 src=r2816-2847" % (MM_OP, mm[0][0], mm[-1][0]))

    # ---- V2: peer-rule combinatorics (pure) ----
    peers = {L: mm_peers(L) for L in range(NLANES)}
    assert all(len(set(p)) == 2 and all(0 <= x < NLANES for x in p)
               for p in peers.values())
    reads = Counter(s for p in peers.values() for s in p)
    assert sorted(reads.values()) == [2] * NLANES, sorted(reads.values())
    gathers = {}
    for g in range(4):
        u = sorted({s for i in range(8) for s in peers[8 * g + i]})
        assert u == sorted(8 * k + 2 * g + t for k in range(4)
                           for t in (0, 1)), (g, u)
        gathers[g] = u
    print("V2 peer-rule: 32 lanes x2 distinct peers; source-reads "
          "uniform x2 (no drop/triple)")
    for g in range(4):
        print("   octet-%d gathers %s" % (g, gathers[g]))

    # ---- V3: source cover -> MMA-D ----
    site_srcs = {}
    for ln, _d, s in mm:
        j = nearest_def(ls, s, ln - 1)
        assert j is not None, ("nodef", s)
        l = ls[j].strip()
        m = re.match(r"(mma\.sync\.aligned\.m16n8k32\.\S+)\s+\{([^}]*)\},?",
                     l)
        assert m and s in re.findall(r"%r\d+", m.group(2)), (s, j + 1,
                                                             l[:80])
        site_srcs.setdefault(j + 1, []).append(s)
    assert len(site_srcs) == 16, len(site_srcs)
    for site, fed in sorted(site_srcs.items()):
        l = ls[site - 1].strip()
        dregs = re.findall(r"%r\d+",
                           re.match(r"mma\.sync\S*\s+\{([^}]*)\},?",
                                    l).group(1))
        assert sorted(fed) == sorted(dregs) and len(dregs) == 2, \
            (site, fed, dregs)
    print("V3 source-cover: 32 srcs -> 16 m16n8k32 MMA-D sites x2 regs, "
          "D-braces fully consumed "
          "(sites@%d-@%d)" % (min(site_srcs), max(site_srcs)))

    # ---- V4: dest cover -> quant -> packs -> wave-1 B ----
    cvts = []  # (ln, rs, rsrc)
    for i, l in enumerate(ls):
        m = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(%rs\d+),\s*"
                     r"(%r\d+);?$", l.strip())
        if m:
            cvts.append((i + 1, m.group(1), m.group(2)))
    mmdest_set = set(dests)
    cvts = [c for c in cvts if c[2] in mmdest_set]
    assert len(cvts) == 32, len(cvts)
    rs_set = sorted((c[1] for c in cvts), key=lambda r: int(r[3:]))
    assert [int(r[3:]) for r in rs_set] == list(range(280, 312)), rs_set
    assert sorted((c[2] for c in cvts)) == dests, "cvt-src != mm-dest"
    packs = []  # (ln, breg, rs0, rs1)
    rs_cover = Counter()
    for i, l in enumerate(ls):
        m = re.match(r"mov\.b32\s+(%r\d+),\s*\{(%rs\d+),\s*(%rs\d+)\};?$",
                     l.strip().strip("{}").strip())
        if m and m.group(2) in rs_set:
            assert m.group(3) in rs_set, (i + 1, l.strip()[:60])
            packs.append((i + 1, m.group(1), m.group(2), m.group(3)))
            rs_cover[m.group(2)] += 1
            rs_cover[m.group(3)] += 1
    assert len(packs) == 16, len(packs)
    assert sorted(rs_cover.values()) == [1] * 32, sorted(rs_cover.values())
    bregs = sorted((p[1] for p in packs), key=lambda r: int(r[2:]))
    sites = parse_mma_full(ls)
    w1 = [(ln, D, A, B, C) for (ln, D, A, B, C) in sites
          if 12270 <= ln <= 12319]
    assert len(w1) == 8, len(w1)
    w1b = sorted((b for (_ln, _D, _A, B, _C) in w1 for b in B),
                 key=lambda r: int(r[2:]))
    assert w1b == bregs, (w1b, bregs)
    print("V4 dest-cover: 32 dests -> 32 %s -> %%rs280-311 x1 each -> "
          "16 packs@%d-@%d -> wave-1 B (8 sites 12270-12319)" %
          (CVT_OP, packs[0][0], packs[-1][0]))

    # ---- V5: consumer groups + no-redef + boundary tiles ----
    tile0 = frozenset(bregs)
    consumers, groups = [], {}
    for (ln, _D, _A, B, _C) in sites:
        if ln >= 12270:
            if set(B) <= tile0:
                consumers.append(ln)
            else:
                groups.setdefault(frozenset(B), []).append(ln)
    t0groups = sorted(consumers)
    # exact consumer-site partition for tile 0: 8 + 8 + 16
    assert t0groups == [12270, 12277, 12284, 12291, 12298, 12305, 12312,
                        12319, 12326, 12333, 12340, 12347, 12354, 12361,
                        12368, 12375, 14082, 14089, 14096, 14103, 14110,
                        14117, 14124, 14131, 14138, 14145, 14152, 14159,
                        14166, 14173, 14180, 14187], t0groups
    packln = {p[1]: p[0] for p in packs}
    for b in bregs:
        assert def_walk(ls, b, 12270 - 1) == packln[b] - 1, b
    defshape = re.compile(r"(?:mov|cvt|prmt|mul|mad|fma|add|sub|cvta|ld)"
                          r"\.\S+\s+(\{[^}]*\}|%r\d+|%rs\d+)")
    mmashape = re.compile(r"mma\.sync\S*\s+\{([^}]*)\},?")
    for b in bregs:
        for ln in range(packs[-1][0] + 1, 14188):
            l = ls[ln - 1]
            s = l.strip().strip("{}").strip()
            m = defshape.match(s)
            if m and b in re.findall(r"%r\d+", m.group(1)):
                raise AssertionError(("redef", b, ln, l.strip()[:80]))
            m2 = mmashape.match(l.strip())
            if m2 and b in re.findall(r"%r\d+", m2.group(1)):
                raise AssertionError(("mma-redef", b, ln))
    print("V5 consumers: tile-0 B shared by 32 O-sites "
          "(12270-12375 x16 + 14082-14187 x16); nearest-def=packs; "
          "no redef to @14187")
    # Boundary tiles: per-pair first-defs, merged into maximal
    # contiguous-B runs (one line per tile; reporting only).
    pairdefs = []
    for bset, lns in sorted(groups.items(), key=lambda x: min(x[1])):
        bset = sorted(bset, key=lambda r: int(r[2:]))
        j = nearest_def(ls, bset[0], lns[0] - 1)
        while j is not None:
            s = ls[j].strip().strip("{}").strip()
            if defshape.match(s) or mmashape.match(ls[j].strip()):
                break
            j = nearest_def(ls, bset[0], j)
        assert j is not None, ("nodef", bset[0])
        assert not any(r in mmdests_all(ls) for r in bset), (bset, lns[0])
        pairdefs.append((bset, lns, j + 1, ls[j].strip()))
    runs = []
    for bset, lns, defln, dtxt in pairdefs:
        bmin, bmax, op = int(bset[0][2:]), int(bset[-1][2:]), dtxt.split()[0]
        if runs and (defln in runs[-1][3] or
                     (set(runs[-1][4]) == {op} and
                      bmin - runs[-1][1] <= 8)):
            runs[-1][1] = bmax
            runs[-1][2] += len(lns)
            runs[-1][3].append(defln)
            runs[-1][4].append(op)
        else:
            runs.append([bset[0], bmax, len(lns), [defln], [op]])
    for b0, b1, nsites, deflns, ops in runs:
        assert len(set(ops)) == 1, (b0, ops)
        print("   boundary tile B=%s..%%r%d %d sites defs@%s: %s" %
              (b0, b1, nsites,
               ",".join(str(d) for d in sorted(set(deflns))), ops[0]))
    print("VCOVER-OK")


def mmdests_all(ls):
    """All movmatrix dest regs kernel-wide (boundary tiles must avoid)."""
    out = set()
    for l in ls:
        m = re.search(r"movmatrix\.\S+\s+(%r\d+),\s*(%r\d+)", l)
        if m:
            out.add(m.group(1))
    return out


if __name__ == "__main__":
    main()
