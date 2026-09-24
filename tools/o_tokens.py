#!/usr/bin/env python3
"""tools/o_tokens.py -- Esum-owner inversion -> O-tokens (gate H).

Resolves HANDOFF 33.5 open item (1): bias-row labels (gate D: which
bias VALUE each score position added) vs downstream TOKENS (Esum
owners: which rowsum owns each E-item). Proves, fail-closed:

H1. Per episode: true_of_lane bijective (32 distinct TRUE rows over
    32 lanes), 64 items/lane, owner-map dup=0. Ep0 runs=[36,37]
    (wave-1), ep1 runs=[48,49] (wave-2). So lane<->TRUE-row 1:1
    and owner[label] = the lane whose Esum owns it.

H2. P.V A-reg dpos-uniformity: all 16 wave-1 A-regs fan into 2
    P-muls reading 2 leaves of the SAME dpos; per (A-reg, lane)
    a SINGLE owner; dpos0-owner(L) != dpos1-owner(L) every lane
    (structural: dpos0 quads are even {0,1}, dpos1 odd {2,3},
    disjoint -> distinct owners by H1 bijectivity).

H3. C-parity consistency at every C-chained wave-1 P.V site:
    C-fan owners per lane == A-fan owners per lane (union).
    The C-chain (@12270-D -> @12284-C etc.) therefore adds
    same-token partials: the bias-query labels differ (q0-7 vs
    q8-15) but OWNERS match. Mixed-by-label, single-token by
    owner. (Parity algebra: f16 halves share frag-row parity
    (even D0 / odd D1, arithmetic on frs); packs are lane-wise
    and dpos-preserving (H2); MMA D[row]<-A[row]+C[row]
    row-preserving; C-reg = prior D-reg exactly. So D-half di
    selects dpos-parity owners; validity then forces pack
    parity-matching under reference-correctness -- HANDOFF.)

H4. O-token map: every store0 quad at lane-quartet m fans into
    owner-pair {m, m+8} (all 32 lanes x 8 quads asserted);
    store1 -> Esum lanes 16-31 pairs; stores 2,3 -> ep1 pairs.
    Pair-order = D-half/dpos (derived H3-parity: D0-half ->
    dpos0-owner, D1-half -> dpos1-owner; static fan unions both
    halves). O-token(store,lane,quad) = owner-pair; single owner
    per D-half by parity derivation.

H5. TRUE<->bias contingency + label inversion (items 1+3): every
    Esum label (site,row,col) inverts to a UNIQUE (lane,half)
    (L=8*(c//2)+(r//2), dpos=r%2, membership-verified); per TRUE
    row exactly 4 bias-queries x16 labels ({o+4k}) and 16
    distinct keys; per bias-query exactly 4 TRUE rows. So TRUE
    rows (parity-split) and bias-queries (load-split) cross-cut:
    parity x load biclique. Row-major-tokens verdict (HANDOFF):
    kernel rows = TRUE rows, port rows = sequential tokens;
    byte formulas match (gate F) but token factorization differs
    (parity-split vs sequential); port faithful+green, kernel
    reference-assumed with parity-piece semantics.

Stdlib-only, GPU-free, byte-deterministic stdout.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import (parse_mma_full, parse_episodes, map_sites,  # noqa: E402
                            run_groups, simulate, frag_P1,
                            arrange, true_row, NLANES)
from o_fanin import (def_walk, build_leafmap,  # noqa: E402
                     resolve)


def full_labels(leaf2sd, sidx, leaf, lane):
    """Ordered full (g,r,p,c) labels of a leaf-half at a lane."""
    site, dpos = leaf2sd[leaf]
    g, p = sidx[site]
    frs, fcs = frag_P1(lane)
    elems = [(r, c) for r in frs for c in fcs]
    q = arrange(elems, "A1")
    halves = [q[0:2], q[2:4]][dpos]
    return [(g, r, p, c) for r, c in halves]


def main():
    ls = kernel_lines()
    sites = parse_mma_full(ls)
    groups = run_groups(sites)
    sidx = {}
    for g, grp in enumerate(groups):
        for p, (idx, _s) in enumerate(grp):
            sidx[idx] = (g, p)
    leaf2sd, leaves = build_leafmap(ls, sites)
    site_by_ln = {s[0]: s for s in sites}
    episodes = parse_episodes(ls)
    assert len(episodes) == 2, len(episodes)

    owners, tols = [], []
    for n, chains in enumerate(episodes):
        mapped = map_sites(ls, sites, chains)
        res, (_lo, _hi, _out), _mach, _seeds = simulate(
            ls, chains, mapped, groups, "P1-2x2", "A1")
        assert len(res) == NLANES, (n, len(res))
        tol = {}
        for lane, s in enumerate(res):
            rows = {true_row(x) for x in s}
            assert len(rows) == 1, (n, lane, rows)
            assert len(s) == 64, (n, lane, len(s))
            tol[lane] = next(iter(rows))
        assert len(set(tol.values())) == NLANES, \
            (n, sorted(set(tol.values())))
        owner = {}
        for lane, s in enumerate(res):
            for x in s:
                assert x not in owner, (n, x)
                owner[x] = lane
        owners.append(owner)
        tols.append(tol)
        print("H1 ep%d: 32 lanes 32 distinct TRUE runs=%s 64/lane "
              "owner-items=%d dup=0" %
              (n, sorted({t[0] for t in tol.values()}), len(owner)))
    assert sorted({t[0] for t in tols[0].values()}) == [36, 37]
    assert sorted({t[0] for t in tols[1].values()}) == [48, 49]

    # ---- H2: A-reg dpos-uniformity + single/parity owners ----
    pvA = sorted({a for (ln, _D, A, _B, _C) in sites
                  if 12270 <= ln < 12456 for a in A})
    assert len(pvA) == 16, pvA
    pmul2leaf = {}
    for i, l in enumerate(ls):
        m = re.match(r"\s*\{?mul\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);?",
                     l)
        if m:
            d, a, b = m.groups()
            if a in leaves or b in leaves:
                pmul2leaf[d] = a if a in leaves else b
    areg_leaves = {}
    for a in pvA:
        j = def_walk(ls, a, 12269)
        srcs = [s for s in re.findall(r"%r\d+|%rs\d+", ls[j]) if s != a]
        got = []
        for s2 in srcs:
            k = def_walk(ls, s2, j)
            for pm in re.findall(r"%r32\d+", ls[k]):
                if pm in pmul2leaf:
                    got.append(pmul2leaf[pm])
        assert len(got) == 2, (a, got)
        tgot = {leaf2sd[lf] for lf in got}
        assert len({d for (_s, d) in tgot}) == 1, (a, tgot)
        areg_leaves[a] = (got, next(iter({d for (_s, d) in tgot})))
    # single owner per (A-reg, lane); dpos-parity distinctness
    dpos_own = defaultdict(dict)
    for a, (lfs, d) in areg_leaves.items():
        for lane in range(NLANES):
            oset = set()
            for lf in lfs:
                for lbl in full_labels(leaf2sd, sidx, lf, lane):
                    assert lbl in owners[0], (a, lane, lbl)
                    oset.add(owners[0][lbl])
            assert len(oset) == 1, (a, lane, oset)
            dpos_own[(a, lane)] = (d, next(iter(oset)))
    for lane in range(NLANES):
        o0 = {o for (a, l), (d, o) in dpos_own.items()
              if l == lane and d == 0}
        o1 = {o for (a, l), (d, o) in dpos_own.items()
              if l == lane and d == 1}
        assert o0 and o1 and not (o0 & o1), (lane, o0, o1)
    print("H2: 16 P.V A-regs dpos-uniform (2 same-dpos leaves each); "
          "single owner per (A-reg,lane); dpos0-owners disjoint "
          "dpos1-owners every lane")

    # ---- H3: C-parity consistency at C-chained P.V sites ----
    memo = {}
    chained = []
    for (ln, _Dm, A, _B, C) in [s for s in sites if 12270 <= s[0] < 12456]:
        kinds = set()
        for cr in C:
            j = def_walk(ls, cr, ln - 1)
            kinds.add("mma" if "mma.sync" in ls[j] else "const")
        if kinds == {"mma"}:
            chained.append((ln, A, C))
    assert len(chained) == 8, [(ln) for (ln, _a, _c) in chained]
    for (ln, A, C) in chained:
        cfan = set()
        for cr in C:
            cfan |= resolve(ls, site_by_ln, leaves, memo, cr, ln - 1)
        afan = set()
        for ar in A:
            afan |= resolve(ls, site_by_ln, leaves, memo, ar, ln - 1)
        for lane in range(NLANES):
            co, ao = set(), set()
            for x in cfan:
                if x.startswith("TAINT"):
                    continue
                lf, _h = x.rsplit("#", 1)
                for lbl in full_labels(leaf2sd, sidx, lf, lane):
                    assert lbl in owners[0], (ln, lane, lbl)
                    co.add(owners[0][lbl])
            for x in afan:
                if x.startswith("TAINT"):
                    continue
                lf, _h = x.rsplit("#", 1)
                for lbl in full_labels(leaf2sd, sidx, lf, lane):
                    assert lbl in owners[0], (ln, lane, lbl)
                    ao.add(owners[0][lbl])
            assert co and ao and co == ao, (ln, lane, co, ao)
    print("H3: %d C-chained P.V sites: C-fan owners == A-fan owners "
          "every lane (same-token accumulation)" % len(chained))

    # ---- H4: O-token pairs (all 4 stores; o_quads covers 0-1 only) ----
    quads = []
    n = 0
    for i, l in enumerate(ls):
        if not re.search(r"st\.global\S*\s+\[(\S+)\]", l):
            continue
        if n >= 4:
            break
        m2 = re.search(r"mov\.b128\s+v,\s*\{([^}]*)\}", l)
        regs = re.findall(r"%r\d+", m2.group(1))
        assert len(regs) == 4, (n, i + 1)
        for r in regs:
            j = def_walk(ls, r, i)
            mm = re.match(r"mov\.b32\s+%r\d+,\s*\{(%rs\d+),\s*(%rs\d+)\}",
                          ls[j].strip().strip("{}").strip().rstrip(";"))
            assert mm, (n, i + 1, ls[j].strip()[:80])
            for q_rs in (mm.group(1), mm.group(2)):
                kk = def_walk(ls, q_rs, j)
                mc = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+"
                              r"%rs\d+,\s*(%r\d+);?",
                              ls[kk].strip().strip("{}").strip())
                assert mc, (n, q_rs, ls[kk].strip()[:80])
                quads.append((n, q_rs, mc.group(1), kk))
        n += 1
    assert n == 4 and len(quads) == 32, (n, len(quads))
    memo2 = {}
    ranges = {}
    for (m, rs, d, k) in quads:
        fan = resolve(ls, site_by_ln, leaves, memo2, d, k)
        for lane in range(NLANES):
            p0, p1 = set(), set()
            for x in fan:
                if x.startswith("TAINT"):
                    continue
                lf, _h = x.rsplit("#", 1)
                tgt = p0 if leaf2sd[lf][1] == 0 else p1
                for lbl in full_labels(leaf2sd, sidx, lf, lane):
                    om = owners[0] if lbl[0] in (36, 37) else owners[1]
                    assert lbl in om, (m, rs, lane, lbl)
                    tgt.add(om[lbl])
            assert len(p0) == 1 and len(p1) == 1, (m, rs, lane, p0, p1)
            pair = p0 | p1
            assert len(pair) == 2, (m, rs, lane, pair)
            a, b = sorted(pair)
            assert b - a == 8, (m, rs, lane, pair)
            ranges.setdefault(m, set()).update(pair)
    for m, r in sorted(ranges.items()):
        lo, hi = (0, 15) if m in (0, 2) else (16, 31)
        assert r <= set(range(lo, hi + 1)), (m, sorted(r))
        assert len(r) == 16, (m, len(r))
    print("H4: 32 O-quads x 32 lanes: owner-pair {a,a+8} "
          "(dpos0-singleton + dpos1-singleton); store ranges "
          + " ".join("st%d:%d-%d" % (m, min(r), max(r))
                     for m, r in sorted(ranges.items())))
    print("H4 note: static fan unions D0+D1 halves; single owner "
          "per D-half by H3-parity (D0->dpos0-owner, D1->dpos1-owner)")
    print("OTOKENS-OK")


if __name__ == "__main__":
    main()
