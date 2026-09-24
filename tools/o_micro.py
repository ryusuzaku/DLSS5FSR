#!/usr/bin/env python3
"""tools/o_micro.py -- frag-row micro-geometry oracle (gate M).

Closes HANDOFF 33.5 open item (2): which token per (lane,frag-row),
i.e. the sub-quad slot order. Method: premise-scoring through the
Esum validity machinery (no new premises invented: P1/P2/P3 x
A1/A2/A3 from esum_ownership), plus a trace-grounded negative
result explaining why the Stage-1 tex walk CANNOT be the row
oracle (and what it is instead).

M1. Baseline: P1A1-valid on BOTH episodes (rows==1 every lane,
    64 items/lane, 32 distinct TRUE rows, owner dup=0) -- the H1
    checks, recomputed here for gate independence.
M2. Alternatives P1A2/P1A3/P2A1/P3A1 on ep0 (wave-1): each must
    FAIL >=1 H1 check (exclusion needs only one failing episode).
    Fail-closed: a passing alternative fails this gate loudly
    (ambiguity would be a finding, not a pass).
M3. Half-combinatorics (pure, no trace): under the forced A1,
    every lane's halves pair same-row/same-colhalf elements, and
    both slots of a half share one TRUE row. With H5's single
    owner per D-half: sub-quad TOKEN order == half order (PROVEN);
    within-half order is dim order (token-irrelevant).
M4. Tex negative (trace-grounded): QK-site B-side (K tile) fans
    are lane-uniform (all 32 lanes x all members -> ONE identical
    TEX+LOAD union: broadcast tile, lanes split in hardware
    layout only) and tex coords are int-machine-invisible (float
    chain) -> tex gives tile IDENTITY, never (lane,row). The
    HANDOFF-prescribed "tex per (lane,row)" oracle is therefore
    replaced by M1/M2 (decisive) with tex as corroboration.
M4b. V-side intra-lane order: cvt f16x2 pairing re-asserted (which
    halves travel together); which peer element lands lo vs hi is
    NOT statically decidable (hardware-layout axiom) and REMAINS
    OPEN -- value-irrelevant for the port (port recomputes V;
    kernel-internal routing never compared, HANDOFF 33.5).

Stdlib-only, GPU-free, byte-deterministic stdout. Slowest gate
(~10 min: 6 Esum simulates + 1 addr-machine run); runbook 14.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import (parse_mma_full, parse_episodes, map_sites,  # noqa: E402
                            run_groups, simulate, frag_P1,
                            arrange, true_row, NLANES)
from qk_addrs import (AddrMachine, RowWalker, qk_def_walk,  # noqa: E402
                      parse_mma_all)

ALTS = ["P1A2", "P1A3", "P2A1", "P3A1"]


def check_episode(ls, chains, mapped, groups, premise, arrangement):
    """(ok, detail): H1 checks without asserting (exclusion-safe)."""
    try:
        res, (_lo, _hi, _out), _mach, _seeds = simulate(
            ls, chains, mapped, groups, premise, arrangement)
    except Exception as e:  # noqa: BLE001 -- exclusion path reports
        return False, "EXC:%s:%s" % (type(e).__name__, str(e)[:60])
    rows1 = sum(1 for s in res if len({true_row(x) for x in s}) == 1)
    len64 = sum(1 for s in res if len(s) == 64)
    tol = {}
    dup = 0
    owner = {}
    for lane, s in enumerate(res):
        rows = {true_row(x) for x in s}
        if len(rows) == 1:
            tol[lane] = next(iter(rows))
        for x in s:
            if x in owner:
                dup += 1
            else:
                owner[x] = lane
    bij = len(set(tol.values())) == NLANES and len(tol) == NLANES
    ok = rows1 == 32 and len64 == 32 and bij and dup == 0
    return ok, ("rows1=%d/32 len64=%d/32 bij=%s dup=%d" %
                (rows1, len64, bij, dup))


def main():
    ls = kernel_lines()
    sites = parse_mma_full(ls)
    groups = run_groups(sites)
    episodes = parse_episodes(ls)
    assert len(episodes) == 2, len(episodes)
    mapped_eps = [map_sites(ls, sites, ch) for ch in episodes]

    # ---- M1: P1A1 baseline, both episodes ----
    for n in (0, 1):
        ok, det = check_episode(ls, episodes[n], mapped_eps[n], groups,
                                "P1-2x2", "A1")
        assert ok, ("P1A1 baseline failed", n, det)
        print("M1 ep%d P1A1: %s -> VALID" % (n, det))

    # ---- M2: alternatives excluded on ep0 ----
    for alt in ALTS:
        prem = {"P1": "P1-2x2", "P2": "P2-1x4", "P3": "P3-4x1"}[alt[:2]]
        ok, det = check_episode(ls, episodes[0], mapped_eps[0], groups,
                                prem, alt[2:])
        assert not ok, ("alternative PASSES (ambiguity)", alt, det)
        print("M2 %s ep0: %s -> EXCLUDED" % (alt, det))

    # ---- M3: A1 half-combinatorics (pure) ----
    same = 0
    for lane in range(NLANES):
        frs, fcs = frag_P1(lane)
        elems = [(r, c) for r in frs for c in fcs]
        q = arrange(elems, "A1")
        for h in (q[0:2], q[2:4]):
            assert h[0][0] == h[1][0], (lane, h)  # same frag-row
            assert h[0][1] // 2 == h[1][1] // 2, (lane, h)  # same colhalf
            g = p = 0
            assert true_row((g, h[0][0], p, h[0][1])) == \
                true_row((g, h[1][0], p, h[1][1])), (lane, h)
            same += 1
    assert same == 64, same
    print("M3 half-combinatorics: 64/64 A1-halves same-row+same-colhalf, "
          "one TRUE row per half")
    print("M3 verdict: A1-row-grouping FORCED (sole valid premise); "
          "sub-quad token order == half order (H5); within-half order "
          "is dim order (token-irrelevant)")

    # ---- M4: tex negative (trace-grounded) ----
    am = AddrMachine()
    try:
        am.run(ls, 0, 12000)
        print("M4 addr-machine: clean to 12000")
    except (KeyError, AssertionError) as e:
        print("M4 addr-machine stopped:", str(e)[:120])
    wk = RowWalker(ls, parse_mma_all(ls), am)
    wk.walkC = False
    qk0 = min(s for s in sites if 10839 <= s[0] <= 10944)
    ln, _D, _A, B, _C = qk0
    assert len(B) == 2, B
    fans = set()
    nwalk = 0
    for lane in range(NLANES):
        for b in B:
            j = qk_def_walk(ls, b, ln - 1)
            m = re.match(r"mov\.b32\s+%r\d+,\s*\{([^}]*)\}$",
                         ls[j].strip().strip("{}").strip().rstrip(";"))
            members = re.findall(r"%rs\d+", m.group(1)) if m else [b]
            assert len(members) == 2, (b, j + 1)
            for mb in members:
                fans.add(frozenset(wk.walk(mb, j, lane)))
                nwalk += 1
    assert nwalk == 128, nwalk
    assert len(fans) == 1, [sorted(f) for f in fans]
    fan = next(iter(fans))
    kinds = sorted({it[0] for it in fan})
    assert "TAINT" not in kinds, sorted(fan)
    assert "TEX" in kinds and "LOAD" in kinds, kinds
    ntex = sum(1 for it in fan if it[0] == "TEX")
    print("M4 site%d B-side: %d walks -> ONE fan %s (broadcast K tile; "
          "lanes split in hardware layout only)" % (ln, nwalk, kinds))
    for it in sorted(fan):
        if it[0] != "TEX":
            continue
        _k, tob, pos = it
        cands = [(j, cr) for (j, p), (o, cr) in wk.texcoords.items()
                 if o == tob and p == pos]
        assert len(cands) == 1, (it, cands)
        j, cr = cands[0]
        invis = [c for c in cr
                 for lane in range(NLANES)
                 if am.v[lane].get(c) is None]
        assert len(invis) == len(cr) * NLANES, (it, cr)
        print("M4 tex@%d pos%d coords=%s int-invisible all 32 lanes "
              "(float chain) -> tile identity only, never (lane,row)"
              % (j + 1, pos, cr))
    print("M4 verdict: tex-walk CANNOT resolve frag-rows (lane-uniform "
          "fans + unevaluated coords); M1/M2 premise-scoring is the "
          "decisive oracle (tex corroborates tile identity)")
    print("M4 piped-fallbacks: %d (deterministic count)" % len(wk.piped))

    # ---- M4b: V-side intra-lane order (pairing proven, order open) ----
    mm_dests = set()
    for i, l in enumerate(ls):
        m = re.search(r"movmatrix\.\S+\s+(%r\d+),\s*(%r\d+)", l)
        if m:
            mm_dests.add(m.group(1))
    vcvts = []
    for i, l in enumerate(ls):
        m = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(%rs\d+),\s*"
                     r"(%r\d+);?$", l.strip())
        if m and m.group(2) in mm_dests:
            vcvts.append((i + 1, m.group(1), m.group(2)))
    assert len(vcvts) == 32, len(vcvts)
    print("M4b V-side: 32 cvt f16x2 pairs re-asserted (which halves "
          "travel together is trace-fixed)")
    print("M4b verdict: which peer element lands lo vs hi is NOT "
          "statically decidable (every walk unions the pair) -> "
          "REMAINS-OPEN hardware-layout axiom; value-irrelevant for "
          "the port (port recomputes V; kernel routing never "
          "compared, HANDOFF 33.5)")

    print("MMICRO-OK")


if __name__ == "__main__":
    main()
