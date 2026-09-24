#!/usr/bin/env python3
"""tools/o_fanin.py -- TRUE row -> query-row mapping via the O path.

Method: backward walk from the output stores' e4m3 values through the
quantizer into O-MMA collector D halves, following A (quantized P) and
C (split-K accumulation) links to score-episode leaves. Each leaf half
has a proven TRUE row at every lane (esum_ownership, P1+A1). If all
leaves fanning into one collector half share a single TRUE row at lane
L, that TRUE row IS the query row lane L stores there -- and the store
address gives the tile-relative query-row offset
(slab*16 + lane//2, e4m3 32-col rows). Fail-closed: unknown producer
ops abort; multi-row fan-ins are reported, never forced.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import (parse_mma_full, parse_episodes, map_sites,  # noqa: E402
                            nearest_def, run_groups, true_row,
                            frag_P1, arrange)

NLANES = 32


def def_walk(ls, reg, use):
    """Nearest preceding line DEFINING `reg` (dest position), skipping
    pure uses (MMA operand braces, source operands). Fail-closed."""
    j = use
    while True:
        j = nearest_def(ls, reg, j)
        assert j is not None, ("no def", reg, use)
        l = ls[j].strip().strip("{}").strip().rstrip(";")
        m = re.match(r"(?:mov|and|or|xor|add|sub|shl|shr|selp|cvt|cvta|"
                     r"mul|mad|div|rem|fma|max|min|abs|neg|sqrt|rsqrt|rcp|sin|cos|"
                     r"ex2|lg2|setp|ld|shfl|prmt|tex|movmatrix)"
                     r"\.[a-z0-9.]+\s+"
                     r"(\{[^}]*\}|%rd\d+|%r\d+|%rs\d+|[a-z]\w*),?", l)
        if m:
            d = m.group(1)
            if d.startswith("{"):
                if reg in re.findall(r"%rd\d+|%r\d+|%rs\d+|[a-z]\w*", d):
                    return j  # pack defines its members
            elif d == reg:
                return j
        # An MMA mnemonic line defines its D brace (first group).
        m = re.match(r"mma\.sync\S*\s+\{([^}]*)\},?", ls[j].strip())
        if m and reg in re.findall(r"%r\d+", m.group(1)):
            return j
        # Otherwise a pure use (MMA brace operand, source position):
        # keep searching backward (nearest_def scans strictly before j).


def def_walk_fwd(ls, reg, use):
    """Nearest FOLLOWING line defining `reg` (software-pipeline carried
    regs: used at the loop head before their body def). Same def-shape
    recognition as def_walk. Fail-closed."""
    want = re.compile(r"(?:^|[{\s,;])" + re.escape(reg) + r"(?:$|[\s,}])")
    for j in range(use + 1, len(ls)):
        if not want.search(ls[j]):
            continue
        l = ls[j].strip().strip("{}").strip().rstrip(";")
        m = re.match(r"(?:mov|and|or|xor|add|sub|shl|shr|selp|cvt|cvta|"
                     r"mul|mad|div|rem|fma|max|min|abs|neg|sqrt|rsqrt|rcp|sin|cos|"
                     r"ex2|lg2|setp|ld|shfl|prmt|tex|movmatrix)"
                     r"\.[a-z0-9.]+\s+"
                     r"(\{[^}]*\}|%rd\d+|%r\d+|%rs\d+|[a-z]\w*),?", l)
        if m:
            d = m.group(1)
            if d.startswith("{"):
                if reg in re.findall(r"%rd\d+|%r\d+|%rs\d+|[a-z]\w*", d):
                    return j
            elif d == reg:
                return j
        m = re.match(r"mma\.sync\S*\s+\{([^}]*)\},?", ls[j].strip())
        if m and reg in re.findall(r"%r\d+", m.group(1)):
            return j
    assert False, ("no fwd def", reg, use)


def is_leaf_def(ls, leaves, reg, use):
    """True iff `reg`'s nearest preceding def is a score-leaf magic-add
    for a known leaf (def-shape, never name-only: registers are reused)."""
    j = def_walk(ls, reg, use)
    l = ls[j].strip().strip("{}").strip().rstrip(";")
    m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*2146992128$", l)
    return bool(m and m.group(1) == reg and reg in leaves)


def build_leafmap(ls, sites):
    """(leaf2sd, leaves) from the Esum episodes (module-level so other
    tools can reuse the identical map)."""
    eps = parse_episodes(ls)
    leaf2sd, leaves = {}, set()
    for ep in eps:
        for _E, _D, lf, _ln, site, dpos in map_sites(ls, sites, ep):
            assert lf not in leaf2sd, ("leaf reuse", lf)
            leaf2sd[lf] = (site, dpos)
            leaves.add(lf)
    return leaf2sd, leaves


def resolve(ls, site_by_ln, leaves, memo, reg, use):
    """Frozenset of score-leaves feeding `reg` (defined before `use`).
    Module-level (same logic as the original main() closure) so other
    tools can reuse the identical fan-in."""
    key = (reg, use)
    if key in memo:
        return memo[key]
    j = def_walk(ls, reg, use)
    l = ls[j].strip().strip("{}").strip().rstrip(";")
    out = None
    # Producer shape is read from the FOUND def line (register reuse
    # across waves makes name-based dispatch unsound).
    m = re.match(r"mov\.b32\s+(%r\d+),\s*\{([^}]*)\}$", l)
    if m and m.group(1) == reg:
        members = re.findall(r"%r\d+|%rs\d+|[a-z]\w+", m.group(2))
        assert members, ("empty pack", j + 1, l)
        out = frozenset()
        for mb in members:
            out |= resolve(ls, site_by_ln, leaves, memo, mb, j)
    m = re.match(r"cvt\.\S+\s+(\S+),\s*(\S+)$", l)
    if m and out is None:
        # Value conversion preserves provenance (row-labels flow
        # through; laneid/const sources resolve to empty below).
        src = m.group(2).rstrip(";")
        if src == "%laneid":
            out = frozenset()
        else:
            try:
                int(src.rstrip("U"), 0)
                out = frozenset()  # numeric immediate
            except ValueError:
                out = resolve(ls, site_by_ln, leaves, memo, src, j)
    m = re.match(r"mul\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)$", l)
    if m and m.group(1) == reg and out is None:
        d, a, b = m.groups()
        # Side-id by DEF SHAPE (register reuse across waves makes
        # name-based checks unsound: ep1's rcp broadcast reuses an
        # earlier leaf's register number).
        la = is_leaf_def(ls, leaves, a, j)
        lb = is_leaf_def(ls, leaves, b, j)
        if la != lb:
            lf = a if la else b
            out = frozenset([lf + "#lo", lf + "#hi"])
        else:
            out = frozenset(["TAINT:mul@%d" % (j + 1)])
    m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*2146992128$", l)
    if m and m.group(1) == reg and out is None:
        assert reg in leaves, ("stray magic-add", j + 1, l)
        out = frozenset([reg + "#lo", reg + "#hi"])
    m = re.match(r"mov\.\S+\s+(\S+?),\s*(\S+)$", l)
    if m and m.group(1) == reg and out is None:
        src = m.group(2)
        assert not src.startswith("{"), ("unhandled pack", j + 1, l)
        if src == "%laneid":
            out = frozenset()
        else:
            try:
                int(src.rstrip("U"), 0)
                out = frozenset()  # const init
            except ValueError:
                out = resolve(ls, site_by_ln, leaves, memo, src, j)
    if out is None:
        # Otherwise must be an MMA D: A fan-in + C chain, nearest
        # preceding site (compiler reuse across waves). Anything
        # else (foreign GEMM-D, stray ALU) is TAINT, not forced.
        hits = [(ln, s) for ln, s in site_by_ln.items()
                if ln - 1 <= j and reg in s[1]]
        if not hits:
            out = frozenset(["TAINT:op@%d:%s" % (j + 1, l[:40])])
        else:
            ln, s = max(hits, key=lambda h: h[0])
            arows = frozenset()
            for ar in s[2]:
                arows |= resolve(ls, site_by_ln, leaves, memo, ar, ln - 1)
            crows = frozenset()
            for cr in s[4]:
                crows |= resolve(ls, site_by_ln, leaves, memo, cr, ln - 1)
            out = arows | crows
    memo[key] = out
    return out


def leaf_true_rows(leaf, site, dpos, sidx):
    """TRUE rows of a leaf b32's halves at every lane (P1+A1)."""
    g, p = sidx[site]
    rows = {}
    for lane in range(NLANES):
        frs, fcs = frag_P1(lane)
        elems = [(r, c) for r in frs for c in fcs]
        q = arrange(elems, "A1")
        halves = [q[0:2], q[2:4]][dpos]
        rset = {true_row((g, r, p, c)) for r, c in halves}
        rows[lane] = rset
    return rows


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
    memo = {}

    def fanin(reg, use):
        return resolve(ls, site_by_ln, leaves, memo, reg, use)

    # stores: (store_line, [pack regs]) -> %rs chain
    stores = []
    for i, l in enumerate(ls):
        m = re.search(r"st\.global\S*\s+\[(\S+)\]", l)
        if m:
            m2 = re.search(r"mov\.b128\s+v,\s*\{([^}]*)\}", l)
            stores.append((i + 1, m2.group(1) if m2 else "?"))
    persite(ls, sites, sidx, leaf2sd, leaves, 12270, 12470)
    print("global stores: %d" % len(stores))
    for n, (ln, packs) in enumerate(stores):
        regs = re.findall(r"%r\d+", packs)
        assert len(regs) == 4, (ln, packs)
        halves = []  # (rs, pack_def_idx) in store-byte order
        for r in regs:
            j = def_walk(ls, r, ln - 1)
            m = re.match(r"mov\.b32\s+%r\d+,\s*\{(%rs\d+),\s*(%rs\d+)\}",
                         ls[j].strip().strip("{}").strip().rstrip(";"))
            assert m, (ln, r, ls[j].strip()[:80])
            halves.extend([(m.group(1), j), (m.group(2), j)])
        assert len(halves) == 8, halves
        print("-- store %d @%d: %d e4m3 quads" % (n, ln, len(halves)))
        for q, (rs, jpack) in enumerate(halves):
            j = def_walk(ls, rs, jpack)
            l = ls[j].strip().strip("{}").strip()
            m = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+%rs\d+,"
                         r"\s*(%r\d+);?", l)
            assert m, (ln, rs, l[:80])
            dsrc = m.group(1)
            fan = fanin(dsrc, j)
            # fan: leaf halves (+ TAINT markers for foreign-phase
            # values such as the collector C addend). Evaluate TRUE
            # rows of the leaf part at lanes 0 and 1.
            taint = sorted({x for x in fan if x.startswith("TAINT")})
            lfset = {x.rsplit("#", 1)[0] for x in fan} - set(taint)
            assert lfset <= leaves, ("non-leaf fan-in", n, q,
                                     sorted(lfset)[:4])
            perlane = {}
            multi = 0
            for lf in lfset:
                site, dpos = leaf2sd[lf]
                tr = leaf_true_rows(lf, site, dpos, sidx)
                for lane, rset in tr.items():
                    perlane.setdefault(lane, set()).update(rset)
                    if len(rset) > 1:
                        multi += 1
            # summary: distinct TRUE-row SETS across lanes; lane rows
            sets = {}
            for lane, rset in perlane.items():
                sets.setdefault(tuple(sorted(rset)), []).append(lane)
            desc = "; ".join("%s->L%s" % (list(k), v) for k, v in
                             sorted(sets.items()))
            print("   quad %d <- %s fanleaves=%d multi=%d taint=%s" %
                  (q, dsrc, len(lfset), multi,
                   ",".join(t.split(":")[0] for t in taint) if taint
                   else "-"))
            print("      lanes: %s" % desc)


def persite(ls, sites, sidx, leaf2sd, leaves, lo, hi):
    """Per O-site A-fan-in (NO C union): do all sites see one TRUE row?"""
    from esum_ownership import nearest_def as _nd  # noqa
    site_by_ln = {s[0]: s for s in sites}
    memo = {}

    def resolve(reg, use):
        key = (reg, use)
        if key in memo:
            return memo[key]
        j = def_walk(ls, reg, use)
        l = ls[j].strip().strip("{}").strip().rstrip(";")
        out = None
        m = re.match(r"mov\.b32\s+(%r\d+),\s*\{([^}]*)\}$", l)
        if m and m.group(1) == reg:
            members = re.findall(r"%r\d+|%rs\d+|[a-z]\w+", m.group(2))
            out = frozenset()
            for mb in members:
                out |= resolve(mb, j)
        m = re.match(r"cvt\.\S+\s+(\S+),\s*(\S+)$", l)
        if m and out is None:
            src = m.group(2).rstrip(";")
            if src == "%laneid":
                out = frozenset()
            else:
                try:
                    int(src.rstrip("U"), 0)
                    out = frozenset()
                except ValueError:
                    out = resolve(src, j)
        m = re.match(r"mul\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)$", l)
        if m and m.group(1) == reg and out is None:
            d, a, b = m.groups()
            la = is_leaf_def(ls, leaves, a, j)
            lb = is_leaf_def(ls, leaves, b, j)
            if la != lb:
                lf = a if la else b
                out = frozenset([lf + "#lo", lf + "#hi"])
            else:
                out = frozenset(["TAINT:mul@%d" % (j + 1)])
        m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*2146992128$", l)
        if m and m.group(1) == reg and out is None:
            out = frozenset([reg + "#lo", reg + "#hi"]) \
                if reg in leaves else frozenset(["TAINT:leaf?@%d" % (j + 1)])
        m = re.match(r"mov\.\S+\s+(\S+?),\s*(\S+)$", l)
        if m and m.group(1) == reg and out is None:
            src = m.group(2)
            if src.startswith("{"):
                out = frozenset(["TAINT:pack@%d" % (j + 1)])
            elif src == "%laneid":
                out = frozenset()
            else:
                try:
                    int(src.rstrip("U"), 0)
                    out = frozenset()
                except ValueError:
                    out = resolve(src, j)
        if out is None:
            out = frozenset(["TAINT:op@%d" % (j + 1)])
        memo[key] = out
        return out

    print("== per-O-site A-rows (lanes 0,1,8,16) ==")
    for ln in range(lo, hi + 1):
        if ln not in site_by_ln:
            continue
        _ln, D, A, B, C = site_by_ln[ln]
        fan = frozenset()
        for ar in A:
            fan |= resolve(ar, ln - 1)
        taint = sorted({x for x in fan if x.startswith("TAINT")})
        lfset = {x.rsplit("#", 1)[0] for x in fan} - set(taint)
        rowstr = []
        for lane in (0, 1, 8, 16):
            rset = set()
            for lf in lfset:
                if lf in leaf2sd:
                    site, dpos = leaf2sd[lf]
                    rset |= leaf_true_rows(lf, site, dpos, sidx)[lane]
            rowstr.append("L%d=%s" % (lane, sorted(rset)))
        print("site %d A-leaves=%d taint=%s %s" %
              (ln, len(lfset), taint if taint else "-", " ".join(rowstr)))


if __name__ == "__main__":
    main()
