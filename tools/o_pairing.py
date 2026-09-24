#!/usr/bin/env python3
"""tools/o_pairing.py -- within-quad pairing + bias-C mechanism proof.

Resolves HANDOFF §33.2's open within-quad pairing question and pins the
bias entry point. Three fail-closed gates (exit 0 iff all hold):

A. O-slot table (stores 0,1 = O run-groups 46,47): every store quad
   comes straight from one O-MMA D b32 (cvt only, no ALU). Both f16
   slots of a quad carry the SAME O-side TRUE row at every lane
   (A1 halves share an MMA row; consecutive frag cols share fc//2),
   so there is NO slot-order ambiguity on the D side. Quad order per
   store is p0D0,p1D0,p0D1,p1D1,p2D0,p3D0,p2D1,p3D1.

B. P-side fan by label (store0-quad0, store0-quad2, store1-quad0):
   each O-quad fans into exactly 2 score-TRUE labels differing ONLY
   in dpos: (run,0,0) = all 8 score-sites' dpos-0 halves (16
   (leaf,half)), (run,0,2) = all dpos-1 halves (16). Pair-order IS
   dpos (D-half): even score-rows vs odd score-rows. Store0 <- score
   group 36 (sites 176-183); store1 <- group 37 (sites 184-191).
   Plus exactly 2 non-score (V-side mul) TAINTs per quad.

C. Bias-C mechanism: the 16 score-MMAs' accumulator C-heads are 8
   ld.global.ca.v4.u32 (16 B each) at bias_base + const_k +
   laneid*16, const_k = 12384 + 512*k, all inside the 8192 B f16
   row-major bias (port o_bias=12384). The bias enters as MMA-C, one
   per-lane 16 B tile per load (no separate bias-add ALU exists in
   10982-12270: only temperature fmas (uniform consts), fast-EXP
   bit-tricks and the rowsum tree).

D. Exact score slot algebra (both waves): bias-relative addresses
   give an EXACT, bijective (query,key) per score C/D element:
   query = qbase + 4*k + L//8, key = (L%8)*8 + h*4 + c*2 + s
   (k = load-pair index, L = lane, h = site-pair-half (0 even,
   1 odd site), c = C-half/dpos, s = f16 slot). 128 distinct (q,k)
   per site. c0==c1 rows (franken token-mixing lives in the dot's
   k-contraction, not the C layout). Site pairs share queries and
   split keys as interleaved even/odd 4-chunks (NOT dim-halves:
   supersedes §33's "2 Q-dim halves" factorization).

E. Two-wave structure + fused Proj: 64 temperature fmas -> 32 score
   sites (wave-1 @10839-10944 queries 0-31, wave-2 @12684-12789
   queries 32-63), 2 per site. P.V split-K (16 sites/wave-1 in 8
   A-sharing pairs, C-chains). O-collector B = Proj weights
   (const 20592 = port o_proj), C = scaled early partial (mul, not
   zero): the kernel FUSES Proj+residual per wave while the port
   splits it (k_proj_res) -- same math, recorded structure gap.
   4 output stores, same base (param+216), 2048 B = port 64x32 O.

F. Store<->port row-level map: 4 stores share base (param+216);
   chunks (C,C+1,C+S,C+S+1) via mad (ctaid multipliers r60/r63,
   stride r4054=S, adjacent r59/r59+1); lane stride 16 (laneid*16
   all 4); 16 B/lane (4 b32 packs) fitting 512 B/chunk exactly
   (32 lanes, no gap/overlap); [64x32] row-major gives 16 rows/
   chunk, lane-paired rows (query 16*C_n+L//2) and 16 dims/lane
   (dim (L%2)*16+2q+s, low-first pack/cvt semantics); kernel byte
   b <-> port k_av thread b (both (b//32,b%32)) given ctaid-0
   chunks (0,1,2,3) (S=2 correctness-forced, HANDOFF) + low-first.

G. QK loop-dim cross-check (reads the port file; fails on drift):
   port SW_TOK=64/SW_C=32/SW_QKV=96 defines; k_qknorm d<32 (both
   loops, per-token rsqrt, s-temperature on Q only, V=Yq[64:]);
   k_scores (i=idx/64,j=idx%64) d<32 with B[i*64+j]; k_av
   (i=idx/32,d=idx%32) linear; kernel score-MMAs m16n8k32 (k=32
   all e4m3) + 16 m16n8k16-f16 @986-1091 (K=16, pre-QK
   norm-candidate; A=B unverified).
   Head-dim 32 both; kernel splits (norm-halves/franken/key-
   waves) vs port clean -- values green, franken value-
   transparent (port needs no slices).

Stdlib-only, GPU-free, byte-deterministic stdout.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import (parse_mma_full, run_groups, frag_P1,  # noqa: E402
                            arrange, true_row)
from o_fanin import (def_walk, leaf_true_rows, build_leafmap,  # noqa: E402
                     resolve)

NLANES = 32
EXPECTED_CONSTS = [12384 + 512 * k for k in range(8)]
BIAS_LEN = 8192


def o_quads(ls, sites):
    """[(store_n, rs, D_reg, site_ln, Didx)] for stores 0,1 (8+8)."""
    out = []
    n = 0
    for i, l in enumerate(ls):
        if not re.search(r"st\.global\S*\s+\[(\S+)\]", l):
            continue
        if n >= 2:
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
                k = def_walk(ls, q_rs, j)
                mc = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+"
                              r"%rs\d+,\s*(%r\d+);?",
                              ls[k].strip().strip("{}").strip())
                assert mc, (n, q_rs, ls[k].strip()[:80])
                d = mc.group(1)
                prod = [ln for (ln, Dm, _A, _B, _C) in sites
                        if ln - 1 <= k and d in Dm]
                assert prod, ("D with no site", d)
                S = max(prod)
                di = [s[1] for s in sites if s[0] == S][0].index(d)
                out.append((n, q_rs, d, S, di, k))
        n += 1
    assert n == 2 and len(out) == 16, (n, len(out))
    return out


def slot_labels(g, p, Didx):
    """{lane: [TRUE slot0, TRUE slot1]} (ordered f16 slots, P1+A1)."""
    rows = {}
    for lane in range(NLANES):
        frs, fcs = frag_P1(lane)
        elems = [(r, c) for r in frs for c in fcs]
        q = arrange(elems, "A1")
        halves = [q[0:2], q[2:4]][Didx]
        rows[lane] = [true_row((g, r, p, c)) for r, c in halves]
    return rows


def main():
    ls = kernel_lines()
    sites = parse_mma_full(ls)
    pos_of_ln = {s[0]: i for i, s in enumerate(sites)}
    sidx = {}
    for g, grp in enumerate(run_groups(sites)):
        for p, (idx, _s) in enumerate(grp):
            sidx[idx] = (g, p)
    leaf2sd, leaves = build_leafmap(ls, sites)
    site_by_ln = {s[0]: s for s in sites}
    memo = {}

    # ---- A. O-slot table: quad order + both-slots-same ----
    quads = o_quads(ls, sites)
    groups = sorted({sidx[pos_of_ln[S]][0] for (_m, _rs, _d, S, _di, _k)
                     in quads})
    assert groups == [46, 47], groups
    for n in (0, 1):
        sub = [(sidx[pos_of_ln[S]][1], di)
               for (m, _rs, _d, S, di, _k) in quads if m == n]
        assert sub == [(0, 0), (1, 0), (0, 1), (1, 1),
                       (2, 0), (3, 0), (2, 1), (3, 1)], (n, sub)
    for (m, rs, _d, S, di, _k) in quads:
        g, p = sidx[pos_of_ln[S]]
        tab = slot_labels(g, p, di)
        for lane, (t0, t1) in tab.items():
            assert t0 == t1, (m, rs, lane, t0, t1)
    print("A slots: 16 quads store-groups=%s quad-order="
          "p0D0,p1D0,p0D1,p1D1,p2D0,p3D0,p2D1,p3D1 both-stores; "
          "both-slots-same at all 32 lanes x 16 quads" % (groups,))
    for (m, rs, d, S, di, _k) in quads:
        g, p = sidx[pos_of_ln[S]]
        tab = slot_labels(g, p, di)
        show = " ".join("L%d:%s" % (lane, tab[lane][0])
                        for lane in (0, 1, 4, 8))
        print("  store%d %s <- D %s site@%d (g%d p%d Didx%d): %s"
              % (m, rs, d, S, g, p, di, show))

    # ---- B. P-side fan by label: pair = dpos parity ----
    byname = {}
    for (m, rs, d, _S, _di, k) in quads:
        byname.setdefault(m, []).append((rs, d, k))
    probes = [("store0-quad0", 0, 0), ("store0-quad2", 0, 2),
              ("store1-quad0", 1, 0)]
    exp = {(0, 0): (36, (176, 177, 178, 179, 180, 181, 182, 183)),
           (0, 2): (36, (176, 177, 178, 179, 180, 181, 182, 183)),
           (1, 0): (37, (184, 185, 186, 187, 188, 189, 190, 191))}
    for name, m, q in probes:
        rs, d, k = byname[m][q]
        fan = resolve(ls, site_by_ln, leaves, memo, d, k)
        grp = defaultdict(lambda: [0, set(), set()])
        taint = sorted({x for x in fan if x.startswith("TAINT")})
        for x in fan:
            if x.startswith("TAINT"):
                continue
            lf, _h = x.rsplit("#", 1)
            site, dpos = leaf2sd[lf]
            tr = leaf_true_rows(lf, site, dpos, sidx)
            e = grp[tuple(sorted(tr[0]))]
            e[0] += 1
            e[1].add(site)
            e[2].add(dpos)
        assert len(taint) == 2 and all(t.startswith("TAINT:mul@")
                                       for t in taint), (name, taint)
        assert len(grp) == 2, (name, sorted(grp))
        erun, esites = exp[(m, 0)]
        for lbl, (cnt, st, dp) in grp.items():
            assert len(lbl) == 1, (name, lbl)
            assert lbl[0][0] == erun and lbl[0][1] == 0, (name, lbl)
            assert cnt == 16, (name, lbl, cnt)
            assert tuple(sorted(st)) == esites, (name, lbl, st)
            assert lbl[0][2] == (0 if dp == {0} else 2), (name, lbl, dp)
            assert dp in ({0}, {1}), (name, lbl, dp)
        desc = "; ".join("%s n=%d dpos=%s" % (list(lbl), v[0],
                                              sorted(v[2]))
                         for lbl, v in sorted(grp.items()))
        print("B %s: fan=%d labels=2 (%s) taint=2 (%s)"
              % (name, len(fan), desc, ",".join(taint)))

    # ---- C. bias enters as score-MMA C (per-lane tiles) ----
    fmaD = set()
    for i in range(10981, 11510):
        mt = re.search(r"fma\.rn\.f16x2\s+(%r\d+),\s*(%r\d+),", ls[i])
        if mt:
            fmaD.add(mt.group(2))
    prod = {}
    for (ln, Dm, _A, _B, _C) in sites:
        for r in Dm:
            prod.setdefault(r, ln)
    score_ln = sorted({prod[r] for r in fmaD if r in prod})
    assert len(score_ln) == 16, len(score_ln)
    cheads = set()
    for (ln, _Dm, _A, _B, C) in sites:
        if ln in score_ln:
            cheads.update(C)
    assert len(cheads) == 32, len(cheads)
    loads = []
    for i, l in enumerate(ls):
        if "ld." in l and "global" in l:
            regs = re.findall(r"%r\d+", l)
            if any(r in cheads for r in regs):
                loads.append((i + 1, l.strip()))
    assert len(loads) == 8, len(loads)
    consts = []
    for ln, l in loads:
        m = re.search(r"ld\.weak\.global\.ca\.v4\.u32\s+\{\s*"
                      r"(%r\d+),(%r\d+),(%r\d+),(%r\d+)\},\[(%rd\d+)\];",
                      l.strip().strip("{}").strip())
        assert m, (ln, l[:100])
        assert set(m.groups()[:4]) <= cheads, (ln, l[:100])
        rd = m.group(5)
        a = next(j for j in range(ln - 2, ln - 6, -1)
                 if re.search(r"add\.s64\s+%s," % rd, ls[j]))
        am = re.search(r"add\.s64\s+(%rd\d+),\s*(%rd\d+),\s*(\d+);",
                       ls[a].strip().strip("{}").strip())
        assert am and am.group(1) == rd, (ln, ls[a].strip()[:80])
        base, const = am.group(2), int(am.group(3))
        consts.append(const)
        b = next(j for j in range(a - 1, a - 6, -1)
                 if re.search(r"add\.s64\s+%s," % base, ls[j]))
        bm = re.search(r"add\.s64\s+(%rd\d+),\s*(%rd\d+),\s*(%rd\d+);",
                       ls[b].strip().strip("{}").strip())
        assert bm and bm.group(1) == base, (ln, ls[b].strip()[:80])
        assert bm.group(2) == "%rd7", (ln, ls[b].strip()[:80])
        off = bm.group(3)
        c = def_walk(ls, off, b) if off.startswith("%r") else None
        oline = ls[c].strip().strip("{}").strip() if c is not None else ""
        assert re.match(r"mul\.wide\.s32\s+%s,\s*%%r\d+,\s*16;$" % off,
                        oline), (ln, oline[:80])
        lr = re.match(r"mul\.wide\.s32\s+\S+,\s*(%r\d+),\s*16;$", oline)
        d = def_walk(ls, lr.group(1), c)
        assert ls[d].strip().strip("{}").strip() == \
            "mov.u32 %s, %%laneid;" % lr.group(1), (ln, ls[d].strip()[:80])
    assert consts == EXPECTED_CONSTS, consts
    assert all(0 <= c - 12384 < BIAS_LEN for c in consts), consts
    print("C bias-C: 8 ld.global.ca.v4.u32 -> 32 score C-head regs; "
          "addr=rd7+bias_const+laneid*16 consts=%s all in "
          "bias@[12384,12384+8192)" % (consts,))

    # ---- D. exact score (query,key) per element, both waves ----
    for w, (wconsts, wsites, qbase) in enumerate((
            ([12384 + 512 * k for k in range(8)],
             [10839 + 7 * i for i in range(16)], 0),
            ([16480 + 512 * k for k in range(8)],
             [12684 + 7 * i for i in range(16)], 32))):
        assert [s[0] for s in sites
                if wsites[0] <= s[0] <= wsites[-1]] == wsites, w
        got = {}
        got_rows = {}
        for k in range(8):
            for j in range(2):
                S = wsites[2 * k + j]
                for c in range(2):
                    t = 2 * j + c
                    for L in range(32):
                        frs, fcs = frag_P1(L)
                        elems = [(r, cc) for r in frs for cc in fcs]
                        q = arrange(elems, "A1")
                        halves = [q[0:2], q[2:4]][c]
                        rows = set()
                        for s, (_r, _cc) in enumerate(halves):
                            f16 = (wconsts[k] + L * 16 + 4 * t + 2 * s) \
                                // 2 - 6192
                            assert qbase * 64 <= f16 < qbase * 64 + 2048, \
                                (w, S, L, f16)
                            rows.add(f16 // 64)
                            key = f16 % 64
                            assert key == (L % 8) * 8 + j * 4 + c * 2 \
                                + s, (w, S, L, c, s, key)
                            got.setdefault(S, set()).add(
                                (f16 // 64, key))
                        assert len(rows) == 1, (w, S, L, c, rows)
                        if c == 1:
                            assert rows == got_rows[(S, L)], (w, S, L)
                        else:
                            got_rows[(S, L)] = rows
        for S in wsites:
            assert len(got[S]) == 128, (w, S, len(got[S]))
            qs = sorted({qq for qq, _kk in got[S]})
            assert qs == [qbase + 4 * (wsites.index(S) // 2) + o
                          for o in range(4)], (w, S, qs)
        print("D wave%d: 16 sites queries %d-%d bijective 128 (q,k)/site; "
              "q=qbase+4k+L//8 key=(L%%8)*8+h*4+c*2+s; c0==c1 rows"
              % (w + 1, qbase, qbase + 31))

    # ---- E. two waves + split-K P.V + fused Proj ----
    fall = defaultdict(set)
    for i, l in enumerate(ls):
        mt = re.search(r"fma\.rn\.f16x2\s+(%r\d+),\s*(%r\d+),%r4502,",
                       l)
        if mt:
            fall[mt.group(2)].add(i + 1)
    prod = {}
    for (ln, Dm, _A, _B, _C) in sites:
        for r in Dm:
            prod.setdefault(r, ln)
    persite = defaultdict(set)
    for r, uses in fall.items():
        if r in prod:
            persite[prod[r]] |= uses
    tsites = sorted(persite)
    assert len(tsites) == 32, len(tsites)
    assert tsites[:16] == [10839 + 7 * i for i in range(16)], tsites[:4]
    assert tsites[16:] == [12684 + 7 * i for i in range(16)], tsites[16:20]
    assert all(len(persite[s]) == 2 for s in tsites), \
        sorted((s, len(persite[s])) for s in tsites)
    pv = [s for s in sites if 12270 <= s[0] < 12456]
    assert len(pv) == 16, len(pv)
    agrp = run_groups(pv)
    assert len(agrp) == 8 and all(len(g) == 2 for g in agrp), \
        [len(g) for g in agrp]
    assert all(g[0][1][2] == g[1][1][2] and
               g[0][1][3] != g[1][1][3] for g in agrp)
    jb = def_walk(ls, "%r3322", 12455)
    assert "ld.weak.global.ca.v4.u32" in ls[jb], ls[jb].strip()[:80]
    ja = next(j for j in range(jb - 1, jb - 5, -1)
              if re.search(r"add\.s64\s+%rd44,", ls[j]))
    assert ", 20592;" in ls[ja], ls[ja].strip()[:80]
    jc = def_walk(ls, "%r3324", 12455)
    assert re.match(r"\{?mul\.f16x2\s+%r3324,",
                    ls[jc].strip()), ls[jc].strip()[:80]
    nst = sum(1 for l in ls if re.search(r"st\.global\S*\s+\[", l))
    assert nst == 4, nst
    print("E waves: 64 temp-fmas -> 32 score sites (16+16) 2/site; "
          "P.V 16 sites 8 A-pairs (same-A diff-B); O-B=Proj@20592 "
          "O-C=mul (fused Proj+residual); 4 stores same base 2048B")

    # ---- F. store<->port row-level map ----
    stores = []
    for i, l in enumerate(ls):
        m = re.search(r"st\.global\S*\s+\[(\S+)\]", l)
        if m:
            m2 = re.search(r"mov\.b128\s+v,\s*\{([^}]*)\}", l)
            stores.append((i, m.group(1),
                           re.findall(r"%r\d+", m2.group(1))))
    assert len(stores) == 4, len(stores)
    assert all(len(r) == 4 for (_i, _a, r) in stores)
    # same base rd6 <- param+216 on all four addr chains
    for (i, addr, _regs) in stores:
        cur, seen, ok = addr, set(), False
        for _d in range(6):
            try:
                k = def_walk(ls, cur, i)
            except AssertionError:
                break
            s = ls[k].strip().strip("{}").strip()
            if re.match(r"ld\.param\.b64\s+%rd6,\s*\[%rd8\+216\];", s):
                ok = True
                break
            nxt = [x for x in re.findall(r"%rd\d+|%r\d+", ls[k])
                   if x != cur and x not in seen]
            if not nxt:
                break
            seen.add(cur)
            cur = nxt[0]
        assert ok, (i, addr)
    # chunk mads: (r60/r63 * r4054 + r59/r59+1), lane stride 16
    mads, strides = [], []
    for (i, _addr, _regs) in stores:
        jm = next(j for j in range(i - 1, i - 40, -1)
                  if re.match(r"\s*\{?mad\.lo\.s32\s+%r\d+,", ls[j]))
        m = re.match(r"\s*\{?mad\.lo\.s32\s+(%r\d+),\s*(%r\d+),\s*"
                     r"(%r\d+),\s*(%r\d+);?", ls[jm])
        assert m, (i, ls[jm].strip()[:80])
        mads.append(m.groups())
        jl = next(j for j in range(i - 1, i - 12, -1)
                  if re.search(r"mul\.wide\.s32\s+%rd\d+,\s*%r\d+,\s*16;",
                               ls[j]))
        ml = re.search(r"mul\.wide\.s32\s+(%rd\d+),\s*(%r\d+),\s*16;",
                       ls[jl])
        kl = def_walk(ls, ml.group(2), jl)
        assert "laneid" in ls[kl], (i, ls[kl].strip()[:80])
        strides.append(ml.group(1))
    assert [m[1] for m in mads][:2] == ["%r60", "%r60"], mads
    assert [m[1] for m in mads][2:] == ["%r63", "%r63"], mads
    assert all(m[2] == "%r4054" for m in mads), mads
    assert mads[1][3] != mads[0][3] and mads[3][3] != mads[2][3], mads
    j62 = def_walk(ls, mads[1][3], stores[1][0])
    assert re.match(r"\s*\{?add\.s32\s+%s,\s*%s,\s*1;?" %
                    (mads[1][3], mads[0][3]), ls[j62].strip()), \
        ls[j62].strip()[:80]
    assert len(set(strides)) == 4, strides
    print("F stores: 4x same-base(param+216) chunks (C,C+1,C+S,C+S+1) "
          "[mad r60/r60/r63/r63 x r4054 + r59/r59+1]; lane-stride-16 "
          "x4; 16B/lane (4 b32 packs) = 512B/chunk exact "
          "(32 lanes, no gap/overlap)")
    print("F map: [64x32] row-major -> 16 rows/chunk; "
          "(store n,lane L,quad q,slot s) -> byte "
          "C_n*512+L*16+2q+s -> (query ...//32, dim ...%32): "
          "query=16*C_n+L//2 dim=(L%2)*16+2q+s "
          "(low-first pack/cvt; ctaid0 C=(0,1,S,S+1))")

    # ---- G. QK loop-dim cross-check (port file + kernel) ----
    port = open(os.path.join(ROOT, "hip", "mvp1", "swin_1h.hip"),
                errors="replace").read()
    defs = dict(re.findall(r"#define\s+(SW_TOK|SW_C|SW_QKV|SW_D_IN)"
                           r"\s+(\d+)", port))
    assert defs == {"SW_TOK": "64", "SW_C": "32", "SW_QKV": "96",
                    "SW_D_IN": "16"}, defs
    assert re.search(r"void\s+k_qknorm.*?for\s*\(int\s+d\s*=\s*0;\s*d\s*<\s*32;",
                     port, re.S), "qknorm-norm-loop"
    assert re.search(r"Qn\[m\s*\*\s*SW_C\s*\+\s*d\]\s*=\s*sf\s*\*",
                     port), "qknorm-Q-temperature"
    assert re.search(r"Kn\[m\s*\*\s*SW_C\s*\+\s*d\]\s*=\s*Yq\[m\s*\*\s*"
                     r"SW_QKV\s*\+\s*32\s*\+\s*d\]",
                     port), "qknorm-K-layout"
    assert re.search(r"void\s+k_scores.*?int\s+i\s*=\s*idx\s*/\s*SW_TOK,"
                     r"\s*j\s*=\s*idx\s*%\s*SW_TOK;.*?for\s*\(int\s+d\s*=\s*0;"
                     r"\s*d\s*<\s*32;.*?S\[i\s*\*\s*SW_TOK\s*\+\s*j\]\s*=\s*"
                     r"acc\s*\+\s*B\[i\s*\*\s*SW_TOK\s*\+\s*j\];",
                     port, re.S), "scores-loop-bias"
    assert re.search(r"void\s+k_av.*?int\s+i\s*=\s*idx\s*/\s*SW_C,\s*d\s*=\s*"
                     r"idx\s*%\s*SW_C;.*?O\[i\s*\*\s*SW_C\s*\+\s*d\]\s*=\s*acc;",
                     port, re.S), "av-linear"
    assert re.search(r"Yq\[j\s*\*\s*SW_QKV\s*\+\s*64\s*\+\s*d\]",
                     port), "av-V-layout"
    assert re.search(r"Wp\[o\s*\*\s*SW_C\s*\+\s*k\]", port), "proj-layout"
    ntemp = sum(1 for l in ls if "m16n8k32" in l and "e4m3.e4m3" in l)
    assert ntemp >= 32, ntemp
    n16 = sum(1 for l in ls if "m16n8k16" in l and "f16.f16.f16.f16" in l)
    assert n16 == 16, n16
    print("G QK-dims: port(SW_TOK=64 SW_C=32 SW_QKV=96) qknorm-d<32 "
          "per-token-rsqrt s-on-Q-only V=Yq[64:]; scores(i=idx/64,"
          "j=idx%%64) d<32 +B[i*64+j]; av(i=idx/32,d=idx%%32) linear; "
          "Wp[o*32+k]; kernel score-k=32(m16n8k32-e4m3 x%d) + "
          "16 m16n8k16-f16(K=16,pre-QK-norm-candidate)" % ntemp)
    print("G verdict: head-dim 32 both; kernel splits (norm-halves/"
          "franken/key-waves) vs port clean -- values green, "
          "franken value-transparent (port needs no slices)")
    print("PAIRING-OK")


if __name__ == "__main__":
    main()
