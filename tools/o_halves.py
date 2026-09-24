#!/usr/bin/env python3
"""tools/o_halves.py -- quarter-recombination halves-combine proof (gate Q).

Closes HANDOFF 33.12(4): the 16 f16 mmas @986-1091 compute quartered
(S.W-half) products (C=zero, never recombined there); this gate proves
WHERE the quarters recombine on the way into the 6959 QKV contraction.

Q1. Quarter census: 16 f16 m16n8k16 sites @986-1091 step 7; A shared
    r439-442 all sites; B = 8 regs r443-450; C = zero (r4734) all;
    D = 32 regs.
Q2. Quarter fan-out (the split): every quarter D-reg has EXACTLY two
    consumers before any redef: one scale-mul @1482+ (C-path, f16
    exact) and one e4m3-cvt @1720+ (A-path, quantized). No other use.
Q3. 1832-run: 16 sites, C=zero; A = 4 shared groups from quant-quarters
    (Q2 A-path); B = ffn1[0..1024) (bases 0/512, lane*16).
Q4. Halves-combine @2973 first half (16 sites): C = 16 scaled quarters
    1:1 (Q2 C-path); A = 4 tiles from scaled+quantized 1832-D; B =
    ffn2[0..1024). Per site D = C + A.B (operand census).
Q5. Accumulation chain (crossed halves): h1 C-links run 1:1 in
    site order (2973h1-D -> 4209h1-C -> 4209h1-D -> 5445h1-C ->
    5445h1-D -> 6681-C -> 6681-D) while h1 A-tiles resolve to
    quant(scaled prev-h2-D) in full (h2 fresh products feed the h1
    accumulator); second halves run C=zero fresh (their D feed
    min/mul side paths, never the chain). B-windows tile ffn2/ffn1
    slabs (bases pinned).
Q6. 6959-run: 48 sites = 4 A-tiles x 12 B-pairs; C=zero all; A-tile-k
    = quant(6681-sites[4k..4k+3]); B = full qkv window [9312,12384)
    (6 x 512B slabs); D = 3 groups x 16 pairs.
Q7. Downstream classes: D-group-2 (r2816-2847) feeds the 32 movmatrix
    srcs; D-groups-0/1 feed self-square norm muls (+ temp-scale muls).
Q8. Verdict HALVES-OK.

Stdlib-only, GPU-free, byte-deterministic stdout. Needs kernel_lines
+ def_walk only (~1 min); runbook 18.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from o_fanin import def_walk  # noqa: E402

M32 = 0xFFFFFFFF
USE_RE = re.compile(r"%r\d+|%rs\d+|%rd\d+")
BOUND = r"(?:^|[{\s,;])%s(?:$|[\s,};])"


def regs(text):
    return re.findall(r"%r\d+", text)


def show(ls, ln):
    return ls[ln - 1].strip()[:100]


def all_sites(ls):
    return [(i + 1) for i, l in enumerate(ls) if "mma.sync.aligned" in l]


def run_sites(ls, lo, hi):
    return [s for s in all_sites(ls) if lo <= s <= hi]


def operands(ls, s):
    """(D, A, B, C) reg lists for MMA site s (1-based)."""
    return (regs(ls[s - 1]), regs(ls[s]), regs(ls[s + 1]),
            regs(ls[s + 2]))


def next_def(ls, reg, frm):
    """First line > frm defining reg (dest position), or None."""
    want = re.compile(BOUND % re.escape(reg))
    for j in range(frm, len(ls)):
        if not want.search(ls[j]):
            continue
        l = ls[j].strip().strip("{}").strip().rstrip(";")
        m = re.match(r"(?:mov|and|or|xor|add|sub|shl|shr|selp|cvt|cvta|"
                     r"mul|mad|div|rem|fma|max|min|abs|neg|sqrt|rsqrt|rcp|"
                     r"sin|cos|ex2|lg2|setp|ld|shfl|prmt|tex|movmatrix)"
                     r"\.[a-z0-9.]+\s+"
                     r"(\{[^}]*\}|%rd\d+|%r\d+|%rs\d+|[a-z]\w*),?", l)
        if m:
            d = m.group(1)
            if d.startswith("{"):
                if reg in USE_RE.findall(d):
                    return j + 1
            elif d == reg:
                return j + 1
        m = re.match(r"mma\.sync\S*\s+\{([^}]*)\},?", ls[j].strip())
        if m and reg in regs(m.group(1)):
            return j + 1
    return None


def uses_before_redef(ls, reg, frm, include_defs=False):
    """(use_lines, redef_line) for reg after frm (1-based line numbers).

    A use counts only when its nearest preceding def is the def at or
    before frm (generation check): walks each candidate use back with
    def_walk and keeps it iff it resolves at/before frm."""
    want = re.compile(BOUND % re.escape(reg))
    nd = next_def(ls, reg, frm)
    hi = (nd - 1) if nd else len(ls)
    base = def_walk(ls, reg, frm) + 1 if frm > 0 else 0
    out = []
    for j in range(frm, hi):
        if not want.search(ls[j]):
            continue
        ln = j + 1
        if not include_defs and ln == nd:
            continue
        try:
            d = def_walk(ls, reg, ln - 1) + 1
        except AssertionError:
            continue
        if d <= base:
            out.append(ln)
    return out, nd


def lane16_base(ls, load_ln):
    """(base_const) for a v4-global load: lane*16 + rd7 + const shape."""
    t = ls[load_ln - 1].strip()
    m = re.match(r"ld\.weak\.global\.ca\.v4\.u32\s+\{(.*)\},\[(%rd\d+)\];?$",
                 t)
    assert m, ("not v4 load", load_ln, t[:60])
    rd = m.group(2)
    base = None
    lane_ok = False
    for k in range(load_ln - 2, max(0, load_ln - 12), -1):
        tk = ls[k].strip()
        m2 = re.match(r"add\.s64\s+" + re.escape(rd) +
                      r",\s*%rd\d+,\s*(\d+);?$", tk)
        if m2 and base is None:
            base = int(m2.group(1))
        if re.match(r"mul\.wide\.s32\s+%rd\d+,\s*%r\d+,\s*16;?$", tk):
            lane_ok = True
    assert base is not None, ("no base", load_ln)
    assert lane_ok, ("not lane*16", load_ln)
    return base


def check_Q1(ls):
    r = run_sites(ls, 986, 1091)
    assert len(r) == 16, len(r)
    assert all(b - a == 7 for a, b in zip(r, r[1:])), r
    Aset, Bset, Cset, D = set(), set(), set(), []
    for s in r:
        assert "m16n8k16" in ls[s - 1], (s, show(ls, s))
        d, a, b, c = operands(ls, s)
        assert len(d) == len(a) // 2 == len(b) == len(c) == 2, (s, d, a)
        Aset.update(a)
        Bset.update(b)
        Cset.update(c)
        D.extend(d)
    assert Aset == {"%r439", "%r440", "%r441", "%r442", "%r451", "%r452",
                    "%r453", "%r454", "%r455", "%r456", "%r457", "%r458",
                    "%r459", "%r460", "%r461", "%r462"}, sorted(Aset)
    groups = {}
    for s in r:
        groups.setdefault(tuple(regs(ls[s])), []).append(s)
    assert sorted(len(v) for v in groups.values()) == [4, 4, 4, 4], \
        sorted(len(v) for v in groups.values())
    assert Bset == {"%r443", "%r444", "%r445", "%r446", "%r447", "%r448",
                    "%r449", "%r450"}, sorted(Bset)
    assert Cset == {"%r4734"}, Cset
    assert len(D) == 32 and len(set(D)) == 32, len(D)
    print("Q1 quarters: 16 f16 sites @986-1091 step7; A = 4 shared "
          "groups x4; B=r443-450; C=zero; 32 D-regs")
    return r, D


def check_Q2(ls, D):
    scale_to, cvt_to = {}, {}
    for d in D:
        s = def_walk(ls, d, 986 + 2000)
        # def must be an f16-site D brace
        assert 986 <= s + 1 <= 1091, (d, s)
        uses, _nd = uses_before_redef(ls, d, s + 1)
        assert len(uses) == 2, (d, uses)
        kinds = []
        for u in uses:
            t = ls[u - 1].strip()
            if re.match(r"\{mul\.f16x2\s+%r\d+," + re.escape(d) + r",",
                         t):
                kinds.append(("mul", u))
            elif re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+%rs\d+,"
                          r"\s*" + re.escape(d) + r";?$", t):
                kinds.append(("cvt", u))
            else:
                kinds.append(("OTHER", u))
        assert sorted(k for k, _ in kinds) == ["cvt", "mul"], \
            (d, kinds, [show(ls, u) for u in uses])
        for k, u in kinds:
            (scale_to if k == "mul" else cvt_to)[d] = u
    mul_lines = sorted(set(scale_to.values()))
    cvt_lines = sorted(set(cvt_to.values()))
    assert all(1482 <= u <= 1720 for u in mul_lines), mul_lines
    assert all(1720 <= u <= 1832 for u in cvt_lines), cvt_lines
    assert len(mul_lines) == 32 and len(cvt_lines) == 32, \
        (len(mul_lines), len(cvt_lines))
    print("Q2 fan-out: 32/32 D-regs -> exactly {scale-mul @1482-1720, "
          "e4m3-cvt @1720-1832}; no other use before redef")
    return scale_to, cvt_to


def chain_A_src(ls, areg, use_ln):
    """Resolve a chain A-reg to its cvt source D-reg.

    mov.b32 pack <- %rs <- cvt.e4m3 <- D-reg. Returns
    (mov_ln, ((rs, cvt_ln, dreg, def_site_ln), ...)). Fail-closed."""
    jm = def_walk(ls, areg, use_ln - 1)
    t = ls[jm].strip()
    m = re.match(r"mov\.b32\s+" + re.escape(areg) +
                 r",\s*\{([^}]*)\};?$", t)
    assert m, ("A not mov-pack", areg, use_ln, t[:60])
    chain = []
    for rs in USE_RE.findall(m.group(1)):
        assert rs.startswith("%rs"), (areg, rs)
        jc = def_walk(ls, rs, jm)
        tc = ls[jc].strip()
        mc = re.match(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+" +
                      re.escape(rs) + r",\s*(%r\d+);?$", tc)
        assert mc, ("rs not e4m3-cvt", rs, tc[:60])
        dreg = mc.group(1)
        jd = def_walk(ls, dreg, jc)
        chain.append((rs, jc + 1, dreg, jd + 1))
    return jm + 1, tuple(chain)


def check_Q3(ls, cvt_to):
    r = run_sites(ls, 1832, 1937)
    assert len(r) == 16, len(r)
    Aset, Bset = set(), set()
    groups = {}
    for s in r:
        assert "m16n8k32" in ls[s - 1] and "e4m3.e4m3" in ls[s - 1], s
        d, a, b, c = operands(ls, s)
        assert c == ["%r4734", "%r4734"], (s, c)
        Aset.update(a)
        Bset.update(b)
        groups.setdefault(tuple(a), []).append(s)
    assert len(Aset) == 16 and len(groups) == 4, \
        (len(Aset), len(groups))
    assert sorted(len(v) for v in groups.values()) == [4, 4, 4, 4], \
        sorted(len(v) for v in groups.values())
    assert Bset == {"%r593", "%r594", "%r599", "%r600", "%r601", "%r602",
                    "%r603", "%r604"}, sorted(Bset)
    # A-regs <- quant-quarters (the Q2 cvt path, same %rs defs)
    cvt_lines = set(cvt_to.values())
    for a in sorted(Aset):
        _mov, ch = chain_A_src(ls, a, r[0])
        for _rs, cln, _dreg, _dsite in ch:
            assert cln in cvt_lines, (a, cln)
    # B loads: [0,512)/[512,1024) = ffn1 slab0, lane*16
    jb = def_walk(ls, "%r593", r[0] - 1)
    assert ls[jb].strip().startswith("ld.weak.global.ca.v4.u32"), \
        show(ls, jb + 1)
    t = ls[jb].strip()
    m = re.match(r"ld\.weak\.global\.ca\.v4\.u32\s+\{(.*)\},\[(%rd\d+)\];?$",
                 t)
    rd = m.group(2)
    base = None
    for k in range(jb - 1, max(0, jb - 12), -1):
        m2 = re.match(r"add\.s64\s+" + re.escape(rd) +
                      r",\s*%rd\d+,\s*(\d+);?$", ls[k].strip())
        if m2:
            base = int(m2.group(1))
    # base-0 load has NO const add (rd7+lane*16 directly)
    if base is None:
        tk = [ls[k].strip() for k in range(jb - 1, max(0, jb - 12), -1)]
        assert any(re.match(r"add\.s64\s+" + re.escape(rd) +
                            r",\s*%rd\d+,\s*%rd\d+;?$", x) for x in tk), tk
        base = 0
    assert base == 0, base
    assert lane16_base(ls, 1717) == 512, show(ls, 1717)
    print("Q3 1832-run: 16 sites C=zero; A = 4 shared groups from "
          "quant-quarters; B = ffn1[0..1024) (bases 0/512)")
    return r


ZERO_MOV = 964  # 1-based line of mov.b32 %r4734,{low,low} (f16 zero)


def resolve_A_full(ls, areg, use_ln):
    """Full A-path: mov <- rs <- cvt <- mid <- (mul <- prevD | MMA-D).

    Returns (mov_ln, [(rs, cvt_ln, mid, mul_ln_or_None, prevD,
    prevsite_ln)]). Fail-closed."""
    mov, ch = chain_A_src(ls, areg, use_ln)
    out = []
    for rs, cln, mid, _dsite in ch:
        jd = def_walk(ls, mid, cln - 1)
        t = ls[jd].strip()
        # mul dest==mid (cvt reads mid): {mul.f16x2 mid,prevD,scale;
        m = re.match(r"\{mul\.f16x2\s+" + re.escape(mid) +
                     r",(%r\d+),(%r\d+);", t)
        if m:
            prevD, scale = m.group(1), m.group(2)
            jp = def_walk(ls, prevD, jd)
            # scale must be gate-ish: global load or f16x2 ALU (never
            # tex/param/shfl). Record, don't certify values.
            js = def_walk(ls, scale, jd)
            ts = ls[js].strip()
            ok = ts.startswith("ld.global") or \
                re.match(r"\{(mul|fma)\.rn\.f16x2\s", ts) is not None
            assert ok, ("scale-foreign", scale, ts[:70])
            out.append((rs, cln, mid, jd + 1, prevD, jp + 1))
        else:
            jp = jd
            out.append((rs, cln, mid, None, mid, jp + 1))
    return mov, tuple(out)


def split_halves(ls, sites):
    """Split run sites into h1 (C <- prev-D/scaled-quarter) and h2
    (C = zero). Returns (h1, h2, Csrc_map). Fail-closed."""
    h1, h2, cmap = [], [], {}
    for s in sites:
        _d, _a, _b, c = operands(ls, s)
        assert len(c) == 2, (s, c)
        defs = set()
        for reg in c:
            defs.add(def_walk(ls, reg, s - 1) + 1)
        if defs == {ZERO_MOV}:
            h2.append(s)
        else:
            h1.append(s)
            cmap[s] = tuple(def_walk(ls, reg, s - 1) + 1 for reg in c)
    return h1, h2, cmap


def check_Q4(ls, scale_to):
    r = run_sites(ls, 2973, 3208)
    assert len(r) == 32, len(r)
    h1, h2, cmap = split_halves(ls, r)
    assert len(h1) == 16 and len(h2) == 16, (len(h1), len(h2))
    assert h1 == r[:16] and h2 == r[16:], (h1, h2)
    # C-path: 32 C-regs <- 32 scale-muls <- all 32 quarter D-regs
    cprev = {}
    for s in h1:
        for reg, cln in zip(regs(ls[s + 2]), cmap[s]):
            t = ls[cln - 1].strip()
            m = re.match(r"\{mul\.f16x2\s+(%r\d+),(%r\d+),(%r\d+);", t)
            assert m and m.group(1) == reg, (s, reg, t[:70])
            assert cln in set(scale_to.values()), (s, reg, cln)
            cprev[reg] = m.group(2)
    qset = set()
    for s in h1:
        for reg in regs(ls[s + 2]):
            qset.add(cprev[reg])
    # every C-mul source must be a Q2 scale output of a quarter
    assert len(qset) == 32, len(qset)
    # A-path: 4 tiles <- cvt <- 16 muls @2002+ <- 1832-D
    Atiles = {}
    for s in h1:
        Atiles.setdefault(tuple(regs(ls[s])), []).append(s)
    assert sorted(len(v) for v in Atiles.values()) == [4, 4, 4, 4], \
        sorted(len(v) for v in Atiles.values())
    aprev = set()
    for tile in Atiles:
        for areg in tile:
            _mov, full = resolve_A_full(ls, areg, h1[0])
            for _rs, _cl, _mid, mul, prevD, psite in full:
                assert mul is not None and 2002 <= mul <= 2860, \
                    (areg, mul)
                assert 1832 <= psite <= 1937, (areg, psite)
                aprev.add(prevD)
    assert len(aprev) == 32, len(aprev)
    # B windows: h1 <- ffn2[0..1024), h2 <- ffn1[1024..2048)
    b1 = set()
    for s in h1:
        b1.update(regs(ls[s + 1]))
    b2 = set()
    for s in h2:
        b2.update(regs(ls[s + 1]))
    assert len(b1) == 8 and len(b2) == 8 and not (b1 & b2), \
        (len(b1), len(b2))
    jb = def_walk(ls, sorted(b1)[0], h1[0] - 1)
    assert lane16_base(ls, jb + 1) in (4096, 4608), show(ls, jb + 1)
    assert lane16_base(ls, 2849) == 4096 and \
        lane16_base(ls, 2858) == 4608
    assert lane16_base(ls, 3091) == 1024 and \
        lane16_base(ls, 3100) == 1536
    print("Q4 halves-combine @2973h1: 16 sites, C = 16 scaled quarters "
          "1:1, A = 4 tiles <- scaled 1832-D, B = ffn2[0..1024); "
          "h2 fresh C=zero B=ffn1[1024..2048)")
    return r, h1, h2


def D_regs(ls, sites):
    out = []
    for s in sites:
        out.extend(regs(ls[s - 1]))
    return out


def check_link(ls, prev_h1, prev_h2, run_lo, run_hi, b_bases, tag):
    """One accumulation link: run[lo..hi] h1 C <- prev-h1-D and
    h1 A <- scaled prev-h2-D (crossed halves: h2 fresh products feed
    the h1 accumulator).

    h1 = first 16 sites, h2 = last 16, C 1:1 in site order, A-tiles
    (4x4) <- scaled prev-h2-D in full, B-windows = b_bases
    (h1 pair + h2 pair). Returns (run, h1, h2)."""
    r = run_sites(ls, run_lo, run_hi)
    assert len(r) == 32, (tag, len(r))
    h1, h2, _cmap = split_halves(ls, r)
    assert h1 == r[:16] and h2 == r[16:], (tag, h1, h2)
    prevset = set(D_regs(ls, prev_h1))
    assert len(prevset) == 32, (tag, len(prevset))
    prevset2 = set(D_regs(ls, prev_h2))
    assert len(prevset2) == 32, (tag, len(prevset2))
    # C: 32 regs <- prev-h1-D 1:1, site order
    for s, p in zip(h1, prev_h1):
        for reg in regs(ls[s + 2]):
            jd = def_walk(ls, reg, s - 1) + 1
            assert jd == p, (tag, s, reg, jd, p)
    # A: 4 tiles <- scaled prev-h2-D in full (32/32)
    Atiles = {}
    for s in h1:
        Atiles.setdefault(tuple(regs(ls[s])), []).append(s)
    assert sorted(len(v) for v in Atiles.values()) == [4, 4, 4, 4], \
        (tag, sorted(len(v) for v in Atiles.values()))
    got = set()
    for tile in Atiles:
        for areg in tile:
            _mov, full = resolve_A_full(ls, areg, h1[0])
            for _rs, _cl, _mid, mul, prevD, _ps in full:
                assert mul is not None, (tag, areg)
                assert prevD in prevset2, (tag, areg, prevD)
                got.add(prevD)
    assert got == prevset2, (tag, len(got), sorted(prevset2 - got)[:4])
    # h2: C=zero; A <- quant-quarters (direct, no scale step).
    # Resolve every DISTINCT h2 A-reg once (tiles are shared x4).
    seen = set()
    for s in h2:
        for reg in regs(ls[s]):
            if reg in seen:
                continue
            seen.add(reg)
            _mov, full = resolve_A_full(ls, reg, s)
            for _rs, _cl, mid, mul, _pd, ps in full:
                assert mul is None and 986 <= ps <= 1091, (tag, s, mid)
    assert len(seen) == 16, (tag, len(seen))
    # B windows
    b1 = set()
    for s in h1:
        b1.update(regs(ls[s + 1]))
    b2 = set()
    for s in h2:
        b2.update(regs(ls[s + 1]))
    assert len(b1) == 8 and len(b2) == 8 and not (b1 & b2), tag
    got_bases = set()
    for b in sorted(b1) + sorted(b2):
        use = h1[0] if b in b1 else h2[0]
        jb = def_walk(ls, b, use - 1)
        got_bases.add(lane16_base(ls, jb + 1))
    assert got_bases == set(b_bases), (tag, got_bases, b_bases)
    print("Q5 link %s: h1 16/16 C<-prev-h1-D + A<-scaled-prev-h2-D "
          "full; h2 C=zero A<-quarters; B=%s" % (tag, sorted(b_bases)))
    return r, h1, h2


def check_Q5(ls, h1_2973, h2_2973):
    r42, h1_42, h2_42 = check_link(ls, h1_2973, h2_2973, 4209, 4444,
                                   (5120, 5632, 2048, 2560), "4209")
    r54, h1_54, h2_54 = check_link(ls, h1_42, h2_42, 5445, 5680,
                                   (6144, 6656, 3072, 3584), "5445")
    # 6681: 16 sites, all h1 (C <- 5445h1-D 1:1)
    r66 = run_sites(ls, 6681, 6786)
    assert len(r66) == 16, len(r66)
    h1, h2, _cmap = split_halves(ls, r66)
    assert h1 == r66 and h2 == [], (len(h1), len(h2))
    prevset = set(D_regs(ls, h1_54))
    for s, p in zip(h1, h1_54):
        for reg in regs(ls[s + 2]):
            jd = def_walk(ls, reg, s - 1) + 1
            assert jd == p, ("6681", s, reg, jd, p)
    Atiles = {}
    for s in h1:
        Atiles.setdefault(tuple(regs(ls[s])), []).append(s)
    assert sorted(len(v) for v in Atiles.values()) == [4, 4, 4, 4], \
        sorted(len(v) for v in Atiles.values())
    got = set()
    prevset2 = set(D_regs(ls, h2_54))
    assert len(prevset2) == 32, len(prevset2)
    for tile in Atiles:
        for areg in tile:
            _mov, full = resolve_A_full(ls, areg, h1[0])
            for _rs, _cl, _mid, mul, prevD, _ps in full:
                assert mul is not None and prevD in prevset2, \
                    (areg, prevD)
                got.add(prevD)
    assert got == prevset2, len(got)
    b66 = set()
    for s in h1:
        b66.update(regs(ls[s + 1]))
    assert len(b66) == 8, len(b66)
    jb = def_walk(ls, sorted(b66)[0], h1[0] - 1)
    assert lane16_base(ls, jb + 1) in (7168, 7680), show(ls, jb + 1)
    assert lane16_base(ls, 6557) == 7168 and \
        lane16_base(ls, 6566) == 7680
    # h2-D side paths: every fresh-chain D-reg feeds min/mul only
    # (uses resolved to the site's own generation: nearest preceding
    # def of each use must be the site itself).
    for tag, h2 in (("2973h2", h2_2973), ("4209h2", h2_42),
                    ("5445h2", h2_54)):
        for s in h2:
            for reg in regs(ls[s - 1]):
                uses, _nd = uses_before_redef(ls, reg, s)
                assert uses, (tag, reg)
                for u in uses:
                    t = ls[u - 1].strip()
                    m = re.match(r"\{(min|mul)\.f16x2\s+(%r\d+),"
                                 r"(%r\d+),(%r\d+);", t)
                    assert m and reg in (m.group(2), m.group(3),
                                         m.group(4)), \
                        (tag, reg, u, t[:70])
    print("Q5 chain: 4209/5445 links + 6681 end (C/A 1:1, B slabs); "
          "h2-D all min/mul side paths")
    return r66


def check_Q6(ls, r6681):
    r = run_sites(ls, 6959, 7288)
    assert len(r) == 48, len(r)
    Atiles = {}
    for s in r:
        d, a, b, c = operands(ls, s)
        assert c == ["%r4734", "%r4734"], (s, c)
        assert len(d) == 2 and len(a) == 4 and len(b) == 2, (s, d, a)
        Atiles.setdefault(tuple(a), []).append(s)
    assert len(Atiles) == 4, len(Atiles)
    tiles = sorted(Atiles, key=lambda t: Atiles[t][0])
    for k, t in enumerate(tiles):
        assert Atiles[t] == r[12 * k:12 * k + 12], (k, Atiles[t])
    # A-tile-k <- 8 direct cvts <- 6681-sites[4k..4k+3] D, in full
    # (no scale step on this last link: quant(chain-end), HANDOFF 21)
    r66set = set(r6681)
    for k, t in enumerate(tiles):
        got = set()
        for areg in t:
            _mov, full = resolve_A_full(ls, areg, Atiles[t][0])
            for _rs, _cl, _mid, mul, prevD, ps in full:
                assert mul is None and ps in r66set, (k, areg, ps)
                got.add(prevD)
        want = set(D_regs(ls, r6681[4 * k:4 * k + 4]))
        assert got == want, (k, sorted(got ^ want))
    # B: 24 regs <- 6 qkv slabs [9312,12384)
    Bset = set()
    for s in r:
        Bset.update(regs(ls[s + 1]))
    assert len(Bset) == 24, len(Bset)
    bases = {}
    for b in sorted(Bset):
        jb = def_walk(ls, b, r[0] - 1)
        bases[b] = lane16_base(ls, jb + 1)
    assert sorted(set(bases.values())) == [9312, 9824, 10336, 10848,
                                           11360, 11872], \
        sorted(set(bases.values()))
    # D classes by B-pair index within tile: [0..4)/[4..8)/[8..12)
    classes = ([], [], [])
    for k in range(4):
        for j, s in enumerate(r[12 * k:12 * k + 12]):
            classes[j // 4].append(s)
    dnames = [sorted(D_regs(ls, c)) for c in classes]
    flat = dnames[0] + dnames[1] + dnames[2]
    assert len(flat) == 96 and len(set(flat)) == 96, len(flat)
    print("Q6 6959-run: 48 sites = 4 A-tiles x 12 B-pairs, C=zero; "
          "A-tile-k <- 6681-sites[4k..4k+3]; B = qkv [9312,12384); "
          "D groups %s/%s/%s" % (dnames[0][0].split("r")[1],
                                 dnames[1][0].split("r")[1],
                                 dnames[2][0].split("r")[1]))
    return r, classes


def check_Q7(ls, classes):
    g0, g1, g2 = (set(D_regs(ls, c)) for c in classes)
    # group-2 -> the 32 movmatrix srcs (V3 corroboration by name)
    assert g2 == {"%%r%d" % n for n in range(2816, 2848)}, \
        sorted(g2)[:4]
    for s in classes[2]:
        for reg in regs(ls[s - 1]):
            uses, _nd = uses_before_redef(ls, reg, s)
            mm = [u for u in uses
                  if ls[u - 1].strip().startswith("movmatrix.sync")]
            assert mm, (reg, uses)
    # groups-0/1 -> self-square norm muls (+ temp-scale muls)
    nsq = nother = 0
    for cls in (classes[0], classes[1]):
        for s in cls:
            for reg in regs(ls[s - 1]):
                uses, _nd = uses_before_redef(ls, reg, s)
                assert uses, reg
                for u in uses:
                    t = ls[u - 1].strip()
                    m = re.match(r"\{mul\.f16x2\s+(%r\d+),(%r\d+),"
                                 r"(%r\d+);", t)
                    assert m, (reg, u, t[:70])
                    if m.group(2) == reg == m.group(3):
                        nsq += 1
                    else:
                        nother += 1
    assert nsq == 64, nsq
    print("Q7 downstream: group-2 32/32 -> movmatrix srcs; groups-0/1 "
          "64 self-squares + %d scale-muls" % nother)
    return nsq, nother


def main():
    ls = kernel_lines()
    _rq1, D = check_Q1(ls)
    scale_to, cvt_to = check_Q2(ls, D)
    _r3 = check_Q3(ls, cvt_to)
    _r29, h1_29, h2_29 = check_Q4(ls, scale_to)
    r66 = check_Q5(ls, h1_29, h2_29)
    r69, classes = check_Q6(ls, r66)
    check_Q7(ls, classes)
    print("HALVES-OK")


if __name__ == "__main__":
    main()
