#!/usr/bin/env python3
"""tools/esum_ownership.py -- derive thread->row ownership from the Esum
softmax row-reduction shuffles (HANDOFF sections 28-30).

Method (all evidence, no guessing):
  1. Parse every m16n8k32 score-MMA site (D/A/B/C reg lists) and group
     consecutive A-sharing sites into runs.
  2. Find the two exp episodes (`shl.b32 E,U,5` chains: fma(D,u)->max->
     min->shl->leaf add of 0x7FF88000). Each E half is traced to its
     score-MMA D half by nearest-preceding-def walk (fail-closed).
  3. Symbolically execute each episode's window (u-steps, add-trees,
     laneid-bit selp, shfl.idx combine, prmt, max-floor, rcp, all P-muls)
     on a 32-lane machine propagating (run, fragrow, sitecol, fragcol)
     labels per byte. Unsupported ops abort; MMA lines are skipped
     (feeding MMAs precede the window).
  4. Checks: slot partition (each E-slot in exactly one lane's Esum),
     TRUE row/col factorization (TRUE row = (run, fc-pair, fr-quad),
     TRUE col = (fr, sitecol, fc)), and P-normalize routing (every P-half
     scaled by its own TRUE row's Esum, identified by exact set match).
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines, MMA  # noqa: E402  (proven constants)


def parse_mma_full(ls):
    """[(ln, D_regs, A_regs, B_regs, C_regs)] for the e4m3 score form.

    Form: `mma... {D2},` on the mnemonic line, then {A4}, {B2}, {C2}
    on the following lines (bias_lanemap.parse_mma takes only A/B/C).
    """
    sites = []
    for i, l in enumerate(ls):
        if MMA not in l:
            continue
        m0 = re.search(r"\{([^}]*)\}", l)
        regs = [re.findall(r"%r\d+", m0.group(1))]
        for k in range(1, 4):
            m = re.search(r"\{([^}]*)\}", ls[i + k])
            regs.append(re.findall(r"%r\d+", m.group(1)) if m else [])
        assert len(regs[0]) == 2 and len(regs[1]) == 4 and \
            len(regs[2]) == 2 and len(regs[3]) == 2, (i + 1, regs)
        sites.append((i + 1, regs[0], regs[1], regs[2], regs[3]))
    return sites

LEAF_MAGIC = 2146992128  # 0x7FF88000, the leaf addend (proven in section 28)
NLANES = 32


def nearest_def(ls, reg, use_idx):
    """Nearest preceding line mentioning `reg` (exact token match)."""
    want = re.compile(r"(?:^|[{\s,;])" + re.escape(reg) + r"(?:$|[\s,}])")
    for j in range(use_idx - 1, -1, -1):
        if want.search(ls[j]):
            return j
    return None


def parse_episodes(ls):
    """Find exp episodes; each chain -> (eline, E, D, leaf, leafln)."""
    shls = []
    for i, l in enumerate(ls):
        m = re.search(r"shl\.b32\s+(%r\d+),\s*(%r\d+),\s*5;", l)
        if m:
            shls.append((i, m.group(1), m.group(2)))
    eps, cur = [], []
    for item in shls:
        if cur and item[0] - cur[-1][0] > 40:
            eps.append(cur)
            cur = []
        cur.append(item)
    if cur:
        eps.append(cur)
    out = []
    for n, ep in enumerate(eps):
        chains = []
        for i, E, U in ep:
            j = nearest_def(ls, U, i)
            m1 = re.search(r"min\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);",
                           ls[j]) if j is not None else None
            if not (m1 and m1.group(1) == U):
                continue  # not an exp chain (prologue shl etc.)
            j2 = nearest_def(ls, m1.group(2), j)
            m2 = re.search(r"max\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);",
                           ls[j2]) if j2 is not None else None
            assert m2, (n, i)
            j3 = nearest_def(ls, m2.group(2), j2)
            m3 = re.search(r"fma\.rn\.f16x2\s+(%r\d+),\s*(%r\d+),\s*"
                           r"(%r\d+),\s*(%r\d+);", ls[j3]) if j3 else None
            assert m3, (n, i)
            D = m3.group(2)
            leaf = leafln = None
            for k in range(i + 1, min(i + 6, len(ls))):
                m4 = re.search(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*(\d+);",
                               ls[k])
                if m4 and m4.group(2) == E \
                        and int(m4.group(3)) == LEAF_MAGIC:
                    leaf, leafln = m4.group(1), k
                    break
            assert leaf, (n, i, E)
            chains.append((i, E, D, leaf, leafln))
        if chains:
            out.append(chains)
    print("episodes: %d; chains: %s" % (len(out), [len(c) for c in out]))
    return out


def run_groups(sites):
    """Group consecutive A-sharing sites; returns [[(idx, site)...]]."""
    groups, cur = [], []
    for idx, site in enumerate(sites):
        ln, D, A, B, C = site
        if cur and cur[-1][1][2] != A:
            groups.append(cur)
            cur = []
        cur.append((idx, site))
    if cur:
        groups.append(cur)
    return groups


def map_sites(ls, sites, chains):
    """Map each chain D reg -> (site_idx, dpos). Fail-closed."""
    res = []
    for i, E, D, leaf, leafln in chains:
        hits = []
        for idx, (ln, Dm, _A, _B, _C) in enumerate(sites):
            if ln - 1 < i and D in Dm:
                hits.append((idx, ln, Dm.index(D)))
        assert hits, ("D has no feeding site", D)
        # nearest preceding site (compiler reuse across waves -> last wins)
        hits.sort(key=lambda h: h[1])
        res.append((E, D, leaf, leafln, hits[-1][0], hits[-1][2]))
    return res


def blank4():
    return ("v4", [set(), set(), set(), set()])


def as_v4(cell):
    kind, pay = cell
    assert kind == "v4", ("not vector", cell)
    return pay


def parse_imm(tok):
    tok = tok.rstrip("U")
    return int(tok, 16) if tok.lower().startswith("0x") else int(tok)


class Machine:
    """32-lane executor propagating byte-label sets (fail-closed)."""

    def __init__(self):
        self.ival = [{} for _ in range(NLANES)]
        self.vval = [{} for _ in range(NLANES)]
        self.pred = [{} for _ in range(NLANES)]
        self.muls = []  # (line, lane, dest, srca, srcb, alabels, blabels)

    def iget(self, lane, tok):
        if tok == "%laneid":
            return lane
        if re.fullmatch(r"%r\d+|%rd\d+|%rs\d+", tok):
            v = self.ival[lane].get(tok)
            assert v is not None, ("undef int", lane, tok)
            return v
        return parse_imm(tok)

    def vget(self, lane, tok):
        v = self.vval[lane].get(tok)
        return v if v is not None else blank4()

    def vset(self, lane, reg, cell):
        self.vval[lane][reg] = cell

    def op_mov(self, d, s, lane):
        if s == "%laneid":
            self.ival[lane][d] = lane
            return
        if d.startswith("{"):
            regs = re.findall(r"[%a-zA-Z]\w*", d)
            assert len(regs) == 2, d
            _, b = self.vget(lane, s)
            b4 = list(b) + [set()] * (4 - len(b))
            self.vset(lane, regs[0], ("v2", [set(b4[0]), set(b4[1])]))
            self.vset(lane, regs[1], ("v2", [set(b4[2]), set(b4[3])]))
            return
        if s.startswith("{"):
            regs = re.findall(r"[%a-zA-Z]\w*", s)
            assert len(regs) == 2, s
            out = [set(), set(), set(), set()]
            for k, r in enumerate(regs):
                c = self.vval[lane].get(r)
                if c is None:
                    continue
                _, b = c
                out[2 * k] |= b[0]
                out[2 * k + 1] |= b[1]
            self.vset(lane, d, ("v4", out))
            return
        if re.fullmatch(r"%r\d+|%rd\d+|%rs\d+|[a-z]\w*", s):
            if s in self.ival[lane]:
                self.ival[lane][d] = self.ival[lane][s]
                return
            if re.fullmatch(r"[a-z]\w*", s) and s not in self.vval[lane]:
                try:
                    self.ival[lane][d] = parse_imm(s)
                    return
                except ValueError:
                    pass
            self.vset(lane, d, self.vget(lane, s))
            return
        self.ival[lane][d] = parse_imm(s)

    def op_int3(self, op, d, a, b, lane, signed=False):
        x, y = self.iget(lane, a), self.iget(lane, b)
        if op == "and":
            r = x & y
        elif op == "or":
            r = x | y
        elif op == "xor":
            r = x ^ y
        elif op == "add":
            r = x + y
        elif op == "sub":
            r = x - y
        elif op == "shl":
            r = x << y
        elif op == "shr":
            r = (x >> y) if not signed or x >= 0 else -((-x) >> y)
        else:
            raise KeyError(op)
        self.ival[lane][d] = r & 0xFFFFFFFF

    def op_setp(self, cond, p, a, b, lane):
        x, y = self.iget(lane, a), self.iget(lane, b)
        self.pred[lane][p] = (x != y) if cond == "ne" else (x == y)

    def op_selp(self, d, a, b, p, lane):
        assert p in self.pred[lane], ("undef pred", p)
        s = a if self.pred[lane][p] else b
        if s in self.ival[lane]:
            self.ival[lane][d] = self.ival[lane][s]
        else:
            self.vset(lane, d, self.vget(lane, s))

    def op_shfl(self, d, v, idx, lane):
        src = self.iget(lane, idx) % NLANES
        if v in self.ival[src]:
            self.ival[lane][d] = self.ival[src][v]
        else:
            cell = self.vval[src].get(v)
            self.vset(lane, d, ("v4", [set(x) for x in cell[1]])
                      if cell else blank4())

    def op_halves2(self, d, a, b, lane):
        A, B = as_v4(self.vget(lane, a)), as_v4(self.vget(lane, b))
        self.vset(lane, d, ("v4", [A[0] | B[0], A[1] | B[1],
                                   A[2] | B[2], A[3] | B[3]]))

    def op_prmt(self, d, a, b, sel, lane):
        A, B = as_v4(self.vget(lane, a)), as_v4(self.vget(lane, b))
        s = self.iget(lane, sel)
        flat = A + B
        out = [set(), set(), set(), set()]
        for i in range(4):
            idx = (s >> (4 * i)) & 0xF
            assert idx < 8, ("prmt idx", idx)
            out[i] = set(flat[idx])
        self.vset(lane, d, ("v4", out))

    def op_cvt(self, d, s, lane):
        if s == "%laneid":
            self.ival[lane][d] = lane
            return
        if s in self.ival[lane]:
            self.ival[lane][d] = self.ival[lane][s]
            return
        if re.fullmatch(r"%r\d+|%rd\d+|%rs\d+|[a-z]\w*", s) is None:
            try:
                self.ival[lane][d] = parse_imm(s)
                return
            except ValueError:
                pass
        c = self.vval[lane].get(s)
        if c is None:
            self.vset(lane, d, ("v2", [set(), set()]))
            return
        _, b = c
        b4 = list(b) + [set()] * (4 - len(b))
        if d.startswith("%rs") or re.fullmatch(r"[a-z]\w*", d):
            self.vset(lane, d, ("v2", [set(b4[0]) | set(b4[1]),
                                       set(b4[2]) | set(b4[3])]))
        else:
            self.vset(lane, d, ("v4", [set(x) for x in b4]))

    def run_window(self, ls, lo, hi):
        for i in range(lo, hi + 1):
            l = ls[i].strip().strip("{}").strip()
            if not l or l.startswith((".", "/", "{", "}")):
                continue
            l = l.rstrip(";")
            if "mma.sync" in l:
                # feeding MMAs all precede the window; any MMA inside is
                # a later wave writing unseeded D regs -- label-safe skip
                # (its operand-continuation lines are skipped below).
                continue
            if re.fullmatch(r"(%r\d+,?\s*)+\}?,?", l):
                continue  # MMA operand-continuation line
            m = re.match(r"mov\.[bsu](?:32|16)\s+(\{[^}]*\}|%r\d+|%rs\d+),"
                         r"\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_mov(m.group(1), m.group(2).strip(), ln)
                continue
            # leaf add: add.s32 LEAF, E, MAGIC reinterprets shifted E bits
            # as f16 exp values (proven commuting in section 28) --
            # label-preserving copy, fail-closed.
            m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*(\d+)$", l)
            if m and int(m.group(3)) == LEAF_MAGIC:
                for ln in range(NLANES):
                    src = self.vval[ln].get(m.group(2))
                    assert src is not None, ("leaf src unlabeled", ln, l)
                    _, b = src
                    self.vset(ln, m.group(1),
                              ("v4", [set(x) for x in b]))
                continue
            m = re.match(r"(and|or|xor|add|sub|shl|shr)\.[bsu](32|16)\s+"
                         r"(%r\d+|%rs\d+),\s*(%r\d+|%rs\d+|%laneid),\s*(.+)$",
                         l)
            if m:
                for ln in range(NLANES):
                    # shl on a labeled vector (E-chain shift) preserves
                    # provenance per the section 28 commutation proof.
                    if m.group(1) == "shl" and \
                            m.group(4) in self.vval[ln]:
                        src = self.vval[ln][m.group(4)]
                        _, b = src
                        self.vset(ln, m.group(3),
                                  ("v4", [set(x) for x in b]))
                    else:
                        self.op_int3(m.group(1), m.group(3), m.group(4),
                                     m.group(5).strip(), ln, ".s32" in l)
                continue
            m = re.match(r"fma\.rn\.f16x2\s+(%r\d+),\s*(%r\d+),\s*"
                         r"(%r\d+),\s*(%r\d+)$", l)
            if m:
                for ln in range(NLANES):
                    A = as_v4(self.vget(ln, m.group(2)))
                    B = as_v4(self.vget(ln, m.group(3)))
                    C = as_v4(self.vget(ln, m.group(4)))
                    self.vset(ln, m.group(1),
                              ("v4", [A[0] | B[0] | C[0], A[1] | B[1] | C[1],
                                       A[2] | B[2] | C[2], A[3] | B[3] | C[3]]))
                continue
            m = re.match(r"setp\.(ne|eq)\.[a-z0-9]+\s+(%p\d+),\s*"
                         r"(.+?),\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_setp(m.group(1), m.group(2), m.group(3).strip(),
                                 m.group(4).strip(), ln)
                continue
            m = re.match(r"selp\.b32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*"
                         r"(%p\d+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_selp(m.group(1), m.group(2), m.group(3),
                                 m.group(4), ln)
                continue
            m = re.match(r"shfl\.sync\.idx\.b32\s+(%r\d+)\|%p\d+,\s*"
                         r"(%r\d+),\s*(%r\d+),\s*31", l)
            if m:
                for ln in range(NLANES):
                    self.op_shfl(m.group(1), m.group(2), m.group(3), ln)
                continue
            m = re.match(r"mul\.f16x2\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_halves2(m.group(1), m.group(2), m.group(3), ln)
                    a = self.vget(ln, m.group(2))[1]
                    b = self.vget(ln, m.group(3))[1]
                    self.muls.append((i + 1, ln, m.group(1), m.group(2),
                                      m.group(3), [set(x) for x in a],
                                      [set(x) for x in b]))
                continue
            m = re.match(r"(add|max|min)\.f16x2\s+(%r\d+),\s*"
                         r"(%r\d+),\s*(%r\d+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_halves2(m.group(2), m.group(3), m.group(4), ln)
                continue
            m = re.match(r"prmt\.b32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*"
                         r"(.+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_prmt(m.group(1), m.group(2), m.group(3),
                                 m.group(4).strip(), ln)
                continue
            m = re.match(r"cvt\.[a-z0-9.]+\s+([a-z%]\w*),\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_cvt(m.group(1), m.group(2).strip(), ln)
                continue
            m = re.match(r"rcp\.approx\.ftz\.f32\s+([a-z%]\w*),\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    self.op_cvt(m.group(1), m.group(2).strip(), ln)
                continue
            raise KeyError("unsupported op @%d: %s" % (i + 1, l))


# ------------------------------------------------------------- premises
def frag_P1(lane):
    rows = [2 * (lane % 8), 2 * (lane % 8) + 1]
    cols = [2 * (lane // 8), 2 * (lane // 8) + 1]
    return rows, cols


def frag_P2(lane):
    return [lane % 16], [4 * (lane // 16) + k for k in range(4)]


def frag_P3(lane):
    return [4 * (lane // 8) + k for k in range(4)], [lane % 8]


PREMISES = {"P1-2x2": frag_P1, "P2-1x4": frag_P2, "P3-4x1": frag_P3}


def arrange(elements, mode):
    """Split 4 fragment elements (e00,e01,e10,e11) into (d0lo,d0hi,d1lo,d1hi).
    A1 row-major, A2 col-major, A3 cross."""
    e00, e01, e10, e11 = elements
    if mode == "A1":
        return (e00, e01, e10, e11)
    if mode == "A2":
        return (e00, e10, e01, e11)
    if mode == "A3":
        return (e00, e11, e01, e10)
    raise KeyError(mode)


def find_window(ls, mapped, raw):
    """Reduction window: first chain's u-step -> rcp-block rejoin mov.

    mapped elements are (E, D, leaf, leafln, site, dpos); raw elements
    are (shl_idx, E, D, leaf, leafln). lo backs off 20 lines to include
    the first chain's fma/max/min/shl (chains interleave: chain k+1's
    u-step follows chain k's leaf)."""
    lo = min(i for i, _E, _D, _lf, _ln in raw) - 20
    assert lo >= 0
    rcp_at = [i for i in range(lo, min(lo + 1200, len(ls)))
              if "rcp.approx.ftz.f32" in ls[i]]
    assert rcp_at, "no rcp in window"
    outreg = None
    for i in range(rcp_at[-1] + 1, min(rcp_at[-1] + 12, len(ls))):
        m = re.search(r"mov\.b32\s+(%r\d+),\s*\{", ls[i])
        if m:
            outreg = (m.group(1), i)
            break
    assert outreg, "no rcp rejoin"
    mul_at = [i for i in range(outreg[1] + 1, min(outreg[1] + 801, len(ls)))
              if re.search(r"mul\.f16x2\s+%r\d+,%r\d+,%r\d+;",
                           ls[i].strip().strip("{}").strip())]
    assert mul_at, "no P-muls after rejoin"
    return lo, mul_at[-1], outreg[0]


def simulate(ls, raw, chains, groups, premise, arrangement):
    sidx = {}
    for g, grp in enumerate(groups):
        for p, (idx, _site) in enumerate(grp):
            sidx[idx] = (g, p)
    frag = PREMISES[premise]
    mach = Machine()
    seeds = set()
    for _E, D, _leaf, _leafln, site, dpos in chains:
        g, p = sidx[site]
        for lane in range(NLANES):
            rows, cols = frag(lane)
            elems = [(r, c) for r in rows for c in cols]
            assert len(elems) == 4, (premise, lane, rows, cols)
            e00, e01, e10, e11 = elems
            q = arrange((e00, e01, e10, e11), arrangement)
            halves = [q[0:2], q[2:4]][dpos]
            lo = {(g, r, p, c) for r, c in [halves[0]]}
            hi = {(g, r, p, c) for r, c in [halves[1]]}
            # Seed the score-MMA D b32 halves (one element per f16 half;
            # halves are positional through the u-step fma). Labels flow
            # D -> E -> leaf mechanically through the supported u-step ops.
            seeds.add(next(iter(lo)))
            seeds.add(next(iter(hi)))
            cur = mach.vval[lane].get(D)
            if cur is None:
                mach.vval[lane][D] = ("v4", [set(lo), set(lo),
                                             set(hi), set(hi)])
    lo, hi, outreg = find_window(ls, chains, raw)
    mach.run_window(ls, lo, hi)
    res = []
    for lane in range(NLANES):
        _, b = mach.vval[lane][outreg]
        res.append(set(b[0]) | set(b[1]) | set(b[2]) | set(b[3]))
    return res, (lo, hi, outreg), mach, seeds


def quad_of(fr):
    return (fr % 2) * 2 + (1 if fr >= 8 else 0)


def true_row(lbl):
    run, fr, _sc, fc = lbl
    return (run, fc // 2, quad_of(fr))


def true_col(lbl):
    _run, fr, sc, fc = lbl
    return (fr, sc, fc)


def check_partition(esum_sets, seeds):
    """Every seeded E-item in exactly one lane's Esum (necessary for a
    valid per-row softmax)."""
    seen = {}
    dup = 0
    for lane, s in enumerate(esum_sets):
        for x in s:
            if x in seen:
                dup += 1
            seen[x] = lane
    return len(seen), dup, len(seeds) - len(seen)


def check_pnormalize(esum_sets, muls):
    """Every P-item scaled by the Esum of its own TRUE row.

    Returns (nmuls, nchecked, bad) where bad lists mismatches."""
    lane_of = {}
    for lane, s in enumerate(esum_sets):
        lane_of[frozenset(s)] = lane
    true_of_lane = {}
    for lane, s in enumerate(esum_sets):
        rows = {true_row(x) for x in s}
        assert len(rows) == 1, ("esum spans rows", lane, rows)
        cols = {true_col(x) for x in s}
        assert len(cols) == 64, ("esum cols undisjoint", lane, len(cols))
        true_of_lane[lane] = next(iter(rows))
    bad, nchecked = [], 0
    psrcs = set()
    for line, lane, _d, sa, sb, la, lb in muls:
        pa = set().union(*la) if la else set()
        pb = set().union(*lb) if lb else set()
        # identify the rcp side (full-b32 union must equal some lane's
        # Esum set exactly -- broadcasts preserve the set by construction)
        ka, kb = frozenset(pa), frozenset(pb)
        if ka in lane_of and kb not in lane_of:
            # A is the rcp broadcast; P is side B
            phalves = [set(lb[0]) | set(lb[1]), set(lb[2]) | set(lb[3])]
            elane = lane_of[ka]
            psrcs.add((lane, sb))
        elif kb in lane_of and ka not in lane_of:
            # B is the rcp broadcast; P is side A
            phalves = [set(la[0]) | set(la[1]), set(la[2]) | set(la[3])]
            elane = lane_of[kb]
            psrcs.add((lane, sa))
        else:
            bad.append((line, lane, "side-id", sa, sb))
            continue
        for h, ph in enumerate(phalves):
            prows = {true_row(x) for x in ph}
            nchecked += 1
            if prows != {true_of_lane[elane]}:
                bad.append((line, lane, "row-mismatch/h%d" % h, prows,
                            true_of_lane[elane]))
    return len(muls), nchecked, bad, len(psrcs)


def verdict(lanesets):
    """Grade per-lane Esum label sets. Returns (grade, detail)."""
    grades = []
    for s in lanesets:
        rows = sorted({x[:2] for x in s})
        cols = sorted({x[2:] for x in s})
        n = len(s)
        if len(rows) == 1 and n == 64 and len(cols) == 64:
            grades.append("CLEAN")
        elif len(rows) == 2 and rows[0][1] == rows[1][1] and n == 64:
            c0 = sorted({x[2:] for x in s if x[:2] == rows[0]})
            c1 = sorted({x[2:] for x in s if x[:2] == rows[1]})
            if len(c0) == 32 and len(c1) == 32 \
                    and not (set(c0) & set(c1)):
                grades.append("MERGE")
            else:
                grades.append("FAIL")
        else:
            grades.append("FAIL")
        detail = (len(rows), n, len(cols))
    gset = sorted(set(grades))
    if len(gset) == 1:
        # The naive axis-aligned criterion is EXPECTED to fail everywhere:
        # the proven TRUE row/col factorization (§30) is diagonal in
        # (lane, site, D-half) coordinates, so no axis-aligned premise
        # can reconstruct full rows. "axis:NO" is the green signal here.
        return ("axis:YES-" + gset[0] if gset[0] != "FAIL" else "axis:NO",
                detail)
    return "MIXED+" + "/".join(gset), detail


def main():
    ls = kernel_lines()
    sites = parse_mma_full(ls)
    print("mma sites: %d" % len(sites))
    groups = run_groups(sites)
    print("A-sharing runs: %d sizes=%s" %
          (len(groups), sorted({len(g) for g in groups})))
    episodes = parse_episodes(ls)
    for n, chains in enumerate(episodes):
        mapped = map_sites(ls, sites, chains)
        runs = sorted({sidx for _, _, _, _, sidx, _ in mapped})
        print("-- episode %d: %d chains; feeding site idx range [%d,%d]; "
              "distinct sites %d" %
              (n, len(mapped), min(runs), max(runs),
               len({s for _, _, _, _, s, _ in mapped})))
        for premise in PREMISES:
            for arr in ("A1", "A2", "A3"):
                try:
                    res, (lo, hi, out), mach, seeds = simulate(
                        ls, chains, mapped, groups, premise, arr)
                except (KeyError, AssertionError, IndexError) as e:
                    print("  ep%d %s+%s: SIM-ERROR %s" % (n, premise, arr,
                                                          str(e)[:100]))
                    continue
                g, _d = verdict(res)
                print("  ep%d %s+%s: %s win=[%d,%d] out=%s nmuls=%d" %
                      (n, premise, arr, g, lo + 1, hi + 1, out,
                       len(mach.muls) // NLANES))
                if premise == "P1-2x2" and arr == "A1":
                    cov, dup, miss = check_partition(res, seeds)
                    print("    partition: covered=%d dup=%d missing=%d "
                          "(seeds=%d)" % (cov, dup, miss, len(seeds)))
                    trows = sorted({true_row(x)
                                    for s in res for x in s})
                    print("    TRUE rows: %d distinct; runs=%s" %
                          (len(trows), sorted({t[0] for t in trows})))
                    try:
                        nm, nck, bad, nps = check_pnormalize(res, mach.muls)
                        print("    pnormalize: muls=%d halfchecks=%d "
                              "bad=%d psrcs=%d" % (nm, nck, len(bad), nps))
                        for b in bad[:8]:
                            print("      BAD:", b)
                    except AssertionError as e:
                        print("    pnormalize ASSERT:", str(e)[:160])
                    leaves = {lf for _E, _D, lf, _ln, _s, _d in mapped}
                    psrc = {sa for _, _, _, sa, sb, _, _ in mach.muls} | \
                        {sb for _, _, _, sa, sb, _, _ in mach.muls}
                    print("    P-src coverage: leaves=%d mulsrcs=%d "
                          "leaves-mulsrcs=%d" %
                          (len(leaves), len(psrc), len(leaves - psrc)))


if __name__ == "__main__":
    main()
