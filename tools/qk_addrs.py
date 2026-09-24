#!/usr/bin/env python3
"""tools/qk_addrs.py -- TRUE row->query-row / TRUE col->key-col mapping.

Method: backward row-identity walk from score-MMA A/B fragments through
the (positional, lane-local, shuffle-free) Q/K chains -- quant, QK-norm
muls (loaded uniform scales: row-transparent), QKV GEMMs -- down to
global loads. Load addresses are evaluated per lane by a small
forward int64 machine; (address - tile base) // stride gives query/key
indices. QKV GEMM D<-A row micro-rules are tried in candidate order
(P1A1-straight first: the Esum-proven arrangement) and SCORED
machine-side: every (lane, score-half) must resolve to a SINGLETON
query/key index, and the row pairing must reproduce the O-side
lane-quad<->TRUE-pair structure (tools/o_fanin.py, HANDOFF section 32).
Fail-closed throughout: taint instead of guessing, report instead of
forcing.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import parse_mma_full, run_groups  # noqa: E402
from o_fanin import def_walk  # noqa: E402

NLANES = 32

MMA_F16 = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
MMA_E4M3 = "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"


def parse_mma_all(ls):
    """[(ln, D, A, B, C, kind)] for both score (e4m3) and QKV (f16) forms.
    Arities are identical (D2/A4/B2/C2); geometry shared (m16n8)."""
    sites = []
    for i, l in enumerate(ls):
        kind = None
        if MMA_E4M3 in l:
            kind = "e4m3"
        elif MMA_F16 in l:
            kind = "f16"
        if kind is None:
            continue
        m0 = re.search(r"\{([^}]*)\}", l)
        regs = [re.findall(r"%r\d+", m0.group(1))]
        for k in range(1, 4):
            m = re.search(r"\{([^}]*)\}", ls[i + k])
            regs.append(re.findall(r"%r\d+", m.group(1)) if m else [])
        assert len(regs[0]) == 2 and len(regs[1]) == 4 and \
            len(regs[2]) == 2 and len(regs[3]) == 2, (i + 1, regs)
        sites.append((i + 1, regs[0], regs[1], regs[2], regs[3], kind))
    return sites


def is_fimm(tok):
    return bool(re.match(r"0[fF][0-9a-fA-F]+$", tok))


def qk_mention(ls, reg, j):
    """Tolerant mention test (accepts predicated dests `%rX|%pY`, which
    shared nearest_def skips -- kept local to avoid disturbing the
    pinned esum/o_fanin walks)."""
    return bool(re.search(r"(?:^|[{\s,;])" + re.escape(reg) +
                          r"(?:$|[\s,}|])", ls[j]))


def qk_is_def(ls, reg, j):
    """Def-shape twin of o_fanin.def_walk (plus `|`-suffixed dests)."""
    l = ls[j].strip().strip("{}").strip().rstrip(";")
    m = re.match(r"(?:mov|and|or|xor|add|sub|shl|shr|selp|cvt|cvta|"
                 r"mul|mad|div|rem|fma|max|min|abs|neg|sqrt|rsqrt|rcp|"
                 r"sin|cos|ex2|lg2|setp|ld|shfl|prmt|tex|movmatrix)"
                 r"\.[a-z0-9.]+\s+"
                 r"(\{[^}]*\}|%rd\d+|%r\d+|%rs\d+|[a-z]\w*)"
                 r"(\|%p\d+)?,?", l)
    if m:
        d = m.group(1)
        if d.startswith("{"):
            if reg in re.findall(r"%rd\d+|%r\d+|%rs\d+|[a-z]\w*", d):
                return True
        elif d == reg:
            return True
    m = re.match(r"mma\.sync\S*\s+\{([^}]*)\},?", ls[j].strip())
    if m and reg in re.findall(r"%r\d+", m.group(1)):
        return True
    return False


def qk_def_walk(ls, reg, use, fwd=False):
    """Nearest def, tolerant mentions; fwd=True searches forward
    (pipeline-carried). Fail-closed."""
    rng = range(use + 1, len(ls)) if fwd else range(use - 1, -1, -1)
    for j in rng:
        if qk_mention(ls, reg, j) and qk_is_def(ls, reg, j):
            return j
    assert False, (("no fwd def" if fwd else "no def"), reg, use)


def same_scope(ls, name, def_j, use_j):
    """True iff no `.reg ... name` redeclaration sits between def and
    use (block-scoped hl/hu/fl/fu/fh reuse names across blocks)."""
    pat = re.compile(r"\.reg\.\S+\s+[^;]*\b" + re.escape(name) + r"\b")
    lo, hi = (def_j, use_j) if def_j < use_j else (use_j, def_j)
    return not any(pat.search(ls[k]) for k in range(lo + 1, hi))


def parse_params(ls):
    """Param-struct bases: {reg: struct_offset} from ld.param.b64 lines."""
    out = {}
    for i, l in enumerate(ls[:4000]):
        m = re.search(r"ld\.param\.b64\s+(%rd\d+),\s*\[%rd8\+(\d+)\]",
                      l.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


class AddrMachine:
    """Forward per-lane int64 machine for address expressions."""

    def dirty(self, lane, tok):
        return bool(re.fullmatch(r"%r\d+|%rd\d+|%rs\d+", tok or "")) and \
            tok in self.t[lane]

    def unk(self, lane, tok):
        """Never-assigned reg: loop-carried value under the straight-line
        pass (e.g. prologue load loop) -> treated as tainted."""
        return bool(re.fullmatch(r"%r\d+|%rd\d+|%rs\d+", tok or "")) and \
            tok not in self.v[lane] and tok not in self.t[lane]

    def bad(self, lane, *toks):
        return any(self.dirty(lane, t) or self.unk(lane, t) for t in toks)

    def g(self, lane, tok):
        # Special regs: warp-0 assumptions (tid.y=0, ctaid=0) recorded
        # in run(); %tid.x = lane (warp-relative, matches %laneid).
        if tok == "%laneid":
            return lane
        if tok in ("%tid.x",):
            return lane
        if tok in ("%tid.y", "%tid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z"):
            return 0
        if tok in ("%ntid.x", "%ntid.y", "%ntid.z"):
            return 1
        if re.fullmatch(r"%r\d+|%rd\d+|%rs\d+", tok):
            assert tok not in self.t[lane], ("tainted int", lane, tok)
            x = self.v[lane].get(tok)
            assert x is not None, ("undef int", lane, tok)
            return x
        t = tok.rstrip("U")
        return int(t, 16) if t.lower().startswith("0x") else int(t)

    def __init__(self):
        self.v = [{} for _ in range(NLANES)]
        self.p = [{} for _ in range(NLANES)]
        self.t = [set() for _ in range(NLANES)]  # float-tainted regs
        self.pt = [set() for _ in range(NLANES)]  # tainted predicates

    def seed_params(self, ls, lo, hi):
        """Variation-preserving sentinels: pointer bases -> 0 (additive),
        scalar params -> 1 (multiplicative user sees lane terms)."""
        for i in range(lo, hi + 1):
            l = ls[i].strip()
            m = re.match(r"ld\.param\.b64\s+(%rd\d+),", l)
            if m:
                for ln in range(NLANES):
                    self.v[ln][m.group(1)] = 0
                continue
            m = re.match(r"ld\.param\.v\d+\.b32\s+\{([^}]*)\},", l)
            if m:
                for r in re.findall(r"%r\d+", m.group(1)):
                    for ln in range(NLANES):
                        self.v[ln][r] = 1
                continue
            m = re.match(r"ld\.param\.b32\s+(%r\d+),", l)
            if m:
                for ln in range(NLANES):
                    self.v[ln][m.group(1)] = 1

    def run(self, ls, lo, hi):
        self.seed_params(ls, lo, hi)
        for i in range(lo, hi + 1):
            l = ls[i].strip().strip("{}").strip().rstrip(";")
            if not l or l.startswith((".", "/", "{", "}", "$", "@", ")")) \
                    or ".param" in l or l.startswith("ld.param"):
                continue
            if l.startswith(("ld.", "st.", "red.", "atom.", "bar.",
                             "tex.", "suld.", "sust.", "cp.async",
                             "membar", "fence")):
                continue  # memory ops: values out of scope for addresses
            if l.startswith("bra") or l.startswith("@"):
                # straight-line assumption (see module docstring): branches
                # guard stores/exits; predicated math would need auditing
                # if address results ever look inconsistent.
                continue
            if re.fullmatch(r"(%r\d+,?\s*|%rd\d+,?\s*|%rs\d+,?\s*)+\}?,?",
                             l):
                continue  # MMA operand-continuation line
            if "mma.sync" in l:
                continue  # MMA value path out of scope for addresses
            m = re.match(r"mov\.b64\s+\{([^}]*)\},\s*(%rd\d+)$", l)
            if m:
                members = [x.strip() for x in m.group(1).split(",")]
                assert len(members) == 2, l
                for ln in range(NLANES):
                    if self.bad(ln, m.group(2)):
                        for mb in members:
                            if mb != "_" and re.fullmatch(r"%r\d+|%rs\d+",
                                                          mb):
                                self.t[ln].add(mb)
                        continue
                    v = self.g(ln, m.group(2))
                    if members[0] != "_":
                        self.v[ln][members[0]] = v & 0xFFFFFFFF
                        self.t[ln].discard(members[0])
                    if members[1] != "_":
                        self.v[ln][members[1]] = (v >> 32) & 0xFFFFFFFF
                        self.t[ln].discard(members[1])
                continue
            m = re.match(r"mov\.b64\s+(%rd\d+),\s*\{([^}]*)\}$", l)
            if m:
                members = [x.strip() for x in m.group(2).split(",")]
                assert len(members) == 2, l
                for ln in range(NLANES):
                    if any(self.bad(ln, mb) for mb in members
                           if mb != "_"):
                        self.t[ln].add(m.group(1))
                        continue
                    lo = self.g(ln, members[0]) if members[0] != "_" else 0
                    hi = self.g(ln, members[1]) if members[1] != "_" else 0
                    self.v[ln][m.group(1)] = (lo | (hi << 32)) & \
                        0xFFFFFFFFFFFFFFFF
                    self.t[ln].discard(m.group(1))
                continue
            m = re.match(r"mov\.[ubs](?:32|64|16)\s+(%r\d+|%rd\d+|%rs\d+),"
                         r"\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    s = m.group(2).strip()
                    if self.dirty(ln, s):
                        self.t[ln].add(m.group(1))
                    elif re.fullmatch(r"%r\d+|%rd\d+|%rs\d+|%laneid|%tid\.x",
                                       s):
                        self.v[ln][m.group(1)] = self.g(ln, s)
                        self.t[ln].discard(m.group(1))
                    else:
                        try:
                            self.v[ln][m.group(1)] = self.g(ln, s)
                            self.t[ln].discard(m.group(1))
                        except ValueError:
                            # symbolic base (param struct symbol) -> 0
                            self.v[ln][m.group(1)] = 0
                            self.t[ln].discard(m.group(1))
                continue
            m = re.match(r"(and|or|xor|add|sub|shl|shr|mul\.wide|mul\.lo|"
                         r"mul\.hi|div|rem)\.[ubs](?:32|64|16)\s+"
                         r"(%r\d+|%rd\d+|%rs\d+),\s*(.+?),\s*(.+)$", l)
            if m:
                op, d, a, b = m.group(1), m.group(2), m.group(3), \
                    m.group(4).strip()
                for ln in range(NLANES):
                    if self.bad(ln, a) or self.bad(ln, b.strip()):
                        self.t[ln].add(d)
                        continue
                    x, y = self.g(ln, a), self.g(ln, b.strip())
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
                        r = (x >> y) if x >= 0 else -((-x) >> y)
                    elif op == "mul.wide":
                        r = (x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)
                    elif op == "mul.lo":
                        r = (x * y) & 0xFFFFFFFF
                    elif op == "mul.hi":
                        r = ((x * y) >> 32) & 0xFFFFFFFF
                    elif op == "div":
                        assert y != 0
                        q = abs(x) // abs(y)
                        r = q if (x >= 0) == (y >= 0) else -q
                    elif op == "rem":
                        assert y != 0
                        q = abs(x) // abs(y)
                        q = q if (x >= 0) == (y >= 0) else -q
                        r = x - q * y
                    else:
                        raise KeyError(op)
                    self.v[ln][d] = r & 0xFFFFFFFFFFFFFFFF
                    self.t[ln].discard(d)
                continue
            m = re.match(r"mad\.lo\.s32\s+(%r\d+),\s*(.+?),\s*(.+?),\s*(.+)$",
                         l)
            if m:
                for ln in range(NLANES):
                    if self.bad(ln, m.group(2).strip()) or \
                            self.bad(ln, m.group(3).strip()) or \
                            self.bad(ln, m.group(4).strip()):
                        self.t[ln].add(m.group(1))
                        continue
                    self.v[ln][m.group(1)] = (self.g(ln, m.group(2)) *
                                              self.g(ln, m.group(3)) +
                                              self.g(ln, m.group(4))) & \
                        0xFFFFFFFFFFFFFFFF
                    self.t[ln].discard(m.group(1))
                continue
            m = re.match(r"cvta\.[a-z0-9.]+\s+(%rd\d+),\s*(%rd\d+)$", l)
            if m:
                for ln in range(NLANES):
                    if self.bad(ln, m.group(2)):
                        self.t[ln].add(m.group(1))
                    else:
                        self.v[ln][m.group(1)] = self.g(ln, m.group(2))
                        self.t[ln].discard(m.group(1))
                continue
            m = re.match(r"cvt\.([a-z0-9.]+)\s+"
                         r"(%r\d+|%rd\d+|%rs\d+|[a-z]\w*),\s*(.+)$", l)
            if m:
                floaty = ("f16" in m.group(1) or "f32" in m.group(1) or
                          "f64" in m.group(1) or "e4m3" in m.group(1) or
                          "bf16" in m.group(1))
                for ln in range(NLANES):
                    if floaty:
                        self.t[ln].add(m.group(2))
                    elif self.bad(ln, m.group(3).strip()):
                        self.t[ln].add(m.group(2))
                    else:
                        self.v[ln][m.group(2)] = self.g(ln, m.group(3).strip())
                        self.t[ln].discard(m.group(2))
                continue
            m = re.match(r"setp\.(gt|ge|lt|le|ne|eq|ltu|leu|gtu|geu)\.([a-z0-9.]+)"
                         r"\s+(%p\d+),\s*(.+?),\s*(.+)$", l)
            if m:
                for ln in range(NLANES):
                    if is_fimm(m.group(4).strip()) or \
                            is_fimm(m.group(5).strip()) or \
                            self.bad(ln, m.group(4).strip()) or \
                            self.bad(ln, m.group(5).strip()):
                        self.pt[ln].add(m.group(3))
                        continue
                    self.pt[ln].discard(m.group(3))
                    x, y = self.g(ln, m.group(4).strip()), \
                        self.g(ln, m.group(5).strip())
                    c = m.group(1)
                    xu, yu = x & 0xFFFFFFFFFFFFFFFF, y & 0xFFFFFFFFFFFFFFFF
                    self.p[ln][m.group(3)] = \
                        (x > y if c == "gt" else x >= y if c == "ge" else
                         x < y if c == "lt" else x <= y if c == "le" else
                         xu > yu if c == "gtu" else xu >= yu if c == "geu" else
                         xu < yu if c == "ltu" else xu <= yu if c == "leu" else
                         x != y if c == "ne" else x == y)
                continue
            m = re.match(r"(\S+)\s+(\{[^}]*\}|%r\d+|%rs\d+|[a-z]\w*),?.*$",
                         l)
            if m and re.search(r"f16|f32|f64|e4m3|bf16", m.group(1)) \
                    and not m.group(1).startswith(("setp", "selp")):
                d = m.group(2)
                regs = re.findall(r"%r\d+|%rs\d+|[a-z]\w+",
                                  d) if d.startswith("{") else [d]
                for ln in range(NLANES):
                    for r in regs:
                        if re.fullmatch(r"%r\d+|%rs\d+|[a-z]\w+", r):
                            self.t[ln].add(r)
                continue
            m = re.match(r"(and|or|xor)\.pred\s+(%p\d+),\s*(%p\d+),\s*(%p\d+)$",
                         l)
            if m:
                for ln in range(NLANES):
                    a = self.p[ln].get(m.group(3), False)
                    b = self.p[ln].get(m.group(4), False)
                    op = m.group(1)
                    self.p[ln][m.group(2)] = (a and b if op == "and" else
                                              a or b if op == "or" else
                                              a != b)
                continue
            m = re.match(r"not\.pred\s+(%p\d+),\s*(%p\d+)$", l)
            if m:
                for ln in range(NLANES):
                    self.p[ln][m.group(1)] = not self.p[ln].get(m.group(2),
                                                               False)
                continue
            m = re.match(r"selp\.[a-z0-9.]+\s+(%r\d+),\s*(.+?),\s*(.+?),"
                         r"\s*(%p\d+)$", l)
            if m:
                for ln in range(NLANES):
                    if m.group(4) in self.pt[ln] or \
                            self.dirty(ln, m.group(2).strip()) or \
                            self.bad(ln, m.group(3).strip()) or \
                            re.match(r"0[fF][0-9a-fA-F]+$",
                                     m.group(2).strip()) or \
                            re.match(r"0[fF][0-9a-fA-F]+$",
                                     m.group(3).strip()):
                        self.t[ln].add(m.group(1))
                        continue
                    assert m.group(4) in self.p[ln], ("undef pred", m.group(4))
                    s = m.group(2).strip() if self.p[ln][m.group(4)] \
                        else m.group(3).strip()
                    self.v[ln][m.group(1)] = self.g(ln, s)
                    self.t[ln].discard(m.group(1))
                continue
            if "shfl." in l:
                continue  # cross-lane values: out of scope for addresses
            if "movmatrix." in l:
                continue  # warp-transpose values: out of scope
            if "prmt." in l:
                continue  # byte permute: data path, not addresses
            m = re.match(r"mov\.b32\s+\{([^}]*)\},\s*(.+)$", l)
            if m and not re.search(r"%r\d+|%rd\d+",
                                   m.group(1).replace("%rs", "")):
                continue  # f16 pack/split or scoped regs: data path
            raise KeyError("addr-op @%d: %s" % (i + 1, l))


def eval_addr(am, ls, line_idx):
    """(base_reg, per-lane offsets or None, const) for the ld/st at line."""
    l = ls[line_idx].strip().strip("{}").strip()
    m = re.search(r"\[\s*(%rd\d+)\s*(?:\+\s*(\d+))?\s*\]", l)
    assert m, ("no addr", line_idx + 1, l[:80])
    base, const = m.group(1), int(m.group(2)) if m.group(2) else 0
    offs = []
    for ln in range(NLANES):
        if base in am.t[ln]:
            return base, None, const
        v = am.v[ln].get(base)
        if v is None:
            return base, None, const
        offs.append(v)
    return base, offs, const


class RowWalker:
    """Backward row-identity walk (b32 granularity) with candidate QKV
    D<-A micro-rules. Row items are (kind, payload): LOAD(base, off),
    CONST (row-free), TAINT(msg). Unions are sound; singletons exact."""

    def __init__(self, ls, sites, am):
        self.ls = ls
        self.sites = sites
        self.site_by_ln = {s[0]: s for s in sites}
        self.am = am
        self.memo = {}
        self.active = set()  # in-progress keys (reduction-cycle cut)
        self.cycles = []  # cut keys, for audit
        self.piped = []  # (reg, use_ln, fwd_def_ln) pipeline-carried fallbacks
        self.texcoords = {}  # (def_idx, dest_pos) -> (texobj, [coord regs])
        self.rule = ("P1A1", "straight")
        self.walkC = True
        self.cskipped = set()
        self.swrites = []  # (addr_reg, addr_const, val_regs, line_idx)
        for i, l in enumerate(ls):
            m = re.search(r"st\.shared\S*\s+\[\s*(%r\d+)\s*(?:\+\s*(-?\d+))?"
                          r"\s*\],\s*\{([^}]*)\}", l.strip().strip("{}").strip())
            if m:
                vals = re.findall(r"%r\d+", m.group(3))
                self.swrites.append((m.group(1), int(m.group(2))
                                     if m.group(2) else 0, vals, i))
        print("st.shared writes: %d" % len(self.swrites))
        # Pass-0 write-window bridge (kernel lines 167/181, idx 166/180).
        # PROOF (in-code, see HANDOFF): head passes k=0,1 max
        # (r5032=tid+32*ntid.y*k exits at >=64); pass>=1 writes never
        # intersect the four read zones ([512,1024) unread; write2+k>=1
        # above all reads). So every read is served by pass 0, whose
        # addresses are loop-invariant (r100 single def @123):
        #   write1 lane WL: [r100(WL), +16) <- {r396,r398,r402,r406}@166
        #   write2 lane WL: [r100(WL)+1024, +16) <- {r410,r413,r417,r419}@180
        self.bridge = {}  # addr -> (value_reg, write_idx, wlane, pass)
        self.readpass = {}  # (read_idx, lane) -> pass (y+4 mapping)
        w1 = ["%r396", "%r398", "%r402", "%r406"]
        w2 = ["%r410", "%r413", "%r417", "%r419"]
        for wl in range(NLANES):
            b = self.am.v[wl].get("%r100")
            if b is None:
                continue
            for k in range(4):
                self.bridge[(b + 4 * k) & 0xFFFFFFFF] = (w1[k], 166, wl, 0)
                self.bridge[(b + 1024 + 4 * k) & 0xFFFFFFFF] = \
                    (w2[k], 180, wl, 0)
                self.bridge[(b + 512 + 4 * k) & 0xFFFFFFFF] = \
                    (w1[k], 166, wl, 1)
                self.bridge[(b + 1536 + 4 * k) & 0xFFFFFFFF] = \
                    (w2[k], 180, wl, 1)
        print("bridge addrs: %d" % len(self.bridge))

    def smem_read(self, addr_reg, lane, read_ln, depth, const=0):
        """Rows for smem addr_reg at lane via the pass-0 bridge table
        (pass>=1 writes provably miss all read zones). The value reg is
        walked at the pass-0 write line under the WRITER lane (swizzle).
        No match -> loud taint (dead zone [512,1024) unread by proof)."""
        if addr_reg in self.am.t[lane]:
            return frozenset([("TAINT", "smemaddr")])
        av = self.am.v[lane].get(addr_reg)
        if av is None:
            return frozenset([("TAINT", "smemaddr")])
        av = (av + const) & 0xFFFFFFFF
        hit = self.bridge.get(av)
        if hit is None:
            return frozenset([("TAINT", "smemnomatch@%d" % read_ln)])
        vreg, widx, wlane, pas = hit
        self.readpass[(read_ln, lane)] = pas
        return self.walk(vreg, widx, wlane, depth + 1)

    def aloads(self, reg, use):
        """A-reg index pairing for D-regs under the candidate rule.

        Returns list of (dreg_pos, areg_pos_list): which A-regs feed
        D-reg position dreg_pos (0/1 of the D pair). P1A1-straight:
        positional pairs."""
        premise, order = self.rule
        if premise == "P1A1":
            pairs = [([0, 1], 0), ([2, 3], 1)] if order == "straight" \
                else [([2, 3], 0), ([0, 1], 1)]
            return pairs
        raise KeyError(self.rule)

    def walk(self, reg, use, lane, depth=0):
        """Row-identity items for (lane, reg): LOAD(base, off), LEAF(reg),
        TAINT(msg). Lane-local (no shuffles in the Q/K chains; a shfl
        encountered here taints loudly). With walkC=False, MMA C inputs
        are recorded in cskipped, not walked (A-path isolation)."""
        assert depth < 500, ("walk too deep", reg, use)
        if re.fullmatch(r"%(laneid|tid\.[xyz]|ntid\.[xyz]|ctaid\.[xyz]|"
                        r"nctaid\.[xyz]|warpid|nwarpid|smid|nsmid|gridid|"
                        r"clock|clock64|lanemask_[a-z]+|envreg\d+)", reg):
            return frozenset()  # launch geometry index: row-free
        key = (reg, use, lane, self.walkC)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            # Reduction cycle (butterfly accumulation re-enters the same
            # def): the cyclic edge adds no new X-rows beyond the seed
            # paths, which are walked non-cyclically. Sound to cut.
            self.cycles.append(key)
            return frozenset()
        self.active.add(key)
        ls = self.ls
        try:
            j = qk_def_walk(ls, reg, use)
        except AssertionError:
            # Software-pipeline carried reg (head use before body def):
            # the entry-pass value is the next textual def. Sound for
            # bridge consumers (pass-tagged at mapping); counted.
            j = qk_def_walk(ls, reg, use, fwd=True)
            self.piped.append((reg, use + 1, j + 1))
        l = ls[j].strip().strip("{}").strip().rstrip(";")
        out = None
        m = re.match(r"mov\.b32\s+(%r\d+),\s*\{([^}]*)\}$", l)
        if m and m.group(1) == reg:
            out = frozenset()
            for mb in re.findall(r"%r\d+|%rs\d+|[a-z]\w+", m.group(2)):
                out |= self.walk(mb, j, lane, depth + 1)
        m = re.match(r"mov\.b32\s+\{([^}]*)\},\s*(%r\d+)$", l)
        if m and out is None:
            # split: scalar dests inherit the whole source rows
            # (b32 granularity; halves share rows under P1A1).
            out = self.walk(m.group(2), j, lane, depth + 1)
        m = re.match(r"mov\.b64\s+\{_,\s*(%r\d+)\},\s*(%rd\d+)$", l)
        if m and m.group(1) == reg and out is None:
            # address upper-half word (clamp bounds, seeds): row-free.
            out = frozenset()
        m = re.match(r"mov\.b64\s+\{(%r\d+),\s*(%r\d+)\},\s*(%rd\d+)$", l)
        if m and out is None and reg in (m.group(1), m.group(2)):
            # address words as floats (clamp bounds): row-free.
            out = frozenset()
        m = re.match(r"shr\.[a-z0-9.]+\s+(%r\d+),\s*(%r\d+),\s*(\d+)$",
                     l)
        if m and m.group(1) == reg and out is None:
            # lane-local bit-split (align/quad idiom): rows preserved.
            out = self.walk(m.group(2), j, lane, depth + 1)
        m = re.match(r"(div|rem)\.\S+\s+(%r\d+),\s*(%r\d+),\s*"
                     r"(%r\d+|0[fF][0-9a-fA-F]+|-?\d+)$", l)
        if m and m.group(2) == reg and out is None:
            out = frozenset()
            for src in (m.group(3), m.group(4)):
                if re.match(r"%r\d+$", src):
                    out |= self.walk(src, j, lane, depth + 1)
        m = re.search(r"shfl\.sync\.idx\.\S+\s+(%r\d+)(?:\|%p\d+)?,\s*"
                      r"(%r\d+),\s*(%r\d+),\s*(\d+),\s*(-?\d+)", l)
        if m and m.group(1) == reg and out is None:
            # indexed gather: value at lane = src[idx(lane)]. The idx
            # reg is int-machine-evaluated; unknown -> loud taint.
            iv = self.am.v[lane].get(m.group(3))
            if iv is not None and 0 <= iv < NLANES and \
                    m.group(3) not in self.am.t[lane]:
                out = self.walk(m.group(2), j, iv, depth + 1)
            else:
                out = frozenset([("TAINT", "shflidx@%d" % (j + 1))])
        m = re.match(r"prmt\.b32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*"
                     r"(0[xX][0-9a-fA-F]+U?|\d+)$", l)
        if m and m.group(1) == reg and out is None:
            # lane-local byte permute: rows preserved.
            out = frozenset()
            for src in (m.group(2), m.group(3)):
                out |= self.walk(src, j, lane, depth + 1)
        m = re.match(r"movmatrix\.sync\.trans\.aligned\.m8n8\.b16\s+"
                     r"(%r\d+),\s*(%r\d+)$", l)
        if m and m.group(1) == reg and out is None:
            # Octet-transpose peer rule (VALIDATED lane-level by
            # tools/o_vcover.py gate V: uniform 2-cover + octet
            # gather/scatter; intra-lane element order stays with the
            # frag-row micro-geometry open item). NOTE: movmatrix is
            # NOT on the Q/K path (QK B-quants predate it); it feeds
            # V-side O-MMAs, so it stays out of the Q/K maps.
            g, i = lane // 8, lane % 8
            out = frozenset()
            for peer in (8 * (i // 2) + 2 * g, 8 * (i // 2) + 2 * g + 1):
                out |= self.walk(m.group(2), j, peer, depth + 1)
        m = re.match(r"mad\.lo\.s32\s+(%r\d+),\s*(%r\d+),\s*"
                     r"(%r\d+|-?\d+),\s*(%r\d+|-?\d+)$", l)
        if m and m.group(1) == reg and out is None:
            # int hash chains (coord hashing): walk reg sources.
            out = frozenset()
            for src in (m.group(2), m.group(3), m.group(4)):
                if re.match(r"%r\d+$", src):
                    out |= self.walk(src, j, lane, depth + 1)
        m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*(?!2146992128$)"
                     r"(\d+)$", l)
        if m and m.group(1) == reg and out is None:
            out = self.walk(m.group(2), j, lane, depth + 1)
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
                    out = self.walk(src, j, lane, depth + 1)
        m = re.match(r"(mul|add|max|min|fma|or|xor|and)\.[a-z0-9.]+\s+"
                     r"(%r\d+),\s*(%r\d+),\s*(%r\d+|0[fF][0-9a-fA-F]+|-?\d+)"
                     r"(,\s*(%r\d+|0[fF][0-9a-fA-F]+|-?\d+))?$", l)
        if m and m.group(2) == reg and out is None and \
                m.group(3) != "2146992128" and m.group(4) != "2146992128":
            out = frozenset()
            for op in [m.group(3), m.group(4)] + ([m.group(6)] if m.group(6)
                                                  else []):
                if re.match(r"%r\d+$", op):
                    out |= self.walk(op, j, lane, depth + 1)
        m = re.match(r"add\.s32\s+(%r\d+),\s*(%r\d+),\s*2146992128$", l)
        if m and m.group(1) == reg and out is None:
            out = frozenset([("LEAF", reg)])
        m = re.match(r"(abs|neg|sqrt|rsqrt|rcp|sin|cos|ex2|lg2)\.\S+\s+"
                     r"(%r\d+|%rs\d+),\s*(%r\d+|%rs\d+)$", l)
        if m and m.group(2) == reg and out is None:
            out = self.walk(m.group(3), j, lane, depth + 1)
        m = re.match(r"shl\.b32\s+(%r\d+),\s*(%r\d+),\s*(\d+)$", l)
        if m and m.group(1) == reg and out is None:
            # pack-hi / stride scaling: rows preserved.
            out = self.walk(m.group(2), j, lane, depth + 1)
        m = re.match(r"shr\.[a-z0-9.]+\s+(%r\d+),\s*(%r\d+),\s*(%r\d+)$",
                     l)
        if m and m.group(1) == reg and out is None:
            # variable shift (hash finalization): lane-local, rows kept.
            out = self.walk(m.group(2), j, lane, depth + 1)
        m = re.match(r"selp\.\S+\s+(%r\d+),\s*(\S+),\s*(\S+),\s*(%p\d+)$",
                     l)
        if m and m.group(1) == reg and out is None:
            out = frozenset()
            for src in (m.group(2), m.group(3)):
                if re.match(r"%r\d+|%rs\d+$", src):
                    out |= self.walk(src, j, lane, depth + 1)
        m = re.match(r"(mul|fma)\.ftz\.f32\s+(%r\d+),\s*(%r\d+),\s*"
                     r"(%r\d+|0[fF][0-9a-fA-F]+)(,\s*(%r\d+|"
                     r"0[fF][0-9a-fA-F]+))?$", l)
        if m and m.group(2) == reg and out is None:
            out = frozenset()
            for src in [m.group(3), m.group(4)] + ([m.group(6)] if
                                                   m.group(6) else []):
                if re.match(r"%r\d+$", src):
                    out |= self.walk(src, j, lane, depth + 1)
        m = re.match(r"(mul|add|sub|div|max|min|fma)\.f16\s+(%rs\d+),\s*"
                     r"(%rs\d+),\s*(%rs\d+)(,\s*(%rs\d+))?$", l)
        if m and m.group(2) == reg and out is None:
            out = frozenset()
            for src in [m.group(3), m.group(4)] + ([m.group(6)] if
                                                   m.group(6) else []):
                out |= self.walk(src, j, lane, depth + 1)
        m = re.match(r"(mov|cvt|rsqrt|sqrt|rcp)\.\S+\s+(fl|fu|hl|hu|fh),"
                     r"\s*(fl|fu|hl|hu|fh|%r\d+|%rs\d+)$", l)
        if m and m.group(2) == reg and out is None:
            if same_scope(ls, reg, j, use):
                out = self.walk(m.group(3), j, lane, depth + 1)
            else:
                out = frozenset([("TAINT", "scope@%d" % (j + 1))])
        m = re.match(r"mov\.\S+\s+(\S+?),\s*(\S+)$", l)
        if m and m.group(1) == reg and out is None:
            src = m.group(2)
            if src.startswith("{"):
                out = frozenset([("TAINT", "pack@%d" % (j + 1))])
            elif src == "%laneid":
                out = frozenset()
            else:
                try:
                    int(src.rstrip("U"), 0)
                    out = frozenset()
                except ValueError:
                    out = self.walk(src, j, lane, depth + 1)
        if out is None and "shfl.sync.bfly" in l:
            m2 = re.search(r"shfl\.sync\.bfly\.\S+\s+(%r\d+),\s*(%r\d+),"
                           r"\s*(%r\d+),\s*(%r\d+),\s*(%r\d+)", l)
            assert m2, ("bfly shape", j + 1, l[:80])
            assert m2.group(1) == reg, ("bfly use-as-def", j + 1, reg)
            masks = {self.am.v[m].get(m2.group(3)) for m in range(NLANES)}
            masks.discard(None)
            if len(masks) == 1:
                out = self.walk(m2.group(2), j, lane ^ next(iter(masks)),
                                depth + 1)
            else:
                out = frozenset([("TAINT", "bflymask@%d" % (j + 1))])
        if out is None and ("shfl." in l or "shfl " in l):
            out = frozenset([("TAINT", "shfl@%d" % (j + 1))])
        if out is None and l.startswith("ld.shared"):
            m2 = re.search(r"\[\s*(%r\d+)\s*(?:\+\s*(\d+))?\s*\]", l)
            assert m2, ("shared ld w/o addr", j + 1, l[:80])
            out = self.smem_read(m2.group(1), lane, j, depth,
                                 int(m2.group(2)) if m2.group(2) else 0)
        if out is None and l.startswith("tex."):
            m2 = re.match(r"tex\.\S+\s+\{([^}]*)\},\s*\[\s*(%rd\d+),\s*"
                          r"\{([^}]*)\}\]$", l)
            assert m2, ("tex shape", j + 1, l[:80])
            dests = re.findall(r"%r\d+", m2.group(1))
            coords = re.findall(r"%r\d+", m2.group(3))
            assert reg in dests, ("tex use-as-def", j + 1, reg)
            self.texcoords[(j, dests.index(reg))] = \
                (m2.group(2), coords)
            out = frozenset([("TEX", m2.group(2), dests.index(reg))])
        if out is None and l.startswith("ld."):
            m2 = re.search(r"\[\s*(%rd\d+)\s*(?:\+\s*(\d+))?\s*\]", l)
            assert m2, ("ld w/o addr", j + 1, l[:80])
            base, const = m2.group(1), int(m2.group(2)) if m2.group(2) \
                else 0
            if base in self.am.t[lane]:
                out = frozenset([("TAINT", "addr@%d" % (j + 1))])
            else:
                v = self.am.v[lane].get(base)
                out = frozenset([("LOAD", base, v + const)]) \
                    if v is not None else \
                    frozenset([("TAINT", "addr@%d" % (j + 1))])
        if out is None:
            # MMA D: A fan-in per candidate rule + C union, same lane;
            # B = weights (row-free hypothesis, verified by scoring).
            hits = [(ln, s) for ln, s in self.site_by_ln.items()
                    if ln - 1 <= j and reg in s[1]]
            if not hits:
                out = frozenset([("TAINT", "op@%d" % (j + 1))])
            else:
                ln, s = max(hits, key=lambda h: h[0])
                dpos = s[1].index(reg)
                pairs = self.aloads(reg, use)
                aregs = s[2]
                want = next(p[0] for p in pairs if p[1] == dpos)
                out = frozenset()
                for ap in want:
                    out |= self.walk(aregs[ap], ln - 1, lane, depth + 1)
                if self.walkC:
                    for cr in s[4]:
                        out |= self.walk(cr, ln - 1, lane, depth + 1)
                else:
                    for cr in s[4]:
                        self.cskipped.add((ln, cr))
        self.active.discard(key)
        self.memo[key] = out
        return out


def main():
    ls = kernel_lines()
    params = parse_params(ls)
    print("param bases: %s" % params)
    sites = parse_mma_all(ls)
    print("mma sites: e4m3=%d f16=%d" %
          (sum(1 for s in sites if s[5] == "e4m3"),
           sum(1 for s in sites if s[5] == "f16")))
    am = AddrMachine()
    try:
        am.run(ls, 0, 12000)
        print("addr machine: clean to 12000")
    except KeyError as e:
        print("addr-machine stopped:", str(e)[:120])
    except AssertionError as e:
        print("addr-machine stopped:", str(e)[:160])
    # Q-driver, run 36, P1A1-straight, A-path only (walkC=False):
    # rs-level walks (A-reg pack members) for lanes 0-3, site 176.
    wk = RowWalker(ls, sites, am)
    wk.walkC = False
    groups = run_groups(parse_mma_full(ls))
    g36 = [x for x in groups if any(i == 176 for i, _ in x)][0]
    try:
        idx, (ln, D, A, B, C) = g36[0]
        print("site176 ln=%d D=%s A=%s B=%s" % (ln, D, A, B))
        e4 = [(s[0], s[2], s[3]) for s in sites if s[5] == "e4m3"]
        print("e4m3 groups: %d" % len(e4))
        print("survey scope: QK+score MMAs (<12456); proj/Esum/O at 12456+"
              " are esum_ownership/o_fanin territory (own gates)")
        scount = {}
        for ln2, A2, B2 in e4:
            if ln2 >= 12456:
                continue
            ab = []
            for side, regs in (("A", A2), ("B", B2)):
                kinds = set()
                for ar in regs:
                    try:
                        j = qk_def_walk(ls, ar, ln2 - 1)
                    except AssertionError:
                        wk.active.clear()
                        kinds.add("NODEF")
                        continue
                    m = re.match(r"mov\.b32\s+%r\d+,\s*\{([^}]*)\}$",
                                 ls[j].strip().strip("{}").strip()
                                 .rstrip(";"))
                    packed = re.findall(r"%rs\d+", m.group(1)) if m else []
                    for mb, uu in ([(mb, j) for mb in packed]
                                   or [(ar, j + 1)]):
                        r = wk.walk(mb, uu, 0)
                        for it in r:
                            kinds.add(it[0])
                ab.append("%s:%s" % (side, ",".join(sorted(kinds))))
            print("  @%d %s" % (ln2, " ".join(ab)))
            scount[" ".join(ab)] = scount.get(" ".join(ab), 0) + 1
        print("survey summary: %s" %
              ", ".join("%dx(%s)" % (v, k)
                        for k, v in sorted(scount.items())))
        nload = ntaint = ntex = 0
        for lane in range(32):
            for ar in A:
                j = qk_def_walk(ls, ar, ln - 1)
                m = re.match(r"mov\.b32\s+%r\d+,\s*\{([^}]*)\}$",
                             ls[j].strip().strip("{}").strip().rstrip(";"))
                members = re.findall(r"%rs\d+", m.group(1))
                for mb in members:
                    r = wk.walk(mb, j, lane)
                    loads = sorted([it for it in r if it[0] == "LOAD"],
                                   key=str)
                    texs = sorted([it for it in r if it[0] == "TEX"],
                                  key=str)
                    other = sorted({it for it in r if it[0] not in
                                    ("LOAD", "TEX")}, key=str)
                    nload += len(loads)
                    ntaint += len([x for x in other if x[0] == "TAINT"])
                    ntex += len(texs)
                    if lane < 4:
                        print("L%d %s.%s: loads=%s tex=%s other=%s" %
                              (lane, ar, mb, loads, texs, other))
        print("totals: loads=%d taints=%d texs=%d cskipped=%d" %
              (nload, ntaint, ntex, len(wk.cskipped)))
        print("texcoords:", len(wk.texcoords), "piped:", len(wk.piped),
              "cycles:", len(wk.cycles))
        for (j, pos), (tob, cr) in sorted(wk.texcoords.items()):
            print("  tex@%d pos%d obj=%s coords=%s :: %s" %
                  (j + 1, pos, tob, cr, ls[j].strip()[:100]))
        for reg, u, d in sorted({(r, uu, dd)
                                 for r, uu, dd in wk.piped}):
            print("  pipe %s use@%d fwddef@%d" % (reg, u, d))
    except AssertionError as e:
        wk.active.clear()
        print("WALK-ASSERT:", str(e)[:200])


if __name__ == "__main__":
    main()
