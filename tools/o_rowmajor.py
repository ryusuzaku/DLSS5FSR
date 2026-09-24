#!/usr/bin/env python3
"""tools/o_rowmajor.py -- row-major token closure (gate R).

Closes the load-bearing row-major token assumption (HANDOFF 33.5
item 4): kernel token t = pixel (t&7, t>>3). Method: absolute texel
composition (int-exact coord formulas + bridge routing + forward
positional walk) checked against gate-D (q,k) pixels.

R1. Coord formulas (mini int evaluator, params symbolic, ctaid
    symbolic): x-int(w) == 8*Cx+(w&7); y-int(w,k) ==
    8*Cy+(w>>3)+4k, all 32 lanes x k in {0,1}; numeric values at
    Cx=Cy=0; separability at 3 CTA points; loop shapes pinned
    (r64/r57/r5032/r5031/r56 inits+steps+exit). Scopes shared with
    section 8: tid.y=0, ntid.y=1, single-warp CTA, interior window.
R2. Fold census: coord-family uses in [1,400) all fold/compare/
    tex/cvt-entry shapes; zero mentions later. Interior
    conditions recorded (x-int<r72, y-int<r6 -- discharged by the
    interior-window scope; the folds are the edge-mirror
    mechanism, consistent with the stated edge=mirror rule).
R3. Window bijectivity: 64 (w,k) -> 64 distinct texels = full 8x8.
R4. Forward positional walk (texel sets; halves/members ordered;
    MMA row-rule (D <- A-row + C full union); bfly peer-union
    with int masks; movmatrix
    peer-union then VSTOP-boundary; shfl-idx/prmt flow tex-free
    lane-union but TEX lane-crossing -> FAIL; other
    shfl/unknown -> FAIL; MMA B-side must be TEX-free (weights);
    f16 QK mmas take the crossed path) from staging seeds
    (zone-exact: pass zones select k) to all QK A/B fragments
    (both waves, 32 sites), k-loop region [150,330) visited per-k
    (k-exact fetches/zones), outside consumers take last-wins
    (k=1) for region defs. Terminal asserts vs gate-D pixels
    (gates C+D, keeper-pinned, cited as the (q,k) input axiom):
    per-(site,lane,member) TEX subset of the slot-algebra pair
    (<=2 texels; >=3 -> FAIL); F (global scales/bias: %r2390
    path + mma-C) and C (lane arithmetic) allowed as token-free
    modulation; X (other textures) / V (V-cone) -> FAIL.
    Coverage: per-site A == 16 query pixels; per-quadrant B ==
    16 key pixels; A-wave == 32 wave queries; B-window == 8x8.
R5. LOAD/scale boundary (counts; FOREIGN modulation uniform at
    all termini, X/V=0; modulation-only cited from section 8) +
    V-side exclusion (VSTOP stops).
R6. Verdict ROWMAJOR-OK (CLOSED under R1 scopes + interior
    conditions).

Stdlib-only, GPU-free, byte-deterministic stdout. Slow gate
(~5 min: addr machine + forward cone); runbook 16.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bias_lanemap import kernel_lines  # noqa: E402
from esum_ownership import parse_mma_full  # noqa: E402
from qk_addrs import AddrMachine, RowWalker, parse_mma_all  # noqa: E402
from o_fanin import def_walk  # noqa: E402

NLANES = 32

# ---------------------------------------------------------------- R1 trees
# Value trees: ('const', n) | ('lane',) | ('cx',) | ('cy',) |
# ('param', name) | ('loopvar', which, lane, k) | ('op', name, args...)
# loopvar which: 'r5032' (= lane+32k) | 'r5031base' (r100 part, k-free)

def E(node):
    return node


def op_tree(*a):
    return ("op",) + a


def eval_int(ls, reg, use_idx, lane, k, memo):
    """Straight-line int value tree for `reg` at use (defs strictly
    before use_idx; r5032/r5031 resolved via k; params symbolic).
    Raises OpError on anything non-int-modelable (fail-closed)."""
    key = (reg, use_idx, lane, k)
    if key in memo:
        return memo[key]
    if reg == "%laneid" or reg == "%tid.x":
        return ("lane",)
    if reg == "%tid.y":
        return ("const", 0)
    if reg == "%ctaid.x":
        return ("cx",)
    if reg == "%ctaid.y":
        return ("cy",)
    if reg == "%ntid.y":
        return ("const", 1)
    if reg == "%r5032":
        return ("loopvar", "r5032", lane, k)
    if reg == "%r5031":
        return ("loopvar", "r5031", lane, k)
    m0 = re.fullmatch(r"-?0[xX][0-9a-fA-F]+U?|-?\d+U?", reg or "")
    if m0:
        t = reg.rstrip("U")
        return ("const", int(t, 16) if t.lower().startswith("0x")
                else int(t))
    from qk_addrs import qk_def_walk as _qdw
    try:
        j = _qdw(ls, reg, use_idx)
    except AssertionError:
        raise OpError(("nodef", reg, use_idx))
    l = ls[j].strip().strip("{}").strip().rstrip(";")
    out = model_line(ls, l, reg, j, lane, k, memo)
    memo[key] = out
    return out


class OpError(Exception):
    pass


def model_line(ls, l, reg, j, lane, k, memo):
    """Value tree for dest `reg` defined by line `l` (def index j)."""
    def ev(tok):
        return eval_int(ls, tok, j, lane, k, memo)
    m = re.match(r"mov\.u32\s+(%r\d+),\s*(\S+)$", l)
    if m and m.group(1) == reg:
        return ev(m.group(2))
    m = re.match(r"mov\.b32\s+(%r\d+),\s*(\S+)$", l)
    if m and m.group(1) == reg:
        src = m.group(2)
        if src.startswith("{") or src.startswith("_"):
            # pack or symbol base: symbol bases are lane-free consts
            # (tin config symbol); packs need not appear in coord chains.
            if src.startswith("_"):
                return ("const", 0)
            raise OpError(("pack", j + 1, l[:60]))
        return ev(src)
    m = re.match(r"(and|or|xor)\.b32\s+(%r\d+),\s*(\S+),\s*(\S+)$", l)
    if m and m.group(2) == reg:
        return ("op", m.group(1), ev(m.group(3)), ev(m.group(4)))
    m = re.match(r"(shl|shr)\.(b32|u32|s32)\s+(%r\d+),\s*(\S+),\s*(\S+)$",
                 l)
    if m and m.group(3) == reg:
        return ("op", m.group(1), ev(m.group(4)), ev(m.group(5)))
    m = re.match(r"(add|sub)\.(s32|u32)\s+(%r\d+),\s*(\S+),\s*(\S+)$", l)
    if m and m.group(3) == reg:
        return ("op", m.group(1), ev(m.group(4)), ev(m.group(5)))
    m = re.match(r"setp\.lt\.(s32|u32)\s+(%p\d+),\s*(\S+),\s*(\S+)$", l)
    if m:
        return ("op", "setp-lt", ev(m.group(3)), ev(m.group(4)))
    m = re.match(r"selp\.b32\s+(%r\d+),\s*(\S+),\s*(\S+),\s*(%p\d+)$", l)
    if m and m.group(1) == reg:
        return ("op", "selp", ("pred", m.group(4)), ev(m.group(2)),
                ev(m.group(3)))
    m = re.match(r"ld\.param\.v2\.b32\s+\{(%r\d+),\s*(%r\d+)\},\s*"
                 r"\[%rd8\+(\d+)\]$", l)
    if m and reg in (m.group(1), m.group(2)):
        return ("param", "p%d" % int(m.group(3)))
    m = re.match(r"ld\.param\.(b32|b64)\s+(%r\w+),\s*\[%rd8\+(\d+)\]$", l)
    if m and m.group(2) == reg:
        return ("param", "p%d" % int(m.group(3)))
    raise OpError(("opaque", j + 1, l[:70]))


def tree_num(t, lane, k, cx=0, cy=0):
    """Numerically evaluate a tree (params stay symbolic -> error)."""
    knd = t[0]
    if knd == "const":
        return t[1]
    if knd == "lane":
        return lane
    if knd == "cx":
        return cx
    if knd == "cy":
        return cy
    if knd == "loopvar":
        if t[1] == "r5032":
            return lane + 32 * k
        raise OpError(("num-loopvar", t[1]))
    if knd == "param":
        raise OpError(("num-param", t[1]))
    if knd == "pred":
        raise OpError(("num-pred",))
    _op, name = t[0], t[1]
    if name == "setp-lt":
        a, b = tree_num(t[2], lane, k, cx, cy), tree_num(t[3], lane, k,
                                                         cx, cy)
        return a < b
    if name == "selp":
        raise OpError(("num-selp",))
    a = tree_num(t[2], lane, k, cx, cy)
    b = tree_num(t[3], lane, k, cx, cy)
    if name == "and":
        return a & b
    if name == "or":
        return a | b
    if name == "xor":
        return a ^ b
    if name == "shl":
        return a << b
    if name == "shr":
        return a >> b
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    raise OpError(("num-op", name))


def strip_line(ls, idx):
    return ls[idx].strip().strip("{}").strip().rstrip(";")


def assert_line(ls, ln, want):
    got = strip_line(ls, ln - 1)
    assert got == want, (ln, got[:100])
    return got


def check_R1(ls):
    """Coord formulas: trees, values, separability, loop shapes."""
    memo = {}
    lane_t = lambda n: ("const", n)  # noqa: E731
    X86 = lambda lane: ("op", "or", ("op", "and", ("lane",),  # noqa: E731
                                     ("const", 7)),
                        ("op", "shl", ("cx",), ("const", 3)))
    P72 = ("param", "p208")
    X88 = lambda lane: ("op", "add", (  # noqa: E731
        "op", "sub", ("op", "shl", P72, ("const", 1)), X86(lane)),
        ("const", -2))
    P6 = ("param", "p208")
    for lane in range(NLANES):
        t = eval_int(ls, "%r89", 106, lane, 0, memo)
        exp = ("op", "selp", ("pred", "%p6"), X86(lane), X88(lane))
        assert t == exp, (lane, t)
        t1 = eval_int(ls, "%r89", 106, lane, 1, memo)
        assert t1 == exp, ("k-indep", lane)
    Y108 = lambda lane, k: ("op", "add", (  # noqa: E731
        "op", "shr", ("loopvar", "r5032", lane, k), ("const", 3)),
        ("op", "shl", ("cy",), ("const", 3)))
    Y110 = lambda lane, k: ("op", "add", (  # noqa: E731
        "op", "sub", ("op", "shl", P6, ("const", 1)), Y108(lane, k)),
        ("const", -2))
    for lane in range(NLANES):
        for k in (0, 1):
            t = eval_int(ls, "%r111", 279, lane, k, memo)
            exp = ("op", "selp", ("pred", "%p7"), Y108(lane, k),
                   Y110(lane, k))
            assert t == exp, (lane, k, t)
    # numeric interior arms at Cx=Cy=0
    for lane in range(NLANES):
        t = eval_int(ls, "%r89", 106, lane, 0, memo)
        assert tree_num(t[3], lane, 0) == (lane & 7), lane
        for k in (0, 1):
            u = eval_int(ls, "%r111", 279, lane, k, memo)
            assert tree_num(u[3], lane, k) == ((lane >> 3) + 4 * k), \
                (lane, k)
    # separability at 3 CTA points
    for cx, cy in ((0, 0), (5, 7), (1, 3)):
        for lane in range(NLANES):
            t = eval_int(ls, "%r89", 106, lane, 0, memo)
            assert tree_num(t[3], lane, 0, cx, cy) - (lane & 7) == \
                8 * cx, (lane, cx)
            for k in (0, 1):
                u = eval_int(ls, "%r111", 279, lane, k, memo)
                assert tree_num(u[3], lane, k, cx, cy) - \
                    ((lane >> 3) + 4 * k) == 8 * cy, (lane, k, cy)
    # loop + base shapes pinned literally
    assert_line(ls, 15, "mov.u32 %r3, %tid.y")
    assert_line(ls, 16, "shl.b32 %r64, %r3, 5")
    assert_line(ls, 17, "mov.u32 %r4, %tid.x")
    assert_line(ls, 18, "add.s32 %r5032, %r64, %r4")
    assert_line(ls, 21, "mov.u32 %r70, %ntid.y")
    assert_line(ls, 119, "shl.b32 %r96, %r3, 9")
    assert_line(ls, 120, "shl.b32 %r97, %r4, 4")
    assert_line(ls, 121, "add.s32 %r98, %r96, %r97")
    got122 = strip_line(ls, 121)
    assert got122.startswith("mov.b32 %r99, _ZZNK6tin3_1"), got122[:60]
    assert_line(ls, 123, "add.s32 %r100, %r98, %r99")
    assert_line(ls, 124, "add.s32 %r5031, %r100, 1024")
    assert_line(ls, 125, "shl.b32 %r56, %r70, 9")
    assert_line(ls, 126, "shl.b32 %r57, %r70, 5")
    assert_line(ls, 182, "add.s32 %r5032, %r5032, %r57")
    assert_line(ls, 183, "add.s32 %r5031, %r5031, %r56")
    assert_line(ls, 184, "setp.lt.u32 %p39, %r5032, 64")
    # fold predicate shapes
    assert_line(ls, 101, "setp.lt.s32 %p6, %r86, %r72")
    s107 = strip_line(ls, 189)
    assert re.match(r"setp\.lt\.(s32|u32)\s+%p7,\s*%r108,\s*%r6$", s107), \
        s107[:80]
    print("R1 coord formulas: x-int=8Cx+(w&7), y-int=8Cy+(w>>3)+4k "
          "all 32 lanes x k=0,1 (trees exact; values + 3-point "
          "separability exact)")
    print("R1 loop shapes pinned (r5032=lane+32k, r5031=r100+1024+512k, "
          "exit r5032<64): scopes tid.y=0 ntid.y=1 (section 8)")


FAMILY = ["%r6", "%r72", "%r73", "%r7", "%r86", "%r87", "%r88", "%r89",
          "%r107", "%r108", "%r109", "%r110", "%r111"]


def check_R2(ls):
    """Fold census: family uses closed; interior conditions recorded."""
    fam = set(FAMILY)
    want = re.compile(r"(?:^|[{\s,;])(%r\d+)(?:$|[\s,}|])")
    nuses = 0
    for i, l in enumerate(ls):
        if i + 1 >= 400:
            for m in want.finditer(l):
                assert m.group(1) not in fam, \
                    ("late family use", i + 1, l.strip()[:80])
            continue
        for m in want.finditer(l):
            reg = m.group(1)
            if reg not in fam:
                continue
            nuses += 1
            s = strip_line(ls, i)
            ok = False
            dm = re.match(r"(?:@%p\d+\s+)?([\w.]+)\s+(\S+?),", s)
            dop = dm.group(1).split(".")[0] if dm else ""
            dest = dm.group(2) if dm else ""
            if dop in ("mov", "and", "or", "shl", "shr", "add", "sub",
                       "selp", "setp", "ld") and \
                    reg in re.findall(r"%r\d+", dest):
                ok = True
            if not ok and i + 1 == 285 and reg in ("%r52",):
                ok = True
            if not ok and re.match(r"cvt\.rn\.f32\.s32\s+"
                                   r"%(r91|r183),\s*" + reg + "$", s):
                ok = True  # float entry (token boundary)
            if not ok and reg not in re.findall(r"%r\d+", dest) and \
                    dop in ("mov", "and", "or", "shl", "shr", "add",
                            "sub", "mul", "xor", "selp", "setp", "cvt",
                            "tex", "ld"):
                # source use inside int/coord machinery (incl. the
                # coord-hash side chain r86->r90->r50->r113->r114...,
                # sink-checked: hashing only, never re-enters coords)
                ok = True
            assert ok, ("family use outside fold machinery", i + 1,
                        s[:90])
    print("R2 fold census: %d family mentions, all fold/compare/tex/"
          "cvt-entry; zero at >=400" % nuses)
    print("R2 interior conditions (scope-discharged): x-int<r72 "
          "(p6), y-int<r6 (p7); folds = edge-mirror")


def check_R3(ls):
    """Window bijectivity at Cx=Cy=0."""
    memo = {}
    pts = set()
    for lane in range(NLANES):
        t = eval_int(ls, "%r89", 106, lane, 0, memo)
        x = tree_num(t[3], lane, 0)
        for k in (0, 1):
            u = eval_int(ls, "%r111", 279, lane, k, memo)
            pts.add((x, tree_num(u[3], lane, k)))
    assert len(pts) == 64, len(pts)
    assert pts == {(x, y) for x in range(8) for y in range(8)}, \
        sorted(pts)[:8]
    print("R3 window cover: 64 (lane,k) -> 64 distinct texels = full 8x8")


# ---------------------------------------------------------------- R4 engine
# Forward values: TEXSET frozenset((x,y)) | CONST | FOREIGN | VSTOP.
# Nodes: (lane, reg, member) with member in {0} (%rs) or {0,1} (b32).
# k-loop region [150,330) (1-based lines) visited per k in {0,1}
# (k-exact LOOPVAR/seeds); outside: single visit, last-wins (k=1)
# for region defs. Packs/cvt positional (little-endian: lo<-first).
# MMA: D-member m <- A-row members m + C-row members m (like-for-like
# halves); B-side must be TEX-free (weights). bfly: peer-union with
# int masks. movmatrix: peer-union (gate-V rule) then VSTOP (stop).
# shfl (non-bfly) / prmt / unknown -> FAIL.

KREGION = (150, 330)
TEXREGS_W1 = ["%r396", "%r398", "%r402", "%r406"]
TEXREGS_W2 = ["%r410", "%r413", "%r417", "%r419"]
TEXMEMBERS = {"%r402", "%r406", "%r410"}  # TEX-carrying write regs
# (zonebase, write regs, zone pass)
ZONES = [(0, TEXREGS_W1, 0), (512, TEXREGS_W1, 1),
         (1024, TEXREGS_W2, 0), (1536, TEXREGS_W2, 1)]


class Fail(Exception):
    pass


def texel_of(wl, pas):
    return ((wl & 7), ((wl >> 3) + 4 * pas))


def parse_operands(s):
    """(op, dest, srcs, pred) from a stripped line."""
    m = re.match(r"(?:@(%p\d+)\s+)?([\w.]+)\s+(.*)$", s)
    if not m:
        return None
    _pred, op, rest = m.groups()
    parts = [p.strip() for p in rest.rstrip(";").split(",")]
    return (op, parts[0] if parts else "", parts[1:] if len(parts) > 1
            else [], _pred)


def is_b32_reg(tok):
    return bool(re.fullmatch(r"%r\d+", tok or ""))


def is_rs_reg(tok):
    return bool(re.fullmatch(r"%rs\d+", tok or ""))


C_ATOM = ("C",)
V_ATOM = ("V",)
F_ATOM = ("F",)


def union_vals(*vals):
    out = set()
    for v in vals:
        out |= set(v)
    if V_ATOM in out:
        return frozenset([V_ATOM])
    return frozenset(out)


def tex_atoms(v):
    return {a for a in v if len(a) == 3 and a[0] == "T"}


def tex_pairs(v):
    return {(a[1], a[2]) for a in v if len(a) == 3 and a[0] == "T"}


QK_W0 = [10839 + 7 * i for i in range(16)]
QK_W1 = [12684 + 7 * i for i in range(16)]


class Fwd:
    """Forward positional token walk. state[(kctx,lane,reg,m)] with
    kctx in {0,1} inside KREGION, {None} outside."""

    def __init__(self, ls, am, bridge):
        self.ls = ls
        self.am = am
        self.bridge = bridge
        self.st = {}
        self.sites = {s[0]: s for s in parse_mma_full(ls)}
        self.skip = set()
        for ln in self.sites:
            self.skip.update([ln + 1, ln + 2, ln + 3])
        # non-site mmas (f16 QK form): crossed path (B tex-free)
        self.xmma = {}
        for i, l in enumerate(ls):
            if "mma.sync.aligned" not in l:
                continue
            if i + 1 in self.sites:
                continue
            regs = []
            m0 = re.search(r"\{([^}]*)\}", l)
            regs.append(re.findall(r"%r\d+", m0.group(1))
                        if m0 else [])
            for kk in range(1, 4):
                m = re.search(r"\{([^}]*)\}", ls[i + kk])
                regs.append(re.findall(r"%r\d+", m.group(1))
                            if m else [])
            if len(regs[0]) != 2 or len(regs[1]) != 4 or \
                    len(regs[2]) != 2 or len(regs[3]) != 2:
                raise Fail(("xmma-arity", i + 1,
                            [len(r) for r in regs]))
            self.xmma[i + 1] = (i + 1, regs[0], regs[1], regs[2],
                                regs[3])
            self.skip.update([i + 2, i + 3, i + 4])
        self.termini = []  # (wave, siteidx, side, reg, lane, m, val)
        self.counts = defaultdict(int)
        self.vstop_lines = []
        self.xtouches = []
        # write shapes pinned
        assert strip_line(ls, 166) == \
            "st.shared.v4.b32 [%r5031+-1024], " \
            "{%r396, %r398, %r402, %r406}", strip_line(ls, 166)[:80]
        assert strip_line(ls, 180) == \
            "st.shared.v4.b32 [%r5031], " \
            "{%r410, %r413, %r417, %r419}", strip_line(ls, 180)[:80]
        # control shapes in k-region
        bras = []
        for i in range(KREGION[0] - 1, KREGION[1] - 1):
            s = strip_line(ls, i)
            if "bra" in s:
                bras.append((i + 1, s[:60]))
        for ln, s in bras:
            assert ln in (185, 186, 328, 332), (ln, s)
        self.counts["kregion-bras"] = len(bras)

    # -- state helpers --
    def get(self, kctx, lane, reg, m):
        if (kctx, lane, reg, m) in self.st:
            return self.st[(kctx, lane, reg, m)]
        if kctx in (0, 1) and (None, lane, reg, m) in self.st:
            return self.st[(None, lane, reg, m)]
        if kctx is None and (1, lane, reg, m) in self.st:
            # last-wins for region defs consumed outside
            return self.st[(1, lane, reg, m)]
        return None

    def put(self, kctx, lane, reg, m, v):
        self.st[(kctx, lane, reg, m)] = v

    def members(self, reg):
        if is_rs_reg(reg):
            return (0,)
        if is_b32_reg(reg):
            return (0, 1)
        return (0,)  # bare f16 names etc: atomic

    def const(self):
        return frozenset([C_ATOM])

    # -- read/write --
    def do_read(self, idx, dest, areg, const):
        for lane in range(NLANES):
            av = self.am.v[lane].get(areg)
            if av is None:
                raise Fail(("read-addr-unknown", idx + 1, areg, lane))
            a = (av + const) & 0xFFFFFFFF
            hit = self.bridge.get(a)
            if hit is None:
                raise Fail(("read-nomatch", idx + 1, lane, a))
            _v, _w, wl, pas = hit
            base = 0 if a < 512 else (512 if a < 1024 else
                                      (1024 if a < 1536 else 1536))
            off = a - base - 16 * wl
            regs = TEXREGS_W1 if base in (0, 512) else TEXREGS_W2
            if off % 4 or off // 4 >= 4:
                raise Fail(("read-badoff", idx + 1, lane, a, off))
            slot = regs[off // 4]
            # slot halves: lo<-posA, hi<-posB from pack state; slot
            # value halves tracked as members of the write reg at
            # (zonek, wl): zone k from pass
            zk = pas
            for m in (0, 1):
                v = self.get(zk, wl, slot, m)
                if v is None:
                    raise Fail(("slot-unset", idx + 1, lane, slot,
                                m, wl, pas))
                self.put(None, lane, dest, m, v)
        self.counts["reads"] += 1

    def do_write(self, idx, areg, regs, kctx):
        # record only (seeds flow via zone model at reads)
        self.counts["writes"] += 1

    # -- op rules --
    def alu(self, kctx, dest, srcs, lane):
        vals = [self.src_val(kctx, lane, s) for s in srcs]
        return union_vals(*vals)

    def src_val(self, kctx, lane, tok):
        if tok is None:
            return self.const()
        if re.fullmatch(r"-?0[xX][0-9a-fA-F]+U?|-?\d+U?|"
                        r"0[fFdD][0-9a-fA-F]+", tok):
            return self.const()
        if tok in ("%laneid", "%tid.x", "%tid.y", "%ctaid.x",
                   "%ctaid.y", "%ntid.x", "%ntid.y", "%ntid.z",
                   "%nctaid.x", "%nctaid.y", "%nctaid.z"):
            return self.const()
        if tok.startswith("_") or re.fullmatch(r"[A-Z]\w*",
                                              tok or ""):
            return self.const()
        if tok.startswith("%rd"):
            return self.const()
        if tok.startswith("%p"):
            return self.const()
        if is_rs_reg(tok):
            v = self.get(kctx, lane, tok, 0)
            if v is None:
                raise Fail(("unset-src", tok, lane))
            return v
        if is_b32_reg(tok):
            ms = [self.get(kctx, lane, tok, m) for m in (0, 1)]
            if any(v is None for v in ms):
                raise Fail(("unset-src", tok, lane))
            return union_vals(*ms)
        if re.fullmatch(r"[a-z]\w*", tok or ""):
            v = self.get(kctx, lane, tok, 0)
            if v is None:
                raise Fail(("unset-src", tok, lane))
            return v
        raise Fail(("bad-src", tok))

    def apply_def(self, kctx, idx, op, dest, srcs, raw):
        base = op.split(".")[0]
        if dest.startswith("%rd") or dest.startswith("%p"):
            return
        if dest.startswith("$") or dest.startswith("."):
            return
        if op.startswith("st.shared") or op.startswith("st.global"):
            self.do_write(idx, dest, srcs, kctx)
            return
        if op.startswith("ld.shared"):
            m = re.match(r"\[\s*(%r\d+)\s*(?:\+\s*(\d+))?\s*\]$",
                         srcs[0] if srcs else "")
            if not m:
                raise Fail(("read-shape", idx + 1))
            self.do_read(idx, dest, m.group(1),
                         int(m.group(2) or 0))
            return
        if ".global." in op or op.startswith("ld.global"):
            m = re.search(r"\{(.*?)\}", raw)
            ds = [x.strip() for x in m.group(1).split(",")] if m \
                else [dest]
            for lane in range(NLANES):
                for dd in ds:
                    for mmbr in self.members(dd):
                        self.put(kctx, lane, dd, mmbr,
                                 frozenset([F_ATOM]))
            self.counts["ld-other"] += 1
            return
        if op.startswith("ld.param") or op.startswith("ld.const"):
            m = re.search(r"\{(.*?)\}", raw)
            ds = [x.strip() for x in m.group(1).split(",")] if m \
                else [dest]
            for lane in range(NLANES):
                for dd in ds:
                    for mmbr in self.members(dd):
                        self.put(kctx, lane, dd, mmbr,
                                 self.const())
            self.counts["ld-other"] += 1
            return
        self.apply_simple(kctx, idx, op, dest, srcs, raw)

    def apply_tex(self, kctx, idx, dest):
        # tex@285 (idx 284): DATA fetches, k-exact texels.
        # all other tex sites: XTEXT markers (audit in R5).
        for lane in range(NLANES):
            for mmbr in self.members(dest):
                if idx == 284:
                    k = kctx if kctx in (0, 1) else 1
                    self.put(kctx, lane, dest, mmbr,
                             frozenset([("T", lane & 7,
                                         ((lane >> 3) + 4 * k))]))
                else:
                    self.put(kctx, lane, dest, mmbr,
                             frozenset([("X", idx + 1)]))
        if idx != 284:
            self.xtouches.append(idx + 1)
        self.counts["tex"] += 1

    # -- mov/cvt/pack ALU --
    def apply_simple(self, kctx, idx, op, dest, srcs, raw):
        base = op.split(".")[0]
        if dest.startswith("%rd"):
            return
        if base == "mov" and "{" in raw:
            sp = re.match(r"mov\.b\d+\s+\{([^}]*)\},\s*(\S+)$", raw)
            if sp:
                ds = [x.strip() for x in sp.group(1).split(",")]
                if len(ds) == 2 and all(
                        is_rs_reg(d) or
                        re.fullmatch(r"[a-z]\w*", d or "")
                        for d in ds) and is_b32_reg(sp.group(2)):
                    for lane in range(NLANES):
                        v = self.src_val(kctx, lane, sp.group(2))
                        # split positional: dest[i] <- src member
                        # (union: member granularity ends here)
                        for d in ds:
                            self.put(kctx, lane, d, 0, v)
                    self.counts["splits"] += 1
                    return
                if not any(is_rs_reg(d) or re.fullmatch(
                        r"[a-z]\w*", d or "") for d in ds):
                    pass  # b64 splinter below
                else:
                    raise Fail(("split-shape", idx + 1))
            ms = re.findall(r"%rs\d+|(?<![%\w.])[a-z]\w*(?=[,}])",
                            raw.split("[")[0])
            if len(ms) == 2 and is_b32_reg(dest):
                for lane in range(NLANES):
                    for m, s in enumerate(ms):
                        self.put(kctx, lane, dest, m,
                                 self.src_val(kctx, lane, s))
                self.counts["packs"] += 1
                return
            if ".b64" in op:
                # 64-bit splinter (address plumbing): token-free
                for lane in range(NLANES):
                    for dd in re.findall(r"%r\d+", raw.split("[")[0]):
                        for m in self.members(dd):
                            self.put(kctx, lane, dd, m,
                                     self.const())
                return
            raise Fail(("pack-shape", idx + 1))
        if base == "mov":
            if (is_rs_reg(dest) or re.fullmatch(r"[a-z]\w*",
                                                dest or "")) and \
                    len(srcs) == 1:
                for lane in range(NLANES):
                    self.put(kctx, lane, dest, 0,
                             self.src_val(kctx, lane, srcs[0]))
                return
            if is_b32_reg(dest) and len(srcs) == 1:
                for lane in range(NLANES):
                    v = self.src_val(kctx, lane, srcs[0])
                    for m in (0, 1):
                        self.put(kctx, lane, dest, m, v)
                return
            raise Fail(("mov-shape", idx + 1, op))
        if base == "cvt":
            if (is_rs_reg(dest) or re.fullmatch(r"[a-z]\w*",
                                                dest or "")) and \
                    len(srcs) == 1:
                for lane in range(NLANES):
                    self.put(kctx, lane, dest, 0,
                             self.src_val(kctx, lane, srcs[0]))
                self.counts["cvts"] += 1
                return
            if is_b32_reg(dest) and len(srcs) == 1:
                # widening (u16->u32): m0<-src, m1<-0; else copy
                for lane in range(NLANES):
                    v = self.src_val(kctx, lane, srcs[0])
                    if ".u16" in op:
                        self.put(kctx, lane, dest, 0, v)
                        self.put(kctx, lane, dest, 1, self.const())
                    else:
                        for m in (0, 1):
                            self.put(kctx, lane, dest, m, v)
                self.counts["cvts"] += 1
                return
            raise Fail(("cvt-shape", idx + 1, op))
        if base in ("add", "sub", "mul", "mad", "fma", "div", "abs",
                    "neg", "and", "or", "xor", "shl", "shr", "min",
                    "max", "ex2", "lg2", "rcp", "sqrt", "rsqrt",
                    "sin", "cos", "selp"):
            for lane in range(NLANES):
                if is_rs_reg(dest) or re.fullmatch(r"[a-z]\w*",
                                                   dest or ""):
                    self.put(kctx, lane, dest, 0,
                             self.alu(kctx, dest, srcs, lane))
                elif is_b32_reg(dest):
                    v = self.alu(kctx, dest, srcs, lane)
                    for m in (0, 1):
                        self.put(kctx, lane, dest, m, v)
                else:
                    raise Fail(("alu-dest", idx + 1, dest))
            self.counts["alu"] += 1
            return
        raise Fail(("unknown-op", idx + 1, op))

    # -- bfly / movmatrix / mma --
    def do_bfly(self, kctx, idx, dest, srcs):
        # shfl.sync.bfly.b32 dest, src, mask, mask2, pred
        if len(srcs) < 2:
            raise Fail(("bfly-shape", idx + 1))
        src, mreg = srcs[0], srcs[1]
        masks = set()
        for lane in range(NLANES):
            mv = self.am.v[lane].get(mreg)
            if mv is None:
                raise Fail(("bfly-mask-unknown", idx + 1, mreg))
            masks.add(mv)
        if len(masks) != 1:
            raise Fail(("bfly-mask-variant", idx + 1))
        mask = next(iter(masks))
        for lane in range(NLANES):
            peer = lane ^ mask
            if peer >= NLANES:
                raise Fail(("bfly-peer", idx + 1, lane, mask))
            for m in self.members(dest):
                a = self.get(kctx, lane, src, m)
                b = self.get(kctx, peer, src, m)
                if a is None or b is None:
                    raise Fail(("bfly-unset", idx + 1, lane))
                self.put(kctx, lane, dest, m, union_vals(a, b))
        self.counts["bfly"] += 1

    def do_lane_move(self, kctx, idx, dest, srcs, tag):
        # shfl.idx (broadcast) / prmt (byte permute, same lane):
        # TEX lane-crossing stays FAIL (fail-closed); tex-free
        # traffic (scales/consts) flows as the lane-union (sound
        # over-approx, exact for uniform broadcasts; asserts only
        # inspect TEX/X/F/V atoms which union preserves).
        dest = dest.split("|")[0]
        got = []
        for tok in srcs:
            t = tok.split("|")[0]
            if re.fullmatch(r"-?0[xX][0-9a-fA-F]+U?|-?\d+U?",
                            t or ""):
                continue
            if t.startswith("%rd") or t.startswith("%p"):
                continue
            if not (is_rs_reg(t) or is_b32_reg(t) or
                    re.fullmatch(r"[a-z]\w*", t or "")):
                raise Fail((tag + "-src", idx + 1, t[:30]))
            per = []
            for lane in range(NLANES):
                per.append(self.src_val(kctx, lane, t))
            got.append(union_vals(*per))
        if any(any(len(a) == 3 and a[0] == "T" for a in v)
               for v in got):
            raise Fail((tag + "-tex", idx + 1))
        u = union_vals(*got) if got else self.const()
        for lane in range(NLANES):
            for m in self.members(dest):
                self.put(kctx, lane, dest, m, u)
        self.counts[tag] += 1

    def do_movmatrix(self, kctx, idx, dest, src):
        # gate-V peer rule, then VSTOP (V cone ends here)
        for lane in range(NLANES):
            for m in self.members(dest):
                self.put(kctx, lane, dest, m, frozenset([V_ATOM]))
        self.vstop_lines.append(idx + 1)
        self.counts["movmatrix"] += 1

    def do_mma(self, kctx, idx, ln, D, A, B, C):
        if len(D) != 2 or len(A) != 4 or len(B) != 2:
            raise Fail(("mma-arity", ln, len(D), len(A), len(B)))
        if ln in QK_W0 or ln in QK_W1:
            wave = 0 if ln in QK_W0 else 1
            si = (QK_W0 if wave == 0 else QK_W1).index(ln)
            for side, regs in (("A", A), ("B", B)):
                for r in regs:
                    for lane in range(NLANES):
                        for m in self.members(r):
                            v = self.get(kctx, lane, r, m)
                            if v is None:
                                raise Fail(("terminus-unset", ln,
                                            side, r, lane, m))
                            self.termini.append(
                                (wave, si, side, r, lane, m, v))
            # poison D (scores mix tokens; downstream out of scope)
            for lane in range(NLANES):
                for d in D:
                    for m in self.members(d):
                        self.put(kctx, lane, d, m,
                                 frozenset([F_ATOM]))
            self.counts["qk-mma"] += 1
            return
        # crossed MMA (norm/proj/V): B must be TEX-free (weights)
        for lane in range(NLANES):
            for r in B:
                for m in self.members(r):
                    v = self.get(kctx, lane, r, m)
                    if v is None:
                        raise Fail(("mma-B-unset", ln, r, lane))
                    if any(len(a) == 3 and a[0] == "T" for a in v):
                        raise Fail(("mma-B-tex", ln, r, lane,
                                    sorted(v)[:4]))
        # D-member m <- A[2m],A[2m+1] member m + C member m
        # (like-for-like halves; FULL union incl. C: the
        # post-softmax path unions fibers/quarters through
        # bfly/movmatrix, so C legitimately carries row texels
        # outside A-row (e.g. bfly mask-16 fiber pairs); the
        # terminal asserts (fiber/query equivalence classes)
        # adjudicate instead of a same-row tripwire)
        for lane in range(NLANES):
            for m in (0, 1):
                for di, d in enumerate(D):
                    acc = set()
                    for a in (A[2 * (di % 2)], A[2 * (di % 2) + 1]):
                        va = self.get(kctx, lane, a, m)
                        if va is None:
                            raise Fail(("mma-A-unset", ln, a,
                                        lane))
                        acc |= set(va)
                    for c in C:
                        vc = self.get(kctx, lane, c, m)
                        if vc is None:
                            raise Fail(("mma-C-unset", ln, c,
                                        lane))
                        if tex_atoms(vc):
                            self.counts["mma-C-tex"] += 1
                        acc |= set(vc)
                    self.put(kctx, lane, d, m, frozenset(acc))
        self.counts["mma"] += 1

    # -- visit --
    SKIPLINE = re.compile(r"^\s*(\$L__\w+:|\.[a-z]+|"
                          r"(@%p\d+\s+)?bra(\.uni)?\s|ret|exit|"
                          r"bar\.|membar|atom\.|nanosleep|trap)")
    PAREONLY = re.compile(r"^\s*[()]+\s*$")

    def visit(self, kctx, idx):
        s = strip_line(self.ls, idx)
        if not s or self.SKIPLINE.match(s) or self.PAREONLY.match(s):
            return True
        if idx + 1 in self.skip:
            return True
        if idx + 1 in self.sites:
            _ln, D, A, B, C = self.sites[idx + 1]
            self.do_mma(kctx, idx, idx + 1, D, A, B, C)
            return True
        if idx + 1 in self.xmma:
            _ln, D, A, B, C = self.xmma[idx + 1]
            self.do_mma(kctx, idx, idx + 1, D, A, B, C)
            return True
        p = parse_operands(s)
        if p is None:
            raise Fail(("parse", idx + 1, s[:70]))
        op, dest, srcs, _pred = p
        if op.startswith("tex."):
            m = re.match(r"tex\.\S+\s+\{([^}]*)\}", s)
            if not m:
                raise Fail(("tex-shape", idx + 1))
            for d in [x.strip() for x in m.group(1).split(",")]:
                self.apply_tex(kctx, idx, d)
            return True
        if ".bfly." in op and op.split(".")[0] == "shfl":
            self.do_bfly(kctx, idx, dest, srcs)
            return True
        if ".idx." in op and op.split(".")[0] == "shfl":
            self.do_lane_move(kctx, idx, dest, srcs, "shfl-idx")
            return True
        if op.startswith("shfl."):
            raise Fail(("shfl", idx + 1, s[:60]))
        if "prmt" in op:
            self.do_lane_move(kctx, idx, dest, srcs, "prmt")
            return True
        if op.startswith("movmatrix."):
            m = re.match(r"movmatrix\.\S+\s+(%r\d+),\s*(%r\d+)$", s)
            if not m:
                raise Fail(("movmatrix-shape", idx + 1))
            self.do_movmatrix(kctx, idx, m.group(1), m.group(2))
            return True
        if op.split(".")[0] in ("setp",):
            return True
        self.apply_def(kctx, idx, op, dest, srcs, s)
        return True

    def run(self):
        n = len(self.ls)
        # outside-low first (param/base defs feed the region)
        for idx in range(0, KREGION[0] - 1):
            self.visit(None, idx)
        # k-region twice (fixpoint for carried uses)
        for k in (0, 1):
            for _round in range(10):
                before = len(self.st)
                for idx in range(KREGION[0] - 1, KREGION[1] - 1):
                    try:
                        self.visit(k, idx)
                    except Fail as e:
                        msg = str(e.args[0]) if e.args else ""
                        if "unset-src" in msg or "slot-unset" in msg:
                            continue
                        raise
                if len(self.st) == before:
                    break
        # outside-high once, line order (through wave-1 QK sites)
        for idx in range(KREGION[1] - 1, min(n, 12900)):
            self.visit(None, idx)

    def unresolved(self):
        return None


def check_R4(ls):
    """Forward walk to QK termini; assert gate-D pixels."""
    am = AddrMachine()
    try:
        am.run(ls, 0, 12000)
        print("R4 addr-machine: clean to 12000")
    except (KeyError, AssertionError) as e:
        print("R4 addr-machine stopped:", str(e)[:120])
    wk0 = RowWalker(ls, parse_mma_all(ls), am)
    fw = Fwd(ls, am, wk0.bridge)
    try:
        fw.run()
    except Fail as e:
        print("R4 FORWARD-FAIL:", str(e.args[0])[:160])
        raise AssertionError("R4 forward failed")
    print("R4 forward: %s" % ", ".join(
        "%s=%d" % kv for kv in sorted(fw.counts.items())))
    # termini asserts vs gate-D pixels. Slot algebra (read off
    # the kernel, verified below): wave w queries = tokens
    # [32w,32w+31] (window rows y=4w..4w+3); A(si) = query group
    # si//8 (x-half 4*(si//8)); B(si) = key quadrant (si%8)//2
    # ((xb,yb) = (4*(q%2),4*(q//2))); lane L = (t,t+16) fiber
    # pair (bfly-16 union of k=0 seeds): L<16 -> x=xb+L//4,
    # y-pair {yb,yb+2}; L>=16 -> x=xb+(L-16)//4, {yb+1,yb+3}.
    nA = nb = 0
    bad = []
    for (wave, si, side, reg, lane, m, v) in fw.termini:
        if side == "A":
            xb, yb = 4 * (si // 8), 4 * wave
        else:
            quad = (si % 8) // 2
            xb, yb = 4 * (quad % 2), 4 * (quad // 2)
        if lane < 16:
            exp = {(xb + lane // 4, yb), (xb + lane // 4, yb + 2)}
        else:
            exp = {(xb + (lane - 16) // 4, yb + 1),
                   (xb + (lane - 16) // 4, yb + 3)}
        tex = tex_pairs(v)
        if side == "A":
            nA += 1
        else:
            nb += 1
        # F (global modulation: %r2390 scale path + mma-C
        # bias/scales) and C (lane arithmetic) are token-free:
        # modulation-only (R5). X (other textures) / V (V-cone
        # breach) still FAIL.
        if any(not (len(a) == 3 and a[0] == "T") and a != C_ATOM
               and a != F_ATOM for a in v):
            bad.append((wave, si, side, reg, lane, m, "xfv",
                        sorted(v)[:6]))
            continue
        if len(tex) > 2:
            bad.append((wave, si, side, reg, lane, m, "wide",
                        sorted(tex)))
            continue
        if not tex <= exp:
            bad.append((wave, si, side, reg, lane, m, "mismatch",
                        sorted(tex), sorted(exp)))
    print("R4 termini: A=%d B=%d member-checks" % (nA, nb))
    if bad:
        print("R4 FIRST-MISMATCH:", bad[0])
        print("R4 mismatches: %d of %d" % (len(bad), nA + nb))
        raise AssertionError("R4 terminal mismatch")
    # coverage: per-site A == 16 query pixels; per-site B ==
    # 16 key pixels (quadrant); B over si 0..7 == full 64 window;
    # A over both groups == 32 wave-query pixels.
    for wave in (0, 1):
        for si in range(16):
            xb, yb = 4 * (si // 8), 4 * wave
            exp = {(x, y) for x in range(xb, xb + 4)
                   for y in range(yb, yb + 4)}
            got = set()
            for (w2, si2, side, _r, _l, _m, v) in fw.termini:
                if w2 == wave and si2 == si and side == "A":
                    got |= tex_pairs(v)
            assert got == exp, ("A-cover", wave, si, sorted(got),
                                sorted(exp))
        for quad in range(4):
            exp = {(x, y)
                   for x in range(4 * (quad % 2), 4 * (quad % 2) + 4)
                   for y in range(4 * (quad // 2),
                                  4 * (quad // 2) + 4)}
            got = set()
            for (w2, si2, side, _r, _l, _m, v) in fw.termini:
                if w2 == wave and side == "B" and \
                        (si2 % 8) // 2 == quad:
                    got |= tex_pairs(v)
            assert got == exp, ("B-cover", wave, quad,
                                sorted(got), sorted(exp))
        gotq = set()
        for (w2, _si, side, _r, _l, _m, v) in fw.termini:
            if w2 == wave and side == "A":
                gotq |= tex_pairs(v)
        assert gotq == {(x, y) for x in range(8)
                        for y in range(4 * wave, 4 * wave + 4)}, \
            ("A-wave", wave, sorted(gotq))
        gotk = set()
        for (w2, si2, side, _r, _l, _m, v) in fw.termini:
            if w2 == wave and side == "B" and si2 < 8:
                gotk |= tex_pairs(v)
        assert gotk == {(x, y) for x in range(8) for y in range(8)}, \
            ("B-window", wave, sorted(gotk))
    print("R4 coverage: 32 sites A exact (16 query-pixels); 4 key "
          "quadrants exact; A-wave=32 queries; B-window=full 8x8")
    nx = len({t for t in fw.xtouches})
    print("R4 boundary: XTEXT sites=%s VSTOP-lines=%d" %
          (sorted(set(fw.xtouches)), len(fw.vstop_lines)))
    return fw


def check_R5(fw):
    nfore = sum(1 for t in fw.termini if F_ATOM in t[6])
    ntot = len(fw.termini)
    assert nfore == ntot, (nfore, ntot)  # uniform single-F pattern
    nxv = sum(1 for t in fw.termini
              for a in t[6]
              if a == V_ATOM or (len(a) == 2 and a[0] == "X"))
    assert nxv == 0, nxv
    print("R5 LOAD/scale boundary: modulation-only (section 8); "
          "V-side excluded at %d movmatrix stops; FOREIGN "
          "modulation at termini=%d/%d (uniform; X/V=0)"
          % (len(fw.vstop_lines), nfore, ntot))


def check_R6():
    print("R6 verdict: kernel token t = pixel (t&7, t>>3) CLOSED "
          "(R1 scopes + R2 interior conditions)")


def main():
    ls = kernel_lines()
    check_R1(ls)
    check_R2(ls)
    check_R3(ls)
    fw = check_R4(ls)
    check_R5(fw)
    check_R6()
    print("ROWMAJOR-OK")


if __name__ == "__main__":
    main()
