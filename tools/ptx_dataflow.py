"""
ptx_dataflow.py - symbolic dataflow over NVCC PTX kernels.

Extends ptx_load_trace.py with:
  * f16 / f16x2 / f32 constant propagation (exact bit patterns),
  * symbolic laneid / ctaid / param-scalar arithmetic, with the compiler's
    signed-div and mod idioms decoded (L/4, L%4, (L/4)%4, ...),
  * memory provenance: every ld.global resolves to (param slot, offset expr),
  * sink reports: mma operands, e4m3 quantisation sites, st.global stores,
  * param-struct decode: every ld.param with offset/width.

Value domain:
  Const(bits, ty)       known bit pattern (b32/b16/f32/f16x2)
  Lin({term: coef}, c)  symbolic int; terms are canonical strings
                        ("L", "L/4", "L%4", "p24", "bx", "(L/4)%4", ...)
  Ptr(slot, lin)        pointer value loaded from param slot, + byte offset
  Load(ptr, size, ty)   value loaded from memory (provenance kept)
  Expr(op, args)        small expression tree (f16 path, shfl, mma, ...)

Usage:
    python ptx_dataflow.py <kernel-name-substring> [--cubin cubin_XX] \
        [--report sinks|loads|params|stores|all] [--full-expr]
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5FSR-nolib" if False else ROOT / "dlss5-analysis" / "cubins"

ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\s*\(")

MAX_TERMS = 12

# ---------------------------------------------------------------- values


class Val:
    pass


class Unknown(Val):
    def __repr__(self):
        return "?"


UNKNOWN = Unknown()


class RoundFix(Val):
    """result of  shr.u32 (shr.s32 x,31), k  - the rounding correction
    added before a signed division.  Carries the dividend register."""

    __slots__ = ("src",)

    def __init__(self, src):
        self.src = src

    def __repr__(self):
        return f"rndfix({self.src})"


class Const(Val):
    __slots__ = ("bits", "ty")

    def __init__(self, bits, ty):
        self.bits = bits & 0xFFFF if ty == "b16" else bits & 0xFFFFFFFF
        self.ty = ty

    def f32(self):
        return struct.unpack("<f", struct.pack("<I", self.bits))[0]

    def halves(self):
        return (self.bits & 0xFFFF, (self.bits >> 16) & 0xFFFF)

    def __repr__(self):
        if self.ty == "f32":
            return f"f32({self.f32()!r})"
        if self.ty == "f16x2":
            lo, hi = self.halves()
            return f"f16x2({f16str(lo)},{f16str(hi)})"
        if self.ty == "b16":
            return f"b16(0x{self.bits:04X})"
        return f"b32(0x{self.bits:08X})"


def f16bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16str(bits: int) -> str:
    try:
        return repr(struct.unpack("<e", struct.pack("<H", bits))[0])
    except Exception:
        return f"0x{bits:04X}"


def f16val(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class Lin(Val):
    __slots__ = ("terms", "c")

    def __init__(self, terms=None, c=0):
        self.terms = {k: v for k, v in (terms or {}).items() if v}
        self.c = c

    def add(self, other: "Lin") -> "Lin":
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0) + v
        if len(t) > MAX_TERMS:
            return UNKNOWN
        return Lin(t, self.c + other.c)

    def scale(self, s: int) -> "Lin":
        return Lin({k: v * s for k, v in self.terms.items()}, self.c * s)

    def is_const(self):
        return not self.terms

    def __repr__(self):
        parts = []
        for k in sorted(self.terms, key=lambda s: (len(s), s)):
            v = self.terms[k]
            parts.append(f"{v}*{k}" if v != 1 else k)
        if self.c or not parts:
            parts.append(str(self.c))
        return " + ".join(parts)


def lin_mul(a: Lin, b: Lin) -> Val:
    if a.is_const():
        return b.scale(a.c)
    if b.is_const():
        return a.scale(b.c)
    t = {}
    for ka, ca in a.terms.items():
        for kb, cb in b.terms.items():
            k = f"{ka}*{kb}" if ka != kb else f"{ka}^2"
            t[k] = t.get(k, 0) + ca * cb
    for ka, ca in a.terms.items():
        t[ka] = t.get(ka, 0) + ca * b.c
    for kb, cb in b.terms.items():
        t[kb] = t.get(kb, 0) + cb * a.c
    t = {k: v for k, v in t.items() if v}
    if len(t) > MAX_TERMS:
        return UNKNOWN
    return Lin(t, a.c * b.c)


L = Lin({"L": 1}, 0)


class Ptr(Val):
    __slots__ = ("slot", "off")

    def __init__(self, slot, off):
        self.slot = slot
        self.off = off

    def __repr__(self):
        return f"{self.slot}[{self.off!r}]"


class Load(Val):
    __slots__ = ("ptr", "size", "ty")

    def __init__(self, ptr, size, ty):
        self.ptr = ptr
        self.size = size
        self.ty = ty

    def __repr__(self):
        return f"ld{self.size}[{self.ptr!r}]"


class Expr(Val):
    __slots__ = ("op", "args")

    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __repr__(self):
        if len(self.args) == 1:
            return f"{self.op}({self.args[0]!r})"
        return f"{self.op}(" + ", ".join(repr(x) for x in self.args) + ")"


# ---------------------------------------------------------------- parsing


def statements(body: str):
    """Yield (lineno, opcode, operands-string) joining continuation lines."""
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        i += 1
        if not stripped or stripped.startswith("//"):
            continue
        if re.match(r"^\.\w+", stripped):
            continue  # directives, .reg/.param decls
        if stripped in ("{", "}", "){"):
            continue
        if stripped.endswith(":") and "$L__" in stripped:
            continue
        stmt = stripped
        while ";" not in stmt and i < len(lines):
            nxt = lines[i].strip()
            i += 1
            if not nxt or nxt.startswith("//"):
                continue
            stmt += " " + nxt
        stmt = re.sub(r"^@!?%p\d+\s+", "", stmt)
        if stmt.startswith("{"):
            stmt = stmt[1:]
        if stmt.endswith("}"):
            stmt = stmt[:-1]
        stmt = stmt.strip()
        m = re.match(r"^([a-z][\w.]*)\s+(.*);$", stmt) or re.match(r"^([a-z][\w.]*)\s*;$", stmt)
        if m:
            yield i, m.group(1), m.group(2) if m.lastindex > 1 else ""


def split_ops(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


REG = re.compile(r"^%([a-z]+)(\d+)$")


def imm_bits(tok: str):
    tok = tok.rstrip("U").rstrip("u")
    if re.match(r"^0f[0-9A-Fa-f]{8}$", tok):
        return int(tok[2:], 16), True
    if re.match(r"^0x[0-9A-Fa-f]+$", tok):
        return int(tok, 16), False
    if re.match(r"^-?\d+$", tok):
        return int(tok) & 0xFFFFFFFF, False
    return None, False


def s32(bits: int) -> int:
    return bits if bits < 0x80000000 else bits - (1 << 32)


# ---------------------------------------------------------------- interpreter


class Dataflow:
    def __init__(self, kname, cubin_name, body, full_expr=False):
        self.kname = kname
        self.cubin = cubin_name
        self.full_expr = full_expr
        self.regs: dict[str, Val] = {}
        self.defs: dict[str, int] = {}
        self.lastop: dict[str, tuple] = {}  # reg -> (op, srcs)
        self.param_loads = []
        self.global_loads = []
        self.sinks = []
        self.stores = []
        self.rsq_sites = []
        self.errs = []
        for lineno, op, ops in statements(body):
            try:
                self.step(lineno, op, ops)
            except Exception as e:
                self.errs.append(f"line {lineno} {op}: {e}")

    # -- helpers ------------------------------------------------------

    def get(self, tok):
        return self.regs.get(tok, UNKNOWN)

    def set(self, tok, val, op=None, srcs=None):
        self.regs[tok] = val
        self.defs[tok] = self.defs.get(tok, 0) + 1
        if op:
            self.lastop[tok] = (op, srcs or [])

    @staticmethod
    def as_lin(v: Val) -> Lin | None:
        if isinstance(v, Lin):
            return v
        if isinstance(v, Const):
            return Lin({}, s32(v.bits))
        return None

    def oplin(self, tok: str) -> Lin | None:
        """operand -> Lin, parsing immediates."""
        if re.match(r"^%\w+$", tok):
            return self.as_lin(self.get(tok))
        b, _ = imm_bits(tok)
        return Lin({}, s32(b)) if b is not None else None

    def lin_div(self, lin: Lin, k: int) -> Val:
        """signed div by 2^k of a laneid-family expression (non-negative)."""
        t = {}
        for key, coef in lin.terms.items():
            nk = f"{key}/{2**k}" if not key.startswith("(") else f"({key})/{2**k}"
            t[nk] = t.get(nk, 0) + coef
        if len(t) > MAX_TERMS:
            return UNKNOWN
        return Lin(t, lin.c >> k)

    # -- one instruction ----------------------------------------------

    def step(self, lineno, op, ops):
        o = split_ops(ops)

        # ---- ld.param ------------------------------------------------
        if op.startswith("ld.param"):
            m = re.match(r"^ld\.param(\.v\d)?\.(b\d+|u\d+|f\d+|s\d+)(\.cn)?$", op)
            width = int(m.group(2)[1:]) if m else 32
            vec = int(m.group(1)[2:]) if m and m.group(1) else 1
            pm = re.match(r"^\[([%\w]+)(?:\+(\d+))?\]$", o[1])
            if pm and re.match(r"^%rd\d+$", pm.group(1)):
                base = self.get(pm.group(1))
                if not isinstance(base, Ptr):
                    for d in re.findall(r"%\w+", o[0]):
                        self.set(d, UNKNOWN)
                    return
                slot, off = base.slot, (base.off.c if isinstance(base.off, Lin) else 0) \
                    + (int(pm.group(2)) if pm.group(2) else 0)
            else:
                slot = pm.group(1) if pm else o[1].strip("[]")
                off = int(pm.group(2)) if pm and pm.group(2) else 0
            dests = re.findall(r"%\w+", o[0])
            self.param_loads.append((off, width * vec, tuple(dests), op))
            for d in dests:
                if width == 64:
                    self.set(d, Ptr(f"{slot}+{off}", Lin({}, 0)))
                else:
                    self.set(d, Lin({f"p{off}": 1}, 0), op, [d])
            return

        # ---- cvta / mov 64 -------------------------------------------
        if op in ("cvta.to.global.u64", "cvta.to.global.u32", "cvta.global"):
            v = self.get(o[1])
            self.set(o[0], v if isinstance(v, Ptr) else UNKNOWN)
            return
        if op in ("mov.u64", "mov.b64"):
            if re.match(r"^%\w+$", o[1]):
                self.set(o[0], self.get(o[1]))
            else:
                self.set(o[0], Ptr(o[1], Lin({}, 0)))  # param-space address
            return
        if op in ("mov.u32", "mov.b32"):
            dst, src = o[0], o[1]
            if dst.startswith("{"):
                lohi = re.findall(r"%\w+", dst)
                v = self.get(src)
                if isinstance(v, Const):
                    b = v.bits
                    self.set(lohi[0], Const(b & 0xFFFF, "b16"))
                    if len(lohi) > 1:
                        self.set(lohi[1], Const((b >> 16) & 0xFFFF, "b16"))
                else:
                    for r in lohi:
                        self.set(r, Expr("half", [v]))
                return
            if src.startswith("{"):
                toks = [t if t.startswith("%") else "%" + t
                        for t in re.findall(r"%?\w+", src)]
                parts = [self.get(t) for t in toks]
                if len(parts) == 2 and all(isinstance(p, Const) for p in parts):
                    ty = "f16x2" if all(p.ty == "b16" for p in parts) else "b32"
                    self.set(dst, Const((parts[0].bits & 0xFFFF) | ((parts[1].bits & 0xFFFF) << 16), ty))
                else:
                    self.set(dst, Expr("pack", parts))
                return
            if re.match(r"^%(ctaid|tid)\.\w+$", src):
                name = src.strip("%").replace(".", "")
                self.set(dst, Lin({name: 1}, 0))
                return
            if src == "%laneid":
                self.set(dst, L)
                return
            bits, isf = imm_bits(src)
            if bits is not None:
                self.set(dst, Const(bits, "f32" if isf else "b32"))
                return
            self.set(dst, self.get(src))
            return
        if op == "mov.b16":
            bits, _ = imm_bits(o[1].rstrip("U"))
            if bits is not None:
                self.set(o[0], Const(bits, "b16"))
            else:
                self.set(o[0], self.get(o[1]))
            return
        if op in ("mov.pred",) or op.startswith("mov.f64"):
            return

        # ---- 32/64-bit int arithmetic --------------------------------
        if op in ("add.s32", "add.u32", "add.s64", "add.u64"):
            va, vb = self.get(o[1]), self.get(o[2])
            # x + rndfix(x)  ==  x   (signed-div rounding idiom)
            if isinstance(vb, RoundFix) and vb.src == o[1]:
                self.set(o[0], va)
                return
            if isinstance(va, RoundFix) and va.src == o[2]:
                self.set(o[0], vb)
                return
            if isinstance(va, Ptr):
                lb = self.as_lin(vb)
                if lb is not None:
                    self.set(o[0], Ptr(va.slot, va.off.add(lb) if isinstance(va.off, Lin) else UNKNOWN))
                    return
            if isinstance(vb, Ptr):
                la = self.as_lin(va)
                if la is not None:
                    self.set(o[0], Ptr(vb.slot, vb.off.add(la) if isinstance(vb.off, Lin) else UNKNOWN))
                    return
            la, lb = self.oplin(o[1]), self.oplin(o[2])
            if la is not None and lb is not None:
                self.set(o[0], la.add(lb))
            else:
                self.set(o[0], va if isinstance(va, (Load, Expr)) else UNKNOWN)
            return
        if op in ("sub.s32", "sub.u32"):
            va, vb = self.get(o[1]), self.get(o[2])
            la, lb = self.oplin(o[1]), self.oplin(o[2])
            if la is not None and lb is not None:
                self.set(o[0], la.add(lb.scale(-1)))
            else:
                self.set(o[0], UNKNOWN)
            return
        if op in ("shl.b32", "shl.b64", "shl.b16"):
            kv = self.get(o[2]) if re.match(r"^%\w+$", o[2]) else None
            kb, _ = imm_bits(o[2])
            k = kv.bits if isinstance(kv, Const) else kb
            if k is None:
                self.set(o[0], UNKNOWN)
                return
            a = self.oplin(o[1])
            self.set(o[0], a.scale(1 << k) if a is not None else UNKNOWN)
            return
        if op == "shr.s32":
            kv = self.get(o[2]) if re.match(r"^%\w+$", o[2]) else None
            kb, _ = imm_bits(o[2])
            k = kv.bits if isinstance(kv, Const) else kb
            if k is None:
                self.set(o[0], UNKNOWN)
                return
            v = self.get(o[1]) if re.match(r"^%\w+$", o[1]) else UNKNOWN
            if isinstance(v, Const):
                self.set(o[0], Const(s32(v.bits) >> k, "b32"), op, [o[1]])
                return
            lin = self.as_lin(v)
            if lin is not None and k in (1, 2, 3, 4, 5):
                self.set(o[0], self.lin_div(lin, k), op, [o[1]])
            elif lin is not None and k >= 16:
                self.set(o[0], Lin({}, 0), op, [o[1]])  # non-negative family
            else:
                self.set(o[0], UNKNOWN, op, [o[1]])
            return
        if op == "shr.u32":
            kv = self.get(o[2]) if re.match(r"^%\w+$", o[2]) else None
            kb, _ = imm_bits(o[2])
            k = kv.bits if isinstance(kv, Const) else kb
            if k is None:
                self.set(o[0], UNKNOWN)
                return
            v = self.get(o[1]) if re.match(r"^%\w+$", o[1]) else UNKNOWN
            if isinstance(v, Const):
                self.set(o[0], Const(v.bits >> k, "b32"))
                return
            lin = self.as_lin(v)
            if lin is not None and k >= 16:
                # laneid/ctaid/size family is far below 2^16; the compiler's
                # round-toward-zero fixups (shr.u32 (shr.s32 x,31), 28..31)
                # and small-range guards all evaluate to 0 here.
                self.set(o[0], Lin({}, 0))
                return
            self.set(o[0], UNKNOWN)
            return
        if op in ("mul.lo.s32", "mul.lo.u32", "mul.wide.s32", "mul.wide.u32"):
            la, lb = self.oplin(o[1]), self.oplin(o[2])
            if la is not None and lb is not None:
                self.set(o[0], lin_mul(la, lb))
            else:
                self.set(o[0], UNKNOWN)
            return
        if op == "mad.lo.s32":
            la, lb, lc = self.oplin(o[1]), self.oplin(o[2]), self.oplin(o[3])
            if la is not None and lb is not None:
                r = lin_mul(la, lb)
                if lc is not None and isinstance(r, Lin):
                    self.set(o[0], r.add(lc))
                else:
                    self.set(o[0], r)
            else:
                self.set(o[0], UNKNOWN)
            return
        if op == "and.b32":
            mask = imm_bits(o[2])[0]
            a = self.oplin(o[1])
            if mask is None or a is None:
                self.set(o[0], UNKNOWN)
                return
            sm = s32(mask)
            if sm < 0 and (~sm) and ((~sm + 1) & ~sm) == 0:  # mask == -2^k
                k = ((~sm) + 1).bit_length() - 1
                mod = Lin({self.mod_key(a, 2 ** k): 1}, 0)
                self.set(o[0], a.add(mod.scale(-1)))  # x - x%2^k
                return
            if mask and (mask & (mask + 1)) == 0:  # mask == 2^k - 1
                k = mask.bit_length()
                self.set(o[0], Lin({self.mod_key(a, 2 ** k): 1}, 0))
                return
            self.set(o[0], UNKNOWN)
            return
        if op in ("or.b32", "xor.b32"):
            self.set(o[0], UNKNOWN)
            return
        if op == "prmt.b32":
            v = self.get(o[1])
            self.set(o[0], Expr("prmt", [v]) if isinstance(v, (Load, Expr, Const)) else UNKNOWN)
            return
        if op.startswith("shfl.sync"):
            v = self.get(o[1])
            self.set(o[0], Expr("shfl", [v]) if isinstance(v, (Load, Expr)) else UNKNOWN)
            return
        if op.startswith("movmatrix"):
            v = self.get(o[1])
            self.set(o[0], Expr("transpose", [v]) if isinstance(v, (Load, Expr)) else UNKNOWN)
            return
        if op == "selp.b32":
            def selval(tok):
                if re.match(r"^%\w+$", tok):
                    return self.get(tok)
                b, _ = imm_bits(tok)
                return Const(b, "b32") if b is not None else UNKNOWN
            a, b = selval(o[1]), selval(o[2])
            if repr(a) == repr(b):
                self.set(o[0], a)
                return
            la, lb = self.as_lin(a), self.as_lin(b)
            if la is not None and lb is not None:
                if la.is_const() and lb.is_const():
                    self.set(o[0], Lin({}, la.c if la.c else lb.c))
                else:
                    self.set(o[0], Lin({f"({la!r}|{lb!r})": 1}, 0))
                return
            self.set(o[0], UNKNOWN)
            return
        if op.startswith("setp") or op.startswith("bra") or op.startswith("vote") \
                or op.startswith("bar") or op.startswith("cvt.rni") or op.startswith("cvt.s") \
                or op.startswith("cvt.u") or op.startswith("testp") or op.startswith("red") \
                or op.startswith("atom"):
            return

        # ---- cvt ------------------------------------------------------
        if op == "cvt.rn.f16.f32":
            v = self.get(o[1])
            dst = o[0] if o[0].startswith("%") else "%" + o[0]
            if isinstance(v, Const) and v.ty in ("f32", "b32"):
                self.set(dst, Const(f16bits(v.f32()), "b16"))
            else:
                self.set(dst, Expr("f32>f16", [v]))
            return
        if op == "cvt.f32.f16":
            v = self.get(o[1])
            if isinstance(v, Const) and v.ty == "b16":
                self.set(o[0], Const(struct.unpack("<I", struct.pack("<f", f16val(v.bits)))[0], "f32"))
            else:
                self.set(o[0], Expr("f16>f32", [v]))
            return
        if op == "cvt.rn.f16x2.f32":
            lo, hi = self.get(o[1]), self.get(o[2])
            if isinstance(lo, Const) and isinstance(hi, Const) and lo.ty == "f32" and hi.ty == "f32":
                self.set(o[0], Const(f16bits(lo.f32()) | (f16bits(hi.f32()) << 16), "f16x2"))
            else:
                self.set(o[0], Expr("f32x2>f16x2", [lo, hi]))
            return
        if op == "cvt.rn.f16x2.e4m3x2":
            v = self.get(o[1])
            if isinstance(v, Const) and v.ty == "b16":
                lo, hi = v.bits & 0xFF, (v.bits >> 8) & 0xFF
                self.set(o[0], Const(f16bits(e4m3val(lo)) | (f16bits(e4m3val(hi)) << 16), "f16x2"))
            else:
                self.set(o[0], Expr("dequant_e4m3", [v]))
            return
        if op == "cvt.rn.satfinite.e4m3x2.f16x2":
            src = self.get(o[1])
            self.sinks.append(f"QUANT  L{lineno}: {self.fmt(src)}")
            self.set(o[0], Expr("quant_e4m3", [src]))
            return
        if op.startswith("cvt."):
            return

        # ---- f16x2 / f32 arithmetic -----------------------------------
        if re.match(r"^(mul|min|max|add|sub|fma\.rn|abs|neg)\.f16x2$", op):
            kind = op.split(".")[0]
            if kind == "fma":
                args = [self.get(o[1]), self.get(o[2]), self.get(o[3])]
            else:
                args = [self.get(o[1]), self.get(o[2])] if len(o) > 2 else [self.get(o[1])]
            if all(isinstance(a, Const) for a in args):
                self.set(o[0], f16x2_eval(op, args))
            else:
                self.set(o[0], Expr(op, args))
            return
        if re.match(r"^(mul|add|sub|min|max|abs|neg)\.f32$|^fma\.rn\.f32$", op):
            args = [self.get(x) for x in o]
            if all(isinstance(a, Const) and a.ty == "f32" for a in args):
                self.set(o[0], f32_eval(op, args))
            else:
                self.set(o[0], Expr(op, args))
            return
        if op in ("rsqrt.approx.ftz.f32", "rcp.approx.ftz.f32") or op.startswith(("ex2.approx", "lg2.approx")):
            self.rsq_sites.append((lineno, op))
            self.set(o[0], Expr(op, [self.get(o[1])]))
            return

        # ---- loads / stores -------------------------------------------
        m = re.match(r"^ld\.(?:weak\.)?global(?:\.[a-z]+)*\.(?:v(\d)\.)?([bfsu])(\d+)$", op)
        if m:
            vec = int(m.group(1) or 1)
            size = int(m.group(3)) // 8 * vec
            am = re.match(r"^\[(%rd\d+)([+-]\d+)?\]$", o[1])
            base = self.get(am.group(1)) if am else UNKNOWN
            disp = int(am.group(2)) if am and am.group(2) else 0
            dests = re.findall(r"%\w+", o[0])
            if isinstance(base, Ptr) and isinstance(base.off, Lin):
                off = base.off.add(Lin({}, disp))
                self.global_loads.append((base.slot, off, size, dests))
                for d in dests:
                    self.set(d, Load(Ptr(base.slot, off), size, "raw"))
            else:
                for d in dests:
                    self.set(d, UNKNOWN)
            return
        m = re.match(r"^st\.(?:weak\.)?global(?:\.[a-z]+)*\.(?:v(\d)\.)?([bfsu])(\d+)$", op)
        if m:
            am = re.match(r"^\[(%rd\d+)([+-]\d+)?\]$", o[1])
            base = self.get(am.group(1)) if am else UNKNOWN
            disp = int(am.group(2)) if am and am.group(2) else 0
            if isinstance(base, Ptr) and isinstance(base.off, Lin):
                self.stores.append((base.slot, base.off.add(Lin({}, disp)),
                                    int(m.group(3)) // 8 * int(m.group(1) or 1), self.get(o[0])))
            return

        # ---- mma ------------------------------------------------------
        if op.startswith("mma.sync"):
            dests = re.findall(r"%\w+", o[0])
            A = re.findall(r"%\w+", o[1])
            B = re.findall(r"%\w+", o[2])
            C = re.findall(r"%\w+", o[3])
            shape = op.split(".")[2]
            self.sinks.append(
                f"MMA {shape} L{lineno}:\n    A = {self.fmt_multi(A)}\n"
                f"    B = {self.fmt_multi(B)}\n    C = {self.fmt_multi(C)}")
            for d in dests:
                self.set(d, Expr("mma", [self.fmt_multi(A), self.fmt_multi(B), self.fmt_multi(C)]))
            return

    @staticmethod
    def mod_key(a: Lin, m: int) -> str:
        if list(a.terms) == ["L"] and a.terms["L"] == 1 and a.c == 0:
            return f"L%{m}" if m != 2 else "L%2"
        return f"({a!r})%{m}"

    def fmt(self, v: Val) -> str:
        s = repr(v)
        if self.full_expr or len(s) < 200:
            return s
        return s[:197] + "..."

    def fmt_multi(self, regs) -> str:
        vs = [self.get(r) for r in regs]
        rep = {repr(v) for v in vs}
        if len(rep) == 1:
            return f"{len(regs)}x {self.fmt(vs[0])}"
        return "[" + ", ".join(self.fmt(v) for v in vs) + "]"


def e4m3val(b: int) -> float:
    """e4m3 (OCP FP8, no inf, NaN only 0x7F/0xFF) -> float."""
    sign = -1.0 if b & 0x80 else 1.0
    exp = (b >> 3) & 0xF
    frac = b & 7
    if exp == 0:
        return sign * frac / 8.0 * 2**-6
    if exp == 0xF and frac == 7:
        return float("nan")
    return sign * (1 + frac / 8.0) * 2 ** (exp - 7)


def f16x2_eval(op, args):
    kind = op.split(".")[0]
    if kind == "abs":
        (a,) = args
        lo, hi = a.halves()
        return Const((lo & 0x7FFF) | ((hi & 0x7FFF) << 16), "f16x2")
    if kind == "neg":
        (a,) = args
        lo, hi = a.halves()
        return Const(((lo ^ 0x8000) & 0xFFFF) | (((hi ^ 0x8000) & 0xFFFF) << 16), "f16x2")
    pairs = [a.halves() for a in args]
    out = []
    for lane in (0, 1):
        vals = [f16val(p[lane]) for p in pairs]
        if kind == "mul":
            r = vals[0] * vals[1]
        elif kind == "min":
            r = min(vals[0], vals[1])
        elif kind == "max":
            r = max(vals[0], vals[1])
        elif kind == "add":
            r = vals[0] + vals[1]
        elif kind == "sub":
            r = vals[0] - vals[1]
        elif kind == "fma":
            r = vals[0] * vals[1] + vals[2]
        else:
            return UNKNOWN
        out.append(f16bits(r) & 0xFFFF)
    return Const(out[0] | (out[1] << 16), "f16x2")


def f32_eval(op, args):
    kind = op.split(".")[0]
    vals = [a.f32() for a in args]
    if kind == "mul":
        r = vals[0] * vals[1]
    elif kind == "add":
        r = vals[0] + vals[1]
    elif kind == "sub":
        r = vals[0] - vals[1]
    elif kind == "fma":
        r = vals[0] * vals[1] + vals[2]
    elif kind == "min":
        r = min(vals)
    elif kind == "max":
        r = max(vals)
    elif kind == "abs":
        r = abs(vals[0])
    else:
        return UNKNOWN
    return Const(struct.unpack("<I", struct.pack("<f", r))[0], "f32")


# ---------------------------------------------------------------- driver


def find_kernels(name_filter, cubin_filter=None):
    files = sorted(CUBINS.glob("*_ptx.ptx"))
    if cubin_filter:
        files = [f for f in files if cubin_filter in f.name]
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        starts = [(m.start(), m.group(1)) for m in ENTRY_RE.finditer(text)]
        for i, (pos, kname) in enumerate(starts):
            if name_filter not in kname:
                continue
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            yield kname, p.name, text[pos:end]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    filt = args[0]
    cubin = args[args.index("--cubin") + 1] if "--cubin" in args else None
    report = args[args.index("--report") + 1] if "--report" in args else "all"
    full = "--full-expr" in args

    for kname, pname, body in find_kernels(filt, cubin):
        print(f"\n################ {kname}  [{pname}]")
        df = Dataflow(kname, pname, body, full_expr=full)

        if df.errs:
            print(f"!! {len(df.errs)} parse gaps (first 5): {df.errs[:5]}")

        if report in ("params", "all"):
            print(f"\n-- param struct loads ({len(df.param_loads)}) --")
            for off, width, regs, op in sorted(set(df.param_loads)):
                print(f"  +{off:<4} w={width:<3} {op:26} -> {','.join(regs)}")

        if report in ("loads", "all"):
            print(f"\n-- global loads by base ({len(df.global_loads)}) --")
            bybase = {}
            for slot, off, size, dests in df.global_loads:
                bybase.setdefault(slot, []).append((off, size))
            for slot, lst in sorted(bybase.items()):
                print(f"  base {slot}: {len(lst)} loads")
                consts = sorted({off.c for off, _ in lst if off.is_const()})
                if consts:
                    print(f"    const offsets ({len(consts)} distinct): {consts[:48]}")
                syms = [off for off, _ in lst if not off.is_const()]
                if syms:
                    shapes = sorted({repr(off) for off in syms})
                    print(f"    symbolic shapes ({len(syms)} loads, {len(shapes)} distinct):")
                    for s in shapes[:16]:
                        print(f"      {s}")

        if report in ("sinks", "all"):
            print(f"\n-- mma / quantise sinks ({len(df.sinks)}) --")
            seen = set()
            for s in df.sinks:
                key = re.sub(r"L\d+", "", s)
                if key not in seen:
                    seen.add(key)
                    print(s)

        if report in ("stores", "all"):
            print(f"\n-- global stores ({len(df.stores)}) --")
            bybase = {}
            for slot, off, size, src in df.stores:
                bybase.setdefault(slot, []).append((off, size, src))
            for slot, lst in sorted(bybase.items()):
                print(f"  base {slot}: {len(lst)} stores")
                for off, size, src in lst[:10]:
                    print(f"    [{off}]  {size}B  <= {df.fmt(src)}")

        if df.rsq_sites:
            print(f"\n-- rsqrt/rcp/ex2 sites: {len(df.rsq_sites)}")


if __name__ == "__main__":
    main()
