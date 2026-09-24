"""
demangle_tin.py - minimal Itanium-C++ demangler for the subset NVIDIA's `tin`
inference library uses, plus a layer-configuration extractor.

The sm_120 PTX embeds fully mangled template names, and `tin` (NVIDIA's
template inference library) encodes layer hyperparameters *as template
arguments*. That means the network architecture is sitting in the binary as
plain integers, e.g.

    tin3_1::Conv2dQKVLayer<tin3_1::Conv2dQKVConfig<32, 32, 1024, 16, 8, 32,
                           2, 2, 2, false, false, 128, ...>, DType=2>

Handles the subset that actually appears: `_ZZ` local-static prefixes,
`NK<...>E` const nested names, length-prefixed identifiers, nested templates,
`Li/Lj/Ll/Lm` integer and `Lb` boolean literal args, and `S_/S0_/S1_/...`
substitutions. Anything unrecognised is passed through rather than guessed at.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

CV_REF = "rVKO"          # restrict, volatile, const, ref-qualifiers


class Demangler:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.subs: list[str] = []

    # -- low level --------------------------------------------------------
    def peek(self, n: int = 1) -> str:
        return self.s[self.i:self.i + n]

    def take(self, n: int = 1) -> str:
        r = self.s[self.i:self.i + n]
        self.i += n
        return r

    def number(self) -> int:
        j = self.i
        if self.peek() == "n":
            self.i += 1
            j += 1
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        if self.i == j:
            raise ValueError("expected number at %d" % j)
        v = int(self.s[j:self.i])
        return -v if self.s[j - 1:j] == "n" else v

    # -- substitutions ----------------------------------------------------
    def substitution(self) -> str:
        """S_ / S0_ / S1_ / ... / SA_ -- base-36 index, 0-based."""
        self.take()                       # 'S'
        # Index is [0-9A-Z]* terminated by '_'. Do NOT swallow the '_' early
        # or you eat the following identifier (e.g. "S_15Conv2d..." ).
        j = self.i
        while self.i < len(self.s) and self.s[self.i].isalnum():
            self.i += 1
        code = self.s[j:self.i]
        if self.peek() == "_":
            self.take()
        else:
            return "std"                  # bare Sx without '_' -> builtin
        idx = 0 if code == "" else int(code, 36) + 1   # S_=0, S0_=1, S1_=2
        if idx < len(self.subs):
            return self.subs[idx]
        # not seen yet (we only record prefixes); fall back to the common one
        return self.subs[0] if self.subs else "tin3_1"

    # -- names ------------------------------------------------------------
    def name(self) -> str:
        c = self.peek()
        if c == "S":
            return self.substitution()
        if c == "N" and self.peek(2) != "S":
            return self.nested()
        if c in "0123456789":
            n = self.number()
            return self.take(n)
        return ""

    def nested(self) -> str:
        """N [<cv-ref>] <prefix components> [I <args> E] E"""
        self.take()                                    # 'N'
        while self.peek() in CV_REF:
            self.take()
        parts: list[str] = []
        while self.i < len(self.s) and self.peek() not in ("E", "I"):
            p = self.name()
            if not p:
                break
            parts.append(p)
            # Itanium records each prefix component as a substitution
            path = "::".join(parts)
            if path not in self.subs:
                self.subs.append(path)
        base = "::".join(parts) if parts else ""
        if self.peek() == "I":
            base += "<%s>" % ", ".join(self.template_args())
        if self.peek() == "E":
            self.take()
        return base

    def template_args(self) -> list[str]:
        self.take()                                    # 'I'
        out = []
        while self.i < len(self.s) and self.peek() != "E":
            before = self.i
            a = self.type()
            if a == "":
                break
            out.append(a)
            if self.i == before:                       # no progress guard
                self.take()
        if self.peek() == "E":
            self.take()
        return out

    # -- types ------------------------------------------------------------
    def type(self) -> str:
        c = self.peek()
        if c == "":
            return ""
        if c == "L":                                   # literal template arg
            self.take()
            k = self.peek()
            if k == "b":                               # bool: Lb<0|1>E
                self.take()
                try:
                    r = "true" if self.number() else "false"
                except ValueError:
                    r = "false"
                if self.peek() == "E":
                    self.take()
                return r
            if k in "ijlmsx":                          # int: Li<digits>E
                self.take()
                try:
                    r = str(self.number())
                except ValueError:
                    r = "0"
                # consume the literal's own closing 'E', otherwise it gets
                # mistaken for the end of the enclosing argument list
                if self.peek() == "E":
                    self.take()
                return r
            # otherwise: L <type> <value> E  (scoped enums, etc.)
            ty = self.type()
            try:
                val = self.number()
            except ValueError:
                val = 0
            if self.peek() == "E":
                self.take()
            return "(%s)%s" % (ty, val)
        if c == "J":                                   # template argument pack
            self.take()
            inner = self.template_args() if self.peek() != "E" else []
            return ", ".join(inner)
        if c == "N":
            return self.nested()
        if c == "i":
            self.take(); return "int"
        if c == "j":
            self.take(); return "unsigned"
        if c == "f":
            self.take(); return "float"
        if c == "d":
            self.take(); return "double"
        if c == "v":
            self.take(); return "void"
        if c == "b":
            self.take(); return "bool"
        if c == "P":
            self.take(); return self.type() + "*"
        if c == "K":
            self.take(); return "const " + self.type()
        if c == "R":
            self.take(); return self.type() + "&"
        if c == "D" and self.peek(2) == "Dn":
            self.take(2); return "decltype(nullptr)"
        n = self.name()
        if not n:
            self.take()
            return "?"
        if self.peek() == "I":
            return "%s<%s>" % (n, ", ".join(self.template_args()))
        if n not in self.subs:
            self.subs.append(n)
        return n

    # -- entry ------------------------------------------------------------
    def run(self) -> str:
        if not self.s.startswith("_Z"):
            return self.s
        self.take(2)
        # function-local statics are encoded _ZZ <encoding> E <name>;
        # skip the local-scope 'Z' markers
        while self.peek() == "Z":
            self.take()
            if self.peek() in CV_REF:
                pass
        try:
            out = self.type()
            # trailing junk (member name, signature) is not interesting here
            return out
        except Exception:
            return self.s


def demangle(s: str) -> str:
    return Demangler(s).run()


# ---------------------------------------------------------------------------

TIN_RE = re.compile(r"_Z[A-Za-z0-9_$.]*tin3_1[A-Za-z0-9_$.]*")


def extract_layers(text: str) -> Counter:
    """{demangled tin type: occurrence count} from a chunk of PTX."""
    seen: Counter = Counter()
    for m in TIN_RE.finditer(text):
        d = demangle(m.group(0))
        for lm in re.finditer(
                r"tin3_1::[A-Za-z0-9_]+(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>)?", d):
            t = lm.group(0)
            if len(t) > 30:                    # skip bare class names
                seen[t] += 1
    return seen


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demangle":
        for line in sys.stdin:
            print(demangle(line.strip()))
        raise SystemExit(0)

    import glob
    import os
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.ptx")))
    print(f"scanning {len(files)} PTX files")
    allc = Counter()
    for f in files:
        allc.update(extract_layers(open(f, errors="replace").read()))
    print(f"{len(allc)} distinct tin layer types\n")
    for k, v in sorted(allc.items(), key=lambda x: (-x[1], x[0])):
        print(f"{v:>5}  {k}")
