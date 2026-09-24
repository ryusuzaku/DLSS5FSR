"""
ptx_offsets.py - recover the fused-tensor sub-matrix offsets from the PTX.

The fused Swin kernels (cc_tinlayout_fused_swin_{1,2,4,8}h_{32,64,128,256}_*)
take a SINGLE weight pointer plus a handful of ints. They therefore have to
address each sub-matrix with a compile-time literal offset, e.g.

    add.s64 %rd499, %rd12, 57344;

so the fused-tensor layout is recoverable straight from the instruction stream.

Usage:
    python ptx_offsets.py [--kernel <substring>] [--min 1024]
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5-analysis" / "cubins"

ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\s*\(")
# add.s64 d, a, imm   /  add.s32 d, a, imm   /  mad.lo.s64 d, a, b, imm
IMM_RE = re.compile(
    r"\b(add|mad)\.(?:lo|wide|hi)?\.?(?:s|u)(?:16|32|64)\s+[^;]*?,\s*(-?\d+)\s*;"
)


def iter_kernels(ptx_text: str):
    """Yield (name, body) for each .visible .entry in the file."""
    starts = [(m.start(), m.group(1)) for m in ENTRY_RE.finditer(ptx_text)]
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(ptx_text)
        yield name, ptx_text[pos:end]


def main() -> None:
    args = sys.argv[1:]
    want = ""
    if "--kernel" in args:
        want = args[args.index("--kernel") + 1]
    min_imm = 1024
    if "--min" in args:
        min_imm = int(args[args.index("--min") + 1])

    # {kernel: Counter({imm: n})}
    found: dict[str, Counter] = {}
    for p in sorted(CUBINS.glob("*_ptx.ptx")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, body in iter_kernels(text):
            if want and want not in name:
                continue
            if not name.startswith("cc_tinlayout_fused"):
                continue
            c = found.setdefault(name, Counter())
            for m in IMM_RE.finditer(body):
                v = int(m.group(2))
                if v >= min_imm:
                    c[v] += 1
            found[name] = c

    for name in sorted(found):
        c = found[name]
        if not c:
            continue
        print(f"\n=== {name} ===")
        for imm, n in sorted(c.items()):
            print(f"   {imm:>10}  (0x{imm:X})   x{n}")

    print(f"\n{len(found)} fused kernels with immediates >= {min_imm}")


if __name__ == "__main__":
    main()
