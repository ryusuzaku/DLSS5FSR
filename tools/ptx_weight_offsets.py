"""
ptx_weight_offsets.py - recover the fused-tensor sub-matrix offsets by tracing
the kernel's parameter pointers forward through the PTX instruction stream.

The fused Swin kernels (cc_tinlayout_fused_swin_*) receive ONE weights pointer
plus a few ints, so every sub-matrix has to be addressed with a compile-time
literal offset. Those literals are what we want, e.g.

    ld.param.b64  %rd10, [param_0+16];     <- weights base
    add.s64       %rd499, %rd10, 57344;    <- sub-matrix #2 begins at +57344

The naive approach (histogram every large immediate in the kernel) drowns in
unrolled activation-tile strides. This instead propagates only from registers
loaded out of the parameter block, keeping offsets that are applied directly to
a parameter-derived pointer.

Usage:
    python ptx_weight_offsets.py <kernel-substring> [--min 256] [--ctx]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5-analysis" / "cubins"

ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\s*\(")
PARAM_STRUCT_RE = re.compile(r"\.param\s+\.align\s+\d+\s+\.b8\s+\S+\[(\d+)\]")

# ld.param.b64 %rd10, [param_0+16];
LD_PARAM_RE = re.compile(r"ld\.param\.b64\s+(%\w+),\s*\[(\w+)\+(\d+)\]\s*;")
# cvta.to.global.u64 %rd1, %rd8;
CVTA_RE = re.compile(r"cvta\.to\.global\.u64\s+(%\w+),\s+(%\w+)\s*;")
# add.s64 %rd499, %rd12, 57344;
ADD_IMM_RE = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(-?\d+)\s*;")
# add.s64 %rd12, %rd11, %rd10;   (loop index + base, no immediate)
ADD_REG_RE = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(%\w+)\s*;")


def iter_kernels(text: str):
    starts = [(m.start(), m.group(1)) for m in ENTRY_RE.finditer(text)]
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        yield name, text[pos:end]


def analyse(name: str, body: str, min_imm: int):
    """Return {param_slot: Counter({offset: n})} of literal offsets applied to
    pointers loaded out of the parameter block."""
    # register -> (param slot it came from, accumulated literal offset)
    origin: dict[str, tuple[str, int]] = {}
    hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for line in body.splitlines():
        line = line.strip()

        m = LD_PARAM_RE.match(line)
        if m:
            reg, slot, off = m.group(1), m.group(2), int(m.group(3))
            origin[reg] = (f"{slot}+{off}", 0)
            continue

        m = CVTA_RE.match(line)
        if m and m.group(2) in origin:
            origin[m.group(1)] = origin[m.group(2)]
            continue

        m = ADD_REG_RE.match(line)
        if m:
            dst, a, b = m.group(1), m.group(2), m.group(3)
            # one side is the pointer, the other a loop index / thread id
            if a in origin and b not in origin:
                origin[dst] = origin[a]
            elif b in origin and a not in origin:
                origin[dst] = origin[b]
            continue

        m = ADD_IMM_RE.match(line)
        if m:
            dst, src, imm = m.group(1), m.group(2), int(m.group(3))
            if src in origin:
                slot, acc = origin[src]
                total = acc + imm
                origin[dst] = (slot, total)
                if min_imm <= imm:
                    hits[slot][total] += 1
            continue

    return hits


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    want = args[0]
    min_imm = 256
    if "--min" in args:
        min_imm = int(args[args.index("--min") + 1])

    for p in sorted(CUBINS.glob("*_ptx.ptx")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, body in iter_kernels(text):
            if want not in name or not name.startswith(("cc_tinlayout_fused",
                                                         "cc_split_swin",
                                                         "cc_vit")):
                continue
            m = PARAM_STRUCT_RE.search(body)
            size = m.group(1) if m else "?"
            hits = analyse(name, body, min_imm)
            if not hits:
                continue
            print(f"\n=== {name}   (param struct {size} B)  [{p.name}] ===")
            for slot in sorted(hits, key=lambda s: int(s.split("+")[1])):
                offs = hits[slot]
                print(f"  param {slot}:")
                for off in sorted(offs):
                    print(f"      +{off:<9} (0x{off:X})  x{offs[off]}")


if __name__ == "__main__":
    main()
