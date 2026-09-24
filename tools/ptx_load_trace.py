"""
ptx_load_trace.py - resolve every global load in a kernel back to
(param slot, constant offset).

Builds a symbolic value map: each 64-bit register becomes either
(param_slot, const) or UNKNOWN, propagating through cvta / add.s64 /
mul.wide.u32 / shl.b64. Then reports, for each ld.global, which parameter
buffer it reads and at what offset.

The fused Swin kernels keep their weights at compile-time literal offsets, so
the resulting offset histogram is the fused tensor's sub-matrix layout.

Usage:
    python ptx_load_trace.py <kernel-name> [--all-cubins]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5-analysis" / "cubins"

ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\s*\(")

LD_PARAM = re.compile(r"ld\.param\.b64\s+(%\w+),\s*\[(\w+)\+(\d+)\]\s*;")
CVTA = re.compile(r"cvta\.to\.global\.u64\s+(%\w+),\s+(%\w+)\s*;")
MOV = re.compile(r"mov\.u64\s+(%\w+),\s+(%\w+)\s*;")
ADD_IMM = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(-?\d+)\s*;")
ADD_REG = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(%\w+)\s*;")
MUL_IMM = re.compile(r"mul\.wide\.u32\s+(%\w+),\s+(%\w+),\s+(\d+)\s*;")
SHL = re.compile(r"shl\.b64\s+(%\w+),\s+(%\w+),\s+(\d+)\s*;")
# NOTE: must not require the mnemonic to start with "ld.global" -- the weight
# bulk stream is `ld.weak.global.ca.v4.u32` / `ld.weak.global.cg.v4.u32`, which
# a naive `ld\.global` prefix silently drops (same trap class as HANDOFF §73D,
# load direction). 65 ld.global.b32 + 64 ld.weak.global.ca.v4.u32 + 4
# ld.weak.global.cg.v4.u32 = the kernel's real 133 global loads.
LD_GLOBAL = re.compile(r"(ld\.(?:weak\.)?global[\w.]*)\s+(\{[^}]*\}|%r\w+|%rd\w+|%f\w+|%rs\w+|%p\w+),\s*\[(%rd\d+)(?:([+-]\d+))?\]\s*;")

UNKNOWN = ("?", None)


def trace(name_filter: str, all_cubins: bool = False):
    files = sorted(CUBINS.glob("*_ptx.ptx"))
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        starts = [(m.start(), m.group(1)) for m in ENTRY_RE.finditer(text)]
        for i, (pos, kname) in enumerate(starts):
            if name_filter not in kname:
                continue
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            body = text[pos:end]

            vals: dict[str, tuple] = {}
            loads: list[tuple[str, str, int | None]] = []

            for line in body.splitlines():
                line = line.strip()

                m = LD_PARAM.match(line)
                if m:
                    vals[m.group(1)] = (f"{m.group(2)}+{m.group(3)}", 0)
                    continue
                m = CVTA.match(line) or MOV.match(line)
                if m and m.group(2) in vals:
                    vals[m.group(1)] = vals[m.group(2)]
                    continue
                m = ADD_IMM.match(line)
                if m and m.group(2) in vals:
                    slot, off = vals[m.group(2)]
                    vals[m.group(1)] = (slot, off + int(m.group(3)))
                    continue
                m = ADD_REG.match(line)
                if m:
                    a, b = vals.get(m.group(2)), vals.get(m.group(3))
                    if a and not b:
                        vals[m.group(1)] = a
                    elif b and not a:
                        vals[m.group(1)] = b
                    continue
                m = MUL_IMM.match(line)
                if m and m.group(2) in vals:
                    slot, off = vals[m.group(2)]
                    vals[m.group(1)] = (slot, off * int(m.group(3))
                                        if off is not None else None)
                    continue
                m = SHL.match(line)
                if m and m.group(2) in vals:
                    slot, off = vals[m.group(2)]
                    vals[m.group(1)] = (slot, off * (1 << int(m.group(3)))
                                        if off is not None else None)
                    continue
                m = LD_GLOBAL.match(line)
                if m:
                    base = vals.get(m.group(3), UNKNOWN)
                    disp = int(m.group(4)) if m.group(4) else 0
                    off = None if base[1] is None else base[1] + disp
                    loads.append((m.group(1), base[0], off))

            if loads:
                yield kname, p.name, loads
            if not all_cubins:
                pass


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    filt = args[0]
    allc = "--all-cubins" in args

    seen = set()
    for kname, pname, loads in trace(filt, allc):
        if kname in seen:
            continue
        seen.add(kname)
        print(f"\n=== {kname}   [{pname}]  {len(loads)} global loads ===")
        by_slot: dict[str, list[int]] = {}
        for kind, slot, off in loads:
            by_slot.setdefault(f"{slot}  ({kind})", []).append(off)
        for slot in sorted(by_slot):
            offs = sorted(o for o in by_slot[slot] if o is not None)
            unknown = len(by_slot[slot]) - len(offs)
            print(f"  {slot}:  {len(offs)} const" + (f", {unknown} dynamic" if unknown else ""))
            if offs:
                print(f"      min={min(offs)}  max={max(offs)}")
                print(f"      {offs[:60]}")


if __name__ == "__main__":
    main()
