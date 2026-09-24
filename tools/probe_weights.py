"""
probe_weights.py - figure out how the 147 MB .rsrc blob is laid out.

Strategy: the tensor table has to be somewhere. Look for it by
  1. hexdumping the head of the blob
  2. hunting for a plausible record count (e.g. 152)
  3. looking for a fixed-stride record array (offset/size/dtype/dims)
  4. checking whether the tail of the blob is a directory appended after data
"""

from __future__ import annotations

import sys
import os
import re
import json
import struct
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pestruct as P


def hexdump(b: bytes, base: int = 0, width: int = 16) -> str:
    out = []
    for i in range(0, len(b), width):
        row = b[i:i + width]
        h = " ".join(f"{x:02X}" for x in row)
        a = "".join(chr(x) if 32 <= x < 127 else "." for x in row)
        out.append(f"{base + i:08X}  {h:<{width * 3}}  {a}")
    return "\n".join(out)


def printable_runs(b: bytes, min_len: int = 6):
    for m in re.finditer(rb"[ -~]{%d,}" % min_len, b):
        yield m.start(), m.group(0).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("--head", type=int, default=512)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pe = P.parse_pe(args.dll)
    sec = max(pe.sections, key=lambda s: s.rawsize)
    print(f"[*] largest section .{sec.name}: {sec.rawsize:,} bytes "
          f"@ file 0x{sec.rawoff:X}")

    print(f"\n[*] image directories:")
    for i, (rva, size) in enumerate(pe.directories):
        if rva:
            print(f"    [{i:>2}] rva=0x{rva:X} size={size:,}")

    head = pe.data[sec.rawoff:sec.rawoff + args.head]
    print(f"\n=== HEAD of .{sec.name} ===")
    print(hexdump(head, sec.rawoff))

    tail_off = sec.rawoff + max(0, sec.rawsize - args.tail)
    print(f"\n=== TAIL of .{sec.name} ===")
    print(hexdump(pe.data[tail_off:tail_off + args.tail], tail_off))

    # --- hunt for a record count ----------------------------------------
    print(f"\n=== RECORD COUNT HUNT ===")
    blob_start = sec.rawoff
    region = pe.data[blob_start:blob_start + min(sec.rawsize, 1 << 16)]
    for want in (152, 0x98):
        hits = [i for i in range(0, len(region) - 4)
                if struct.unpack_from("<I", region, i)[0] == want]
        print(f"  u32 {want}: {len(hits)} hits in first 64 KB "
              f"-> {[hex(blob_start + h) for h in hits[:16]]}")

    # --- printable strings in the blob head (a name/index would show up) --
    print(f"\n=== STRINGS in first 64 KB of blob ===")
    n = 0
    for off, s in printable_runs(region, 8):
        print(f"    +{off:#07x}  {s}")
        n += 1
        if n > 40:
            print("    ...")
            break

    # --- look for a fixed-stride record array ----------------------------
    # heuristic: many small u32s that are near-monotonically increasing
    # (offsets) or a repeated (a,b) pattern
    print(f"\n=== STRIDE SCAN (u64 pairs, offsets/sizes) ===")
    for stride in (16, 24, 32, 40, 48, 56, 64):
        vals = []
        for i in range(0, 4096, stride):
            if i + 16 > len(region):
                break
            a, b = struct.unpack_from("<QQ", region, i)
            vals.append((a, b))
        if not vals:
            continue
        inc = sum(1 for i in range(1, len(vals))
                  if vals[i][0] > vals[i - 1][0])
        print(f"    stride {stride:>3}: monotonic-offset ratio "
              f"{inc / max(1, len(vals) - 1):.2f}  first={vals[:3]}")

    # --- is the blob a known container? ----------------------------------
    print(f"\n=== CONTAINER CHECK ===")
    magic = pe.data[sec.rawoff:sec.rawoff + 16]
    print(f"    first 16 bytes: {magic.hex(' ')}")
    for name, sig in (("zstd", b"\x28\xb5\x2f\xfd"), ("elf", b"\x7fELF"),
                      ("zip", b"PK\x03\x04"), ("gzip", b"\x1f\x8b"),
                      ("lz4", b"\x04\x22\x4d\x18")):
        print(f"    {name:<6}: {magic[:4] == sig}")

    # --- entropy profile (finds table vs data boundaries) ----------------
    print(f"\n=== ENTROPY PROFILE (1 MB windows) ===")
    import math
    step = 1 << 20
    for w in range(0, sec.rawsize, max(step, sec.rawsize // 24)):
        chunk = pe.data[sec.rawoff + w:sec.rawoff + w + (1 << 16)]
        if not chunk:
            break
        c = Counter(chunk)
        ent = -sum((v / len(chunk)) * math.log2(v / len(chunk))
                   for v in c.values())
        print(f"    +{w / (1 << 20):>7.1f} MB  entropy={ent:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
