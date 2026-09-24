#!/usr/bin/env python3
"""Map the dtype regions of a resolved tensor, 256 bytes at a time.

Why: the layout rules (§4/§208) say a tensor interleaves e4m3 weight regions with
fp16 gate/bias regions, and the transitions add an e4m3 projection. The *boundaries*
between those regions are what pins a layout, and they are visible from the bytes
alone -- provided the probe is calibrated, which is why this tool is meant to be run
on a tensor whose map is already known before it is trusted on a new one.

Two signals per window:

  * `0x7F`/`0xFF`: e4m3 has no infinity and only those two NaN codes, so trained
    e4m3 weights essentially never contain them. fp16 data has no such constraint.
    A window with zero hits is e4m3-CONSISTENT; a window with hits is NOT e4m3.
    (Calibrated in S208/S209: the C=32 stage's own A region = 0 hits in 8192 B.)
  * `f16max`: decode the window as fp16 and take max|v|. e4m3 bytes read as fp16
    produce wild magnitudes (tens to thousands); genuine fp16 weights stay small.

Verdict per window: "e4m3" (0 hits), "fp16" (hits and a small f16max), else "?". A
third state matters for the transitions: an appened projection is e4m3 too, so the
*e4m3/f16 alternation* is the map, not the presence of e4m3.

    python tools/dtype_map.py dlss5-analysis/tensors/tensor_091.bin --from 0 --to 22720
"""
import argparse
import struct
import sys

S = 256


def analyse(data, off, n):
    b = data[off:off + n]
    hits = b.count(0x7F) + b.count(0xFF)
    try:
        v = struct.unpack_from("<%de" % (len(b) // 2), b, 0)
        mx = max(abs(x) for x in v if x == x)
    except Exception:
        mx = float("nan")
    if hits == 0:
        verdict = "e4m3"
    elif mx <= 8.0:
        verdict = "fp16"
    else:
        verdict = "?"
    return hits, mx, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tensor")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=0)
    ap.add_argument("--window", type=int, default=S)
    ap.add_argument("--quiet", action="store_true", help="only print verdict changes")
    a = ap.parse_args()

    data = open(a.tensor, "rb").read()
    hi = a.hi or len(data)
    print("== %s: %d B; map %d..%d, %d B windows" % (a.tensor, len(data), a.lo, hi,
                                                     a.window))
    prev = None
    print("   %-16s %-8s %-6s %s" % ("offset", "hits", "f16max", "verdict"))
    for off in range(a.lo, min(hi, len(data)) - a.window + 1, a.window):
        hits, mx, verdict = analyse(data, off, a.window)
        if a.quiet and verdict == prev:
            prev = verdict
            continue
        print("   [%6d..%-6d] %-8d %-6.2f %s"
              % (off, off + a.window, hits, mx, verdict))
        prev = verdict
    tail = len(data) % a.window
    if tail and hi >= len(data) - tail:
        hits, mx, verdict = analyse(data, len(data) - tail, tail)
        print("   [%6d..%-6d] %-8d %-6.2f %-10s (partial tail %d B)"
              % (len(data) - tail, len(data), hits, mx, verdict, tail))


if __name__ == "__main__":
    sys.exit(main())
