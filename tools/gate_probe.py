#!/usr/bin/env python3
"""Find fp16 gate blocks in a resolved tensor, and describe the region structure.

Why this exists: the fused-stage layout (S158/S170) is anchored by its two fp16
*gate* blocks, and the transitions (S201) proved the stage rules are NOT
transferable, so every new block needs the same measurement. S203's lesson was that
a probe must be calibrated against a known answer before it is trusted, so this tool
carries its calibration in the command line:

    # calibrate first -- these offsets are known from S170 and S203
    python tools/gate_probe.py dlss5-analysis/tensors/tensor_001.bin --expect 8208,20592
    python tools/gate_probe.py dlss5-analysis/tensors/tensor_000.bin --expect 9232,21616
    # then the unknown
    python tools/gate_probe.py dlss5-analysis/tensors/tensor_091.bin

The probe, and why it is NOT "all values in (0,1]": the S203 probe used exactly
that and it reads correctly on the stem (tensor_000), but tensor_001's gate1 holds
-0.1455 and -0.3894 and its gate2 holds -0.0083 -- a few negatives are normal, so an
all-positive test finds nothing at all (measured this session: 0 groups on
tensor_001, i.e. the S203 criterion does not generalise). What separates a gate
block from weight data is that a gate's coefficients are all *substantial* and
clustered near 1:

    gate     median(v) ~ 0.85     count(|v| < 0.05) <= 2
    weights  median(v) ~ 0.00     count(|v| < 0.05) large

Both statistics are printed, plus the best REJECTED window, so the separation is
visible rather than assumed.

No numpy: struct's 'e' decodes half floats.
"""
import argparse
import struct
import sys

GATE_MEDIAN = 0.40
GATE_NZ = 2
NZ_ABS = 0.05


def f16s(data, off, n):
    return struct.unpack_from("<%de" % n, data, off)


def scan(data):
    """Every 64 B window as (offset, median, count(|v|<NZ_ABS), frac in (0,1])."""
    rows = []
    for i in range(0, len(data) // 2 - 32 + 1):
        v = struct.unpack_from("<32e", data, i * 2)
        med = sorted(v)[16]
        nz = sum(1 for x in v if not (x == x) or abs(x) < NZ_ABS)
        f01 = sum(1 for x in v if x == x and 0 < x <= 1) / 32.0
        rows.append((i * 2, med, nz, f01))
    return rows


def is_gate(r):
    return r[1] >= GATE_MEDIAN and r[2] <= GATE_NZ


def groups_of(rows):
    hits = [r[0] for r in rows if is_gate(r)]
    out = []
    for o in hits:
        if out and o - out[-1][1] <= 64:
            out[-1][1] = o
        else:
            out.append([o, o])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tensor")
    ap.add_argument("--expect", default="",
                    help="comma-separated gate offsets the tensor is KNOWN to have; "
                         "a miss means the probe is wrong, not the tensor")
    ap.add_argument("--stats", type=int, default=2048,
                    help="window size for the region statistics (default 2048)")
    ap.add_argument("--no-values", dest="values", action="store_false", default=True,
                    help="do not print the 32 values of each group")
    a = ap.parse_args()

    data = open(a.tensor, "rb").read()
    print("== %s: %d bytes, %d fp16" % (a.tensor, len(data), len(data) // 2))

    rows = scan(data)
    groups = groups_of(rows)
    print("gate groups (median >= %.2f and count(|v|<%.2f) <= %d): %d"
          % (GATE_MEDIAN, NZ_ABS, GATE_NZ, len(groups)))
    for g in groups:
        vals = f16s(data, g[0], 32)
        print("   @%d..%d (%d B)  median %.4f  max %.4f  min %.4f  frac(0,1] %.3f"
              % (g[0], g[1] + 64, g[1] + 64 - g[0], sorted(vals)[16], max(vals),
                 min(vals), sum(1 for x in vals if 0 < x <= 1) / 32.0))
        if a.values:
            print("      " + " ".join("%.4g" % x for x in vals))

    if a.expect:
        want = [int(x) for x in a.expect.split(",") if x.strip()]
        for w in want:
            near = [g for g in groups if g[0] - 8 <= w <= g[1] + 72 or abs(g[0] - w) <= 8]
            print("   expect @%d -> %s" % (w, ("FOUND (@%d)" % near[0][0]) if near
                                           else "*** NOT FOUND ***"))
        rej = [r for r in rows if not is_gate(r)]
        if rej:
            best = max(rej, key=lambda r: (r[1], -r[2]))
            print("   separation: best REJECTED window @%d median %.4f nz %d"
                  % (best[0], best[1], best[2]))
        else:
            print("   separation: *** nothing rejected -- the criterion is useless ***")

    S = a.stats
    print("\nregion stats, %d B windows:" % S)
    for off in range(0, len(data) - S + 1, S):
        vals = f16s(data, off, S // 2)
        fin = [v for v in vals if v == v]
        print("   [%6d..%-6d] max %9.4f min %9.4f  frac(0,1] %.3f  frac neg %.3f  nf %d"
              % (off, off + S, max(fin), min(fin),
                 sum(1 for v in fin if 0 < v <= 1) / float(len(vals)),
                 sum(1 for v in fin if v < 0) / float(len(vals)),
                 len(vals) - len(fin)))
    tail = len(data) % S
    if tail:
        vals = f16s(data, len(data) - tail, tail // 2)
        fin = [v for v in vals if v == v]
        print("   [%6d..%-6d] max %9.4f min %9.4f  *** partial tail %d B ***"
              % (len(data) - tail, len(data), max(fin), min(fin), tail))


if __name__ == "__main__":
    sys.exit(main())
