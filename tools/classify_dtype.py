"""
classify_dtype.py - decide, per tensor, whether the payload is FP16, FP8-E4M3
or FP8-E5M2.

The container only stores element counts, not dtypes, so we classify
statistically: decode a sample under each hypothesis and score how much the
result looks like trained neural-network weights (roughly zero-mean, small
standard deviation, no NaN/Inf, sane dynamic range).
"""

from __future__ import annotations

import sys
import os
import math
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pestruct as P


def fp8_e4m3(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x07
    if e == 0:
        return s * (m / 8.0) * 2.0 ** (1 - 7)
    if e == 0xF and m == 7:
        return math.nan      # E4M3FN: only 0x7f/0xff are NaN; max finite is 448.
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def fp8_e5m2(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 2) & 0x1F
    m = b & 0x03
    if e == 0:
        return s * (m / 4.0) * 2.0 ** (1 - 15)
    if e == 0x1F:
        return math.inf if m == 0 else math.nan
    return s * (1.0 + m / 4.0) * 2.0 ** (e - 15)


_LUT_E4M3 = [fp8_e4m3(i) for i in range(256)]
_LUT_E5M2 = [fp8_e5m2(i) for i in range(256)]


def _moments(vals):
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return None
    n = len(finite)
    mu = sum(finite) / n
    var = sum((v - mu) ** 2 for v in finite) / n
    return mu, math.sqrt(var), min(finite), max(finite), len(finite) / len(vals)


def score(m) -> float:
    """Higher is better. Reward plausible NN weight statistics."""
    if m is None:
        return -1e9
    mu, sd, lo, hi, finite_ratio = m
    if finite_ratio < 0.999:
        return -1e9
    s = 0.0
    # std in a believable band for conv/linear weights
    if 1e-5 <= sd <= 10.0:
        s += 10.0
    else:
        s -= 10.0 + abs(math.log10(max(sd, 1e-12)))
    # mean near zero relative to spread
    if sd > 0:
        s -= min(10.0, abs(mu) / sd)
    # bounded dynamic range
    if max(abs(lo), abs(hi)) < 1e3:
        s += 5.0
    else:
        s -= 5.0 + math.log10(max(abs(lo), abs(hi)))
    # symmetric-ish
    s -= 2.0 * abs(abs(lo) - abs(hi)) / max(abs(hi), 1e-12)
    return s


def classify(raw: bytes) -> tuple[str, dict]:
    n = len(raw)
    f16 = struct.unpack_from(f"<{n // 2}e", raw, 0)
    f32 = struct.unpack_from(f"<{n // 4}f", raw, 0) if n >= 4 else ()
    e4 = [_LUT_E4M3[b] for b in raw]
    e5 = [_LUT_E5M2[b] for b in raw]
    i8 = [float(v - 256 if v > 127 else v) for v in raw]
    bf16 = [struct.unpack("<f", struct.pack("<I", h << 16))[0]
            for h in struct.unpack_from(f"<{n // 2}H", raw, 0)]
    cand = {
        "fp16": _moments(f16),
        "fp8_e4m3": _moments(e4),
        "fp8_e5m2": _moments(e5),
        "fp32": _moments(f32),
        "int8": _moments(i8),
        "bf16": _moments(bf16),
    }
    scored = {k: score(v) for k, v in cand.items()}
    best = max(scored, key=scored.get)
    detail = {}
    for k, v in cand.items():
        if v:
            detail[k] = {"mean": v[0], "std": v[1], "min": v[2], "max": v[3]}
        detail[k + "_score"] = scored[k]
    return best, detail


if __name__ == "__main__":
    dll = sys.argv[1]
    pe = P.parse_pe(dll)
    res = [r for r in pe.resources if r.type_name == "RT_RCDATA"
           and r.name != "#1"]
    r = max(res, key=lambda x: x.size)
    base = r.file_offset
    total = struct.unpack_from("<Q", pe.data, base)[0]
    p = 8
    counts = {}
    rows = []
    while p < total:
        nl = struct.unpack_from("<Q", pe.data, base + p)[0]
        name = pe.data[base + p + 8:base + p + 8 + nl].decode("ascii", "replace")
        q = p + 8 + nl
        a = struct.unpack_from("<Q", pe.data, base + q)[0]
        c = struct.unpack_from("<Q", pe.data, base + q + 16)[0]
        ds = q + 28
        dl = a - 20
        sample = pe.data[base + ds:base + ds + min(c, 65536)]
        best, detail = classify(sample)
        counts[best] = counts.get(best, 0) + 1
        rows.append((name, c, c // 2, best, detail[best]["std"]))
        p = ds + dl

    for name, nbytes, nel, dt, sd in rows:
        print(f"{name:<30} {nbytes:>10,} B  {nel:>10,} el  {dt:<10} std={sd:.5f}")
    print("\n=== dtype histogram ===")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v}")
