"""
resolve_tensors.py - determine, per tensor, where the weight data actually
starts and what dtype it is.

Why this exists: the container stores no dtype field, and some tensors carry a
header prefix before the weight data (the ViT QKV tensor has 128 bytes of it).
Decoding from the wrong offset still produces plausible numbers. Known ViT
layouts use PTX-backed matrix boundaries; other tensors get statistical dtype
estimates only. Statistics cannot establish exact prefix/suffix boundaries.

  mean ~ 0, std in [0.001, 0.5], max|v| in [0.005, 8], essentially zero NaN

Writes model.resolved.json alongside model.json with `data_offset` and a
confidence flag per tensor.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
from typing import Callable

OFFSETS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048)
SAMPLE = 200_000


def e4m3(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x07
    if e == 0:
        return s * (m / 8.0) * 2.0 ** (1 - 7)
    if e == 0xF and m == 7:
        return math.nan
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def e5m2(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 2) & 0x1F
    m = b & 0x03
    if e == 0:
        return s * (m / 4.0) * 2.0 ** (1 - 15)
    if e == 0x1F:
        return math.inf * s if m == 0 else math.nan
    return s * (1.0 + m / 4.0) * 2.0 ** (e - 15)


LUT_E4M3 = [e4m3(i) for i in range(256)]
LUT_E5M2 = [e5m2(i) for i in range(256)]


def decode_fp16(buf: bytes, n: int):
    return list(struct.unpack_from("<%de" % (n // 2), buf, 0))


def decode(buf: bytes, dtype: str):
    n = min(len(buf), SAMPLE)
    if dtype == "fp8_e4m3":
        return [LUT_E4M3[b] for b in buf[:n]]
    if dtype == "fp8_e5m2":
        return [LUT_E5M2[b] for b in buf[:n]]
    if dtype == "fp16":
        return decode_fp16(buf, n - (n % 2))
    return []


def score(vals) -> tuple[float, dict]:
    if not vals:
        return -1e9, {}
    fin = [v for v in vals if math.isfinite(v)]
    frac = len(fin) / len(vals)
    if frac < 0.9999 or len(fin) < 16:
        return -1e9, {"finite_frac": frac}
    mu = sum(fin) / len(fin)
    var = sum((v - mu) ** 2 for v in fin) / len(fin)
    sd = math.sqrt(var)
    mx = max(abs(min(fin)), abs(max(fin)))
    s = 0.0
    s += 10.0 if 0.001 <= sd <= 0.5 else -10.0 - abs(math.log10(max(sd, 1e-9)))
    s += 8.0 if 0.005 <= mx <= 8.0 else -8.0 - abs(math.log10(max(mx, 1e-9)))
    if sd > 0:
        s -= min(10.0, abs(mu) / sd)
    return s, {"mean": mu, "std": sd, "absmax": mx, "finite_frac": frac}


DTYPES = ("fp8_e4m3", "fp8_e5m2", "fp16")


def vit_layout(name: str, raw: bytes) -> dict | None:
    """Verified payload boundaries; semantic roles are supported by PTX layouts.

    See dlss5-analysis/VIT_VERIFICATION.md. This requires corrected extraction
    (payload_layout_version=2); never compensate for an old byte-shifted dump.
    """
    match = re.fullmatch(r"block(3[1-8])\.layer([0-4])\.layer", name)
    if not match:
        return None
    layer = int(match[2])
    sizes = {0: (0, 4194304, 16), 1: (0, 4194304, 2048),
             2: (128, 3145728, 0), 3: (0, 2, 0),
             4: (0, 1048576, 2048)}
    off, size, tail = sizes[layer]
    if len(raw) != off + size + tail:
        raise ValueError(f"{name}: unexpected payload size {len(raw)}")
    dtype = "unknown" if layer == 3 else "fp8_e4m3"
    segments = []
    if layer == 2:
        values = struct.unpack_from("<32f", raw)
        if not all(math.isfinite(v) and v >= 2.0 ** -126 for v in values):
            raise ValueError(f"{name}: invalid QKV prefix; check extraction alignment")
        segments.append({"offset": 0, "bytes": 128, "dtype": "fp32",
                         "role": "qkv_head_coefficients", "count": 32})
    segments.append({"offset": off, "bytes": size, "dtype": dtype,
                     "role": "scalar" if layer == 3 else "weights"})
    if layer == 3:
        segments[-1]["candidate_interpretation"] = "fp16"
    if tail:
        if layer == 0:
            if any(raw[size:]):
                raise ValueError(f"{name}: expected a 16-byte zero suffix")
            segments.append({"offset": size, "bytes": tail, "role": "zero_suffix"})
        else:
            values = struct.unpack_from("<1024e", raw, size)
            if not all(math.isfinite(v) and 0 < v <= 1 for v in values):
                raise ValueError(f"{name}: invalid residual coefficients; check alignment")
            segments.append({"offset": size, "bytes": tail, "dtype": "fp16",
                             "role": "residual_coefficients", "count": 1024})
    _, stats = score(decode(raw[off:off + size], dtype))
    return {"data_offset": off, "dtype": dtype,
            "confidence": "unresolved" if layer == 3 else "verified_layout",
            "offset_confidence": "structural", "stats": stats,
            "bytes_of_data": size, "segments": segments,
            "evidence": "VIT_VERIFICATION.md"}


def resolve(path: str, name: str = "") -> dict:
    raw = open(path, "rb").read()
    known = vit_layout(name, raw)
    if known is not None:
        return known
    # IMPORTANT: take the FIRST acceptable offset in ascending order, not the
    # best-scoring one. Any offset inside a weight matrix also looks sane, so
    # picking the maximum score silently skips real data. The header is a
    # PREFIX, so the data starts at the earliest offset that decodes sanely.
    threshold = 15.0
    for off in OFFSETS:
        if off >= len(raw):
            break
        chunk = raw[off:]
        if len(chunk) < 64:
            break
        best_here = (-1e9, "unknown", {})
        for dt in DTYPES:
            sc, st = score(decode(chunk, dt))
            if sc > best_here[0]:
                best_here = (sc, dt, st)
        sc, dt, st = best_here
        if sc > threshold:
            return {
                "data_offset": off,
                "dtype": dt,
                "confidence": "statistical",
                "offset_confidence": "unverified",
                "boundary_note": "Plausible dtype only; short headers and suffixes may be undetectable.",
                "stats": st,
                "bytes_of_data": len(raw) - off,
            }
    # nothing sane anywhere: report the least-bad attempt at offset 0
    best = (-1e9, "unknown", {})
    for dt in DTYPES:
        sc, st = score(decode(raw, dt))
        if sc > best[0]:
            best = (sc, dt, st)
    return {
        "data_offset": 0,
        "dtype": best[1] if best[0] > -1e8 else "unknown",
        "confidence": "unresolved",
        "offset_confidence": "unverified",
        "stats": best[2],
        "bytes_of_data": len(raw),
    }


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "dlss5-analysis"
    model = json.load(open(os.path.join(root, "model.json")))
    if model.get("payload_layout_version") != 2:
        raise ValueError("Re-extract with the corrected extract_dlssnr.py first: old dumps skip one payload byte.")
    tensors = model["tensors"]

    conf = {}
    for t in tensors:
        p = os.path.join(root, "tensors", "tensor_%03d.bin" % t["index"])
        if not os.path.exists(p):
            continue
        r = resolve(p, t["name"])
        t["resource_data_offset"] = t["data_offset"]
        t["data_offset"] = r["data_offset"]
        t["offset_confidence"] = r["offset_confidence"]
        t["dtype_resolved"] = r["dtype"]
        t["dtype_confidence"] = r["confidence"]
        t["dtype_stats"] = r["stats"]
        t["data_bytes"] = r["bytes_of_data"]
        for key in ("segments", "evidence", "boundary_note"):
            if key in r:
                t[key] = r[key]
        conf[r["confidence"]] = conf.get(r["confidence"], 0) + 1

    out = os.path.join(root, "model.resolved.json")
    json.dump(model, open(out, "w"), indent=2)

    print(f"{len(tensors)} tensors")
    for k in ("verified_layout", "statistical", "unresolved"):
        print(f"  {k:<12} {conf.get(k, 0)}")
    print()
    print(f"{'idx':>4} {'name':<26}{'off':>6} {'dtype':<10}{'std':>9}"
          f"{'absmax':>9} {'conf':<10}")
    for t in tensors:
        s = t.get("dtype_stats", {})
        print(f"{t['index']:>4} {t['name']:<26}{t.get('data_offset',0):>6} "
              f"{t.get('dtype_resolved','?'):<10}"
              f"{s.get('std', float('nan')):>9.4f}{s.get('absmax', float('nan')):>9.3f} "
              f"{t.get('dtype_confidence','?'):<10}")
    print(f"\n[+] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
