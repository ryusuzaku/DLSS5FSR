"""
fused_decode.py - decompose the single-tensor "fused" stage blocks.

The stage families (689232 / 197184 / 61760 / 20672 bytes) are several weight
matrices concatenated into one payload. We break them by combining two sources
of truth:

  1. AUTHORITATIVE channel counts harvested from the sm_120 PTX template
     arguments (`CrazyCuckooFusedSwin2d{2,4}HConfig<C, ...>`).
  2. The structural separator rule: each sub-matrix is followed by either
     - fp16 gate coefficients: 2 * out_channels bytes, all positive, <= 1.0
     - a small f32 scale block

Usage:
    python fused_decode.py              # survey all single-tensor blocks
    python fused_decode.py --solve      # brute-force the decompositions
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "dlss5-analysis"
TENSORS = ANA / "tensors"

FAMILIES = {
    689232: [15, 16, 17, 18, 19, 20, 21, 49, 50, 51, 52, 53, 54, 55],
    197184: [9, 10, 11, 12, 13, 57, 58, 59, 60, 61],
    61760: [5, 6, 7, 63, 64, 65],
    20672: [1, 2, 3, 67, 68, 69],
}


def load_index() -> dict[str, dict]:
    m = json.loads((ANA / "model.resolved.json").read_text(encoding="utf-8"))
    return {t["name"]: t for t in m["tensors"]}


def read_tensor(idx: dict, name: str) -> bytes:
    i = idx[name]["index"]
    return (TENSORS / f"tensor_{i:03d}.bin").read_bytes()


# -- data views ------------------------------------------------------------

def as_fp16(buf: bytes):
    return struct.unpack(f"<{len(buf)//2}e", buf[: len(buf) // 2 * 2])


def as_f32(buf: bytes):
    return struct.unpack(f"<{len(buf)//4}f", buf[: len(buf) // 4 * 4])


def gate_score(buf: bytes) -> tuple[bool, float, float]:
    """Is `buf` plausibly fp16 gate coefficients? (positive, <= 1.0)."""
    if len(buf) < 4:
        return False, 0.0, 0.0
    vals = as_fp16(buf)
    if not vals:
        return False, 0.0, 0.0
    pos = sum(1 for v in vals if 0.0 < v <= 1.0)
    frac = pos / len(vals)
    return frac > 0.95, frac, max(vals)


def scale_score(buf: bytes) -> tuple[bool, float, float]:
    """Is `buf` plausibly a small f32 scale block (positive, modest range)?"""
    if len(buf) < 8:
        return False, 0.0, 0.0
    vals = as_f32(buf)
    if not vals:
        return False, 0.0, 0.0
    pos = sum(1 for v in vals if v > 0.0)
    frac = pos / len(vals)
    return frac > 0.95, frac, max(vals)


# -- search ----------------------------------------------------------------

def find_separators(data: bytes, min_len: int = 64, max_len: int = 4096):
    """Scan every 4-byte-aligned offset for a run that decodes as fp16 gates
    or f32 scales. Returns candidate [start, end, kind, frac]."""
    out = []
    n = len(data)
    for off in range(0, n - min_len, 4):
        for ln in (min_len, 128, 256, 512, 1024, 2048):
            if ln > max_len or off + ln > n:
                continue
            ok, frac, mx = gate_score(data[off : off + ln])
            if ok:
                out.append((off, off + ln, "fp16gate", round(frac, 3), round(mx, 4)))
            ok, frac, mx = scale_score(data[off : off + ln])
            if ok:
                out.append((off, off + ln, "f32scale", round(frac, 3), round(mx, 4)))
    return merge_runs(out)


def merge_runs(runs):
    """Collapse overlapping runs of the same kind, keeping the widest."""
    runs = sorted(runs, key=lambda r: (r[0], r[1]))
    merged = []
    for r in runs:
        if merged and r[0] < merged[-1][1] and r[2] == merged[-1][2]:
            if r[1] > merged[-1][1]:
                merged[-1] = (merged[-1][0], r[1], r[2], r[3], r[4])
        else:
            merged.append(list(r) if False else r)
    return merged


# -- sub-matrix solver -----------------------------------------------------

def divisors(n: int, lo: int = 16, hi: int = 8192):
    return [d for d in range(lo, hi + 1) if n % d == 0]


def solve(total: int, channels, ffn_ratios, max_extra: int = 4096, n_sub: int = 4):
    """Find (sub-matrix products) + residue == total.

    A CrazyCuckoo fused Swin block holds: QKV (3C x C), attn proj (C x C),
    FFN expand (fC x C), FFN contract (C x fC). We allow the FFN ratio to vary
    and require the leftover 'extra' (gates + scales) to be a small positive
    number that is itself a plausible sum of 2*out fp16 gate blocks.
    """
    sols = []
    for C in channels:
        qkv = 3 * C * C
        proj = C * C
        for f in ffn_ratios:
            hid = int(C * f)
            if hid <= 0:
                continue
            wt = qkv + proj + 2 * hid * C
            extra = total - wt
            if 0 <= extra <= max_extra:
                sols.append(
                    {
                        "C": C,
                        "ffn_ratio": f,
                        "ffn_hidden": hid,
                        "weights": wt,
                        "extra": extra,
                        "parts": {
                            "qkv_3CxC": qkv,
                            "attnproj_CxC": proj,
                            "ffn_expand": hid * C,
                            "ffn_contract": hid * C,
                        },
                    }
                )
    return sols


def main() -> None:
    idx = load_index()
    mode = "--solve" in sys.argv

    if mode:
        channels = [64, 128, 256, 512, 1024]
        ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        for total, blocks in FAMILIES.items():
            print(f"\n=== {total} bytes  (blocks {blocks[0]}..{blocks[-1]}) ===")
            for s in solve(total, channels, ratios):
                print(f"  C={s['C']:<5} ffn={s['ffn_ratio']:<4} "
                      f"hidden={s['ffn_hidden']:<5} weights={s['weights']:<8} "
                      f"extra={s['extra']:<6} {s['parts']}")
        return

    # survey: tails + separators for one representative of each family
    for total, blocks in FAMILIES.items():
        b = blocks[0]
        name = f"block{b}.layer0.layer"
        data = read_tensor(idx, name)
        print(f"\n=== block{b}  ({total} bytes) ===")
        for ln in (64, 128, 256, 512, 1024):
            tail = data[-ln:]
            g = gate_score(tail)
            s = scale_score(tail)
            print(f"  tail {ln:5d}: fp16gate ok={g[0]} frac={g[1]:.2f} max={g[2]:.3f}"
                  f" | f32scale ok={s[0]} frac={s[1]:.2f} max={s[2]:.3f}")
        seps = find_separators(data, min_len=64, max_len=2048)
        print(f"  {len(seps)} candidate separator runs; widest 12:")
        for r in sorted(seps, key=lambda r: -(r[1] - r[0]))[:12]:
            print(f"    [{r[0]:8d}, {r[1]:8d}) len={r[1]-r[0]:5d} {r[2]}")


if __name__ == "__main__":
    main()
