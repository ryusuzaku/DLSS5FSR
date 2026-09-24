"""
scan_separators.py - find the coefficient blocks inside a fused weight tensor.

Every sub-matrix in this model is followed by a short run of coefficients that
decodes cleanly as either

  * fp16 gates   -- 2 * out_channels bytes, all values in (0, 1]
  * f32 scales   -- a small positive block (32 values on the ViT QKV,
                    16 values on the Swin FFN)

Weight data is fp8 and reinterprets as noise, so a run of consecutive
in-range fp16 / f32 values is a separator: it marks the end of one sub-matrix
and the start of the next.

Usage:
    python scan_separators.py [--all] [--min-run 16]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "dlss5-analysis"
TENSORS = ANA / "tensors"

# block -> (expected channels, family size)
STAGES = {
    1: (32, 20672),
    5: (64, 61760),
    9: (128, 197184),
    15: (256, 689232),
}


def load_index() -> dict[str, dict]:
    m = json.loads((ANA / "model.resolved.json").read_text(encoding="utf-8"))
    return {t["name"]: t for t in m["tensors"]}


def read_tensor(idx: dict, name: str) -> np.ndarray:
    i = idx[name]["index"]
    return np.frombuffer((TENSORS / f"tensor_{i:03d}.bin").read_bytes(), dtype=np.uint8)


def runs(mask: np.ndarray, min_run: int):
    """Yield (start, length) for each True-run of at least min_run."""
    if not mask.any():
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for s, e in zip(idx[0::2], idx[1::2]):
        if e - s >= min_run:
            yield int(s), int(e - s)


def scan(data: np.ndarray, min_run: int = 16):
    n = len(data)
    out = []

    # fp16 at every byte offset (parity matters: the block may not be 2-aligned)
    for off in range(0, 2):
        body = data[off : n - ((n - off) % 2)]
        v = body.view(np.float16).astype(np.float32)
        ok = np.isfinite(v) & (v > 0) & (v <= 1.0)
        for s, ln in runs(ok, min_run):
            seg = v[s : s + ln]
            out.append(("fp16gate", off + s * 2, ln * 2, float(seg.min()),
                        float(seg.max()), float(seg.mean())))

    # f32 at every 4-byte phase
    for off in range(0, 4):
        body = data[off : n - ((n - off) % 4)]
        v = body.view(np.float32)
        ok = np.isfinite(v) & (v > 0) & (v < 1e4) & (np.abs(v) > 1e-6)
        for s, ln in runs(ok, min_run):
            seg = v[s : s + ln]
            out.append(("f32scale", off + s * 4, ln * 4, float(seg.min()),
                        float(seg.max()), float(seg.mean())))

    return sorted(out, key=lambda r: r[1])


def main() -> None:
    args = sys.argv[1:]
    min_run = 16
    if "--min-run" in args:
        min_run = int(args[args.index("--min-run") + 1])

    idx = load_index()
    for block, (C, size) in STAGES.items():
        name = f"block{block}.layer0.layer"
        data = read_tensor(idx, name)
        assert len(data) == size, (len(data), size)
        print(f"\n=== block{block}  C={C}  ({size} B) ===")
        res = scan(data, min_run)
        # drop fp16 runs that are subsumed by a longer f32 run and vice versa
        if not res:
            print("  no separator runs found")
            continue
        print(f"  {len(res)} runs >= {min_run} values")
        for kind, off, nbytes, lo, hi, mean in res[:25]:
            end = off + nbytes
            print(f"    {kind:9s} [{off:>8}, {end:>8})  {nbytes:>6} B  "
                  f"vals[{lo:.4g}, {hi:.4g}] mean {mean:.4g}   "
                  f"tail-gap {size - end}")
        print(f"  ... tensor ends at {size}; last run ends at "
              f"{max(r[1] + r[2] for r in res)}")


if __name__ == "__main__":
    main()
