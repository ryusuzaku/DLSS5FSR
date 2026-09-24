#!/usr/bin/env python3
"""Compare two arms of the in-game A/B numerically (HANDOFF S204/S205/S223).

The A/B is judged on the model's own field, and the shim can write what it
produced per frame -- `DumpFrames=1`, `DumpEvery=n`, one `DumpDir` per arm
(ab_arm_A.bat / ab_arm_B.bat / ab_arm_C.bat). This turns the two piles of
screenshots into numbers, which is the half of the instrument that does not
depend on someone's eye.

Only the **out** frames are read. The in/out pair is written at two different
resolutions (the render subrect and the output subrect), so a pixel-wise
residual between them would be a measurement of the resampler, not of the
model.

  The decisive number is RATIO = |A - B| / |A - A'|: the mean absolute
  difference between the two arms at the SAME frame index, divided by the
  difference between two frames of the SAME arm at the same static spot. That
  second term is the session's own drift and noise floor -- if the flag's
  effect is no bigger than the drift, the flag is not the variable (S205's
  null-result branch, diagnosed rather than interpreted).

  rough  the share of the picture's energy above a 2-pixel scale. A field that
         is a function of the image is low-frequency by construction: the
         front-end reads 64 tokens on an 8x8 grid. Splotches are rough.

  Luma   0..1 mean, and the fraction of pixels pinned to 0 or 1 (saturation is
         what a wrong weight reading tends to produce).

Usage:
    python tools/ab_compare.py <dirA> <dirB> [--label-a A] [--label-b B] [--max N]

This is an instrument, not a check: nothing here is a tolerance, and it exits 0
unless it could not read anything at all.
"""

import argparse
import os
import struct
import sys

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def read_bmp(path):
    """24-bit BGR bottom-up BMP, as src/ngx/gpu.cpp writes them."""
    d = open(path, "rb").read()
    if d[:2] != b"BM":
        raise SystemExit(f"{path}: not a BMP")
    off, = struct.unpack_from("<I", d, 10)
    w, = struct.unpack_from("<i", d, 18)
    h, = struct.unpack_from("<i", d, 22)
    bits, = struct.unpack_from("<H", d, 28)
    if bits != 24:
        raise SystemExit(f"{path}: {bits}-bit, expected 24")
    bottom_up = h > 0
    h = abs(h)
    row = ((w * 3 + 3) // 4) * 4
    raw = np.frombuffer(d, dtype=np.uint8, count=row * h, offset=off)
    raw = raw.reshape(h, row)[:, : w * 3].reshape(h, w, 3)
    if bottom_up:
        raw = raw[::-1]
    return raw[:, :, ::-1].astype(np.float32) / 255.0     # BGR -> RGB


def box2(a):
    h, w = a.shape[0] & ~1, a.shape[1] & ~1
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def up2(a, h, w):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)[:h, :w]


def luma(img):
    return img @ LUMA


def stats(l):
    """(rough, mean_luma, clipped_fraction) for one frame's luma plane."""
    # Odd widths are common in real frames (991 px here), and box2 drops the
    # last row/column pair -- so crop first rather than let the subtraction
    # broadcast-fail on a one-column difference.
    h, w = l.shape[0] & ~1, l.shape[1] & ~1
    l = l[:h, :w]
    hi = l - up2(box2(l), h, w)
    spread = float(l.std())
    rough = float(np.abs(hi).mean() / spread) if spread > 1e-9 else float("nan")
    clipped = float(((l <= 0.002) | (l >= 0.998)).mean())
    return rough, float(l.mean()), clipped


def collect(d):
    """{frame index: luma array} for one arm's dump directory."""
    out = {}
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not (name.startswith("nr_") and name.endswith("_out.bmp")):
            continue
        idx = name[3:-8]
        if idx.isdigit():
            out[int(idx)] = luma(read_bmp(os.path.join(d, name)))
    return out


def mad(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return float(np.abs(a[:h, :w] - b[:h, :w]).mean())


def report(label, d, limit):
    fr = collect(d)
    if not fr:
        print(f"{label}: no nr_*_out.bmp in {d}")
        return {}
    idx = sorted(fr)[:limit]
    rows = {i: stats(fr[i]) for i in idx}
    r = np.array(list(rows.values()))
    print(f"{label}: {len(fr)} out frame(s), {len(idx)} read   frames {sorted(fr)[:4]}"
          f"{' ...' if len(fr) > 4 else ''}")
    if len(r):
        print("   rough %.4f   luma %.4f   clipped %.4f   (medians)"
              % tuple(np.nanmedian(r, axis=0)))
    return {i: fr[i] for i in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--max", type=int, default=12)
    a = ap.parse_args()

    fa = report(a.label_a, a.a, a.max)
    fb = report(a.label_b, a.b, a.max)
    if not fa or not fb:
        print("\n   Give each arm its own DumpDir (ab_arm_*.bat do) and let both "
              "runs reach the same frame numbers.")
        return 1

    common = sorted(set(fa) & set(fb))
    print()
    if common:
        cross = np.mean([mad(fa[i], fb[i]) for i in common])
        print("   same-index frames compared: %d" % len(common))
        print("   |%s - %s| at the same frame index : %.5f   <- the flag's effect"
              % (a.label_a, a.label_b, cross))
    for lbl, f in ((a.label_a, fa), (a.label_b, fb)):
        idx = sorted(f)
        if len(idx) > 1:
            within = np.mean([mad(f[idx[i]], f[idx[i + 1]])
                              for i in range(len(idx) - 1)])
            print("   |%s - %s'| across neighbouring frames : %.5f   <- the arm's own drift"
                  % (lbl, lbl, within))
    print()
    print("   If the cross-arm number is no larger than the drift, the flag is not")
    print("   the variable (S205). If it is much larger, read 'rough': a smooth")
    print("   field that tracks the scene beats a rough one that does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
