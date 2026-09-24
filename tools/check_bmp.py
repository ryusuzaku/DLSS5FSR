#!/usr/bin/env python3
"""Check a shim dump against the analytic value it should hold.

The harness verifies pixels through its own readback. This checks the shim's
BMP writer instead, which is a separate path: it copies on the caller's
command list, unpacks R10G10B10A2 and BGRA, and writes bottom-up.

    python tools/check_bmp.py build/nr_0000_out.bmp [--mode proxy|frame]

  proxy  the DebugView=1 picture: the frame after the encode's knee, decoded
         back to frame units. Identical to the frame below the knee, short of
         the rolled-off highlights above it.
  frame  the plain resampled frame, which is what every strength-0
         configuration must produce.

Exits non-zero if anything is more than a couple of counts out.
"""

import argparse
import math
import struct
import sys

SRC_W, SRC_H = 320, 180
DST_W, DST_H = 1280, 720

LUMA = (0.2126, 0.7152, 0.0722)
KNEE = 0.75


def srgb_to_linear(v):
    v = min(max(v, 0.0), 1.0)
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def linear_to_srgb(v):
    v = min(max(v, 0.0), 1.0)
    return v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055


def read_bmp(path):
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
    out = bytearray(w * h * 3)
    for y in range(h):
        src = off + y * row
        iy = (h - 1 - y) if bottom_up else y
        out[iy * w * 3:(iy + 1) * w * 3] = d[src:src + w * 3]
    return w, h, out


def at(px, w, x, y, c):
    """Channel c (0=R) at image position (x,y). BMP rows are stored BGR."""
    return px[(y * w + x) * 3 + (2 - c)]


def source_at(x, y, w, h):
    """The harness's test pattern, 0..1, at a pixel of a w x h image.

    At the source size this is the pattern itself; at the output size it is
    the analytic bilinear of the 4x resample.
    """
    sx = max((x + 0.5) * SRC_W / w - 0.5, 0.0)
    sy = max((y + 0.5) * SRC_H / h - 0.5, 0.0)
    # B is a hard step at the halfway column, so it is only analytic well
    # away from the transition.
    return [sx / (SRC_W - 1), sy / (SRC_H - 1), 0.0 if x < w / 2 else 1.0]


def expected(x, y, mode, w, h):
    v = source_at(x, y, w, h)
    if mode == "proxy":
        luma = sum(LUMA[c] * v[c] for c in range(3))
        if luma > KNEE:
            rolled = KNEE + 0.25 * (1.0 - math.exp(-(luma - KNEE) / 0.25))
            v = [c * rolled / luma for c in v]
        # The debug view decodes what the encode produced and rescales it.
        v = [srgb_to_linear(linear_to_srgb(c)) for c in v]
    return [c * 255.0 for c in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", choices=("proxy", "frame"), default="proxy")
    ap.add_argument("--tol", type=int, default=3)
    a = ap.parse_args()

    w, h, px = read_bmp(a.path)
    print(f"{a.path}  {w}x{h}  mode={a.mode}")
    if (w, h) not in ((DST_W, DST_H), (SRC_W, SRC_H)):
        print(f"  note: neither {DST_W}x{DST_H} nor {SRC_W}x{SRC_H}")

    for label, x, y in (("TL", 0, 0), ("TR", w - 1, 0),
                        ("BL", 0, h - 1), ("BR", w - 1, h - 1),
                        ("mid", w // 2, h // 2)):
        print(f"  {label:<4} = {tuple(at(px, w, x, y, c) for c in range(3))}")

    # The blue step is not analytic across its transition, which at the source
    # is one texel wide and at the output is one source texel's worth.
    guard = 2 if w == SRC_W else 24

    worst = 0
    worst_at = None
    n = 0
    for y in range(1, h, max(1, h // 14)):
        for x in range(1, w, max(1, w // 24)):
            if abs(x - w // 2) <= guard:
                continue
            e = expected(x, y, a.mode, w, h)
            for c in range(3):
                d = abs(at(px, w, x, y, c) - round(e[c]))
                if d > worst:
                    worst, worst_at = d, (x, y, "RGB"[c], at(px, w, x, y, c),
                                          round(e[c], 1))
            n += 1

    print(f"  {n} samples, worst error vs analytic {a.mode}: {worst}")
    if worst_at:
        print(f"    at x={worst_at[0]} y={worst_at[1]} {worst_at[2]}: "
              f"got {worst_at[3]}, want {worst_at[4]}")
    if worst > a.tol:
        print("  FAIL")
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
