#!/usr/bin/env python3
"""Extract block 70's four weight pieces from the fused tensor (S227).

`tensor_150` (21808 B) is not a new container: it is a C=32 stage tensor (20672 B)
with a 128-byte block inserted at 8272 and a 1024-byte coefficient tail appended.
The reference port's `Development/native_post70_reference.py` unpacks it as

    ordinary[:0x2050] = raw[:0x2050]              # 0 .. 8272
    ordinary[0x2060:] = raw[0x20d0:0x5130]        # 8288 .. 20672  <- 8400 .. 20784
    sm[order] = raw[0x2050:0x2090].view(f2)       # 8272 .. 8336,   32 f16
    ss[order] = raw[0x2090:0x20d0].view(f2)       # 8336 .. 8400,   32 f16
    head[bits(512,[2,5,6,7]), bits(512,[0,1,3,4,8])] = raw[0x5130:].view(f2)

so: the body's weights are the 20672-byte stage-shaped part (the pieces S4 already
pins -- A at 0, gate1 at 8208, B at 8288, gate2 at 20592), `sm`/`ss` are the two
32-channel merge scales, and `head` is a [16][32] matrix of latent->RGB
coefficients of which three rows are used. The arithmetic closes exactly:
20672 - 16 + 128 + 1024 = 21808.

This tool extracts all four, applies the two order conventions, and **verifies
itself** three ways, because a mis-sliced extractor looks exactly like a working
one until the kernel is written against it:

  1. the sizes and the 1136-byte identity above;
  2. a round trip -- repacking the four pieces must reproduce the input byte for
     byte, including the stage's 16-byte constant pad, whose value is read from a
     real stage tensor rather than assumed (the head's tensor does not contain it:
     the scales occupy that slot);
  3. the regional dtype profile -- the body part must show e4m3 in A and qkv
     (no 0x7F/0xFF) and f16-like bytes in the bias region, the signature that
     distinguishes a stage tensor from everything else (S226).

Usage:
    python tools/head70_weights.py [--tensor <bin>] [--stage <bin>] [--out <dir>]
Writes body.bin / sm.bin / ss.bin / coeff.bin into --out (default: beside the
input, suffixed `.head70`). Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

TENSOR_BYTES = 21808
STAGE_BYTES = 20672
INSERT_AT = 0x2050          # 8272 -- where a stage has its 16-byte constant pad
INSERT_LEN = 0x80           # 128  -- sm (64 B) + ss (64 B) replace it
TAIL_AT = 0x5130            # 20784
PAD_AT = INSERT_AT          # the stage's pad, 16 bytes, at the same offset

# from native_post70_reference.py, verbatim
ORDER = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                  16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31])
COEFF_ROWS = (2, 5, 6, 7)
COEFF_COLS = (0, 1, 3, 4, 8)


def bits(count: int, positions) -> np.ndarray:
    """The reference port's index map: out bit b takes input bit positions[b]."""
    i = np.arange(count, dtype=np.int64)
    out = np.zeros(count, dtype=np.int64)
    for b, p in enumerate(positions):
        out |= ((i >> p) & 1) << b
    return out


def extract(raw: bytes, pad: bytes):
    if len(raw) != TENSOR_BYTES:
        raise SystemExit("tensor is %d B, expected %d" % (len(raw), TENSOR_BYTES))
    body = bytearray(STAGE_BYTES)
    body[:INSERT_AT] = raw[:INSERT_AT]
    body[INSERT_AT + 16:] = raw[INSERT_AT + INSERT_LEN:TAIL_AT]
    sm_file = np.frombuffer(raw[INSERT_AT:INSERT_AT + 64], dtype="<f2")
    ss_file = np.frombuffer(raw[INSERT_AT + 64:INSERT_AT + INSERT_LEN], dtype="<f2")
    sm = np.empty(32, np.float32); sm[ORDER] = sm_file
    ss = np.empty(32, np.float32); ss[ORDER] = ss_file
    coeff = np.empty((16, 32), np.float32)
    coeff[bits(512, COEFF_ROWS), bits(512, COEFF_COLS)] = \
        np.frombuffer(raw[TAIL_AT:], dtype="<f2")
    return bytes(body), sm, ss, coeff, pad


def repack(body: bytes, sm, ss, coeff, pad: bytes) -> bytes:
    """The inverse of extract(), so the round trip is a real check.

    Note what is NOT here: the stage's 16-byte pad. The head's tensor does not
    contain it -- `sm` occupies the slot it would sit in -- so the raw is
    [0,8272) + sm + ss + the stage's [8288, 20672) + the coefficients, and the pad
    exists only in the reconstructed stage-shaped body. `pad` is asserted zero
    there; a non-zero pad would mean this layout is not the one in S227.
    """
    if pad != b"\0" * 16:
        raise SystemExit("stage pad is not zero (%s) -- layout assumption broken" % pad.hex())
    out = bytearray(TENSOR_BYTES)
    out[:INSERT_AT] = body[:INSERT_AT]
    # forward is `sm[ORDER] = file`, so the inverse GATHERS at the same indices.
    smf = sm[ORDER]
    ssf = ss[ORDER]
    out[INSERT_AT:INSERT_AT + 64] = smf.astype("<f2").tobytes()
    out[INSERT_AT + 64:INSERT_AT + INSERT_LEN] = ssf.astype("<f2").tobytes()
    out[INSERT_AT + INSERT_LEN:TAIL_AT] = body[INSERT_AT + 16:]
    rows, cols = bits(512, COEFF_ROWS), bits(512, COEFF_COLS)
    # `coeff[rows, cols] = file` is a scatter FROM file order, so the inverse is a
    # plain gather at the same index array -- the tail's order IS the gather order.
    out[TAIL_AT:] = coeff[rows, cols].astype("<f2").tobytes()
    return bytes(out)


def profile(seg: bytes, label: str) -> bool:
    """e4m3 regions carry no 0x7F/0xFF; f16 ones do. Returns True if as expected."""
    x = np.frombuffer(seg, dtype=np.uint8)
    nan = int((x == 0x7F).sum()) + int((x == 0xFF).sum())
    print("   %-22s %6d B   0x7F+0xFF = %4d   distinct %3d" % (label, len(x), nan, len(np.unique(x))))
    return nan


def main() -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tens = os.path.join(here, "dlss5-analysis", "tensors")
    ap.add_argument("--tensor", default=os.path.join(tens, "tensor_150.bin"))
    ap.add_argument("--stage", default=os.path.join(tens, "tensor_001.bin"),
                    help="a real C=32 stage tensor, for the pad value and the profile")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    raw = open(a.tensor, "rb").read()
    stage = open(a.stage, "rb").read()
    pad = stage[PAD_AT:PAD_AT + 16]
    print("tensor %s (%d B)   stage %s (%d B)" % (os.path.basename(a.tensor), len(raw),
                                                  os.path.basename(a.stage), len(stage)))
    print("the pad the head replaces: %d B at %d = %s" % (len(pad), PAD_AT, pad.hex()))

    body, sm, ss, coeff, pad = extract(raw, pad)

    print("\nchecks")
    ok = True
    ident = STAGE_BYTES - 16 + INSERT_LEN + 1024
    print("   arithmetic: (%d - 16) + %d + 1024 = %d   tensor %d   %s"
          % (STAGE_BYTES, INSERT_LEN, ident, TENSOR_BYTES,
             "OK" if ident == TENSOR_BYTES else "FAIL"))
    ok = ok and ident == TENSOR_BYTES

    back = repack(body, sm, ss, coeff, pad)
    rt = back == raw
    print("   round trip: repacked == input   %s" % ("OK" if rt else "FAIL"))
    ok = ok and rt

    print("   regional profile of the body (e4m3 regions should be 0, the bias not):")
    a_region = profile(body[0:8192], "A (FFN w1+w2)")
    qkv = profile(body[8288:11360], "B (qkv)")
    bias = profile(body[11360:19552], "B (bias)")
    print("   body vs stage on the bias region: %d vs %d (both must be non-zero)"
          % (bias, profile(stage[11360:19552], "stage bias, reference")))
    prof_ok = a_region == 0 and qkv == 0 and bias > 0
    print("   profile %s" % ("OK" if prof_ok else "FAIL"))
    ok = ok and prof_ok

    out = a.out or (a.tensor + ".head70")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "body.bin"), "wb").write(body)
    sm.astype("<f4").tofile(os.path.join(out, "sm.bin"))
    ss.astype("<f4").tofile(os.path.join(out, "ss.bin"))
    coeff.astype("<f4").tofile(os.path.join(out, "coeff.bin"))
    print("\nwrote %s/{body.bin,sm.bin,ss.bin,coeff.bin}" % out)
    print("sm  range %.5f .. %.5f    ss range %.5f .. %.5f" % (sm.min(), sm.max(), ss.min(), ss.max()))
    print("coeff rows used (0,2,4 of the 16):  %s" % np.round(coeff[[0, 2, 4], :4], 5).tolist())
    print("\n%s" % ("all checks passed" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
