#!/usr/bin/env python3
"""The head (block 70) as a reference implementation (S237).

Spec: the reference port's `Development/native_post70_reference.py`, functions
`post()` and `aligned()`. Their pipeline, in their order:

    merged   = H(H(upsample2(main) * sm) + skip * ss)          # the merge pass
    tiles    = merged -> 8x8 windows of 64x32                  # window order
    features = block(tiles, body, raw_output=True)             # THE C=32 BODY
    value    = aligned(features[:, :16], head[:, :16], 0)
    value    = aligned(features[:, 16:], head[:, 16:], value)  # accumulate
    base     = color * 0.125 - 0.0625
    out      = clip((value * input_scale + base) * 8 + 0.5, 0, 1)

`main` is the model latent at half resolution, `skip` is a 32-channel feature at
full resolution, `color` is 3-channel RGB. `sm`/`ss`/`head` and the body's
weights all come out of `tensor_150` (see tools/head70_weights.py, S230).

BODY LIMITATION (S241). This uses the preblock primitives through
`carve(CARVE_FUSED32)`, as `swin1h_ref.cmd_realweights()` does. It is NOT the
live fused C=32 body's reading: that uses mode-2 FFN and k-major QKV, whereas
this path uses dense FFN and out-major QKV. Upstream block() also unpacks
attention/gates through recovered maps absent here. The body is a hypothesis;
the outer-pass device tests do not establish its equivalence to either body.

WHAT THIS IS AND IS NOT. It is a port of their *reference*, so a later device
comparison tests our kernel against our port -- it does not by itself establish
that the port is the model's function. `aligned()` is transcribed faithfully
because it models a hardware accumulate (a fixed-point grid from the largest
exponent, 27-bit, truncated); a plain fp32 sum would be a different function and
would fail comparison for the wrong reason.

Usage:
    python tools/head70_reference.py                 # self-test on synthetic input
    python tools/head70_reference.py --tensor <bin>  # self-test with another tensor
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swin1h_ref as R          # noqa: E402  (the oracle home)
import head70_weights as HW     # noqa: E402  (S230's verified extractor)

# Their H and F, verbatim in meaning: an f16 round trip and an e4m3 quantisation.
H = lambda x: np.asarray(x, np.float16).astype(np.float32)          # noqa: E731


def F(x):
    """e4m3 quantise, matching their `F = e4m3fn(quantize(x))`."""
    flat = R.f32_list_to_e4m3(np.asarray(x, np.float32).ravel().tolist())
    return np.array([R.e4m3_decode(b) for b in flat], np.float32).reshape(
        np.asarray(x).shape)


def aligned(a, b, acc, acc_exponent_offset=1, truncate_acc=False):
    """Their `aligned()`, transcribed: a hardware-style accumulate.

    The products are summed on a fixed-point grid set by the largest exponent in
    the accumulation (`exp2(e - 27)`, i.e. 27 bits), with optional accumulator
    truncation. A plain fp32 sum is a different function.
    """
    result = np.empty_like(acc)
    for start in range(0, len(a), 4096):
        x = a[start:start + 4096]
        product = x[:, None, :].astype(np.float64) * b[None, :, :].astype(np.float64)
        e = (np.frexp(np.abs(x))[1][:, None, :]
             + np.frexp(np.abs(b))[1][None, :, :])
        e = np.max(np.where(product != 0, e, -1000), axis=-1)
        initial = acc[start:start + 4096].astype(np.float64)
        if acc_exponent_offset is not None:
            e = np.maximum(e, np.where(initial != 0,
                                       np.frexp(np.abs(initial))[1]
                                       + acc_exponent_offset, -1000))
        e = np.where(e == -1000, 0, e)
        quantum = np.exp2(e - 27)
        if truncate_acc:
            initial = np.trunc(initial / quantum) * quantum
        result[start:start + 4096] = H(
            np.trunc(product / quantum[..., None]).sum(-1) * quantum + initial)
    return result


def merge(main, skip, sm, ss):
    """H(H(upsample2(main) * sm) + skip * ss) -- nearest-neighbour 2x, as theirs."""
    up = np.repeat(np.repeat(main, 2, 0), 2, 1)
    return H(H(up * sm) + skip * ss)


def windowise(merged):
    """Their reshape: 8x8 windows, each read as 64 tokens of 32 channels."""
    hh, ww = merged.shape[:2]
    if hh % 8 or ww % 8:
        raise ValueError("post70 needs a multiple of 8: %dx%d" % (hh, ww))
    return merged.reshape(hh // 8, 8, ww // 8, 8, 32).transpose(
        0, 2, 1, 3, 4).reshape(-1, 64, 32)


def dewindowise(features, hh, ww):
    h8, w8 = hh // 8, ww // 8
    return features.reshape(h8, w8, 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(hh, ww, 32)


def body_forward(tiles, W, qk_scale):
    """One C=32 preblock over [nwin][64][32] tokens -- the same call sequence
    cmd_realweights() runs on a real fused tensor, in S21's FFN-first order.

    The slices use swin1h_ref's MODULE constants (C=32, TOK=64, QKV=96, H1=128 --
    the preblock regime that ffn_forward/qkv_forward themselves use), not the
    C64_* ones, which describe the encoder's C=64 blocks.
    """
    C, H1, QKV = R.C, R.H1, R.QKV
    Wq = [W["qkv"][i * C:(i + 1) * C] for i in range(QKV)]
    Wp = [W["proj"][i * C:(i + 1) * C] for i in range(C)]
    W1 = [W["ffn1"][i * C:(i + 1) * C] for i in range(H1)]
    W2 = [W["ffn2"][i * H1:(i + 1) * H1] for i in range(C)]
    out = []
    for win in tiles:
        Xb = [[float(v) for v in row] for row in win]
        Yf, _, _ = R.ffn_forward(Xb, W1, W2, W["gate_ffn"])
        Q, K, V, _ = R.qkv_forward(Yf, Wq)
        Qn, Kn, _, _, _, _ = R.qknorm_forward(Q, K, qk_scale)
        O, S, _, _ = R.attn_forward(Qn, Kn, V, bias=W["bias"])
        Yp, _ = R.proj_forward(O, Wp, Yf, W["gate_attn"])
        out.append(Yp)
    return np.array(out, np.float32)


def post(main, skip, color, params, W, qk_scale, input_scale=0.03125):
    """Their post(), with the body supplied by our oracle."""
    body, sm, ss, coeff = params
    h, w, c = skip.shape
    if c != 32 or main.shape != (h // 2, w // 2, 32) or color.shape != (h, w, 3):
        raise ValueError("post70 shape")
    if h % 8 or w % 8:
        raise ValueError("post70 needs multiples of 8")
    merged = merge(main, skip, sm, ss)
    tiles = windowise(merged)
    feats = body_forward(tiles, W, qk_scale)
    hh, ww = merged.shape[:2]
    features = dewindowise(feats, hh, ww).reshape(-1, 32)
    return finish(features.reshape(h, w, 32), color, coeff, input_scale)


def finish(features, color, coeff, input_scale=0.03125):
    """Logical HWC body features -> RGB; independently device-tested outer pass."""
    h, w, c = features.shape
    if c != 32 or color.shape != (h, w, 3) or coeff.shape != (3, 32):
        raise ValueError("post70 finish shape")
    features = features.reshape(-1, 32)
    value = aligned(features[:, :16], coeff[:, :16], np.zeros((h * w, 3), np.float32))
    value = aligned(features[:, 16:], coeff[:, 16:], value)
    base = np.float32(color.reshape(-1, 3).astype(np.float64) * .125 - .0625)
    encoded = np.float32(value.astype(np.float64) * np.float32(input_scale)
                         + base.astype(np.float64))
    return np.clip(np.float32(encoded.astype(np.float64) * 8 + .5),
                   0, 1).reshape(h, w, 3)


def load(tensor, stage):
    """The head's four pieces (S230) plus the body's weights carved from it.

    `coeff` comes back as all 16 rows; their `unpack()` returns `head[[0,2,4]]`,
    i.e. only the three rows the rgb pass uses, so select them here -- passing 16
    rows would broadcast the accumulator wrongly rather than fail loudly.
    """
    raw = open(tensor, "rb").read()
    pad = open(stage, "rb").read()[HW.PAD_AT:HW.PAD_AT + 16]
    body, sm, ss, coeff, _ = HW.extract(raw, pad)
    coeff = coeff[[0, 2, 4]]
    tmp = tensor + ".head70-body.bin"
    open(tmp, "wb").write(body)
    W = R.carve(tmp, R.CARVE_FUSED32)
    os.remove(tmp)
    qk_scale = struct.unpack("<f", body[19552:19556])[0]
    return (body, sm, ss, coeff), W, qk_scale


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default=os.path.join(
        here, "dlss5-analysis", "tensors", "tensor_150.bin"))
    ap.add_argument("--stage", default=os.path.join(
        here, "dlss5-analysis", "tensors", "tensor_001.bin"))
    ap.add_argument("--crosscheck", action="store_true",
                    help="also compare the body's wiring against load_pre()'s path")
    a = ap.parse_args()
    crosscheck = a.crosscheck

    params, W, qk = load(a.tensor, a.stage)
    body, sm, ss, coeff = params
    print("body %d B   sm/ss 32+32   coeff %s   QK s = %.4f"
          % (len(body), coeff.shape, qk))

    # Synthetic input: a deterministic latent, a skip field and a colour field.
    st = 0x5EED1234
    def rnd(n, lo, hi):
        nonlocal st
        out = []
        for _ in range(n):
            st = (st * 1103515245 + 12345) & 0xFFFFFFFF
            out.append(lo + (hi - lo) * ((st >> 8) / 16777216.0))
        return np.array(out, np.float32)
    h, w = 16, 16
    main_in = rnd((h // 2) * (w // 2) * 32, -1.0, 1.0).reshape(h // 2, w // 2, 32)
    skip = rnd(h * w * 32, -1.0, 1.0).reshape(h, w, 32)
    color = rnd(h * w * 3, 0.0, 1.0).reshape(h, w, 3)

    out = post(main_in, skip, color, params, W, qk)
    merged = merge(main_in, skip, sm, ss)
    tiles = windowise(merged)
    feats = dewindowise(body_forward(tiles, W, qk), h, w)

    print("merge  range %.4f .. %.4f   finite %s"
          % (merged.min(), merged.max(), bool(np.isfinite(merged).all())))
    print("body   32-ch maxabs %.4f   finite %s"
          % (np.abs(feats).max(), bool(np.isfinite(feats).all())))
    ok = True
    checks = [
        ("output shape is h x w x 3", out.shape == (h, w, 3)),
        ("output finite", bool(np.isfinite(out).all())),
        ("output within [0,1] (the clip)", bool(out.min() >= 0 and out.max() <= 1)),
        ("merge finite", bool(np.isfinite(merged).all())),
        ("body finite", bool(np.isfinite(feats).all())),
    ]
    # Determinism, and that the input actually matters (a constant output would
    # pass every check above and mean nothing).
    again = post(main_in, skip, color, params, W, qk)
    checks.append(("deterministic", bool(np.array_equal(out, again))))
    other = post(main_in * -1.0, skip, color, params, W, qk)
    checks.append(("output moves when the latent does",
                   float(np.abs(out - other).max()) > 1e-3))
    # The body's wiring, cross-checked against the path the b2/b3 checks use.
    # run_pre_block()'s five stages are the same sequence body_forward() calls, but
    # it consumes PRE-SLICED lists built by load_pre(), while body_forward() slices
    # from a carve dict itself. Both sides here use CARVE_PRE on tensor_000, so what
    # this verifies is precisely: the same keys and the same slicing indices produce
    # the same numbers as the established path. It is a CONVENTION check -- it does
    # not validate the R.* primitives (device-checked elsewhere) and it is NOT a
    # two-table check, because the table is shared.
    if crosscheck:
        try:
            import emit_yp_golden as E
            P = E.load_pre()
            W0 = R.carve(os.path.join(os.path.dirname(os.path.abspath(E.__file__)),
                                      "..", "dlss5-analysis", "tensors",
                                      "tensor_000.bin"), R.CARVE_PRE)
            tok = [[float(v) for v in row] for row in tiles[0]]
            A = None
            Yf, _, _ = R.ffn_forward(tok, P["W1"], P["W2"], P["gate_ffn"])
            Q, K, V, _ = R.qkv_forward(Yf, P["Wq"])
            Qn, Kn, _, _, _, _ = R.qknorm_forward(Q, K, P["s_real"])
            O, _, _, _ = R.attn_forward(Qn, Kn, V, bias=P["bias"])
            A, _ = R.proj_forward(O, P["Wp"], Yf, P["gate_attn"])
            B = body_forward(tiles[:1], W0, P["s_real"])[0]
            same = np.array_equal(np.array(A, np.float32), np.array(B, np.float32))
            print("   %-40s %s" % ("body wiring agrees with load_pre()'s path"
                                   if same else "body wiring DIFFERS from load_pre()", "OK" if same else "FAIL"))
            ok = ok and same
        except FileNotFoundError as e:
            # Only a missing input may skip. Any other failure is a failure --
            # a swallowed exception that still prints OK is a false green, which
            # is the one thing this project works hardest to avoid.
            print("   %-40s SKIP (no input: %s)"
                  % ("body wiring cross-check", e))
        except Exception as e:                      # noqa: BLE001
            print("   %-40s FAIL (%s: %s)"
                  % ("body wiring cross-check", type(e).__name__, e))
            ok = False

    for name, good in checks:
        print("   %-40s %s" % (name, "OK" if good else "FAIL"))
        ok = ok and good
    print("\noutput mean %.4f  min %.4f  max %.4f  nonzero-bytes %d"
          % (out.mean(), out.min(), out.max(), int((out > 0).sum())))
    print("%s" % ("all checks passed" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
