#!/usr/bin/env python3
"""16-token ViT attention candidate; no original 16-token oracle exists here."""
import numpy as np

from native_c32_reference import F, H


def reference(q, k, v):
    if any(a.shape != (16, 1024) or a.dtype != np.float32 for a in (q, k, v)):
        raise ValueError('expected three float32 arrays of 16 x 1024')
    qb, kb, vb = (a.reshape(16, 32, 32).transpose(1, 0, 2) for a in (q, k, v))
    scores = H(qb @ kb.transpose(0, 2, 1))
    coefficient = np.array([0x2dbb], np.uint16).view(np.float16).astype(np.float32)[0]
    affine = np.clip(H(scores * coefficient + np.float32(1.708984375)),
                     1.439453125, 1.9775390625)
    bits = affine.astype(np.float16).view(np.uint16).astype(np.uint32)
    exponents = (((bits << 4) + 0x4000) & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
    denominator = H(exponents.astype(np.float64).sum(axis=-1, keepdims=True))
    numerator = H(F(exponents).astype(np.float64) @ vb.astype(np.float64))
    output = F(H(numerator * H(1 / denominator))).transpose(1, 0, 2).reshape(16, 1024)
    if not all(np.isfinite(a).all() for a in (scores, exponents, output)):
        raise ValueError('nonfinite 16-token candidate attention')
    return scores, exponents, output
