"""Check the chunked ViT attention kernel for 128, 192 and 256 tokens.

The scalar reference is the upstream native attention contract (verified by
its author at 64/128/256 tokens) with kernel-order score sums; 192 tokens
extends the same chunk loop. Inputs are random FP8 Q/K/V.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ref/dlss5-port/Development'))
from native_c32_reference import H, F
from native_c32_softmax_sum import denominator


def reference(q, k, v):
    n = q.shape[0]
    q, k, v = [a.reshape(n, 32, 32).transpose(1, 0, 2) for a in (q, k, v)]
    dots = np.zeros((32, n, n), np.float32)
    for c in range(32):
        dots += q[:, :, c, None]*k[:, None, :, c]
    score = H(dots)
    coefficient = np.array([0x2dbb], np.uint16).view(np.float16).astype(np.float32)[0]
    affine = np.clip(H(score*coefficient+np.float32(1.708984375)), 1.439453125, 1.9775390625)
    b = affine.astype(np.float16).view(np.uint16).astype(np.uint32)
    exp = (((b << 4)+0x4000) & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
    key_order = np.zeros(64, np.int32)
    for bit, destination in enumerate([4, 0, 1, 3, 2, 5]):
        key_order |= ((np.arange(64) >> bit) & 1) << destination
    den = denominator(exp[..., :64][..., np.argsort(key_order)])
    for start in range(64, n, 64):
        den = H(den+denominator(exp[..., start:start+64][..., np.argsort(key_order)]))
    numerator = H(F(exp[..., :32]).astype(np.float64)@v[:, :32].astype(np.float64))
    for start in range(32, n, 32):
        numerator = H(numerator.astype(np.float64) +
                      F(exp[..., start:start+32]).astype(np.float64)@v[:, start:start+32].astype(np.float64))
    attention = F(H(numerator*H(1/den))).transpose(1, 0, 2).reshape(n, 1024)
    return score, exp, attention


def main():
    out = ROOT/'build/vit_attention_chunks_check'
    for tokens in (64, 128, 192, 256):
        rng = np.random.default_rng(1900+tokens)
        q, k, v = (F(rng.normal(0, .5, (tokens, 1024)).astype(np.float32)) for _ in range(3))
        score, exp, attention = reference(q, k, v)
        folder = out/str(tokens)
        folder.mkdir(parents=True, exist_ok=True)
        np.stack([q, k, v], 1).astype('<f4').tofile(folder/'qkv.f32')
        score.astype('<f4').tofile(folder/'scores.f32')
        exp.astype('<f4').tofile(folder/'exponents.f32')
        attention.astype('<f4').tofile(folder/'attention.f32')
        r = subprocess.run([str(ROOT/'build/vit_attention_chunks_test.exe'), str(folder), str(tokens)],
                           capture_output=True, text=True)
        print(tokens, 'tokens:', r.stdout.strip().replace('\n', ' | '))
        if r.returncode:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
