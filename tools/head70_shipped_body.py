"""Explicit shipped-mode2 C32 body candidate for the head (S242).

This is the live fused recipe, NOT native_c32_reference.block: no QK norm,
sequential half softmax sum, FP8 residual, and float32 final projection.
Keep this separate from head70_reference.body_forward. Its purpose is a
device/oracle diagnostic of reusing the shipped body on tensor_150.
"""
from contextlib import contextmanager
import struct
import numpy as np
import swin1h_ref as R
from wload_map import mode2_dense


def fnv(raw):
    h = 2166136261
    for b in raw:
        h = ((h ^ b) * 16777619) & 0xffffffff
    return h


def rewrite(body):
    if len(body) != 20672:
        raise ValueError("head body must be a 20672-byte stage")
    return mode2_dense(body, 32, 128, 32)


@contextmanager
def dimensions():
    settings = dict(C64_C=32, C64_W=32, C64_K=32, C64_H1P=128,
                    C64_HEADS=1, C64_TOK=64, C64_DIM=32)
    previous = {k: getattr(R, k) for k in settings}
    try:
        for k, v in settings.items():
            setattr(R, k, v)
        yield
    finally:
        for k, v in previous.items():
            setattr(R, k, v)


def weights(body):
    raw = rewrite(body)
    fp8 = lambda b: np.asarray([R.e4m3_decode(v) for v in b], np.float32)
    half = lambda off, n: np.frombuffer(raw, '<f2', n, off).astype(np.float32)
    return dict(w1=fp8(raw[:4096]).reshape(128, 32).T[None].tolist(),
                w2=fp8(raw[4096:8192]).reshape(32, 128).T[None].tolist(),
                g1=half(8208, 32).tolist(),
                qkv=fp8(raw[8288:11360]).reshape(1, 1, 32, 96).tolist(),
                bias=half(11360, 4096).reshape(1, 64, 64).tolist(),
                temp=[struct.unpack_from('<f', raw, 19552)[0]],
                proj=fp8(raw[19568:20592]).reshape(32, 32).T.tolist(),
                g2=half(20592, 32).tolist())


def quant(x):
    a = np.asarray(x, np.float32)
    b = R.f32_list_to_e4m3(a.ravel().tolist())
    return np.asarray([R.e4m3_decode(v) for v in b], np.float32).reshape(a.shape)


def forward(tiles, body):
    """Return stage traces, with a window dimension on every array."""
    if tiles.ndim != 3 or tiles.shape[1:] != (64, 32):
        raise ValueError("expected [window][64][32]")
    w = weights(body)
    trace = {k: [] for k in ('ffn', 'qkv', 'scores', 'prob', 'context', 'body')}
    with dimensions():
        for tile in tiles:
            x = quant(tile).tolist()
            yf = R.c64_ffn2_forward(x, w['w1'], w['w2'], w['g1'])
            yfe = quant(yf).tolist()
            qkv = R.c64_qkv_forward(yfe, w['qkv'])
            q, k, v = [quant(np.asarray(qkv)[..., i:i+32]).tolist()
                       for i in (0, 32, 64)]
            scores = R.c64_scores_forward(q, k, w['temp'], w['bias'])
            _, prob = R.c64_exp_forward(scores)
            ctx = R.c64_ctx_forward(quant(prob).tolist(), v)
            out = R.c64_proj_forward(quant(ctx)[0].tolist(), w['proj'], w['g2'], yfe)
            for name, value in zip(trace, (yf, qkv[0], scores[0], prob[0], ctx[0], out)):
                trace[name].append(value)
    return {k: np.asarray(v, np.float32) for k, v in trace.items()}
