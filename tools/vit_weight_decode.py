"""Decode candidate ViT records without materializing full address-index arrays.

This implements the same bit coordinates as the vendored native ViT reference;
it does not establish an independent physical-layout or original-kernel oracle.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref/dlss5-port/Development'))
from decode_tinlayout_global import e4m3fn
from unpack_vit_matrices import axis_permutation, MATRIX_OUTPUT_TO_RAW


# Preserve every reference code, including signed zero and SATFINITE 0x7f/ff.
_FP8 = e4m3fn(np.arange(256, dtype=np.uint8))
_FP8.flags.writeable = False
_SKIP_ORDER = axis_permutation(1024, MATRIX_OUTPUT_TO_RAW)
_SKIP_ORDER.flags.writeable = False


def _logical_codes(raw, inputs, outputs):
    if (inputs, outputs) not in ((1024, 4096), (4096, 1024), (1024, 1024)):
        raise ValueError('unsupported ViT matrix shape')
    if raw.ndim != 1 or raw.size != inputs * outputs:
        raise ValueError('wrong ViT matrix size')
    ib, ob = inputs.bit_length() - 1, outputs.bit_length() - 1
    output_bits = [6, 3, 9, 7, 8] + list(range(10, ob + 5))
    input_bits = [0, 1, 2, 4, 5] + list(range(ob + 5, ib + ob))
    # A C-order binary reshape exposes raw bits MSB first. Put output bits
    # first (also MSB first), then input bits, to obtain [output, input].
    order = list(reversed(output_bits)) + list(reversed(input_bits))
    axes = [ib + ob - 1 - bit for bit in order]
    return raw.reshape((2,) * (ib + ob)).transpose(axes).reshape(outputs, inputs)


def matrix(raw, inputs, outputs):
    if raw.dtype != np.uint8:
        raise ValueError('ViT packed matrix must contain uint8 codes')
    return _FP8[_logical_codes(raw, inputs, outputs)]


def unpack_residual(raw, inputs):
    if inputs not in (1024, 4096) or len(raw) != inputs * 1024 + 2048:
        raise ValueError('wrong ViT residual record size')
    count = inputs * 1024
    weights = matrix(np.frombuffer(raw, np.uint8, count=count), inputs, 1024)
    skip = np.frombuffer(raw, '<f2', offset=count).astype(np.float32)[_SKIP_ORDER]
    return weights, skip


def unpack_qkv(raw):
    if len(raw) != 3145856:
        raise ValueError('wrong ViT QKV record size')
    scales = np.frombuffer(raw, '<f4', count=32).copy()
    if not np.isfinite(scales).all():
        raise ValueError('nonfinite ViT QKV scales')
    packed = np.frombuffer(raw, np.uint8, offset=128).reshape(-1, 3, 1024)
    matrices = [matrix(packed[:, part, :].ravel(), 1024, 1024) for part in range(3)]
    return matrices, scales
