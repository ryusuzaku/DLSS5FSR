"""Candidate C32 front end: stem, block0, pool, encoder blocks 1-4, block4 downsample.

Weights come from the original tensors: block0 (tensor_000) is a C32 body with
a 16x32 FP16 input stem inserted at 0x2010; blocks 1-3 are plain C32 bodies;
block4 (tensor_091) carries its C32->C64 downsample at 0x50B0. The C32 bodies
use the public QMMA layout (as decoder66-69); the stem and downsample layouts
were recovered by exact comparison with the public graph's coefficients.
The noise texture is the public graph's constant. Everything runs in the
public (peer) channel basis and is converted to the resident chain's native
basis only at its outputs, with the resident FP8/FP16 conventions. This is a
candidate, not an original-kernel result.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import numpy_helper

from check_block66_peer_candidate import decode, save
from check_c256_ffn_candidate import multiply
from compare_peer_weight_layouts import peer_qmma
from head70_normalized_reference import trace, packed
from native_c32_reference import F, H
from audit_peer_native_c32_basis import peer_to_native, peer_to_native_multihead
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from cached_public_branch import digest

ROOT = Path(__file__).resolve().parents[1]
TENSORS = {0: 0, 1: 1, 2: 12, 3: 44, 4: 91}
SHIFTS = {0: 0, 1: 0, 2: 3, 3: 0, 4: 3}
STEM_AT, DOWN_AT = 0x2010, 0x50B0


def tensor(block):
    path = ROOT / 'dlss5-analysis/tensors' / f'tensor_{TENSORS[block]:03d}.bin'
    return np.fromfile(path, np.uint8), path


def stem_weights(raw):
    """16x32 FP16 stem (row 15 padding); address bits r0 r3 c3 r1 r2 c0 c1 c2 c4."""
    halves = raw[STEM_AT:STEM_AT+1024].view('<u2')
    r = np.arange(16)[:, None]
    c = np.arange(32)[None, :]
    index = ((r & 1) | ((r >> 3 & 1) << 1) | ((c >> 3 & 1) << 2) | ((r >> 1 & 1) << 3) |
             ((r >> 2 & 1) << 4) | ((c & 1) << 5) | ((c >> 1 & 1) << 6) | ((c >> 2 & 1) << 7) |
             ((c >> 4 & 1) << 8))
    return halves[index].view(np.float16).astype(np.float32)[:15]


def load_weights():
    raw0, _ = tensor(0)
    if raw0.size != 21696:
        raise ValueError('unexpected block0 record size')
    bodies = {0: decode(np.concatenate([raw0[:STEM_AT], raw0[STEM_AT+1024:]]))}
    for block in (1, 2, 3, 4):
        raw, _ = tensor(block)
        if raw.size != (22720 if block == 4 else 20672):
            raise ValueError(f'unexpected block{block} record size')
        bodies[block] = decode(raw[:20672])
    raw4, _ = tensor(4)
    matrix = peer_qmma(raw4, DOWN_AT, 64, 32)
    model = onnx.load(str(MODEL))
    public = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer
              if i.name in ('noise', '/MatMul_output_0_half_b', '/graph/down32_64/downsample/conv_weight/MatMul_output_0_half_b')}
    stem = stem_weights(raw0)
    # Recovered layouts must reproduce the public coefficients exactly.
    if (not np.array_equal(stem, public['/MatMul_output_0_half_b'].astype(np.float32)) or
            not np.array_equal(matrix, public['/graph/down32_64/downsample/conv_weight/MatMul_output_0_half_b'].astype(np.float32).T)):
        raise AssertionError('stem/downsample layout no longer matches the public coefficients')
    noise = public['noise'][0].transpose(1, 2, 0).astype(np.float32)
    return dict(stem=stem, bodies=bodies, matrix=matrix, noise=noise)


def windows(image, shift):
    h, w, c = image.shape
    px = 4 if shift & 1 else 0
    py = 4 if shift & 2 else 0
    ww, hh = ((w+px+7)//8)*8, ((h+py+7)//8)*8
    padded = np.pad(image, ((py, hh-h-py), (px, ww-w-px), (0, 0)))
    return padded.reshape(hh//8, 8, ww//8, 8, c).transpose(0, 2, 1, 3, 4).reshape(-1, 64, c)


def image(windowed, h, w, shift):
    c = windowed.shape[-1]
    px = 4 if shift & 1 else 0
    py = 4 if shift & 2 else 0
    ww, hh = ((w+px+7)//8)*8, ((h+py+7)//8)*8
    return windowed.reshape(hh//8, ww//8, 8, 8, c).transpose(0, 2, 1, 3, 4).reshape(hh, ww, c)[py:py+h, px:px+w]


def pool(raw):
    return F(H(H(H(raw[::2, ::2]+raw[::2, 1::2])+H(raw[1::2, ::2]+raw[1::2, 1::2]))*np.float32(.25)))


def noise_for(noise, width, height):
    """Public 256x256 noise, tiled for wider synthetic extents (assumption)."""
    reps = (-(-height//256), -(-width//256), 1)
    return np.tile(noise, reps)[:height, :width]


def run(rgb, weights, out=None):
    """Return every stage; with out, write the harness fixtures."""
    height, width, _ = rgb.shape
    x = H((rgb-np.float32(.5))*np.float32(.125))
    noise = noise_for(weights['noise'], width, height)
    feature = np.concatenate([noise, np.ones((height, width, 1), np.float32), x, x,
                              np.zeros((height, width, 5), np.float32)], -1)
    stem = multiply(H(feature), weights['stem'].T)
    stages = {}
    tokens = windows(stem, 0)
    t0 = trace(tokens, weights['bodies'][0])
    raw0 = image(t0['body'], height, width, 0)
    skip0 = np.clip(raw0, -448, 448)
    current = pool(raw0)
    stages[0] = dict(input=tokens, trace=t0, raw=raw0)
    pre_down = current
    for block in (1, 2, 3, 4):
        tokens = windows(current, SHIFTS[block])
        t = trace(tokens, weights['bodies'][block])
        h, w = current.shape[:2]
        raw = image(t['body'], h, w, SHIFTS[block])
        stages[block] = dict(input=tokens, trace=t, raw=raw, image=F(raw))
        current = F(raw)
    down_pool = pool(stages[4]['raw'])
    down = F(multiply(down_pool.reshape(-1, 32), weights['matrix'])).reshape(height//4, width//4, 64)
    p32, p64 = peer_to_native(np.arange(32)), peer_to_native_multihead(np.arange(64))
    outputs = dict(c64_input=np.empty_like(down), skip4=np.empty_like(current), skip0=np.empty_like(skip0))
    outputs['c64_input'][..., p64] = down
    outputs['skip4'][..., p32] = current
    outputs['skip0'][..., p32] = skip0
    result = dict(stem=stem, pre_down=pre_down, down_pool=down_pool, down=down, stages=stages, **outputs)
    if out is not None:
        write(out, rgb, weights, result)
    return result


def write(out, rgb, weights, r):
    out = Path(out)
    height, width, _ = rgb.shape
    save(out, dict(rgb=rgb, noise=noise_for(weights['noise'], width, height), stem_weights=weights['stem'],
                   stem=r['stem'], pre_down=r['pre_down'], down_pool=r['down_pool'], down=r['down'],
                   c64_input=r['c64_input'], skip4=r['skip4'], skip0=r['skip0'],
                   matrix=weights['matrix']))
    for block, stage in r['stages'].items():
        folder = out / f'block{block}'
        body = dict(input=stage['input'], weights=packed(weights['bodies'][block]), **stage['trace'],
                    output=F(stage['trace']['body']))
        save(folder, body)
        save(folder, dict(raw=stage['raw'], **({'image': stage['image']} if 'image' in stage else {})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared_dir', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    rgb = np.fromfile(a.prepared_dir / 'color_linear.f32', '<f4').reshape(256, 256, 3)
    result = run(rgb, load_weights(), a.output_root)
    print(json.dumps({k: list(v.shape) for k, v in result.items() if hasattr(v, 'shape')}, indent=2))
