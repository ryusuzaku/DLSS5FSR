#!/usr/bin/env python3
"""Extract pinned same-image public encoder8–14 boundaries and C128 bodies."""
from pathlib import Path
import json
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest


OUT = ROOT / 'build/peer_encoder128_inputs'
TEMP = Path.home() / 'DLSS5FSR-build-offload' / 'onnx-extraction'
NODES = {'block8_down': '/graph/down64_128/downsample/Cast_19_output_0',
         **{f'block{block}': f'/graph/enc128.{block-9}/Cast_61_output_0'
            for block in range(9, 14)},
         'block14_skip': '/graph/down128_256/Cast_9_output_0',
         'block14_down': '/graph/down128_256/downsample/Cast_19_output_0'}


def run():
    if digest(MODEL) != MODEL_SHA256:
        raise ValueError('public ONNX model hash differs')
    prior = ROOT / 'build/peer_coherent_head_inputs'
    old = json.loads((prior / 'manifest.json').read_text())
    if old['model_sha256'] != MODEL_SHA256 or old['input_image_sha256'] != digest(IMAGE):
        raise ValueError('prior public image/model differs')
    color = prior / 'color_linear.f32'
    if digest(color) != old['color_linear_sha256']:
        raise ValueError('public color hash differs')
    rgb = np.fromfile(color, '<f4').reshape(256, 256, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
        branch = Path(tmp) / 'encoder128_inputs.onnx'
        onnx.utils.extract_model(str(MODEL), str(branch), ['rgb'], list(NODES.values()))
        branch_hash = digest(branch)
        arrays = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider']).run(
            None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    rounded = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 16, 16, 256) if name == 'block14_down' else (1, 32, 32, 128)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{name} shape/nonfinite: {array.shape}')
        rounded[name] = bool(np.array_equal(array, array.astype('<f2').astype(np.float32)))
        array[0].astype('<f4').tofile(OUT / f'{name}_peer.f32')
    cross_checks = {
        'block14_down': ROOT / 'build/peer_encoder256_inputs/block14_down_peer.f32',
        'block14_skip': ROOT / 'build/peer_decoder56_inputs/skip14_peer.f32',
    }
    for name, earlier in cross_checks.items():
        if digest(OUT / f'{name}_peer.f32') != digest(earlier):
            raise AssertionError(f'{name} differs from independent extraction')
    result = dict(model_sha256=MODEL_SHA256, image_sha256=digest(IMAGE),
                  nodes=NODES, temporary_branch_sha256=branch_hash,
                  tensor_sha256={name: digest(OUT / f'{name}_peer.f32') for name in NODES},
                  half_rounded=rounded, same_inference_call=True,
                  independent_block14_cross_checks_exact=True,
                  original_kernel_executed=False)
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run()
