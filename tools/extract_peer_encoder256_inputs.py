#!/usr/bin/env python3
"""Extract the same-image public C256 encoder and block22 downsample."""
from pathlib import Path
import json
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest


OUT = ROOT / 'build/peer_encoder256_inputs'
TEMP = Path.home() / 'DLSS5FSR-build-offload' / 'onnx-extraction'
NODES = {'block14_down': '/graph/down128_256/downsample/Cast_19_output_0',
         **{f'block{block}': f'/graph/enc256.{block-15}/Cast_61_output_0'
            for block in range(15, 22)},
         'block22_skip': '/graph/down256_512/Cast_9_output_0',
         'block22_down': '/graph/down256_512/downsample/Cast_19_output_0'}


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
        branch = Path(tmp) / 'encoder256_inputs.onnx'
        onnx.utils.extract_model(str(MODEL), str(branch), ['rgb'], list(NODES.values()))
        branch_hash = digest(branch)
        session = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider'])
        arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    rounded = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 8, 8, 512) if name == 'block22_down' else (1, 16, 16, 256)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{name} shape/nonfinite: {array.shape}')
        rounded[name] = bool(np.array_equal(array, array.astype('<f2').astype(np.float32)))
        array[0].astype('<f4').tofile(OUT / f'{name}_peer.f32')
    previous = ROOT / 'build/peer_split512_encoder_inputs'
    old_encoder = json.loads((previous / 'manifest.json').read_text())
    if (old_encoder['model_sha256'] != MODEL_SHA256 or
            old_encoder['image_sha256'] != digest(IMAGE) or
            digest(OUT / 'block22_down_peer.f32') != old_encoder['tensor_sha256']['block22']):
        raise AssertionError('block22 downsample differs from prior encoder extraction')
    skip = ROOT / 'build/peer_decoder48_inputs'
    old_skip = json.loads((skip / 'manifest.json').read_text())
    if digest(OUT / 'block22_skip_peer.f32') != old_skip['tensor_sha256']['skip22']:
        raise AssertionError('block22 skip differs from prior decoder extraction')
    result = dict(model_sha256=MODEL_SHA256, image_sha256=digest(IMAGE),
                  nodes=NODES, temporary_branch_sha256=branch_hash,
                  tensor_sha256={name: digest(OUT / f'{name}_peer.f32') for name in NODES},
                  half_rounded=rounded, same_inference_call=True,
                  prior_block22_down_exact=True, prior_block22_skip_exact=True,
                  original_kernel_executed=False)
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run()
