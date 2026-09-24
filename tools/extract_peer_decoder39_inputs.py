#!/usr/bin/env python3
"""Extract public FP16 ViT38, split-encoder30 skip, and decoder39 boundaries."""
from pathlib import Path
import json
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest


OUT = ROOT / 'build/peer_decoder39_inputs'
TEMP = Path.home() / 'DLSS5FSR-build-offload' / 'onnx-extraction'
NODES = {
    'vit38': '/graph/vit.7/Cast_37_output_0',
    'skip30': '/graph/split_enc.7/Cast_57_output_0',
    'merge39': '/graph/dec_input/Add_output_0',
    'block39': '/graph/dec_input/Cast_10_output_0',
}
SHAPES = {'vit38': (1, 4, 4, 1024),
          'skip30': (1, 8, 8, 512),
          'merge39': (1, 8, 8, 512),
          'block39': (1, 8, 8, 512)}


def run():
    if digest(MODEL) != MODEL_SHA256:
        raise ValueError('public model hash differs')
    prior = ROOT / 'build/peer_coherent_head_inputs'
    old = json.loads((prior / 'manifest.json').read_text())
    if old['model_sha256'] != MODEL_SHA256 or old['input_image_sha256'] != digest(IMAGE):
        raise ValueError('previous inference model/image differs')
    color = prior / 'color_linear.f32'
    if digest(color) != old['color_linear_sha256']:
        raise ValueError('color source hash differs')
    rgb = np.fromfile(color, '<f4').reshape(256, 256, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
        branch = Path(tmp) / 'decoder39_inputs.onnx'
        onnx.utils.extract_model(str(MODEL), str(branch), ['rgb'], list(NODES.values()))
        branch_hash = digest(branch)
        session = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider'])
        arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    for (name, _), array in zip(NODES.items(), arrays):
        if array.shape != SHAPES[name] or not np.isfinite(array).all():
            raise ValueError(f'{name} shape/nonfinite: {array.shape}')
        if name != 'merge39' and not np.array_equal(array, array.astype('<f2').astype(np.float32)):
            raise ValueError(f'{name} is not half-rounded')
        array[0].astype('<f4').tofile(OUT / f'{name}_peer.f32')
    parent = ROOT / 'build/peer_split512_inputs'
    earlier = json.loads((parent / 'manifest.json').read_text())
    if digest(OUT / 'block39_peer.f32') != earlier['tensor_sha256']['block39']:
        raise AssertionError('new block39 differs from prior branch')
    report = dict(model_sha256=MODEL_SHA256, image_sha256=digest(IMAGE),
                  temporary_branch_sha256=branch_hash, nodes=NODES,
                  shapes={name: list(shape) for name, shape in SHAPES.items()},
                  tensor_sha256={name: digest(OUT / f'{name}_peer.f32') for name in NODES},
                  prior_block39_exact=True, same_inference_call=True,
                  model_variant='optimized AMD FP16', original_kernel_executed=False)
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
