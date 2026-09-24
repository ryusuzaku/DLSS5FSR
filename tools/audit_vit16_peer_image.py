#!/usr/bin/env python3
"""Compare each connected 16-token AMD ViT block with the public FP16 model."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from audit_native_vit_logical_map import logical_map
from audit_peer_native_c32_basis import peer_to_native_multihead
from check_split512_peer_image import metrics
from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest


VIT31 = Path.home() / 'DLSS5FSR-build-offload' / 'peer_vit31_candidate16'
CHAIN = Path.home() / 'DLSS5FSR-build-offload' / 'peer_vit16_candidate'
TEMP = Path.home() / 'DLSS5FSR-build-offload' / 'onnx-extraction'
OUT = ROOT / 'build/vit16_peer_audit'
NODES = {block: f'/graph/vit.{block-31}/Cast_37_output_0' for block in range(31, 39)}


def run(vit31=VIT31, chain_root=CHAIN, temp=TEMP, output=OUT):
    if digest(MODEL) != MODEL_SHA256:
        raise ValueError('public model hash differs')
    prior = ROOT / 'build/peer_coherent_head_inputs'
    old = json.loads((prior / 'manifest.json').read_text())
    if old['model_sha256'] != MODEL_SHA256 or old['input_image_sha256'] != digest(IMAGE):
        raise ValueError('public inference model/image differs')
    color = prior / 'color_linear.f32'
    if digest(color) != old['color_linear_sha256']:
        raise ValueError('public RGB source hash differs')
    first = json.loads((vit31 / 'report.json').read_text())
    chain = json.loads((chain_root / 'report.json').read_text())
    for record in (first, chain):
        if record['source_model_sha256'] != MODEL_SHA256 or record['source_image_sha256'] != digest(IMAGE):
            raise ValueError('candidate ViT model/image differs')
    rgb = np.fromfile(color, '<f4').reshape(256, 256, 3)
    temp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp) as tmp:
        branch = Path(tmp) / 'vit16_stages.onnx'
        onnx.utils.extract_model(str(MODEL), str(branch), ['rgb'], list(NODES.values()))
        branch_hash = digest(branch)
        session = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider'])
        arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    inverse = np.argsort(logical_map(16))
    p1024 = peer_to_native_multihead(np.arange(1024))
    comparisons = {}
    public38 = ROOT / 'build/peer_decoder39_inputs/vit38_peer.f32'
    for (block, _), array in zip(NODES.items(), arrays):
        if array.shape != (1, 4, 4, 1024) or not np.isfinite(array).all():
            raise ValueError(f'public ViT{block} shape/nonfinite: {array.shape}')
        path = (vit31 / 'vit31_prefix/projection_device.f32' if block == 31 else
                chain_root / f'vit_block{block}/projection_device.f32')
        expected = (first['vit31_projection_device_sha256'] if block == 31 else
                    chain['handoffs'][block-32]['projection_device_sha256'])
        if digest(path) != expected:
            raise ValueError(f'ViT{block} device hash differs')
        native = np.fromfile(path, '<f4').reshape(16*1024)[inverse].reshape(16, 1024)
        comparisons[f'vit{block}'] = metrics(native[:, p1024], array.reshape(16, 1024))
    if digest(public38) != hashlib.sha256(arrays[-1][0].astype('<f4').tobytes()).hexdigest():
        raise AssertionError('public ViT38 differs from prior extraction')
    result = dict(model_sha256=MODEL_SHA256, image_sha256=digest(IMAGE),
                  temporary_branch_sha256=branch_hash, nodes=NODES,
                  comparisons=comparisons,
                  basis='candidate logical inverse and P1024; original physical map unverified',
                  attention='experimental 16-valid-key reduction; not original oracle',
                  original_kernel_executed=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    for block, item in comparisons.items():
        print(f"{block}: MAE {item['mae']:.7g}, corr {item['correlation']:.7g}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vit31-root', type=Path, default=VIT31)
    parser.add_argument('--chain-root', type=Path, default=CHAIN)
    parser.add_argument('--temp-root', type=Path, default=TEMP)
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.vit31_root, args.chain_root, args.temp_root, args.output_root)
