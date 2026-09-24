#!/usr/bin/env python3
"""Run public block-4 entry and candidate AMD encoder blocks 5-8 on a capture.

The public optimized FP16 ONNX graph supplies only the upstream block-4
boundary. Blocks 5-8 run through the existing HIP/scalar candidate checks.
This is offline, not a live game network or an original-kernel oracle.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_block62_candidate import ROOT, run as run_block
from check_split512_peer_image import metrics
from extract_peer_encoder64_inputs import NODES
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_split_reference import F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, output_root, last_block=8):
    prepared_dir = Path(prepared_dir).resolve()
    output_root = Path(output_root).resolve()
    if last_block not in range(5, 9):
        raise ValueError('last block must be 5..8')
    prepared_path = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_path.read_text())
    color_path = prepared_dir / 'color_linear.f32'
    if (prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_path) or
            color_path.stat().st_size != 256 * 256 * 3 * 4 or
            digest(MODEL) != MODEL_SHA256):
        raise ValueError('prepared input or pinned public model differs')
    rgb = np.fromfile(color_path, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB has nonfinite values')
    output_root.mkdir(parents=True, exist_ok=True)
    public_dir = output_root / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_path = public_dir / 'encoder4_8.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch_path), ['rgb'], list(NODES.values()))
    session = ort.InferenceSession(str(branch_path), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 32, 32, 128) if name == 'block8_down' else (1, 64, 64, 64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    p64 = peer_to_native_multihead(np.arange(64))
    native = np.empty_like(public['block4_down'])
    native[..., p64] = public['block4_down']
    native = F(native)
    boundary = output_root / 'boundary4'
    boundary.mkdir(exist_ok=True)
    previous = boundary / 'output_device.f32'
    native.astype('<f4').tofile(previous)
    native.astype('<f4').tofile(boundary / 'output.f32')
    blocks = []
    for block in range(5, last_block + 1):
        input_sha = digest(previous)
        result = run_block(block=block, previous=previous, width=64, height=64,
                           output_root=output_root / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'candidate block{block} handoff differs')
        answer = np.fromfile(result, '<f4').reshape(64, 64, 64)
        reference_name = 'block8_skip' if block == 8 else f'block{block}'
        comparison = metrics(answer[..., p64], public[reference_name])
        blocks.append(dict(block=block, input_device_sha256=input_sha,
                           output_device_sha256=digest(result),
                           candidate_vs_public_fp16=comparison,
                           hip_scalar_exact=True))
        previous = result
        print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}",
              flush=True)
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_path),
                  color_linear_sha256=digest(color_path),
                  source_model_sha256=MODEL_SHA256,
                  public_branch_sha256=digest(branch_path),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  candidate_boundary4_sha256=digest(boundary / 'output_device.f32'),
                  blocks=blocks, final_device_sha256=digest(previous),
                  original_kernel_executed=False, full_candidate_inference=False,
                  production_wiring=False)
    (output_root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_encoder64')
    parser.add_argument('--last-block', type=int, default=8)
    args = parser.parse_args()
    run(args.prepared_dir, args.output_root, args.last_block)
