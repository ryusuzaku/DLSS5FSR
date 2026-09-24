#!/usr/bin/env python3
"""Continue a captured-input AMD candidate from encoder22 through encoder30.

The pinned public FP16 model supplies only same-image comparison boundaries.
Input to block23 is the verified candidate AMD block22 downsample device output.
This is an offline candidate check, not the original NVIDIA runtime.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_split512_bridge import run_case as run_bridge
from check_split512_peer_image import metrics
from check_split512_spatial_block import ROOT, run_case
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from extract_peer_split512_encoder_inputs import NODES
from native_split_reference import F


SHIFTS = (0, 3, 1, 2, 0, 3, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, encoder22_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    encoder22_dir = Path(encoder22_dir).resolve()
    output_root = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    source_file = encoder22_dir / 'report.json'
    source = json.loads(source_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    device_file = encoder22_dir / 'downsample22/output_device.f32'
    public22_file = encoder22_dir / 'public_boundary/block22_down_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            color_file.stat().st_size != 256 * 256 * 3 * 4 or
            prepared['color_linear_sha256'] != digest(color_file) or
            source['source_model_sha256'] != MODEL_SHA256 or
            source['source_capture_sha256'] != prepared['source_capture_sha256'] or
            source['prepared_manifest_sha256'] != digest(prepared_file) or
            source['color_linear_sha256'] != digest(color_file) or
            source['public_boundary_sha256']['block22_down'] != digest(public22_file) or
            source['downsample22']['output_device_sha256'] != digest(device_file) or
            not source['downsample22']['hip_scalar_exact'] or
            len(source['blocks']) != 4 or len(source['c128_blocks']) != 6 or
            len(source['c256_blocks']) != 8 or
            not all(stage['hip_scalar_exact'] for section in
                    ('blocks', 'c128_blocks', 'c256_blocks') for stage in source[section])):
        raise ValueError('prepared image, model or candidate encoder22 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    output_root.mkdir(parents=True, exist_ok=True)
    public_dir = output_root / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'encoder22_30.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch_file), ['rgb'], list(NODES.values()))
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 4, 4, 1024) if name == 'head30' else (1, 8, 8, 512)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if digest(public_dir / 'block22_peer.f32') != digest(public22_file):
        raise AssertionError('same-frame public block22 cross-extraction differs')
    p512 = peer_to_native_multihead(np.arange(512))
    p1024 = peer_to_native_multihead(np.arange(1024))
    native = np.fromfile(device_file, '<f4').reshape(64, 512)
    if not np.isfinite(native).all() or not np.array_equal(native, F(native)):
        raise ValueError('candidate block22 device tensor is not finite FP8')
    input_comparison = metrics(native[:, p512], public['block22'].reshape(64, 512))
    handoffs = []
    comparisons = {}
    last = None
    for block, shift in zip(range(23, 31), SHIFTS):
        input_sha = hashlib.sha256(native.astype('<f4').tobytes()).hexdigest()
        folder = run_case(block, 8, 8, shift, native, output_root)
        if digest(folder / 'input.f32') != input_sha:
            raise AssertionError(f'block{block} input handoff differs')
        if (folder / 'final_device.f32').read_bytes() != (folder / 'final.f32').read_bytes():
            raise AssertionError(f'block{block} HIP/scalar differs')
        native = np.fromfile(folder / 'final_device.f32', '<f4').reshape(64, 512)
        if not np.isfinite(native).all():
            raise ValueError(f'block{block} nonfinite device result')
        comparisons[f'block{block}'] = metrics(native[:, p512], public[f'block{block}'].reshape(64, 512))
        handoffs.append(dict(block=block, shift=shift, input_sha256=input_sha,
                             device_output_sha256=digest(folder / 'final_device.f32'),
                             hip_scalar_exact=True))
        last = folder
        m = comparisons[f'block{block}']
        print(f"block{block}: MAE {m['mae']:.7g}, corr {m['correlation']:.7g}", flush=True)
    bridge = run_bridge(8, 8, last / 'final_raw_device.f32', output_root / 'block30_head_4x4')
    head = np.fromfile(bridge / 'head_device.f32', '<f4').reshape(16, 1024)
    head_comparison = metrics(head[:, p1024], public['head30'].reshape(16, 1024))
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_file),
                  color_linear_sha256=digest(color_file),
                  source_model_sha256=MODEL_SHA256,
                  source_encoder22_report_sha256=digest(source_file),
                  source_encoder22_device_sha256=digest(device_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block22_public_exact=True,
                  input_vs_public_fp16=input_comparison,
                  comparisons=comparisons, handoffs=handoffs,
                  head30_vs_public_fp16=head_comparison,
                  head30_device_sha256=digest(bridge / 'head_device.f32'),
                  shifts=list(SHIFTS), hip_scalar_exact=True,
                  original_kernel_executed=False, full_candidate_inference=False,
                  production_wiring=False)
    (output_root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('encoder22_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_encoder512')
    args = parser.parse_args()
    run(args.prepared_dir, args.encoder22_dir, args.output_root)
