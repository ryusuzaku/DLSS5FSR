#!/usr/bin/env python3
"""Continue a captured-input candidate from decoder39 through C512 blocks40-47."""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_split512_peer_image import metrics
from check_split512_spatial_block import run_case
from cached_public_branch import extract_cached
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from extract_peer_split512_inputs import NODES


SHIFTS = (0, 3, 1, 2, 0, 3, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, vit39_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    vit39_dir = Path(vit39_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    source_file = vit39_dir / 'report.json'
    source = json.loads(source_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    device_file = vit39_dir / 'decoder39/output_device.f32'
    public39_file = vit39_dir / 'public_boundary/block39_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            source['source_model_sha256'] != MODEL_SHA256 or
            source['source_capture_sha256'] != prepared['source_capture_sha256'] or
            source['prepared_manifest_sha256'] != digest(prepared_file) or
            source['color_linear_sha256'] != digest(color_file) or
            source['public_boundary_sha256']['block39'] != digest(public39_file) or
            source['decoder39_output_device_sha256'] != digest(device_file) or
            not source['hip_scalar_exact'] or not source['same_frame_skip30_public_exact'] or
            len(source['vit_handoffs']) != 8 or
            not all(handoff['hip_scalar_exact'] for handoff in source['vit_handoffs'])):
        raise ValueError('prepared image or candidate decoder39 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'decoder39_47.onnx'
    extract_cached(MODEL, branch_file, ['rgb'], list(NODES.values()), MODEL_SHA256)
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        if array.shape != (1, 8, 8, 512) or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if digest(public_dir / 'block39_peer.f32') != digest(public39_file):
        raise AssertionError('same-frame public decoder39 cross-extraction differs')
    p512 = peer_to_native_multihead(np.arange(512))
    native = np.fromfile(device_file, '<f4').reshape(64, 512)
    if not np.isfinite(native).all():
        raise ValueError('candidate decoder39 device output nonfinite')
    input_comparison = metrics(native[:, p512], public['block39'].reshape(64, 512))
    handoffs = []
    comparisons = {}
    for block, shift in zip(range(40, 48), SHIFTS):
        input_sha = hashlib.sha256(native.astype('<f4').tobytes()).hexdigest()
        folder = run_case(block, 8, 8, shift, native, out)
        if (digest(folder / 'input.f32') != input_sha or
                (folder / 'final_device.f32').read_bytes() != (folder / 'final.f32').read_bytes()):
            raise AssertionError(f'block{block} input handoff or HIP/scalar differs')
        native = np.fromfile(folder / 'final_device.f32', '<f4').reshape(64, 512)
        if not np.isfinite(native).all():
            raise ValueError(f'block{block} nonfinite device result')
        comparisons[f'block{block}'] = metrics(native[:, p512], public[f'block{block}'].reshape(64, 512))
        handoffs.append(dict(block=block, shift=shift, input_sha256=input_sha,
                             device_output_sha256=digest(folder / 'final_device.f32'),
                             hip_scalar_exact=True))
        m = comparisons[f'block{block}']
        print(f"block{block}: MAE {m['mae']:.7g}, corr {m['correlation']:.7g}", flush=True)
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_file),
                  color_linear_sha256=digest(color_file),
                  source_model_sha256=MODEL_SHA256,
                  source_vit39_report_sha256=digest(source_file),
                  source_decoder39_device_sha256=digest(device_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block39_public_exact=True,
                  input_vs_public_fp16=input_comparison,
                  comparisons=comparisons, handoffs=handoffs,
                  final_device_sha256=handoffs[-1]['device_output_sha256'],
                  shifts=list(SHIFTS), hip_scalar_exact=True,
                  original_kernel_executed=False, full_candidate_inference=False,
                  production_wiring=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('vit39_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_decoder512')
    args = parser.parse_args()
    run(args.prepared_dir, args.vit39_dir, args.output_root)
