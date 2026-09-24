#!/usr/bin/env python3
"""Continue a captured-input candidate through ViT31-38 and decoder entry39."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

from audit_native_vit_logical_map import audit, logical_map
from audit_peer_native_c32_basis import peer_index, peer_to_native_multihead
from check_split512_peer_image import metrics
from check_vit_expand_chain import run_block
from extract_peer_decoder39_inputs import NODES, SHAPES
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_split_reference import bits
from recover_vit_bridge_ptx import ROOT

sys.path.insert(0, str(ROOT / 'ref/dlss5-port/Development'))
import native_decoder_entry_reference as D
from native_c32_reference import F, H


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, encoder30_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    encoder30_dir = Path(encoder30_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    source_file = encoder30_dir / 'report.json'
    source = json.loads(source_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    head_file = encoder30_dir / 'block30_head_4x4/head_device.f32'
    skip_file = encoder30_dir / 'block30-8x8-s2/final_device.f32'
    public30_file = encoder30_dir / 'public_boundary/block30_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            source['source_model_sha256'] != MODEL_SHA256 or
            source['source_capture_sha256'] != prepared['source_capture_sha256'] or
            source['prepared_manifest_sha256'] != digest(prepared_file) or
            source['color_linear_sha256'] != digest(color_file) or
            source['public_boundary_sha256']['block30'] != digest(public30_file) or
            source['head30_device_sha256'] != digest(head_file) or
            source['handoffs'][-1]['device_output_sha256'] != digest(skip_file) or
            not source['hip_scalar_exact'] or not source['same_frame_block22_public_exact'] or
            len(source['handoffs']) != 8 or
            not all(handoff['hip_scalar_exact'] for handoff in source['handoffs'])):
        raise ValueError('prepared image or candidate encoder30 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'vit38_decoder39.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch_file), ['rgb'], list(NODES.values()))
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        if array.shape != SHAPES[name] or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if digest(public_dir / 'skip30_peer.f32') != digest(public30_file):
        raise AssertionError('same-frame public encoder30 skip cross-extraction differs')
    bridge_audit, _ = audit(4, 4)
    if not all(bridge_audit[k] for k in
               ('cell_ok', 'bank_ok', 'repeat_ok', 'permutation_ok')):
        raise AssertionError('4x4 logical bridge audit differs')
    gather = logical_map(16)
    if not np.array_equal(np.sort(gather), np.arange(16 * 1024)):
        raise AssertionError('ViT logical gather is not bijective')
    map_dir = out / 'vit_map'
    bridge_dir = out / 'vit_bridge'
    map_dir.mkdir(exist_ok=True)
    bridge_dir.mkdir(exist_ok=True)
    gather.astype('<i4').tofile(map_dir / 'hwc-to-vit.i32')
    head = np.fromfile(head_file, '<f4')
    if head.size != len(gather) or not np.isfinite(head).all():
        raise ValueError('wrong/nonfinite candidate encoder head')
    (bridge_dir / 'input.f32').write_bytes(head_file.read_bytes())
    head[gather].astype('<f4').tofile(bridge_dir / 'expected.f32')
    subprocess.run([str(ROOT / 'build/vit_bridge_ptx_test.exe'), str(bridge_dir),
                    str(map_dir), str(len(gather))], cwd=ROOT, check=True)
    if (bridge_dir / 'device.f32').read_bytes() != (bridge_dir / 'expected.f32').read_bytes():
        raise AssertionError('candidate HIP ViT bridge differs')
    previous = bridge_dir / 'device.f32'
    vit_handoffs = []
    for block in range(31, 39):
        folder = run_block(block, 4, 4, derived=True, image_source=previous,
                           output_root=out / f'vit_block{block}')
        if (folder / 'input.f32').read_bytes() != previous.read_bytes():
            raise AssertionError(f'ViT block{block} handoff differs')
        current = folder / 'projection_device.f32'
        if current.read_bytes() != (folder / 'projection.f32').read_bytes():
            raise AssertionError(f'ViT block{block} HIP/scalar differs')
        vit_handoffs.append(dict(block=block, input_sha256=digest(previous),
                                 projection_device_sha256=digest(current),
                                 hip_scalar_exact=True))
        previous = current
        print(f'ViT block{block}: candidate HIP/scalar exact', flush=True)
    inverse = np.argsort(gather).astype('<i4')
    inverse_dir = out / 'inverse_map'
    inverse_bridge = out / 'inverse_bridge'
    inverse_dir.mkdir(exist_ok=True)
    inverse_bridge.mkdir(exist_ok=True)
    inverse.tofile(inverse_dir / 'hwc-to-vit.i32')
    vit = np.fromfile(previous, '<f4')
    (inverse_bridge / 'input.f32').write_bytes(previous.read_bytes())
    vit[inverse].astype('<f4').tofile(inverse_bridge / 'expected.f32')
    subprocess.run([str(ROOT / 'build/vit_bridge_ptx_test.exe'), str(inverse_bridge),
                    str(inverse_dir), str(len(inverse))], cwd=ROOT, check=True)
    if (inverse_bridge / 'device.f32').read_bytes() != (inverse_bridge / 'expected.f32').read_bytes():
        raise AssertionError('candidate HIP ViT inverse differs')
    p1024 = peer_to_native_multihead(np.arange(1024))
    p512 = peer_to_native_multihead(np.arange(512))
    main = np.fromfile(inverse_bridge / 'device.f32', '<f4').reshape(4, 4, 1024)
    vit38_comparison = metrics(main[..., p1024], public['vit38'])
    skip = np.fromfile(skip_file, '<f4').reshape(8, 8, 512)
    count = 512 * 1024
    rows = bits(count, [3, 6, 7, 8, 9, 10, 11, 12, 13])
    cols = bits(count, [1, 0, 4, 5, 2, 14, 15, 16, 17, 18])
    native_index = np.empty((512, 1024), np.int32)
    native_index[rows, cols] = np.arange(count)
    if (np.unique(rows * 1024 + cols).size != count or
            not np.array_equal(peer_index(512, 1024, 512),
                               native_index[np.ix_(p512, p1024)])):
        raise AssertionError('block39 projection index basis differs')
    records = {record['name']: record for record in
               json.loads((ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    index = records['block39.layer0.layer']['index']
    tensor = ROOT / 'dlss5-analysis/tensors' / f'tensor_{index:03d}.bin'
    weights, scale = D.unpack(tensor)
    projected = D.project(main, weights)
    merged_half = H(np.repeat(np.repeat(projected, 2, axis=0), 2, axis=1) + skip * scale)
    result = D.decoder_entry(main, skip, (weights, scale))
    np.testing.assert_array_equal(result, F(merged_half))
    dec = out / 'decoder39'
    dec.mkdir(exist_ok=True)
    for name, array in dict(vit=main, main=main, skip=skip, weights=weights,
                            scale=scale, projected=projected, output=result).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite candidate decoder39 {name}')
        np.asarray(array, '<f4').tofile(dec / f'{name}.f32')
    np.arange(main.size, dtype='<i4').tofile(dec / 'inverse.i32')
    check = subprocess.run([str(ROOT / 'build/decoder39_entry_test.exe'), str(dec), '4', '4'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    (dec / 'hip_check.log').write_text(check.stdout)
    if (check.returncode or check.stdout.count('PASS') != 3 or 'FAIL' in check.stdout or
            (dec / 'output_device.f32').read_bytes() != (dec / 'output.f32').read_bytes()):
        raise RuntimeError(f'candidate decoder39 HIP/scalar differs: {check.stdout[-2000:]}')
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_file),
                  color_linear_sha256=digest(color_file),
                  source_model_sha256=MODEL_SHA256,
                  source_encoder30_report_sha256=digest(source_file),
                  encoder_head_device_sha256=digest(head_file),
                  encoder_skip30_device_sha256=digest(skip_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_skip30_public_exact=True,
                  bridge_map_sha256=digest(map_dir / 'hwc-to-vit.i32'),
                  bridge_device_sha256=digest(bridge_dir / 'device.f32'),
                  vit_handoffs=vit_handoffs,
                  vit38_inverse_device_sha256=digest(inverse_bridge / 'device.f32'),
                  vit38_vs_public_fp16=vit38_comparison,
                  decoder39_tensor_sha256=digest(tensor),
                  decoder39_output_device_sha256=digest(dec / 'output_device.f32'),
                  decoder39_merge_vs_public_fp16=metrics(merged_half[..., p512], public['merge39']),
                  decoder39_output_vs_public_fp16=metrics(result[..., p512], public['block39']),
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_physical_bridge_validated=False,
                  original_16_token_reduction_validated=False,
                  full_candidate_inference=False, production_wiring=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('encoder30_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_vit39')
    args = parser.parse_args()
    run(args.prepared_dir, args.encoder30_dir, args.output_root)
