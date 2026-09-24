#!/usr/bin/env python3
"""Continue one captured-input candidate through decoder blocks66-69.

The encoder4 decoder skip remains a same-image public FP16/FP8 control.
"""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import onnx
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native
from check_block66_peer_candidate import ROOT, run as run_block
from check_upsample66_peer_image import metrics
from decode_tinlayout_global import e4m3fn
from extract_peer_decoder66_inputs import (
    BLOCK65, SKIP4, MERGE66, BLOCK66, BLOCK67, BLOCK68, BLOCK69)
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_split_reference import bits
from native_c64_reference import multiply
from native_c32_reference import H, F


NODES = dict(block65=BLOCK65, skip4=SKIP4, merge66=MERGE66,
             block66=BLOCK66, block67=BLOCK67, block68=BLOCK68, block69=BLOCK69)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, decoder65_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    decoder65_dir = Path(decoder65_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    source_file = decoder65_dir / 'report.json'
    source = json.loads(source_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    block65_file = decoder65_dir / 'block65/output/output_device.f32'
    public65_file = decoder65_dir / 'public_boundary/block65_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            source['source_model_sha256'] != MODEL_SHA256 or
            source['source_capture_sha256'] != prepared['source_capture_sha256'] or
            source['prepared_manifest_sha256'] != digest(prepared_file) or
            source['public_boundary_sha256']['block65'] != digest(public65_file) or
            source['final_device_sha256'] != digest(block65_file) or
            not source['hip_scalar_exact'] or len(source['blocks']) != 4 or
            not all(block['hip_scalar_exact'] for block in source['blocks'])):
        raise ValueError('prepared image or candidate decoder65 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'decoder65_69.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch_file), ['rgb'], list(NODES.values()))
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 64, 64, 64) if name == 'block65' else (1, 128, 128, 32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if digest(public_dir / 'block65_peer.f32') != digest(public65_file):
        raise AssertionError('same-frame public block65 cross-extraction differs')
    p32 = peer_to_native(np.arange(32))
    tensor = ROOT / 'dlss5-analysis/tensors/tensor_144.bin'
    raw = np.fromfile(tensor, np.uint8)
    if raw.size != 22784:
        raise ValueError('wrong block66 tensor size')
    rows = bits(2048, [3, 6, 7, 8, 9])
    cols = bits(2048, [1, 0, 4, 5, 2, 10])
    if np.unique(rows * 64 + cols).size != 2048:
        raise AssertionError('block66 projection map collision')
    matrix = np.empty((32, 64), np.float32)
    matrix[rows, cols] = e4m3fn(raw[0x2000:0x2800])
    order = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                      16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31])
    c = np.arange(32)
    other = (c // 16) * 16 + (c % 8) * 2 + (c % 16 // 8)
    weights = np.empty_like(matrix)
    weights[order] = matrix[other]
    scale = np.empty(32, np.float32)
    scale[order] = raw[0x2860:0x28a0].view('<f2').astype(np.float32)
    x = np.fromfile(block65_file, '<f4').reshape(64, 64, 64)
    skip_peer = F(public['skip4'])
    skip = np.empty_like(skip_peer)
    skip[..., p32] = skip_peer
    low = multiply(x, weights)
    merged = H(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1) + skip * scale)
    prefix = out / 'block66_prefix'
    prefix.mkdir(exist_ok=True)
    for name, array in dict(input=x, weights=weights, scale=scale, skip=skip,
                            low=low, merged=merged).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite candidate block66 {name}')
        np.asarray(array, '<f4').tofile(prefix / f'{name}.f32')
    check = subprocess.run([str(ROOT / 'build/upsample66_prefix_test.exe'), str(prefix), '64', '64'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    (prefix / 'hip_check.log').write_text(check.stdout)
    if (check.returncode or check.stdout.count('PASS') != 2 or 'FAIL' in check.stdout or
            (prefix / 'merged_device.f32').read_bytes() != (prefix / 'merged.f32').read_bytes()):
        raise RuntimeError(f'candidate block66 prefix HIP/scalar differs: {check.stdout[-2000:]}')
    previous = prefix / 'merged_device.f32'
    blocks = []
    for block in range(66, 70):
        input_sha = digest(previous)
        result = run_block(block=block, previous=previous, width=128, height=128,
                           output_root=out / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'block{block} handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(128, 128, 32)
        comparison = metrics(candidate[..., p32], public[f'block{block}'])
        blocks.append(dict(block=block, input_device_sha256=input_sha,
                           output_device_sha256=digest(result),
                           candidate_vs_public_fp16=comparison,
                           hip_scalar_exact=True))
        previous = result
        print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}",
              flush=True)
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_file),
                  color_linear_sha256=digest(color_file),
                  source_model_sha256=MODEL_SHA256,
                  source_decoder65_report_sha256=digest(source_file),
                  source_block65_device_sha256=digest(block65_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block65_public_exact=True,
                  skip4_source='same-frame public ONNX FP8-rounded',
                  block66_tensor_sha256=digest(tensor),
                  block66_prefix_device_sha256=digest(prefix / 'merged_device.f32'),
                  block66_prefix_vs_public_fp16=metrics(merged[..., p32], public['merge66']),
                  blocks=blocks, final_device_sha256=digest(previous),
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_c32_body_map_validated=False,
                  full_candidate_inference=False, production_wiring=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('decoder65_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_decoder32')
    args = parser.parse_args()
    run(args.prepared_dir, args.decoder65_dir, args.output_root)
