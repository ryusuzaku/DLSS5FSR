#!/usr/bin/env python3
"""Continue one captured-input candidate through decoder blocks62-65."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_block62_candidate import ROOT, run as run_block
from check_upsample62_peer_image import metrics, project_sequential_f32
from check_c256_ffn_candidate import bits
from decode_tinlayout_global import e4m3fn
from extract_peer_decoder62_inputs import NODES
from cached_public_branch import extract_cached
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_c32_reference import H, F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, encoder22_dir, decoder61_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    encoder22_dir = Path(encoder22_dir).resolve()
    decoder61_dir = Path(decoder61_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    encoder_file = encoder22_dir / 'report.json'
    encoder = json.loads(encoder_file.read_text())
    decoder_file = decoder61_dir / 'report.json'
    decoder = json.loads(decoder_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    block61_file = decoder61_dir / 'block61/output/output_device.f32'
    skip8_file = encoder22_dir / 'block8/output/output_device.f32'
    public61_file = decoder61_dir / 'public_boundary/block61_peer.f32'
    public8_file = encoder22_dir / 'public_boundary/block8_skip_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            decoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            decoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            encoder['prepared_manifest_sha256'] != digest(prepared_file) or
            decoder['prepared_manifest_sha256'] != digest(prepared_file) or
            encoder['public_boundary_sha256']['block8_skip'] != digest(public8_file) or
            decoder['public_boundary_sha256']['block61'] != digest(public61_file) or
            encoder['blocks'][-1]['output_device_sha256'] != digest(skip8_file) or
            decoder['final_device_sha256'] != digest(block61_file) or
            not decoder['hip_scalar_exact'] or not encoder['blocks'][-1]['hip_scalar_exact']):
        raise ValueError('prepared image or candidate encoder8/decoder61 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'decoder61_65.onnx'
    extract_cached(MODEL, branch_file, ['rgb'], list(NODES.values()), MODEL_SHA256)
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 32, 32, 128) if name == 'block61' else (1, 64, 64, 64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if (digest(public_dir / 'block61_peer.f32') != digest(public61_file) or
            digest(public_dir / 'skip8_peer.f32') != digest(public8_file)):
        raise AssertionError('same-frame public block61/skip8 cross-extraction differs')
    p64 = peer_to_native_multihead(np.arange(64))
    tensor = ROOT / 'dlss5-analysis/tensors/tensor_140.bin'
    raw = np.fromfile(tensor, np.uint8)
    if raw.size != 70048:
        raise ValueError('wrong block62 tensor size')
    count = 2 * 64 * 64
    rows = bits(count, [3] + list(range(6, 11)))
    cols = bits(count, [1, 0, 4, 5, 2] + list(range(11, 13)))
    if np.unique(rows * 128 + cols).size != count:
        raise AssertionError('block62 projection map collision')
    weights = np.empty((64, 128), np.float32)
    weights[rows, cols] = e4m3fn(raw[0x7000:0x9000])
    scale = np.empty(64, np.float32)
    scale[p64] = raw[0x9080:0x9100].view('<f2').astype(np.float32)
    x = np.fromfile(block61_file, '<f4').reshape(32, 32, 128)
    skip = np.fromfile(skip8_file, '<f4').reshape(64, 64, 64)
    low = project_sequential_f32(x, weights)
    merged_half = H(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1) + skip * scale)
    merged = F(merged_half)
    prefix = out / 'block62_prefix'
    prefix.mkdir(exist_ok=True)
    for name, array in dict(input=x, weights=weights, scale=scale, skip=skip,
                            low=low, merged=merged).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite candidate block62 {name}')
        np.asarray(array, '<f4').tofile(prefix / f'{name}.f32')
    check = subprocess.run([str(ROOT / 'build/upsample62_prefix_test.exe'), str(prefix), '32', '32'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    (prefix / 'hip_check.log').write_text(check.stdout)
    if (check.returncode or check.stdout.count('PASS') != 2 or 'FAIL' in check.stdout or
            (prefix / 'merged_device.f32').read_bytes() != (prefix / 'merged.f32').read_bytes()):
        raise RuntimeError(f'candidate block62 prefix HIP/scalar differs: {check.stdout[-2000:]}')
    previous = prefix / 'merged_device.f32'
    blocks = []
    for block in range(62, 66):
        input_sha = digest(previous)
        result = run_block(block=block, previous=previous, width=64, height=64,
                           output_root=out / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'block{block} handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(64, 64, 64)
        comparison = metrics(candidate[..., p64], public[f'block{block}'])
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
                  source_encoder22_report_sha256=digest(encoder_file),
                  source_decoder61_report_sha256=digest(decoder_file),
                  source_block61_device_sha256=digest(block61_file),
                  source_skip8_device_sha256=digest(skip8_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block61_skip8_public_exact=True,
                  block62_tensor_sha256=digest(tensor),
                  block62_prefix_device_sha256=digest(prefix / 'merged_device.f32'),
                  block62_prefix_vs_public_fp16=metrics(merged[..., p64], public['merge62']),
                  blocks=blocks, final_device_sha256=digest(previous),
                  hip_scalar_exact=True, original_kernel_executed=False,
                  full_candidate_inference=False, production_wiring=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('encoder22_dir', type=Path)
    parser.add_argument('decoder61_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_decoder64')
    args = parser.parse_args()
    run(args.prepared_dir, args.encoder22_dir, args.decoder61_dir, args.output_root)
