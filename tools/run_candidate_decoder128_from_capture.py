#!/usr/bin/env python3
"""Continue one captured-input candidate through decoder blocks56-61."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import onnx
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_block56_candidate import ROOT, run as run_block
from check_upsample56_peer_image import metrics
from check_c256_ffn_candidate import bits
from decode_tinlayout_global import e4m3fn
from extract_peer_decoder56_inputs import NODES
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_c64_reference import multiply
from native_c32_reference import H, F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, encoder22_dir, decoder55_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    encoder22_dir = Path(encoder22_dir).resolve()
    decoder55_dir = Path(decoder55_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    encoder_file = encoder22_dir / 'report.json'
    encoder = json.loads(encoder_file.read_text())
    decoder_file = decoder55_dir / 'report.json'
    decoder = json.loads(decoder_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    block55_file = decoder55_dir / 'block55/output/output_device.f32'
    skip14_file = encoder22_dir / 'block14/output/output_device.f32'
    down14_file = encoder22_dir / 'downsample14/output_device.f32'
    public55_file = decoder55_dir / 'public_boundary/block55_peer.f32'
    public14_file = encoder22_dir / 'public_boundary/block14_skip_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            decoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            decoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            encoder['prepared_manifest_sha256'] != digest(prepared_file) or
            decoder['prepared_manifest_sha256'] != digest(prepared_file) or
            encoder['public_boundary_sha256']['block14_skip'] != digest(public14_file) or
            decoder['public_boundary_sha256']['block55'] != digest(public55_file) or
            encoder['c128_blocks'][-1]['output_device_sha256'] != digest(skip14_file) or
            encoder['downsample14']['output_device_sha256'] != digest(down14_file) or
            decoder['final_device_sha256'] != digest(block55_file) or
            not decoder['hip_scalar_exact'] or
            not encoder['c128_blocks'][-1]['hip_scalar_exact'] or
            not encoder['downsample14']['hip_scalar_exact']):
        raise ValueError('prepared image or candidate encoder14/decoder55 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'decoder55_61.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch_file), ['rgb'], list(NODES.values()))
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 16, 16, 256) if name == 'block55' else (1, 32, 32, 128)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if (digest(public_dir / 'block55_peer.f32') != digest(public55_file) or
            digest(public_dir / 'skip14_peer.f32') != digest(public14_file)):
        raise AssertionError('same-frame public block55/skip14 cross-extraction differs')
    p128 = peer_to_native_multihead(np.arange(128))
    tensor = ROOT / 'dlss5-analysis/tensors/tensor_133.bin'
    raw = np.fromfile(tensor, np.uint8)
    if raw.size != 230176:
        raise ValueError('wrong block56 tensor size')
    count = 2 * 128 * 128
    rows = bits(count, [3] + list(range(6, 12)))
    cols = bits(count, [1, 0, 4, 5, 2] + list(range(12, 15)))
    if np.unique(rows * 256 + cols).size != count:
        raise AssertionError('block56 projection map collision')
    weights = np.empty((128, 256), np.float32)
    weights[rows, cols] = e4m3fn(raw[0x18000:0x20000])
    scale = np.empty(128, np.float32)
    scale[p128] = raw[0x20100:0x20200].view('<f2').astype(np.float32)
    x = np.fromfile(block55_file, '<f4').reshape(16, 16, 256)
    skip = np.fromfile(skip14_file, '<f4').reshape(32, 32, 128)
    low = multiply(x, weights)
    merged_half = H(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1) + skip * scale)
    merged = F(merged_half)
    prefix = out / 'block56_prefix'
    prefix.mkdir(exist_ok=True)
    for name, array in dict(input=x, weights=weights, scale=scale, skip=skip,
                            low=low, merged=merged).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite candidate block56 {name}')
        np.asarray(array, '<f4').tofile(prefix / f'{name}.f32')
    check = subprocess.run([str(ROOT / 'build/upsample56_prefix_test.exe'), str(prefix), '16', '16'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    (prefix / 'hip_check.log').write_text(check.stdout)
    if (check.returncode or check.stdout.count('PASS') != 2 or 'FAIL' in check.stdout or
            (prefix / 'merged_device.f32').read_bytes() != (prefix / 'merged.f32').read_bytes()):
        raise RuntimeError(f'candidate block56 prefix HIP/scalar differs: {check.stdout[-2000:]}')
    previous = prefix / 'merged_device.f32'
    blocks = []
    for block in range(56, 62):
        input_sha = digest(previous)
        result = run_block(block=block, previous=previous, width=32, height=32,
                           output_root=out / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'block{block} handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(32, 32, 128)
        comparison = metrics(candidate[..., p128], public[f'block{block}'])
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
                  source_decoder55_report_sha256=digest(decoder_file),
                  source_block55_device_sha256=digest(block55_file),
                  source_skip14_device_sha256=digest(skip14_file),
                  source_down14_device_sha256=digest(down14_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block55_skip14_public_exact=True,
                  block56_tensor_sha256=digest(tensor),
                  block56_prefix_device_sha256=digest(prefix / 'merged_device.f32'),
                  block56_prefix_vs_public_fp16=metrics(merged[..., p128], public['merge56']),
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
    parser.add_argument('decoder55_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_decoder128')
    args = parser.parse_args()
    run(args.prepared_dir, args.encoder22_dir, args.decoder55_dir, args.output_root)
