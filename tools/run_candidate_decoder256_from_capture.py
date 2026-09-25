#!/usr/bin/env python3
"""Continue one captured-input candidate through decoder blocks48-55."""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_index, peer_to_native_multihead
from check_decoder49_candidate import ROOT, run as run_block
from check_upsample48_peer_image import metrics
from decode_tinlayout_global import e4m3fn
from extract_peer_decoder48_inputs import NODES
from cached_public_branch import extract_cached
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_split_reference import bits
from native_c64_reference import multiply
from native_c32_reference import H, F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, encoder22_dir, decoder47_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    encoder22_dir = Path(encoder22_dir).resolve()
    decoder47_dir = Path(decoder47_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    encoder_file = encoder22_dir / 'report.json'
    encoder = json.loads(encoder_file.read_text())
    decoder_file = decoder47_dir / 'report.json'
    decoder = json.loads(decoder_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    block47_file = decoder47_dir / 'block47-8x8-s2/final_device.f32'
    skip22_file = encoder22_dir / 'block22/output/output_device.f32'
    public47_file = decoder47_dir / 'public_boundary/block47_peer.f32'
    public22_file = encoder22_dir / 'public_boundary/block22_skip_peer.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_file) or
            decoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_model_sha256'] != MODEL_SHA256 or
            encoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            decoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            encoder['prepared_manifest_sha256'] != digest(prepared_file) or
            decoder['prepared_manifest_sha256'] != digest(prepared_file) or
            encoder['public_boundary_sha256']['block22_skip'] != digest(public22_file) or
            decoder['public_boundary_sha256']['block47'] != digest(public47_file) or
            encoder['c256_blocks'][-1]['output_device_sha256'] != digest(skip22_file) or
            decoder['final_device_sha256'] != digest(block47_file) or
            not decoder['hip_scalar_exact'] or
            not encoder['c256_blocks'][-1]['hip_scalar_exact']):
        raise ValueError('prepared image or candidate encoder22/decoder47 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    public_dir = out / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    branch_file = public_dir / 'decoder47_55.onnx'
    extract_cached(MODEL, branch_file, ['rgb'], list(NODES.values()), MODEL_SHA256)
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = (1, 8, 8, 512) if name == 'block47' else (1, 16, 16, 256)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    if (digest(public_dir / 'block47_peer.f32') != digest(public47_file) or
            digest(public_dir / 'skip22_peer.f32') != digest(public22_file)):
        raise AssertionError('same-frame public block47/skip22 cross-extraction differs')
    p512 = peer_to_native_multihead(np.arange(512))
    p256 = peer_to_native_multihead(np.arange(256))
    count = 256 * 512
    rows = bits(count, [3] + list(range(6, 13)))
    cols = bits(count, [1, 0, 4, 5, 2] + list(range(13, 17)))
    native_indices = np.empty((256, 512), np.int32)
    native_indices[rows, cols] = np.arange(count)
    if (np.unique(rows * 512 + cols).size != count or
            not np.array_equal(peer_index(256, 512, 256),
                               native_indices[np.ix_(p256, p512)])):
        raise AssertionError('block48 projection index basis differs')
    tensor = ROOT / 'dlss5-analysis/tensors/tensor_124.bin'
    raw = np.fromfile(tensor, np.uint8)
    if raw.size != 820784:
        raise ValueError('wrong block48 tensor size')
    weights = np.empty((256, 512), np.float32)
    weights[rows, cols] = e4m3fn(raw[0x58000:0x78000])
    scale = np.empty(256, np.float32)
    scale[p256] = raw[0x78200:0x78400].view('<f2').astype(np.float32)
    x = np.fromfile(block47_file, '<f4').reshape(8, 8, 512)
    skip = np.fromfile(skip22_file, '<f4').reshape(16, 16, 256)
    low = multiply(x, weights)
    merged_half = H(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1) + skip * scale)
    merged = F(merged_half)
    prefix = out / 'block48_prefix'
    prefix.mkdir(exist_ok=True)
    for name, array in dict(input=x, weights=weights, scale=scale, skip=skip,
                            low=low, merged=merged).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite candidate block48 {name}')
        np.asarray(array, '<f4').tofile(prefix / f'{name}.f32')
    check = subprocess.run([str(ROOT / 'build/upsample48_prefix_test.exe'), str(prefix), '8', '8'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    (prefix / 'hip_check.log').write_text(check.stdout)
    if (check.returncode or check.stdout.count('PASS') != 2 or 'FAIL' in check.stdout or
            (prefix / 'merged_device.f32').read_bytes() != (prefix / 'merged.f32').read_bytes()):
        raise RuntimeError(f'candidate block48 prefix HIP/scalar differs: {check.stdout[-2000:]}')
    previous = prefix / 'merged_device.f32'
    blocks = []
    for block in range(48, 56):
        input_sha = digest(previous)
        result = run_block(block, previous, width=16, height=16,
                           output_root=out / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'block{block} handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(16, 16, 256)
        comparison = metrics(candidate[..., p256], public[f'block{block}'])
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
                  source_decoder47_report_sha256=digest(decoder_file),
                  source_block47_device_sha256=digest(block47_file),
                  source_skip22_device_sha256=digest(skip22_file),
                  public_branch_sha256=digest(branch_file),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in NODES},
                  same_frame_block47_skip22_public_exact=True,
                  block48_tensor_sha256=digest(tensor),
                  block48_prefix_device_sha256=digest(prefix / 'merged_device.f32'),
                  block48_prefix_vs_public_fp16=metrics(merged[..., p256], public['merge48']),
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
    parser.add_argument('decoder47_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_decoder256')
    args = parser.parse_args()
    run(args.prepared_dir, args.encoder22_dir, args.decoder47_dir, args.output_root)
