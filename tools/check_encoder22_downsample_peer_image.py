#!/usr/bin/env python3
"""Check the candidate AMD block22 raw pool and C256-to-C512 downsample."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_c256_ffn_candidate import ROOT, bits, F, H
from check_split512_peer_image import metrics
from decode_tinlayout_global import e4m3fn
from native_c64_reference import multiply


SOURCE = ROOT / 'build/peer_encoder256_inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(encoder_root, output_root):
    src = json.loads((SOURCE / 'manifest.json').read_text())
    encoder = Path(encoder_root).resolve()
    chain = json.loads((encoder / 'report.json').read_text())
    if (chain['source_model_sha256'] != src['model_sha256'] or
            chain['source_image_sha256'] != src['image_sha256'] or
            chain['blocks'][-1]['block'] != 21 or not chain['hip_scalar_exact']):
        raise ValueError('candidate encoder15-21 ancestry differs')
    block22 = encoder / 'block22'
    body = json.loads((block22 / 'manifest.json').read_text())
    prior = encoder / 'block21/output/output_device.f32'
    skip = block22 / 'output/output_device.f32'
    raw_path = block22 / 'raw_output/output_device.f32'
    if (digest(prior) != chain['final_device_sha256'] or
            digest(prior) != body['input_device_sha256'] or
            digest(skip) != body['output_device_sha256'] or
            digest(raw_path) != body['raw_output_device_sha256']):
        raise ValueError('block22 body handoff/hash differs')
    records = {r['name']: r for r in json.loads((ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    index = records['block22.layer0.layer']['index']
    tensor = ROOT / 'dlss5-analysis/tensors' / f'tensor_{index:03d}.bin'
    data = np.fromfile(tensor, np.uint8)
    if data.size != 820288 or data.size - 0xa8440 != 131072:
        raise ValueError('wrong block22 downsample record extent')
    positions = np.arange(2*256*256, dtype=np.int32)
    inputs = bits(len(positions), [1, 0, 4, 5, 2, 14, 15, 16])
    outputs = bits(len(positions), [3, 6, 7, 8, 9, 10, 11, 12, 13])
    if np.unique(outputs*256+inputs).size != len(positions):
        raise ValueError('candidate downsample map collides')
    matrix = np.empty((512, 256), np.float32)
    matrix[outputs, inputs] = e4m3fn(data[0xa8440:])
    raw = np.fromfile(raw_path, '<f4').reshape(16, 16, 256)
    rows_top = H(raw[::2, ::2] + raw[::2, 1::2])
    rows_bottom = H(raw[1::2, ::2] + raw[1::2, 1::2])
    pool = F(H(H(rows_top + rows_bottom)*np.float32(.25)))
    output = F(multiply(pool.reshape(64, 256), matrix)).reshape(8, 8, 512)
    if not np.isfinite(output).all():
        raise ValueError('nonfinite candidate downsample')
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, array in dict(raw=raw, pool=pool, matrix=matrix, output=output).items():
        np.asarray(array, '<f4').tofile(out / f'{name}.f32')
    exe = ROOT / 'build/encoder256_downsample_test.exe'
    subprocess.run([str(exe), str(out), '16', '16'], cwd=ROOT, check=True)
    if ((out / 'pool_device.f32').read_bytes() != (out / 'pool.f32').read_bytes() or
            (out / 'output_device.f32').read_bytes() != (out / 'output.f32').read_bytes()):
        raise AssertionError('block22 downsample HIP/scalar differs')
    p256 = peer_to_native_multihead(np.arange(256))
    p512 = peer_to_native_multihead(np.arange(512))
    public_skip = np.fromfile(SOURCE / 'block22_skip_peer.f32', '<f4').reshape(16, 16, 256)
    public_down = np.fromfile(SOURCE / 'block22_down_peer.f32', '<f4').reshape(8, 8, 512)
    candidate_skip = np.fromfile(skip, '<f4').reshape(16, 16, 256)
    result = dict(source_model_sha256=src['model_sha256'],
                  source_image_sha256=src['image_sha256'],
                  block21_device_sha256=digest(prior),
                  block22_skip_device_sha256=digest(skip),
                  block22_skip_device_path=str(skip),
                  block22_raw_device_sha256=digest(raw_path),
                  block22_down_device_sha256=digest(out / 'output_device.f32'),
                  block22_tensor_sha256=digest(tensor),
                  skip22_vs_public_fp16=metrics(candidate_skip[..., p256], public_skip),
                  down22_vs_public_fp16=metrics(output[..., p512], public_down),
                  map='C256 downsample address-bit extension from measured C64/C128 maps; original C256 map unverified',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_encoder256_candidate')
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_encoder22_down_candidate')
    args = parser.parse_args()
    run(args.encoder_root, args.output_root)
