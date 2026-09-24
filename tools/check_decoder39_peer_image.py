#!/usr/bin/env python3
"""Check AMD block39 on same-image public ViT38 and split-encoder30 inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref/dlss5-port/Development'))
from audit_peer_native_c32_basis import peer_index, peer_to_native_multihead
from native_split_reference import bits
import native_decoder_entry_reference as D
from native_c32_reference import F, H

SOURCE = ROOT / 'build/peer_decoder39_inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference):
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        raise ValueError('comparison shape/nonfinite')
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return dict(values=int(delta.size), exact=int(np.count_nonzero(delta == 0)),
                mae=float(delta.mean()), max_abs=float(delta.max()),
                correlation=float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]))


def run(output_root=None, candidate_skip30=None):
    if candidate_skip30 is not None and output_root is None:
        raise ValueError('candidate skip requires a distinct --output-root')
    src = json.loads((SOURCE / 'manifest.json').read_text())
    for name in ('vit38', 'skip30', 'merge39', 'block39'):
        if digest(SOURCE / f'{name}_peer.f32') != src['tensor_sha256'][name]:
            raise ValueError(f'public {name} hash differs')
    if not src['prior_block39_exact'] or not src['same_inference_call']:
        raise ValueError('public block39 provenance differs')
    p1024 = peer_to_native_multihead(np.arange(1024))
    p512 = peer_to_native_multihead(np.arange(512))
    peer_vit = np.fromfile(SOURCE / 'vit38_peer.f32', '<f4').reshape(4, 4, 1024)
    peer_skip = np.fromfile(SOURCE / 'skip30_peer.f32', '<f4').reshape(8, 8, 512)
    main = np.empty_like(peer_vit); main[..., p1024] = peer_vit; main = F(main)
    if candidate_skip30 is None:
        skip = np.empty_like(peer_skip); skip[..., p512] = peer_skip; skip = F(skip)
        candidate_skip_sha256 = None
    else:
        encoder = Path(candidate_skip30).resolve()
        ancestry = json.loads((encoder / 'report.json').read_text())
        if (ancestry['source_model_sha256'] != src['model_sha256'] or
                ancestry['source_image_sha256'] != src['image_sha256']):
            raise ValueError('candidate encoder model/image differs')
        candidate = encoder / 'block30-8x8-s2/final_device.f32'
        if digest(candidate) != ancestry['handoffs'][-1]['device_output_sha256']:
            raise ValueError('candidate encoder skip device hash differs')
        skip = np.fromfile(candidate, '<f4').reshape(8, 8, 512)
        candidate_skip_sha256 = digest(candidate)
    count = 512 * 1024
    rows = bits(count, [3, 6, 7, 8, 9, 10, 11, 12, 13])
    cols = bits(count, [1, 0, 4, 5, 2, 14, 15, 16, 17, 18])
    if np.unique(rows * 1024 + cols).size != count:
        raise ValueError('block39 projection raw-index map has a collision')
    native_index = np.empty((512, 1024), np.int32)
    native_index[rows, cols] = np.arange(count)
    if not np.array_equal(peer_index(512, 1024, 512), native_index[np.ix_(p512, p1024)]):
        raise AssertionError('block39 C1024/C512 public/native raw-index basis differs')
    records = {record['name']: record for record in
               json.loads((ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    record = records['block39.layer0.layer']
    raw_path = ROOT / 'dlss5-analysis/tensors' / f"tensor_{record['index']:03d}.bin"
    weights, scale = D.unpack(raw_path)
    projected = D.project(main, weights)
    merged_half = H(np.repeat(np.repeat(projected, 2, axis=0), 2, axis=1) + skip * scale)
    output = D.decoder_entry(main, skip, (weights, scale))
    np.testing.assert_array_equal(output, F(merged_half))
    out = Path(output_root) if output_root is not None else (
        Path.home() / 'DLSS5FSR-build-offload' / 'decoder39_peer_image')
    out.mkdir(parents=True, exist_ok=True)
    # This fixture starts in logical HWC order, so the inverse kernel's
    # control permutation is identity. It does not validate the native ViT
    # physical-view bridge from an upstream AMD ViT output.
    vit = main.copy()
    inverse = np.arange(vit.size, dtype='<i4')
    for name, array in dict(vit=vit, main=main, skip=skip, weights=weights,
                            scale=scale, projected=projected, output=output).items():
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite block39 fixture: {name}')
        np.asarray(array, dtype='<f4').tofile(out / f'{name}.f32')
    inverse.tofile(out / 'inverse.i32')
    exe = ROOT / 'build/decoder39_entry_test.exe'
    result = subprocess.run([str(exe), str(out), '4', '4'], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS') != 3 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out / 'output_device.f32').read_bytes() != (out / 'output.f32').read_bytes():
        raise AssertionError('block39 AMD/scalar output differs')
    public_merge = np.fromfile(SOURCE / 'merge39_peer.f32', '<f4').reshape(8, 8, 512)
    public_output = np.fromfile(SOURCE / 'block39_peer.f32', '<f4').reshape(8, 8, 512)
    report = dict(case='same_image_public_vit38_candidate_skip30' if candidate_skip30 is not None
                  else 'same_image_public_vit38_skip30',
                  source_model_sha256=src['model_sha256'],
                  source_image_sha256=src['image_sha256'],
                  source_vit38_peer_sha256=src['tensor_sha256']['vit38'],
                  source_skip30_peer_sha256=src['tensor_sha256']['skip30'],
                  candidate_skip30_device_sha256=candidate_skip_sha256,
                  block39_tensor_sha256=digest(raw_path),
                  peer_native_projection_indices_exact=count,
                  input_vit_native_sha256=digest(out / 'main.f32'),
                  skip_native_sha256=digest(out / 'skip.f32'),
                  output_device_sha256=digest(out / 'output_device.f32'),
                  hip_scalar_stages_exact=3,
                  merged_half_vs_public_fp16=metrics(merged_half[..., p512], public_merge),
                  output_fp8_vs_public_fp16=metrics(output[..., p512], public_output),
                  basis='C1024/C512 P relation exact for block39 projection; public logical ViT and candidate encoder skip'
                  if candidate_skip30 is not None else
                  'C1024/C512 P relation exact for block39 projection; public logical ViT and skip inputs',
                  native_vit_physical_bridge_validated=False,
                  original_kernel_executed=False, original_runtime_validation=False)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout, end='')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--candidate-skip30', type=Path,
                        help='encoder candidate output root, using its block30 AMD device skip')
    args = parser.parse_args()
    run(args.output_root, args.candidate_skip30)
