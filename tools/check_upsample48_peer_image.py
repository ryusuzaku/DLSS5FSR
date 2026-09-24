#!/usr/bin/env python3
"""Check block48 projection on same-image public C512 and C256 boundaries.

The public model is FP16 and this candidate uses native-style FP8 input
rounding. Exact HIP/scalar agreement is not original NVIDIA-kernel parity.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ref/dlss5-port/Development'))
from native_split_reference import bits
from native_c64_reference import multiply
from native_c32_reference import H, F
from decode_tinlayout_global import e4m3fn
from audit_peer_native_c32_basis import peer_index, peer_to_native_multihead

SOURCE = ROOT / 'build/peer_decoder48_inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(values=int(d.size), exact=int(np.count_nonzero(d == 0)),
                mae=float(d.mean()), max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))


def run():
    src = json.loads((SOURCE / 'manifest.json').read_text())
    for name in ('block47', 'skip22', 'merge48'):
        if digest(SOURCE / f'{name}_peer.f32') != src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    p512 = peer_to_native_multihead(np.arange(512))
    p256 = peer_to_native_multihead(np.arange(256))
    xpeer = F(np.fromfile(SOURCE / 'block47_peer.f32', '<f4').reshape(8, 8, 512))
    speer = F(np.fromfile(SOURCE / 'skip22_peer.f32', '<f4').reshape(16, 16, 256))
    x = np.empty_like(xpeer); x[..., p512] = xpeer
    skip = np.empty_like(speer); skip[..., p256] = speer

    raw_path = ROOT / 'dlss5-analysis/tensors/tensor_124.bin'
    raw = np.fromfile(raw_path, np.uint8)
    if raw.size != 820784:
        raise ValueError('wrong block48 tensor size')
    count = 256 * 512
    rows = bits(count, [3] + list(range(6, 13)))
    cols = bits(count, [1, 0, 4, 5, 2] + list(range(13, 17)))
    if np.unique(rows * 512 + cols).size != count:
        raise ValueError('projection map collision')
    native_indices = np.empty((256, 512), np.int32)
    native_indices[rows, cols] = np.arange(count)
    peer_indices = peer_index(256, 512, 256)
    if not np.array_equal(peer_indices, native_indices[np.ix_(p256, p512)]):
        raise AssertionError('C512-to-C256 peer/native projection index basis differs')
    weights = np.empty((256, 512), np.float32)
    weights[rows, cols] = e4m3fn(raw[0x58000:0x78000])
    scale = np.empty(256, np.float32)
    scale[p256] = raw[0x78200:0x78400].view('<f2').astype(np.float32)
    low = multiply(x, weights)
    merged_half = H(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1) + skip * scale)
    merged = F(merged_half)
    out = ROOT / 'build/upsample48_prefix_derived/image_fp8'
    out.mkdir(parents=True, exist_ok=True)
    for name, array in dict(input=x, weights=weights, scale=scale, skip=skip,
                            low=low, merged=merged).items():
        np.asarray(array, dtype='<f4').tofile(out / f'{name}.f32')
    result = subprocess.run([str(ROOT / 'build/upsample48_prefix_test.exe'), str(out), '8', '8'],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out / 'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS') != 2 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out / 'merged_device.f32').read_bytes() != (out / 'merged.f32').read_bytes():
        raise AssertionError('prefix HIP/scalar merge differs')
    public = np.fromfile(SOURCE / 'merge48_peer.f32', '<f4').reshape(16, 16, 256)
    report = dict(case='image_fp8', input_extent=[8, 8, 512],
                  output_extent=[16, 16, 256],
                  source_model_sha256=src['model_sha256'],
                  source_block47_peer_sha256=src['tensor_sha256']['block47'],
                  source_skip22_peer_sha256=src['tensor_sha256']['skip22'],
                  block48_tensor_sha256=digest(raw_path),
                  peer_native_projection_indices_exact=count,
                  input_native_sha256=digest(out / 'input.f32'),
                  skip_native_sha256=digest(out / 'skip.f32'),
                  output_device_sha256=digest(out / 'merged_device.f32'),
                  hip_projection_merge_exact=True,
                  merged_half_vs_public_fp16=metrics(merged_half[..., p256], public),
                  merged_fp8_vs_public_fp16=metrics(merged[..., p256], public),
                  basis='candidate C512/C256 P extension; index relation exact',
                  original_kernel_executed=False, original_runtime_validation=False)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout, end='')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
