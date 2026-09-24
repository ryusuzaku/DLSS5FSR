#!/usr/bin/env python3
"""Check block48 projection on same-image public C512 and C256 boundaries.

The public model is FP16 and this candidate uses native-style FP8 input
rounding. Exact HIP/scalar agreement is not original NVIDIA-kernel parity.
"""
from pathlib import Path
import argparse
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


def run(split512=False, output_root=None, amd_block39=False, chain_report=None, block39_dir=None):
    if chain_report is not None:
        if block39_dir is None or output_root is None:
            raise ValueError('custom chain requires --block39-dir and --output-root')
        amd_block39=True
    if amd_block39:
        split512=True
    src = json.loads((SOURCE / 'manifest.json').read_text())
    for name in ('block47', 'skip22', 'merge48'):
        if digest(SOURCE / f'{name}_peer.f32') != src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    p512 = peer_to_native_multihead(np.arange(512))
    p256 = peer_to_native_multihead(np.arange(256))
    xpeer = F(np.fromfile(SOURCE / 'block47_peer.f32', '<f4').reshape(8, 8, 512))
    speer = F(np.fromfile(SOURCE / 'skip22_peer.f32', '<f4').reshape(16, 16, 256))
    x = np.empty_like(xpeer); x[..., p512] = xpeer
    source_block47_native_sha256 = None
    if split512:
        report_path=(Path(chain_report) if chain_report is not None else
                     ROOT / 'build/split512_peer_image_from38_report.json' if amd_block39 else
                     ROOT / 'build/split512_peer_image_report.json')
        chain = json.loads(report_path.read_text())
        if chain_report is not None and chain['source_case'] != 'AMD block39 from public ViT38/candidate encoder skip30':
            raise ValueError('custom C512 chain lacks candidate encoder skip provenance')
        if chain['source_model_sha256'] != src['model_sha256'] or chain['source_image_sha256'] != src['image_sha256']:
            raise ValueError('split512 source model/image differs')
        if amd_block39:
            entry_dir=(Path(block39_dir) if block39_dir is not None else
                       Path.home() / 'DLSS5FSR-build-offload' / 'decoder39_peer_image')
            entry = json.loads((entry_dir / 'manifest.json').read_text())
            if block39_dir is not None and entry['case'] != 'same_image_public_vit38_candidate_skip30':
                raise ValueError('custom block39 fixture lacks candidate encoder skip provenance')
            if chain['source_block39_native_sha256'] != entry['output_device_sha256']:
                raise ValueError('AMD block39 handoff differs')
        candidate = Path(chain['output_root']) / 'block47-8x8-s2' / 'final_device.f32'
        if digest(candidate) != chain['final_device_sha256']:
            raise ValueError('split512 block47 device hash differs')
        x = np.fromfile(candidate, '<f4').reshape(8, 8, 512)
        source_block47_native_sha256 = digest(candidate)
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
    out = Path(output_root) if output_root is not None else (ROOT / 'build/upsample48_prefix_derived/image_fp8')
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
    report = dict(case='image_fp8_encoder_skip30' if chain_report is not None else
                  'image_fp8_from38' if amd_block39 else 'image_fp8_from39' if split512 else 'image_fp8', input_extent=[8, 8, 512],
                  output_extent=[16, 16, 256],
                  source_model_sha256=src['model_sha256'],
                  source_block47_peer_sha256=src['tensor_sha256']['block47'],
                  source_block47_native_sha256=source_block47_native_sha256,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split512', action='store_true',
                        help='use the connected C512 candidate block47 device output')
    parser.add_argument('--amd-block39', action='store_true',
                        help='use the C512 chain started from AMD block39')
    parser.add_argument('--output-root', type=Path,
                        help='fixture directory (use a spacious drive for --split512)')
    parser.add_argument('--chain-report', type=Path,
                        help='custom C512 chain report, e.g. one using a candidate encoder skip')
    parser.add_argument('--block39-dir', type=Path,
                        help='AMD block39 fixture that supplied the custom chain')
    args = parser.parse_args()
    if (args.split512 or args.amd_block39) and args.output_root is None:
        args.output_root = Path.home() / 'DLSS5FSR-build-offload' / ('upsample48_from38' if args.amd_block39 else 'upsample48_from39')
    run(args.split512, args.output_root, args.amd_block39, args.chain_report, args.block39_dir)
