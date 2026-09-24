#!/usr/bin/env python3
"""Continue the public block14 downsample through candidate AMD C256 blocks15–21."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_decoder49_candidate import ROOT, run as run_block
from check_split512_peer_image import metrics
from native_split_reference import F


SOURCE = ROOT / 'build/peer_encoder256_inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output_root, last_block=21):
    if last_block not in range(15, 22):
        raise ValueError('last block must be 15..21')
    source = json.loads((SOURCE / 'manifest.json').read_text())
    for name, sha in source['tensor_sha256'].items():
        if digest(SOURCE / f'{name}_peer.f32') != sha:
            raise ValueError(f'public {name} hash differs')
    if not source['prior_block22_down_exact'] or not source['prior_block22_skip_exact']:
        raise ValueError('public block22 cross-extraction not verified')
    p256 = peer_to_native_multihead(np.arange(256))
    public = np.fromfile(SOURCE / 'block14_down_peer.f32', '<f4').reshape(16, 16, 256)
    native = np.empty_like(public)
    native[..., p256] = public
    native = F(native)
    out = Path(output_root).resolve()
    boundary = out / 'boundary14'
    boundary.mkdir(parents=True, exist_ok=True)
    previous = boundary / 'output_device.f32'
    native.astype('<f4').tofile(previous)
    native.astype('<f4').tofile(boundary / 'output.f32')
    comparisons = {'block14_down': metrics(native[..., p256], public)}
    blocks = []
    for block in range(15, last_block + 1):
        first_hash = digest(previous)
        result = run_block(block, previous, width=16, height=16,
                           output_root=out / f'block{block}')
        report = json.loads((result.parents[1] / 'manifest.json').read_text())
        if report['input_device_sha256'] != first_hash or report['output_device_sha256'] != digest(result):
            raise AssertionError(f'block{block} device handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(16, 16, 256)
        reference = np.fromfile(SOURCE / f'block{block}_peer.f32', '<f4').reshape(16, 16, 256)
        comparison = metrics(candidate[..., p256], reference)
        comparisons[f'block{block}'] = comparison
        blocks.append(dict(block=block, shift=report['shift'], windows=report['windows'],
                           input_sha256=first_hash, output_sha256=digest(result),
                           candidate_vs_public_fp16=comparison))
        previous = result
        print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}", flush=True)
    result = dict(source_model_sha256=source['model_sha256'],
                  source_image_sha256=source['image_sha256'],
                  source_block14_down_sha256=source['tensor_sha256']['block14_down'],
                  boundary_device_sha256=digest(boundary / 'output_device.f32'),
                  final_device_sha256=digest(previous),
                  blocks=blocks, comparisons=comparisons,
                  schedule='native 0,3,1,2 repeat',
                  basis='candidate C256 extension of measured C64/C128 maps',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_encoder256_candidate')
    parser.add_argument('--last-block', type=int, default=21)
    args = parser.parse_args()
    run(args.output_root, args.last_block)
