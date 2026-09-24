#!/usr/bin/env python3
"""Run same-image C128 encoder9–14 through exact AMD/scalar stage checks."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_block56_candidate import ROOT, run as run_block
from check_split512_peer_image import metrics
from native_split_reference import F


SOURCE = ROOT / 'build/peer_encoder128_inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output_root, last_block=14):
    if last_block not in range(9, 15):
        raise ValueError('last block must be 9..14')
    source = json.loads((SOURCE / 'manifest.json').read_text())
    if not source['independent_block14_cross_checks_exact']:
        raise ValueError('public block14 cross-extractions not verified')
    for name, sha in source['tensor_sha256'].items():
        if digest(SOURCE / f'{name}_peer.f32') != sha:
            raise ValueError(f'public {name} hash differs')
    p128 = peer_to_native_multihead(np.arange(128))
    public = np.fromfile(SOURCE / 'block8_down_peer.f32', '<f4').reshape(32, 32, 128)
    native = np.empty_like(public)
    native[..., p128] = public
    native = F(native)
    out = Path(output_root).resolve()
    boundary = out / 'boundary8'
    boundary.mkdir(parents=True, exist_ok=True)
    previous = boundary / 'output_device.f32'
    native.astype('<f4').tofile(previous)
    native.astype('<f4').tofile(boundary / 'output.f32')
    comparisons = {'block8_down': metrics(native[..., p128], public)}
    blocks = []
    for block in range(9, last_block + 1):
        first_hash = digest(previous)
        result = run_block(block=block, previous=previous, width=32, height=32,
                           output_root=out / f'block{block}')
        report = json.loads((result.parents[1] / 'manifest.json').read_text())
        if report['input_device_sha256'] != first_hash or report['output_device_sha256'] != digest(result):
            raise AssertionError(f'block{block} device handoff differs')
        candidate = np.fromfile(result, '<f4').reshape(32, 32, 128)
        reference_name = 'block14_skip' if block == 14 else f'block{block}'
        reference = np.fromfile(SOURCE / f'{reference_name}_peer.f32', '<f4').reshape(32, 32, 128)
        comparison = metrics(candidate[..., p128], reference)
        comparisons[reference_name] = comparison
        blocks.append(dict(block=block, shift=report['shift'], windows=report['windows'],
                           input_sha256=first_hash, output_sha256=digest(result),
                           candidate_vs_public_fp16=comparison))
        previous = result
        print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}", flush=True)
    result = dict(source_model_sha256=source['model_sha256'],
                  source_image_sha256=source['image_sha256'],
                  source_block8_down_sha256=source['tensor_sha256']['block8_down'],
                  boundary_device_sha256=digest(boundary / 'output_device.f32'),
                  final_device_sha256=digest(previous),
                  blocks=blocks, comparisons=comparisons,
                  schedule='native front-chain 0,3,1,2,0,3',
                  basis='measured C128 coefficient maps; PTX-supported candidate residual order',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_encoder128_candidate')
    parser.add_argument('--last-block', type=int, default=14)
    args = parser.parse_args()
    run(args.output_root, args.last_block)
