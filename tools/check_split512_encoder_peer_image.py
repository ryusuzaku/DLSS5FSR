#!/usr/bin/env python3
"""Run native-schedule AMD C512 encoder blocks23..30 on the public image."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_split512_spatial_block import ROOT, run_case
from check_split512_bridge import run_case as run_bridge
from check_split512_peer_image import metrics
from native_split_reference import F


SOURCE = ROOT / 'build/peer_split512_encoder_inputs'
SHIFTS = (0, 3, 1, 2, 0, 3, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output_root, unshifted_control=False, candidate_block22_down=None):
    source = json.loads((SOURCE / 'manifest.json').read_text())
    if not source['prior_skip30_exact']:
        raise ValueError('public encoder/decoder skip cross-check absent')
    for name, sha in source['tensor_sha256'].items():
        if digest(SOURCE / f'{name}_peer.f32') != sha:
            raise ValueError(f'{name} source hash differs')
    p512 = peer_to_native_multihead(np.arange(512))
    p1024 = peer_to_native_multihead(np.arange(1024))
    first = np.fromfile(SOURCE / 'block22_peer.f32', '<f4').reshape(64, 512)
    candidate_down_sha256 = None
    if candidate_block22_down is None:
        native = np.empty_like(first)
        native[:, p512] = first
        native = F(native)
    else:
        upstream = Path(candidate_block22_down).resolve()
        ancestry = json.loads((upstream / 'report.json').read_text())
        if (ancestry['source_model_sha256'] != source['model_sha256'] or
                ancestry['source_image_sha256'] != source['image_sha256'] or
                not ancestry['hip_scalar_exact']):
            raise ValueError('candidate block22 downsample ancestry differs')
        candidate = upstream / 'output_device.f32'
        if digest(candidate) != ancestry['block22_down_device_sha256']:
            raise ValueError('candidate block22 downsample hash differs')
        native = np.fromfile(candidate, '<f4').reshape(64, 512)
        candidate_down_sha256 = digest(candidate)
    first_metrics = metrics(native[:, p512], first)
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shifts = (0,) * 8 if unshifted_control else SHIFTS
    comparisons = {}
    handoffs = []
    last = None
    for block, shift in zip(range(23, 31), shifts):
        raw = native.astype('<f4').tobytes()
        folder = run_case(block, 8, 8, shift, native, out)
        if (folder / 'input.f32').read_bytes() != raw:
            raise AssertionError(f'block{block} device input handoff changed')
        native = np.fromfile(folder / 'final_device.f32', '<f4').reshape(64, 512)
        np.testing.assert_array_equal(native, np.fromfile(folder / 'final.f32', '<f4').reshape(64, 512))
        reference = np.fromfile(SOURCE / f'block{block}_peer.f32', '<f4').reshape(64, 512)
        comparisons[f'block{block}'] = metrics(native[:, p512], reference)
        handoffs.append(dict(block=block, input_sha256=hashlib.sha256(raw).hexdigest(),
                             device_output_sha256=digest(folder / 'final_device.f32')))
        last = folder
        print(f"block{block}: MAE {comparisons[f'block{block}']['mae']:.7g}, "
              f"corr {comparisons[f'block{block}']['correlation']:.7g}", flush=True)
    bridge = run_bridge(8, 8, last / 'final_raw_device.f32', out / 'block30_head_4x4')
    head = np.fromfile(bridge / 'head_device.f32', '<f4').reshape(16, 1024)
    public_head = np.fromfile(SOURCE / 'head30_peer.f32', '<f4').reshape(16, 1024)
    head_metrics = metrics(head[:, p1024], public_head)
    print(f"head30: MAE {head_metrics['mae']:.7g}, corr {head_metrics['correlation']:.7g}", flush=True)
    report = dict(source_model_sha256=source['model_sha256'], source_image_sha256=source['image_sha256'],
                  source_block22_sha256=source['tensor_sha256']['block22'],
                  extent=[8, 8, 512], shifts=list(shifts),
                  schedule='public zero-shift control' if unshifted_control else 'native 0,3,1,2 repeat',
                  input_vs_public_fp16=first_metrics,
                  comparisons=comparisons, head30_vs_public_fp16=head_metrics,
                  head30_device_sha256=digest(bridge / 'head_device.f32'),
                  handoffs=handoffs,
                  source_case='candidate AMD block22 downsample' if candidate_down_sha256 else 'public block22 FP8-rounded',
                  candidate_block22_down_device_sha256=candidate_down_sha256,
                  peer_to_native_channels=p512.tolist(), peer_to_native_head_channels=p1024.tolist(),
                  candidate_basis='P512/P1024 extension, not original logical-map recovery',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--unshifted-control', action='store_true')
    parser.add_argument('--candidate-block22-down', type=Path,
                        help='candidate encoder22 downsample fixture root')
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = Path.home() / 'DLSS5FSR-build-offload' / (
            'peer_split512_encoder_unshifted' if args.unshifted_control else
            'peer_split512_encoder_candidate')
    run(args.output_root, args.unshifted_control, args.candidate_block22_down)
