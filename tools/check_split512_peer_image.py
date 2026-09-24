#!/usr/bin/env python3
"""Run the C512 AMD candidate from same-image public block39 through block47."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_split512_spatial_block import ROOT, run_case
from native_split_reference import F


SOURCE = ROOT / 'build/peer_split512_inputs'
NATIVE_SHIFTS = (0, 3, 1, 2, 0, 3, 1, 2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference):
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        raise ValueError('comparison shape/nonfinite')
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return dict(values=int(delta.size), exact=int(np.count_nonzero(delta == 0)),
                mae=float(delta.mean()), max_abs=float(delta.max()),
                correlation=float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]))


def run(output_root, through=47, teacher_forced=False, unshifted_control=False):
    if through not in range(40, 48):
        raise ValueError('through must be block40..47')
    source = json.loads((SOURCE / 'manifest.json').read_text())
    for block in range(39, through + 1):
        name = f'block{block}'
        if digest(SOURCE / f'{name}_peer.f32') != source['tensor_sha256'][name]:
            raise ValueError(f'{name} source hash differs')
    if not source['prior_block47_exact']:
        raise ValueError('public block47 cross-check absent')
    p512 = peer_to_native_multihead(np.arange(512))
    peer = np.fromfile(SOURCE / 'block39_peer.f32', '<f4').reshape(64, 512)
    native = np.empty_like(peer)
    native[:, p512] = peer
    native = F(native)
    input_rounding = metrics(native[:, p512], peer)
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    handoffs = []
    comparisons = {}
    shifts = (0,) * 8 if unshifted_control else NATIVE_SHIFTS
    for block, shift in zip(range(40, through + 1), shifts):
        raw = native.astype('<f4').tobytes()
        folder = run_case(block, 8, 8, shift, native, out)
        if (folder / 'input.f32').read_bytes() != raw:
            raise AssertionError(f'block{block} device input handoff changed')
        handoffs.append(dict(block=block, input_sha256=hashlib.sha256(raw).hexdigest(),
                             device_output_sha256=digest(folder / 'final_device.f32')))
        native = np.fromfile(folder / 'final_device.f32', '<f4').reshape(64, 512)
        np.testing.assert_array_equal(native, np.fromfile(folder / 'final.f32', '<f4').reshape(64, 512))
        reference = np.fromfile(SOURCE / f'block{block}_peer.f32', '<f4').reshape(64, 512)
        comparisons[f'block{block}'] = metrics(native[:, p512], reference)
        print(f"block{block}: MAE {comparisons[f'block{block}']['mae']:.7g}, "
              f"corr {comparisons[f'block{block}']['correlation']:.7g}")
    teacher = {}
    if teacher_forced:
        forced_root = out / 'teacher_forced'
        for block, shift in zip(range(40, through + 1), shifts):
            previous = np.fromfile(SOURCE / f'block{block-1}_peer.f32', '<f4').reshape(64, 512)
            forced = np.empty_like(previous)
            forced[:, p512] = previous
            forced = F(forced)
            folder = run_case(block, 8, 8, shift, forced, forced_root)
            device = np.fromfile(folder / 'final_device.f32', '<f4').reshape(64, 512)
            reference = np.fromfile(SOURCE / f'block{block}_peer.f32', '<f4').reshape(64, 512)
            teacher[f'block{block}'] = dict(
                input_rounding=metrics(forced[:, p512], previous),
                output_vs_public_fp16=metrics(device[:, p512], reference),
                device_sha256=digest(folder / 'final_device.f32'))
            print(f"teacher block{block}: MAE {teacher[f'block{block}']['output_vs_public_fp16']['mae']:.7g}, "
                  f"corr {teacher[f'block{block}']['output_vs_public_fp16']['correlation']:.7g}")
    report = dict(source_model_sha256=source['model_sha256'],
                  source_image_sha256=source['image_sha256'],
                  source_block39_sha256=source['tensor_sha256']['block39'],
                  extent=[8, 8, 512], shifts=list(shifts[:through-39]),
                  window_schedule='public ONNX unshifted control' if unshifted_control else 'native 0,3,1,2 repeat',
                  input_fp8_rounding_vs_public_fp16=input_rounding,
                  peer_to_native_channels=p512.tolist(), handoffs=handoffs,
                  candidate_vs_public_fp16=comparisons, teacher_forced=teacher,
                  output_root=str(out), final_device_sha256=handoffs[-1]['device_output_sha256'],
                  basis='candidate C512 P extension; not original-map recovery',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    summary = ROOT / ('build/split512_peer_image_unshifted_report.json' if unshifted_control
                      else 'build/split512_peer_image_report.json')
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(f'report: {summary}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_split512_candidate')
    parser.add_argument('--through', type=int, default=47)
    parser.add_argument('--teacher-forced', action='store_true',
                        help='also check each block from its public FP16 predecessor')
    parser.add_argument('--unshifted-control', action='store_true',
                        help='run zero-shift public-model control, not the native schedule')
    args = parser.parse_args()
    if args.unshifted_control and args.output_root == Path.home() / 'DLSS5FSR-build-offload' / 'peer_split512_candidate':
        args.output_root = Path.home() / 'DLSS5FSR-build-offload' / 'peer_split512_unshifted_control'
    run(args.output_root, args.through, args.teacher_forced, args.unshifted_control)
