#!/usr/bin/env python3
"""Run the candidate C256 decoder from one same-image block48 prefix.

The public ONNX outputs provide a comparison, not original native-kernel parity.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_decoder49_candidate import ROOT, digest, run as run_block

SOURCE = ROOT / 'build/peer_decoder48_inputs'
OUT = ROOT / 'build/peer_decoder48_candidate'


def metrics(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(values=int(d.size), exact=int(np.count_nonzero(d == 0)),
                mae=float(d.mean()), max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))


def run(last_block=55, split512=False, output_root=None, prefix_root=None,
        amd_block39=False, encoder_skip30=False, candidate_vit16=False,
        candidate_encoder22=False):
    if (encoder_skip30 or candidate_vit16 or candidate_encoder22) and (output_root is None or prefix_root is None):
        raise ValueError('custom candidate requires explicit prefix/output roots')
    if amd_block39 or encoder_skip30 or candidate_vit16 or candidate_encoder22:
        split512=True
    if last_block not in range(48, 56):
        raise ValueError('last block must be 48..55')
    src = json.loads((SOURCE / 'manifest.json').read_text())
    prefix = (Path(prefix_root) if prefix_root is not None else
              Path.home() / 'DLSS5FSR-build-offload' / ('upsample48_from38' if amd_block39 else 'upsample48_from39') if split512
              else ROOT / 'build/upsample48_prefix_derived/image_fp8')
    prefix_report = json.loads((prefix / 'manifest.json').read_text())
    expected_case=('image_fp8_candidate_encoder22' if candidate_encoder22 else
                   'image_fp8_candidate_vit16' if candidate_vit16 else
                   'image_fp8_encoder_skip30' if encoder_skip30 else
                   'image_fp8_from38' if amd_block39 else 'image_fp8_from39' if split512 else 'image_fp8')
    if prefix_report['case'] != expected_case:
        raise ValueError('block48 prefix case differs')
    if prefix_report['source_model_sha256'] != src['model_sha256']:
        raise ValueError('block48 prefix model differs')
    if (prefix_report['source_block47_peer_sha256'] != src['tensor_sha256']['block47'] or
            prefix_report['source_skip22_peer_sha256'] != src['tensor_sha256']['skip22']):
        raise ValueError('block48 prefix public boundaries differ')
    previous = prefix / 'merged_device.f32'
    if digest(previous) != prefix_report['output_device_sha256']:
        raise ValueError('block48 prefix device hash differs')
    p256 = peer_to_native_multihead(np.arange(256))
    out = (Path(output_root) if output_root is not None else
           (Path.home() / 'DLSS5FSR-build-offload' / ('peer_decoder48_from38' if amd_block39 else 'peer_decoder48_from39') if split512 else OUT))
    blocks = []
    for block in range(48, last_block + 1):
        first_hash = digest(previous)
        result = run_block(block, previous, width=16, height=16,
                           output_root=out / f'block{block}')
        report = json.loads((result.parents[1] / 'manifest.json').read_text())
        if report['input_device_sha256'] != first_hash or report['output_device_sha256'] != digest(result):
            raise AssertionError(f'block{block} device handoff hash mismatch')
        peer_path = SOURCE / f'block{block}_peer.f32'
        if digest(peer_path) != src['tensor_sha256'][f'block{block}']:
            raise ValueError(f'public block{block} hash differs')
        public = np.fromfile(peer_path, '<f4').reshape(16, 16, 256)
        native = np.fromfile(result, '<f4').reshape(16, 16, 256)
        blocks.append(dict(block=block, shift=report['shift'], windows=report['windows'],
                           input_sha256=first_hash, output_sha256=digest(result),
                           native_fp8_vs_public_fp16=metrics(native[..., p256], public)))
        previous = result
    summary = dict(blocks=blocks, source_model_sha256=src['model_sha256'],
                   source_image_sha256=src['image_sha256'],
                   block48_prefix_sha256=prefix_report['output_device_sha256'],
                   case='image_candidate_encoder22' if candidate_encoder22 else
                        'image_candidate_vit16' if candidate_vit16 else
                        'image_encoder_skip30' if encoder_skip30 else
                        'image_from38' if amd_block39 else 'image_from39' if split512 else 'image_from47',
                   output_root=str(out.resolve()),
                   final_device_sha256=digest(previous),
                   candidate_skip22_device_sha256=prefix_report.get('candidate_skip22_device_sha256'),
                   candidate_basis='C256 extension of measured C64/C128 maps',
                   same_image_encoder22_skip=True,
                   candidate_encoder22_skip=candidate_encoder22,
                   original_kernel_executed=False,
                   original_runtime_validation=False)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--last-block', type=int, choices=range(48, 56), default=55)
    parser.add_argument('--split512', action='store_true',
                        help='consume the C512 candidate-derived block48 prefix')
    parser.add_argument('--amd-block39', action='store_true',
                        help='consume block48 derived from AMD block39')
    parser.add_argument('--encoder-skip30', action='store_true',
                        help='consume the block48 prefix using the candidate encoder skip')
    parser.add_argument('--candidate-vit16', action='store_true',
                        help='consume the block48 prefix using candidate 16-token ViT')
    parser.add_argument('--candidate-encoder22', action='store_true',
                        help='consume the block48 prefix using candidate encoder22 skip')
    parser.add_argument('--output-root', type=Path, help='candidate block fixture directory')
    parser.add_argument('--prefix-root', type=Path, help='block48 prefix fixture directory')
    args = parser.parse_args()
    run(args.last_block, args.split512, args.output_root, args.prefix_root,
        args.amd_block39, args.encoder_skip30, args.candidate_vit16,
        args.candidate_encoder22)
