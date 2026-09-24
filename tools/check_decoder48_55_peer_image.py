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


def run(last_block=55):
    if last_block not in range(48, 56):
        raise ValueError('last block must be 48..55')
    src = json.loads((SOURCE / 'manifest.json').read_text())
    prefix = ROOT / 'build/upsample48_prefix_derived/image_fp8'
    prefix_report = json.loads((prefix / 'manifest.json').read_text())
    previous = prefix / 'merged_device.f32'
    if digest(previous) != prefix_report['output_device_sha256']:
        raise ValueError('block48 prefix device hash differs')
    p256 = peer_to_native_multihead(np.arange(256))
    blocks = []
    for block in range(48, last_block + 1):
        first_hash = digest(previous)
        result = run_block(block, previous, width=16, height=16,
                           output_root=OUT / f'block{block}')
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
                   block48_prefix_sha256=prefix_report['output_device_sha256'],
                   final_device_sha256=digest(previous),
                   candidate_basis='C256 extension of measured C64/C128 maps',
                   same_image_encoder22_skip=True, original_kernel_executed=False,
                   original_runtime_validation=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--last-block', type=int, choices=range(48, 56), default=55)
    run(parser.parse_args().last_block)
