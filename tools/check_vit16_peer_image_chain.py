#!/usr/bin/env python3
"""Continue the 4x4 AMD candidate from ViT31 through ViT38 and inverse view."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np

from audit_native_vit_logical_map import logical_map
from audit_peer_native_c32_basis import peer_to_native_multihead
from check_vit_expand_chain import run_block
from check_split512_peer_image import metrics
from recover_vit_bridge_ptx import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(vit31_root, output_root):
    source = Path(vit31_root).resolve()
    prior = json.loads((source / 'report.json').read_text())
    if not prior['hip_scalar_exact'] or prior['original_physical_bridge_validated']:
        raise ValueError('ViT31 candidate provenance differs')
    previous = source / 'vit31_prefix/projection_device.f32'
    if digest(previous) != prior['vit31_projection_device_sha256']:
        raise ValueError('ViT31 device projection hash differs')
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    handoffs = []
    for block in range(32, 39):
        folder = run_block(block, 4, 4, derived=True, image_source=previous,
                           output_root=out / f'vit_block{block}')
        if (folder / 'input.f32').read_bytes() != previous.read_bytes():
            raise AssertionError(f'ViT block{block} device handoff differs')
        current = folder / 'projection_device.f32'
        handoffs.append(dict(block=block, input_sha256=digest(previous),
                             projection_device_sha256=digest(current)))
        previous = current
        print(f'ViT block{block}: candidate HIP/scalar exact', flush=True)
    gather = logical_map(16)
    inverse = np.argsort(gather).astype('<i4')
    if not np.array_equal(gather[inverse], np.arange(inverse.size)):
        raise AssertionError('4x4 logical inverse is not bijective')
    mapping = out / 'inverse_map'
    case = out / 'inverse_bridge'
    mapping.mkdir(exist_ok=True)
    case.mkdir(exist_ok=True)
    inverse.tofile(mapping / 'hwc-to-vit.i32')
    vit = np.fromfile(previous, '<f4')
    if vit.size != 16*1024:
        raise ValueError('wrong ViT38 extent')
    (case / 'input.f32').write_bytes(previous.read_bytes())
    vit[inverse].astype('<f4').tofile(case / 'expected.f32')
    exe = ROOT / 'build/vit_bridge_ptx_test.exe'
    subprocess.run([str(exe), str(case), str(mapping), str(inverse.size)], cwd=ROOT, check=True)
    if (case / 'device.f32').read_bytes() != (case / 'expected.f32').read_bytes():
        raise AssertionError('AMD/scalar inverse logical bridge differs')
    public_source = ROOT / 'build/peer_decoder39_inputs'
    peer_record = json.loads((public_source / 'manifest.json').read_text())
    if (prior['source_model_sha256'] != peer_record['model_sha256'] or
            prior['source_image_sha256'] != peer_record['image_sha256'] or
            digest(public_source / 'vit38_peer.f32') != peer_record['tensor_sha256']['vit38']):
        raise ValueError('public ViT38 comparison used a different model/image')
    p1024 = peer_to_native_multihead(np.arange(1024))
    native = np.fromfile(case / 'device.f32', '<f4').reshape(16, 1024)
    public = np.fromfile(public_source / 'vit38_peer.f32', '<f4').reshape(16, 1024)
    comparison = metrics(native[:, p1024], public)
    result = dict(source_model_sha256=prior['source_model_sha256'],
                  source_image_sha256=prior['source_image_sha256'],
                  vit31_projection_device_sha256=prior['vit31_projection_device_sha256'],
                  vit38_projection_device_sha256=digest(previous),
                  vit38_inverse_device_sha256=digest(case / 'device.f32'),
                  vit38_vs_public_fp16=comparison, handoffs=handoffs,
                  attention_contract='experimental 16-valid-key sum; original reduction unverified',
                  logical_bridge='source-derived 4x4 candidate, original physical map unverified',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_runtime_validation=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vit31-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_vit31_candidate16')
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_vit16_candidate')
    args = parser.parse_args()
    run(args.vit31_root, args.output_root)
