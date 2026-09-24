#!/usr/bin/env python3
"""Connect same-image AMD encoder head to the validated ViT31 prefix."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np

from audit_native_vit_logical_map import audit, logical_map
from check_vit_expand_chain import run_block
from recover_vit_bridge_ptx import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(encoder_root, output_root):
    encoder = Path(encoder_root).resolve()
    source = json.loads((encoder / 'report.json').read_text())
    head = encoder / 'block30_head_4x4/head_device.f32'
    if digest(head) != source['head30_device_sha256']:
        raise ValueError('encoder head device hash differs')
    report, _ = audit(4, 4)
    if not all(report[k] for k in ('cell_ok', 'bank_ok', 'repeat_ok', 'permutation_ok')):
        raise ValueError('4x4 C512/ViT logical geometry disagrees with PTX composition')
    gather = logical_map(16)
    if not np.array_equal(np.sort(gather), np.arange(16*1024)):
        raise ValueError('ViT map is not bijective')
    exe = ROOT / 'build/vit_bridge_ptx_test.exe'
    if not exe.is_file():
        raise FileNotFoundError(exe)
    out = Path(output_root).resolve()
    mapping = out / 'map'
    case = out / 'bridge'
    mapping.mkdir(parents=True, exist_ok=True)
    case.mkdir(parents=True, exist_ok=True)
    gather.astype('<i4').tofile(mapping / 'hwc-to-vit.i32')
    x = np.fromfile(head, '<f4')
    if x.size != gather.size or not np.isfinite(x).all():
        raise ValueError('wrong/nonfinite encoder head extent')
    (case / 'input.f32').write_bytes(head.read_bytes())
    x[gather].astype('<f4').tofile(case / 'expected.f32')
    subprocess.run([str(exe), str(case), str(mapping), str(len(gather))], cwd=ROOT, check=True)
    if (case / 'device.f32').read_bytes() != (case / 'expected.f32').read_bytes():
        raise AssertionError('4x4 HIP bridge differs from logical map gather')
    vit = run_block(31, 4, 4, derived=True, image_source=case / 'device.f32',
                    output_root=out / 'vit31_prefix')
    if (vit / 'input.f32').read_bytes() != (case / 'device.f32').read_bytes():
        raise AssertionError('ViT31 input handoff differs')
    result = dict(source_model_sha256=source['source_model_sha256'],
                  source_image_sha256=source['source_image_sha256'],
                  encoder_head_device_sha256=digest(head),
                  map_sha256=digest(mapping / 'hwc-to-vit.i32'),
                  bridge_device_sha256=digest(case / 'device.f32'),
                  vit31_contract_device_sha256=digest(vit / 'contract_device.f32'),
                  vit31_qkv_scalar_sha256=digest(vit / 'qkv.f32'),
                  extent=[4, 4, 1024],
                  scope='ViT31 expand, gated hidden, contract/residual, QKV projection and normalize; attention/projection pending',
                  hip_scalar_exact=True, original_kernel_executed=False,
                  original_physical_bridge_validated=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_split512_encoder_candidate')
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' / 'peer_vit31_prefix')
    args = parser.parse_args()
    run(args.encoder_root, args.output_root)
