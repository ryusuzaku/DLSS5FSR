#!/usr/bin/env python3
"""Connect a block69 latent to a one-window head70 candidate control.

Use synthetic inputs, a mixed public-image/control fixture, or coherent
public-ONNX image-derived latent/skip/color. The C32 body uses public QMMA
decoding in the audited block66 transition basis. The coherent fixture is
still an optimized FP16 public-model candidate, not an original-kernel test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

from audit_peer_native_c32_basis import run as audit_basis, peer_to_native
from check_block66_peer_candidate import decode, save
from head70_normalized_reference import trace, packed
from head70_reference import H, F, merge, windowise, dewindowise, finish
from head70_weights import extract, repack, PAD_AT

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(case='synthetic'):
    folders={'synthetic':'build/head70_peer_connected',
             'image_skip':'build/head70_peer_image_skip',
             'same_image':'build/head70_peer_same_image',
             'same_image_peer_scale':'build/head70_peer_same_image_scale1'}
    if case not in folders:raise ValueError(f'unknown head control {case}')
    out=ROOT/folders[case]
    audit_basis(verbose=False)
    source = ROOT / 'build/decoder69_peer_candidate/seeded/output/output_device.f32'
    map32 = peer_to_native(np.arange(32))
    branch_report=None
    input_scale=1.0 if case=='same_image_peer_scale' else .03125
    coherent=case in ('same_image','same_image_peer_scale')
    if coherent:
        branch=ROOT/'build/peer_coherent_head_inputs'
        branch_report=json.loads((branch/'manifest.json').read_text())
        for name,key in [('skip_peer','skip_peer_sha256'),('latent_peer','latent_peer_sha256'),
                         ('color_linear','color_linear_sha256')]:
            if digest(branch/f'{name}.f32')!=branch_report[key]:
                raise ValueError(f'coherent head source hash differs: {name}')
        main_peer=np.fromfile(branch/'latent_peer.f32','<f4').reshape(128,128,32)[60:64,60:64]
        main=np.empty_like(main_peer);main[...,map32]=main_peer
        peer_skip=np.fromfile(branch/'skip_peer.f32','<f4').reshape(256,256,32)[120:128,120:128]
        skip=np.empty_like(peer_skip);skip[...,map32]=peer_skip
        color=np.fromfile(branch/'color_linear.f32','<f4').reshape(256,256,3)[120:128,120:128].copy()
        main_source='public ONNX block69 latent, blue_marble crop (60,60)'
        skip_source='same-image public ONNX preblock0 skip, crop (120,120)'
        color_source='same-image blue_marble linear RGB, crop (120,120)'
    else:
        main = np.fromfile(source, '<f4').reshape(128, 512, 32)[:4, :4].copy()
        main_source='seeded standalone RX9070XT block69 device control, crop (0,0)'
    if case=='synthetic':
        rng = np.random.default_rng(0x70C32069)
        skip = H(rng.normal(0, 0.25, (8, 8, 32)).astype(np.float32))
        color = rng.uniform(0, 1, (8, 8, 3)).astype(np.float32)
        skip_source = 'deterministic synthetic preblock0 control'
        color_source = 'deterministic synthetic control'
    elif case=='image_skip':
        branch = ROOT/'build/peer_preblock0_skip'
        branch_report=json.loads((branch/'manifest.json').read_text())
        if digest(branch/'skip_peer.f32')!=branch_report['skip_peer_sha256'] or \
           digest(branch/'color_linear.f32')!=branch_report['color_linear_sha256']:
            raise ValueError('image-derived preblock0 source hash differs')
        peer_skip = np.fromfile(branch/'skip_peer.f32','<f4').reshape(256,256,32)[120:128,120:128]
        skip = np.empty_like(peer_skip);skip[...,map32]=peer_skip
        color = np.fromfile(branch/'color_linear.f32','<f4').reshape(256,256,3)[120:128,120:128].copy()
        skip_source = 'public ONNX preblock0 branch, blue_marble crop (120,120)'
        color_source = 'blue_marble linear-RGB crop (120,120)'
    raw = (ROOT / 'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    stage = (ROOT / 'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    ordinary, sm, ss, coeff, pad = extract(raw, stage[PAD_AT:PAD_AT+16])
    if repack(ordinary, sm, ss, coeff, pad) != raw:
        raise AssertionError('head70 extraction does not round trip')
    coeff = coeff[[0, 2, 4]]
    merged = merge(main, skip, sm, ss)
    peer_merged = merged[..., map32]
    peer_windows = windowise(peer_merged)
    weights = decode(np.frombuffer(ordinary, np.uint8))
    stages = trace(peer_windows, weights)
    body_peer = stages['body']
    device_body = out / 'body/body_device.f32'
    body = out / 'body'
    save(body, dict(input=peer_windows, weights=packed(weights), **stages,
                    output=F(body_peer)))
    subprocess.run([str(ROOT/'build/c32_peer_body_test.exe'), str(body), '1'],
                   cwd=ROOT, check=True)
    if device_body.read_bytes() != (body/'body.f32').read_bytes():
        raise AssertionError('head70 peer body device output differs')
    peer_image = dewindowise(np.fromfile(device_body, '<f4').reshape(1, 64, 32), 8, 8)
    features = np.empty_like(peer_image)
    features[..., map32] = peer_image
    rgb = finish(features, color, coeff,input_scale)
    outer = out / 'outer'
    save(outer, dict(main=main, skip=skip, sm=sm, ss=ss, coeff=coeff,
                     color=color, features=windowise(features),
                     merged=windowise(merged), rgb=rgb))
    subprocess.run([str(ROOT/'build/head70_test.exe'), str(outer), '8', '8', str(input_scale)],
                   cwd=ROOT, check=True)
    report = dict(main_source=main_source,
                  main_crop_sha256=hashlib.sha256(np.asarray(main,dtype='<f4').tobytes()).hexdigest(),
                  head_tensor_sha256=hashlib.sha256(raw).hexdigest(),
                  body_device_sha256=digest(device_body), rgb_sha256=digest(outer/'rgb.f32'),
                  windows=1, output_extent=[8, 8, 3], input_scale=input_scale,
                  body_stages_checked=12,
                  outer_stages_checked=['merge/window', 'finish/dewindow'],
                  preblock0_skip=skip_source,color=color_source,
                  skip_sha256=hashlib.sha256(np.asarray(skip,dtype='<f4').tobytes()).hexdigest(),
                  main_skip_common_frame=coherent,
                  original_kernel_executed=False, original_runtime_validation=False)
    if not coherent:report['source_block69_sha256']=digest(source)
    if branch_report is not None:
        report['peer_preblock0_skip_sha256']=branch_report['skip_peer_sha256']
        report['peer_model_sha256']=branch_report['model_sha256']
        if coherent:report['peer_block69_latent_sha256']=branch_report['latent_peer_sha256']
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('synthetic','image_skip','same_image','same_image_peer_scale'),default='synthetic')
    args=parser.parse_args();run(args.case)
