#!/usr/bin/env python3
"""Measure head-stage drift against the optimized public FP16 ONNX graph.

This is diagnostic. The graph's FP16/FP32 arithmetic and this HIP candidate's
FP8/half arithmetic differ by design; no original NVIDIA result is involved.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

from audit_peer_native_c32_basis import peer_to_native
from check_block66_peer_candidate import decode, save
from head70_normalized_reference import trace, packed
from head70_reference import F,H
from head70_weights import extract, PAD_AT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/peer_head_intermediate_audit'
CROP=slice(120,128)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual,target):
    if actual.shape!=target.shape or not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError('bad comparison shape/nonfinite')
    delta=np.abs(actual.astype(np.float64)-target.astype(np.float64))
    return dict(values=int(delta.size),exact=int(np.count_nonzero(delta==0)),
                mae=float(delta.mean()),max_abs=float(delta.max()),
                rmse=float(np.sqrt(np.mean(delta*delta))),
                correlation=float(np.corrcoef(actual.ravel(),target.ravel())[0,1]))


def run():
    source=ROOT/'build/peer_coherent_head_inputs'
    report=json.loads((source/'manifest.json').read_text())
    local=ROOT/'build/head70_peer_same_image'
    local_report=json.loads((local/'manifest.json').read_text())
    for name,key in [('fused','fused_peer_sha256'),('body','body_peer_sha256')]:
        if digest(source/f'{name}_peer.f32')!=report[key]:
            raise ValueError(f'public {name} hash differs')
    for name,key in [('enhanced','enhanced_rgb_sha256'),('final','final_rgb_sha256')]:
        if digest(source/f'{name}_rgb.f32')!=report[key]:
            raise ValueError(f'public {name} RGB hash differs')
    if report['model_sha256']!=local_report['peer_model_sha256'] or \
       not local_report['main_skip_common_frame']:
        raise ValueError('head fixture and public intermediates do not share input model')
    p=peer_to_native(np.arange(32))
    public_fused=np.fromfile(source/'fused_peer.f32','<f4').reshape(256,256,32)[CROP,CROP].reshape(64,32)
    public_body=np.fromfile(source/'body_peer.f32','<f4').reshape(256,256,32)[CROP,CROP].reshape(64,32)
    local_fused=np.fromfile(local/'outer/merged.f32','<f4').reshape(64,32)[:,p]
    local_body=np.fromfile(local/'body/body_device.f32','<f4').reshape(64,32)
    scale1=ROOT/'build/head70_peer_same_image_scale1'
    scale1_report=json.loads((scale1/'manifest.json').read_text())
    if scale1_report['input_scale']!=1.0 or \
       scale1_report['body_device_sha256']!=local_report['body_device_sha256']:
        raise ValueError('public-scale fixture uses different body or scale')
    public_enhanced=np.fromfile(source/'enhanced_rgb.f32','<f4').reshape(256,256,3)[CROP,CROP]
    public_final=np.fromfile(source/'final_rgb.f32','<f4').reshape(256,256,3)[CROP,CROP]
    color=np.fromfile(source/'color_linear.f32','<f4').reshape(256,256,3)[CROP,CROP]
    local_enhanced=np.fromfile(scale1/'outer/rgb.f32','<f4').reshape(8,8,3)
    blend=np.float32(.73974609375)
    local_final=local_enhanced+blend*(color-local_enhanced)
    raw=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    pad=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()[PAD_AT:PAD_AT+16]
    ordinary,*_=extract(raw,pad)
    weights=decode(np.frombuffer(ordinary,np.uint8))
    # The optimized ONNX fuse is float32. The original HIP head enters its
    # body after a half-rounded merge, so impose that boundary explicitly.
    native_entry=H(public_fused).reshape(1,64,32)
    stages=trace(native_entry,weights)
    body=OUT/'public_fused_body';save(body,dict(input=native_entry,
                                               weights=packed(weights),**stages,
                                               output=F(stages['body'])))
    subprocess.run([str(ROOT/'build/c32_peer_body_test.exe'),str(body),'1'],
                   cwd=ROOT,check=True)
    same_input_device=body/'body_device.f32'
    if same_input_device.read_bytes()!=(body/'body.f32').read_bytes():
        raise AssertionError('same-input HIP/scalar body differs')
    same_input=np.fromfile(same_input_device,'<f4').reshape(64,32)
    audit=dict(public_model_sha256=report['model_sha256'],
               public_fused_sha256=report['fused_peer_sha256'],
               public_body_sha256=report['body_peer_sha256'],
               local_merged_sha256=digest(local/'outer/merged.f32'),
               local_body_device_sha256=digest(local/'body/body_device.f32'),
               same_input_body_device_sha256=digest(same_input_device),
               crop_xy=[120,120],crop_extent=[8,8,32],
               merged_local_vs_public=metrics(local_fused,public_fused),
               half_entry_vs_public_fused=metrics(native_entry.reshape(64,32),public_fused),
               body_local_vs_public=metrics(local_body,public_body),
               body_same_input_vs_public=metrics(same_input,public_body),
               enhanced_scale1_vs_public=metrics(local_enhanced,public_enhanced),
               blended_scale1_vs_public_final=metrics(local_final,public_final),
               input_color_vs_public_final=metrics(color,public_final),
               public_blend_scale=float(blend),
               same_input_hip_scalar_exact=True,
               comparison='optimized public FP16 graph vs candidate HIP FP8/half arithmetic; public fused values half-rounded at native body entry',
               original_kernel_executed=False,original_runtime_validation=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':run()
