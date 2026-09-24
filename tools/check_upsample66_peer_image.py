#!/usr/bin/env python3
"""Check block66 C64->C32 prefix on same-image public FP16 inputs.

The peer-half and native-style FP8 input-boundary cases are separate
diagnostics. Neither is an original NVIDIA-kernel oracle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
from native_split_reference import bits
from native_c64_reference import multiply
from native_c32_reference import H,F
from decode_tinlayout_global import e4m3fn
from audit_peer_native_c32_basis import run as audit_basis,peer_to_native,peer_to_native_multihead

SOURCE=ROOT/'build/peer_decoder66_inputs'
FROM39=Path.home()/'DLSS5FSR-build-offload'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def run(case):
    if case not in ('image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'):raise ValueError('bad case')
    audit_basis(verbose=False)
    src=json.loads((SOURCE/'manifest.json').read_text())
    for name in ('block65','skip4','merge66'):
        if digest(SOURCE/f'{name}_peer.f32')!=src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    xpeer=np.fromfile(SOURCE/'block65_peer.f32','<f4').reshape(64,64,64)
    speer=np.fromfile(SOURCE/'skip4_peer.f32','<f4').reshape(128,128,32)
    if case=='image_fp8':xpeer=F(xpeer);speer=F(speer)
    if case in ('from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'):speer=F(speer)
    p64=peer_to_native_multihead(np.arange(64))
    p32=peer_to_native(np.arange(32))
    x=np.empty_like(xpeer);x[...,p64]=xpeer
    upstream=None
    if case in ('from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'):
        upstream_case='image_fp8' if case=='from62_fp8' else case
        audit_dir=('peer_decoder62_tail_audit' if case=='from62_fp8' else
                   'peer_decoder62_tail_audit_from56' if case=='from56_fp8' else
                   'peer_decoder62_tail_audit_from48' if case=='from48_fp8' else
                   f'peer_decoder62_tail_audit_{case.removesuffix("_fp8")}')
        audit_path=ROOT/'build'/audit_dir/'manifest.json'
        audit=json.loads(audit_path.read_text())
        input_path=(FROM39/f'peer_decoder62_{case.removesuffix("_fp8")}'/'block65/output/output_device.f32'
                    if case in ('from39_fp8','from38_fp8') else
                    ROOT/'build/decoder65_candidate_derived'/upstream_case/'output/output_device.f32')
        if audit['source_model_sha256']!=src['model_sha256'] or \
           audit['source_image_sha256']!=src['image_sha256'] or \
           digest(input_path)!=audit['block_output_stages']['block65']['candidate_native_sha256']:
            raise ValueError('upstream block65 provenance/hash differs')
        x=np.fromfile(input_path,'<f4').reshape(64,64,64)
        upstream=dict(audit_sha256=digest(audit_path),
                      block65_native_sha256=digest(input_path),
                      boundary=('AMD candidate block62-65 from public block61/skip8'
                                if case=='from62_fp8' else
                                'AMD candidate block56-65 from public block55/skip14 and skip8'
                                if case=='from56_fp8' else
                                'AMD candidate block48-65 from public block47/skip22/skip14/skip8'
                                if case=='from48_fp8' else
                                'AMD candidate block40-65 from public block39 and same-image skips'
                                if case=='from39_fp8' else
                                'AMD candidate block39-65 from public ViT38/skip30 and same-image skips'))
    skip=np.empty_like(speer);skip[...,p32]=speer
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_144.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=22784:raise ValueError('wrong block66 tensor size')
    rows=bits(2048,[3,6,7,8,9]);cols=bits(2048,[1,0,4,5,2,10])
    if np.unique(rows*64+cols).size!=2048:raise ValueError('projection map collision')
    matrix=np.empty((32,64),np.float32)
    matrix[rows,cols]=e4m3fn(raw[0x2000:0x2800])
    order=np.array([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
                    16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31])
    c=np.arange(32);other=(c//16)*16+(c%8)*2+(c%16//8)
    weights=np.empty_like(matrix);weights[order]=matrix[other]
    scale=np.empty(32,np.float32)
    scale[order]=raw[0x2860:0x28a0].view('<f2').astype(np.float32)
    low=multiply(x,weights)
    merged=H(np.repeat(np.repeat(low,2,axis=0),2,axis=1)+skip*scale)
    out=(FROM39/f'upsample66_{case.removesuffix("_fp8")}' if case in ('from39_fp8','from38_fp8') else
         ROOT/'build/upsample66_prefix_derived'/case)
    out.mkdir(parents=True,exist_ok=True)
    for name,array in dict(input=x,weights=weights,scale=scale,skip=skip,
                           low=low,merged=merged).items():
        np.asarray(array,dtype='<f4').tofile(out/f'{name}.f32')
    result=subprocess.run([str(ROOT/'build/upsample66_prefix_test.exe'),str(out),'64','64'],
                          cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS')!=2 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out/'merged_device.f32').read_bytes()!=(out/'merged.f32').read_bytes():
        raise AssertionError('prefix HIP/scalar merge differs')
    public=np.fromfile(SOURCE/'merge66_peer.f32','<f4').reshape(128,128,32)
    candidate=merged[...,p32]
    report=dict(case=case,input_extent=[64,64,64],output_extent=[128,128,32],
                source_model_sha256=src['model_sha256'],
                source_block65_peer_sha256=src['tensor_sha256']['block65'],
                upstream_amd_block65=upstream,
                source_skip4_peer_sha256=src['tensor_sha256']['skip4'],
                block66_tensor_sha256=digest(raw_path),
                input_native_sha256=digest(out/'input.f32'),
                skip_native_sha256=digest(out/'skip.f32'),
                output_device_sha256=digest(out/'merged_device.f32'),
                hip_projection_merge_exact=True,
                merged_vs_public_fp16=metrics(candidate,public),
                merged_vs_half_public_fp16=metrics(candidate,H(public)),
                basis='audited C64 P and C32 S; C32 body still candidate',
                original_kernel_executed=False,original_runtime_validation=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,end='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'),default='image_half')
    a=p.parse_args();run(a.case)
