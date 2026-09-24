#!/usr/bin/env python3
"""Check block62 C128->C64 prefix with same-image public FP16 boundaries."""
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
from audit_peer_native_c32_basis import run as audit_basis,peer_to_native_multihead

SOURCE=ROOT/'build/peer_decoder62_inputs'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def project_sequential_f32(x,weights):
    """Mirror the HIP prefix's non-FMA f32 accumulation and 32-wide half steps."""
    if x.dtype!=np.float32 or weights.dtype!=np.float32 or x.shape[-1]!=128 or weights.shape!=(64,128):
        raise ValueError('unexpected projection input shape/dtype')
    flat=x.reshape(-1,128)
    acc=np.zeros((flat.shape[0],64),np.float32)
    for base in range(0,128,32):
        part=np.zeros_like(acc)
        for channel in range(base,base+32):
            product=np.multiply(flat[:,channel,None],weights[None,:,channel],dtype=np.float32)
            part=np.add(part,product,dtype=np.float32)
        acc=H(np.add(acc,part,dtype=np.float32))
    return acc.reshape(*x.shape[:-1],64)


def run(case):
    if case not in ('image_half','image_fp8','from56_fp8'):raise ValueError('bad case')
    audit_basis(verbose=False)
    src=json.loads((SOURCE/'manifest.json').read_text())
    for name in ('block61','skip8','merge62'):
        if digest(SOURCE/f'{name}_peer.f32')!=src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    xpeer=np.fromfile(SOURCE/'block61_peer.f32','<f4').reshape(32,32,128)
    speer=np.fromfile(SOURCE/'skip8_peer.f32','<f4').reshape(64,64,64)
    if case=='image_fp8':xpeer=F(xpeer);speer=F(speer)
    if case=='from56_fp8':speer=F(speer)
    p128=peer_to_native_multihead(np.arange(128))
    p64=peer_to_native_multihead(np.arange(64))
    x=np.empty_like(xpeer);x[...,p128]=xpeer
    upstream=None
    if case=='from56_fp8':
        audit_path=ROOT/'build/peer_decoder56_tail_audit/manifest.json'
        audit=json.loads(audit_path.read_text())
        input_path=ROOT/'build/decoder61_candidate_derived/image_fp8/output/output_device.f32'
        if audit['source_model_sha256']!=src['model_sha256'] or \
           audit['source_image_sha256']!=src['image_sha256'] or \
           digest(input_path)!=audit['block_output_stages']['block61']['candidate_native_sha256']:
            raise ValueError('upstream block61 provenance/hash differs')
        x=np.fromfile(input_path,'<f4').reshape(32,32,128)
        upstream=dict(audit_sha256=digest(audit_path),block61_native_sha256=digest(input_path),
                      boundary='AMD candidate block56-61 from public block55/skip14')
    skip=np.empty_like(speer);skip[...,p64]=speer
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_140.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=70048:raise ValueError('wrong block62 tensor size')
    c=64;ob=6;count=2*c*c
    rows=bits(count,[3]+list(range(6,ob+5)))
    cols=bits(count,[1,0,4,5,2]+list(range(ob+5,2*ob+1)))
    if np.unique(rows*128+cols).size!=count:raise ValueError('projection map collision')
    weights=np.empty((64,128),np.float32)
    weights[rows,cols]=e4m3fn(raw[0x7000:0x9000])
    order=peer_to_native_multihead(np.arange(c))
    scale=np.empty(c,np.float32)
    scale[order]=raw[0x9080:0x9100].view('<f2').astype(np.float32)
    low_numpy=multiply(x,weights)
    low=project_sequential_f32(x,weights)
    numpy_dot_disagreements=int(np.count_nonzero(low_numpy!=low))
    numpy_dot_max_abs=float(np.max(np.abs(low_numpy.astype(np.float64)-low.astype(np.float64))))
    merged_half=H(np.repeat(np.repeat(low,2,axis=0),2,axis=1)+skip*scale)
    merged=F(merged_half)
    out=ROOT/'build/upsample62_prefix_derived'/case
    out.mkdir(parents=True,exist_ok=True)
    for name,array in dict(input=x,weights=weights,scale=scale,skip=skip,
                           low=low,merged=merged).items():
        np.asarray(array,dtype='<f4').tofile(out/f'{name}.f32')
    result=subprocess.run([str(ROOT/'build/upsample62_prefix_test.exe'),str(out),'32','32'],
                          cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS')!=2 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out/'merged_device.f32').read_bytes()!=(out/'merged.f32').read_bytes():
        raise AssertionError('prefix HIP/scalar merge differs')
    public=np.fromfile(SOURCE/'merge62_peer.f32','<f4').reshape(64,64,64)
    report=dict(case=case,input_extent=[32,32,128],output_extent=[64,64,64],
                source_model_sha256=src['model_sha256'],
                source_block61_peer_sha256=src['tensor_sha256']['block61'],
                upstream_amd_block61=upstream,
                source_skip8_peer_sha256=src['tensor_sha256']['skip8'],
                block62_tensor_sha256=digest(raw_path),
                input_native_sha256=digest(out/'input.f32'),
                skip_native_sha256=digest(out/'skip.f32'),
                output_device_sha256=digest(out/'merged_device.f32'),
                hip_projection_merge_exact=True,
                reference_arithmetic='sequential separate f32 multiply/add, half after every 32 channels (HIP non-FMA contract)',
                numpy_dot_projection_disagreements=numpy_dot_disagreements,
                numpy_dot_projection_max_abs=numpy_dot_max_abs,
                merged_half_vs_public_fp16=metrics(merged_half[...,p64],public),
                merged_fp8_vs_public_fp16=metrics(merged[...,p64],public),
                basis='audited C128/C64 P; C64 attention residual still candidate',
                original_kernel_executed=False,original_runtime_validation=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,end='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('image_half','image_fp8','from56_fp8'),default='image_half')
    a=p.parse_args();run(a.case)
