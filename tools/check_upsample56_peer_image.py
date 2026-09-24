#!/usr/bin/env python3
"""Check block56 C256→C128 prefix on same-image public FP16 inputs.

This is a candidate C256 basis extension with native-style FP8 input
boundaries, not an original NVIDIA-kernel oracle.
"""
from pathlib import Path
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
from audit_peer_native_c32_basis import peer_to_native_multihead

SOURCE=ROOT/'build/peer_decoder56_inputs'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def run():
    src=json.loads((SOURCE/'manifest.json').read_text())
    for name in ('block55','skip14','merge56'):
        if digest(SOURCE/f'{name}_peer.f32')!=src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    xpeer=F(np.fromfile(SOURCE/'block55_peer.f32','<f4').reshape(16,16,256))
    speer=F(np.fromfile(SOURCE/'skip14_peer.f32','<f4').reshape(32,32,128))
    p256=peer_to_native_multihead(np.arange(256))
    p128=peer_to_native_multihead(np.arange(128))
    x=np.empty_like(xpeer);x[...,p256]=xpeer
    skip=np.empty_like(speer);skip[...,p128]=speer
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_133.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=230176:raise ValueError('wrong block56 tensor size')
    c=128;ob=7;count=2*c*c
    rows=bits(count,[3]+list(range(6,ob+5)))
    cols=bits(count,[1,0,4,5,2]+list(range(ob+5,2*ob+1)))
    if np.unique(rows*256+cols).size!=count:raise ValueError('projection map collision')
    weights=np.empty((128,256),np.float32)
    weights[rows,cols]=e4m3fn(raw[0x18000:0x20000])
    scale=np.empty(c,np.float32)
    scale[p128]=raw[0x20100:0x20200].view('<f2').astype(np.float32)
    low=multiply(x,weights)
    merged_half=H(np.repeat(np.repeat(low,2,axis=0),2,axis=1)+skip*scale)
    merged=F(merged_half)
    out=ROOT/'build/upsample56_prefix_derived/image_fp8'
    out.mkdir(parents=True,exist_ok=True)
    for name,array in dict(input=x,weights=weights,scale=scale,skip=skip,
                           low=low,merged=merged).items():
        np.asarray(array,dtype='<f4').tofile(out/f'{name}.f32')
    result=subprocess.run([str(ROOT/'build/upsample56_prefix_test.exe'),str(out),'16','16'],
                          cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS')!=2 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out/'merged_device.f32').read_bytes()!=(out/'merged.f32').read_bytes():
        raise AssertionError('prefix HIP/scalar merge differs')
    public=np.fromfile(SOURCE/'merge56_peer.f32','<f4').reshape(32,32,128)
    report=dict(case='image_fp8',input_extent=[16,16,256],output_extent=[32,32,128],
                source_model_sha256=src['model_sha256'],
                source_block55_peer_sha256=src['tensor_sha256']['block55'],
                source_skip14_peer_sha256=src['tensor_sha256']['skip14'],
                block56_tensor_sha256=digest(raw_path),
                input_native_sha256=digest(out/'input.f32'),
                skip_native_sha256=digest(out/'skip.f32'),
                output_device_sha256=digest(out/'merged_device.f32'),
                hip_projection_merge_exact=True,
                merged_half_vs_public_fp16=metrics(merged_half[...,p128],public),
                merged_fp8_vs_public_fp16=metrics(merged[...,p128],public),
                basis='C256 P extension candidate; C128 body measured except attention residual',
                original_kernel_executed=False,original_runtime_validation=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,end='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
