#!/usr/bin/env python3
"""Check provisional C256 attention through the final residual output.

The C256 matrix/bias maps extend upstream's measured C64/C128 bit rules.
Static PTX supports transferring the FFN residual order into attention,
but the logical C256 channel map has not been validated by an original kernel.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'ref/dlss5-port/Development'
sys.path.insert(0,str(UPSTREAM))
from native_c64_reference import multiply,normalize,denominator
from native_c32_reference import F,H
from decode_tinlayout_global import e4m3fn
from check_c256_ffn_candidate import bits
from audit_c256_residual_ptx import run as audit_residual_ptx


def candidate_attention_maps():
    count=256*256;source=np.arange(count,dtype=np.int32)
    inputs=bits(count,[1,0,4,5,2,13,14,15])
    outputs=bits(count,[3,6,7,8,9,10,11,12])
    if len(np.unique(outputs*256+inputs))!=count:
        raise ValueError('candidate C256 attention matrix map collides')
    bias_count=8*4096
    heads=bits(bias_count,[12,13,14])
    queries=bits(bias_count,[5,6,10,7,1,11])
    keys=bits(bias_count,[0,3,8,4,2,9])
    if len(np.unique(heads*4096+queries*64+keys))!=bias_count:
        raise ValueError('candidate C256 bias map collides')
    offsets=(source//1024)*3072+2048+source%1024
    return inputs,outputs,heads,queries,keys,offsets


def decode_attention(raw,maps):
    if len(raw)==820784:
        qkv_offset,bias_offset,scale_offset,projection_offset,skip_offset=(
            0x78400,0xa8400,0xb8400,0xb8420,0xc8420)
    elif len(raw)==689232:
        qkv_offset,bias_offset,scale_offset,projection_offset,skip_offset=(
            0x58220,0x88220,0x98220,0x98240,0xa8240)
    else:raise ValueError('wrong C256 tensor size')
    inputs,outputs,heads,queries,keys,offsets=maps
    qkv=[]
    for delta in (-2048,-1024,0):
        matrix=np.empty((256,256),np.float32)
        matrix[outputs,inputs]=e4m3fn(raw[qkv_offset+offsets+delta])
        qkv.append(matrix)
    bias=np.empty((8,64,64),np.float32)
    bias[heads,queries,keys]=raw[bias_offset:scale_offset].view('<f2').astype(np.float32)
    scales=raw[scale_offset:scale_offset+32].view('<f4').astype(np.float32)
    projection=np.empty((256,256),np.float32)
    projection[outputs,inputs]=e4m3fn(raw[projection_offset:projection_offset+65536])
    # PTX establishes identical paired-half load/accumulator order for FFN
    # and attention residuals. The FFN raw-to-logical C256 map is still a
    # C64/C128 extension, so this attention order remains a candidate too.
    order=(np.arange(256)//16)*16+(np.arange(256)%8)*2+(np.arange(256)%16//8)
    skip=np.empty(256,np.float32)
    skip[order]=raw[skip_offset:skip_offset+512].view('<f2').astype(np.float32)
    return np.stack(qkv),bias,scales,projection,skip


def reference(feature,weights,bias,scales,projection,skip):
    channels=feature.shape[-1]
    heads=channels//32
    qkv=np.stack([multiply(feature,weights[m]) for m in range(3)],axis=1)
    normalized=np.empty_like(qkv)
    scores=np.empty((heads,64,64),np.float32)
    exponents=np.empty_like(scores)
    probabilities=np.empty_like(scores)
    context=np.empty((64,channels),np.float32)
    for head in range(heads):
        sl=slice(head*32,(head+1)*32)
        q=F(H(normalize(qkv[:,0,sl])*H(scales[head])))
        k=F(normalize(qkv[:,1,sl]))
        v=F(qkv[:,2,sl])
        normalized[:,0,sl]=q;normalized[:,1,sl]=k;normalized[:,2,sl]=v
        score=H(q@k.T+bias[head]);scores[head]=score
        affine=np.clip(H(score*np.float32(.044921875)+np.float32(1.30078125)),
                       1.03125,1.5693359375)
        halfbits=affine.astype(np.float16).view(np.uint16).astype(np.uint32)
        exp=(((halfbits<<5)+0x8000)&65535).astype(np.uint16).view(np.float16).astype(np.float32)
        exponents[head]=exp
        prob=F(H(exp*H(1/denominator(exp))))
        probabilities[head]=prob
        context[:,sl]=F(H(H(prob[:,:32]@v[:32])+prob[:,32:]@v[32:]))
    return dict(qkv=qkv,normalized=normalized,scores=scores,exponents=exponents,
                probabilities=probabilities,context=context,
                projection_linear=multiply(context,projection),
                projection_residual=F(multiply(context,projection,H(feature*skip))))


def run(derived=False,all_windows=False):
    exe=ROOT/'build/c256_attention_candidate_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with Git Bash tools/build_split512_block.sh')
    source_script=UPSTREAM/'derive_native_attention_layout.py'
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_124.bin'
    raw=np.frombuffer(raw_path.read_bytes(),np.uint8)
    evidence=audit_residual_ptx()
    if not evidence['same_relative_addresses_all_8_warps_32_lanes']:
        raise ValueError('PTX residual order transfer failed')
    weights,bias,scales,projection,skip=decode_attention(raw,candidate_attention_maps())
    input_root=ROOT/'build'/('c256_ffn_candidate_derived' if derived else 'c256_ffn_candidate')
    output_root=ROOT/'build'/('c256_attention_candidate_derived' if derived else 'c256_attention_candidate')
    cases=(('seeded',0),) if all_windows else (('zero',0),('seeded',0),('seeded',3))
    for case,shift in cases:
        leaf=f'{case}-s{shift}'+('-all' if all_windows else '')
        source=input_root/leaf/'feature_device.f32'
        if not source.is_file():raise FileNotFoundError('run python tools/check_c256_ffn_candidate.py first')
        if source.read_bytes()!=(source.parent/'feature.f32').read_bytes():
            raise ValueError('candidate C256 FFN device feature differs from reference')
        feature=np.fromfile(source,'<f4').reshape(-1,256)
        if len(feature)%64:raise ValueError('feature token count is not whole windows')
        windows=len(feature)//64
        chunks=[reference(feature[i:i+64],weights,bias,scales,projection,skip)
                for i in range(0,len(feature),64)]
        stages={name:np.concatenate([item[name] for item in chunks],axis=0)
                for name in chunks[0]}
        folder=output_root/leaf
        folder.mkdir(parents=True,exist_ok=True)
        for name,array in dict(feature=feature,qkv_weights=weights,bias=bias,
                               scales=scales,projection_weights=projection,
                               attention_skip=skip,**stages).items():
            if not np.isfinite(array).all():raise ValueError(f'nonfinite {case}-s{shift} {name}')
            np.asarray(array,dtype='<f4').tofile(folder/f'{name}.f32')
        subprocess.run([str(exe),str(folder),str(windows)],cwd=ROOT,check=True)
        for name in ('context','projection_linear','projection_residual'):
            if (folder/f'{name}_device.f32').read_bytes()!=(folder/f'{name}.f32').read_bytes():
                raise AssertionError(f'candidate {name} device bytes differ')
        report=dict(case=case,shift=shift,tokens=len(feature),windows=windows,channels=256,heads=8,
                    tensor_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    input_device_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    candidate_map_source=str(source_script.relative_to(ROOT)),
                    candidate_map_source_sha256=hashlib.sha256(source_script.read_bytes()).hexdigest(),
                    comparison='exact against upstream logical arithmetic under candidate maps',
                    scope='QKV through attention residual/projection; candidate channel order',
                    original_kernel_executed=False,original_runtime_validation=False,
                    map_status='candidate C256 extension of measured C64/C128 bit rules',
                    residual_map_evidence=str((ROOT/'build/c256_residual_ptx_audit.json').relative_to(ROOT)))
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    p.add_argument('--all-windows',action='store_true')
    a=p.parse_args();raise SystemExit(run(a.derived,a.all_windows))
