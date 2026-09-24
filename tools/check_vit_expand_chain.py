#!/usr/bin/env python3
"""Check connected ViT blocks using a declared control or derived bridge."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from recover_vit_bridge_ptx import ROOT,recover
from audit_native_vit_logical_map import logical_map

sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
import native_vit_linear_reference as V
import native_vit_qkv_reference as Q
import native_vit_attention_reference as A
import vit_attention16_candidate as A16
from native_c64_reference import multiply
from native_c32_reference import H


def run_block(block,width=8,height=4,derived=False,image_source=None,output_root=None):
    exe=ROOT/'build/vit_expand_chain_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    if (width,height) not in ((4,4),(8,4),(16,4)):raise ValueError('covered extents are 4x4, 8x4 and 16x4')
    tokens=width*height
    if block==31:
        if image_source is not None:
            if (width,height)!=(4,4) or not derived:
                raise ValueError('image source requires the 4x4 derived-map candidate')
            source_path=Path(image_source)
            if not source_path.is_file():raise FileNotFoundError(source_path)
            head_path=None
        else:
            if width==4:raise ValueError('4x4 requires an explicit image source')
            head_path=ROOT/'build'/('split512_bridge' if width==8 else 'split512_bridge_32x8')/'head_device.f32'
            if not head_path.is_file():raise FileNotFoundError('run python tools/check_split512_bridge.py first')
            head=np.fromfile(head_path,'<f4')
            gather=logical_map(tokens) if derived else recover(width,height)[0]
            if len(head)!=len(gather):raise ValueError('head/map extent mismatch')
            cases='vit_bridge_derived_cases' if derived else 'vit_bridge_ptx_cases'
            source_path=ROOT/'build'/cases/f'block30_head_{width}x{height}'/'device.f32'
            if not source_path.is_file():raise FileNotFoundError(f'run the {cases} bridge check first')
            mapped=np.fromfile(source_path,'<f4')
            if mapped.tobytes()!=np.asarray(head[gather],dtype='<f4').tobytes():
                raise ValueError('HIP bridge output differs bytewise from PTX map gather')
    else:
        if tokens==16 and image_source is not None and derived:
            source_path=Path(image_source)
        elif tokens==64:
            suffix='_derived' if derived else ''
            previous=f'vit_expand_chain_16x4{suffix}' if block==32 else f'vit_block{block-1}_16x4{suffix}'
            source_path=ROOT/'build'/previous/'projection_device.f32'
        else:
            raise ValueError('connected ViT attention requires 64 tokens or explicit 4x4 candidate source')
        if not source_path.is_file():raise FileNotFoundError(f'run block{block-1} first')
    x=np.fromfile(source_path,'<f4')
    if x.size!=tokens*1024:raise ValueError('wrong ViT source extent')
    x=x.reshape(tokens,1024)
    records={v['name']:v for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    record=records[f'block{block}.layer0.layer']
    raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin").read_bytes()
    if len(raw)!=4194320 or any(raw[4194304:]):
        raise ValueError(f'wrong block{block} expansion size or nonzero trailer')
    weight=V.matrix(np.frombuffer(raw[:4194304],np.uint8),1024,4096)
    expanded=multiply(x,weight)
    hidden=V.expand(x,weight)
    contract_record=records[f'block{block}.layer1.layer']
    contract_raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{contract_record['index']:03d}.bin").read_bytes()
    if len(contract_raw)!=4196352:
        raise ValueError(f'wrong block{block} contraction size')
    contract_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{contract_record['index']:03d}.bin"
    contract_weights,contract_skip=V.unpack_residual(contract_path,4096)
    contract=V.residual_projection(hidden,x,contract_weights,contract_skip)
    qkv_record=records[f'block{block}.layer2.layer']
    qkv_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{qkv_record['index']:03d}.bin"
    qkv_raw=qkv_path.read_bytes()
    qkv_weights,qkv_scales=Q.unpack(qkv_path)
    qkv_projected=np.stack([H(multiply(contract[:,:512],m[:,:512])+
                               multiply(contract[:,512:],m[:,512:]))
                            for m in qkv_weights],axis=1).reshape(tokens,3,1024)
    q,k,v=Q.qkv(contract,qkv_weights,qkv_scales)
    qkv=np.stack((q,k,v),axis=1)
    attention_data={}
    if tokens in (16,64):
        qb,kb=(a.reshape(tokens,32,32).transpose(1,0,2) for a in (q,k))
        scores=H(qb@kb.transpose(0,2,1))
        coefficient=np.array([0x2dbb],np.uint16).view(np.float16).astype(np.float32)[0]
        affine=np.clip(H(scores*coefficient+np.float32(1.708984375)),
                       1.439453125,1.9775390625)
        bits=affine.astype(np.float16).view(np.uint16).astype(np.uint32)
        exponents=(((bits<<4)+0x4000)&65535).astype(np.uint16).view(np.float16).astype(np.float32)
        if tokens==16:
            candidate_scores,candidate_exponents,attention=A16.reference(q,k,v)
            np.testing.assert_array_equal(candidate_scores,scores)
            np.testing.assert_array_equal(candidate_exponents,exponents)
        else:
            attention=A.attention(q,k,v)
        projection_record=records[f'block{block}.layer4.layer']
        projection_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{projection_record['index']:03d}.bin"
        projection_raw=projection_path.read_bytes()
        if len(projection_raw)!=1050624:raise ValueError(f'wrong block{block} projection size')
        projection_weights,projection_skip=V.unpack_residual(projection_path,1024)
        projection=V.residual_projection(attention,contract,projection_weights,projection_skip)
        attention_data=dict(scores=scores,exponents=exponents,attention=attention,
                            projection_weights=projection_weights,projection_skip=projection_skip,
                            projection=projection)
    if not all(np.isfinite(a).all() for a in
               (x,weight,expanded,hidden,contract_weights,contract_skip,contract,
                *qkv_weights,qkv_scales,qkv_projected,qkv,*attention_data.values())):
        raise ValueError(f'nonfinite ViT block{block} stage')
    suffix='_derived' if derived else ''
    if output_root is not None:
        folder=Path(output_root)
    elif block==31:
        folder=ROOT/'build'/(('vit_expand_chain' if tokens==32 else 'vit_expand_chain_16x4')+suffix)
    else:
        folder=ROOT/'build'/f'vit_block{block}_16x4{suffix}'
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in (('input',x),('weights',weight),('expanded',expanded),('hidden',hidden),
                       ('contract_weights',contract_weights),('contract_skip',contract_skip),
                       ('contract',contract),('qkv_weights',np.stack(qkv_weights)),
                       ('qkv_scales',qkv_scales),('qkv_projected',qkv_projected),('qkv',qkv),
                       *attention_data.items()):
        np.asarray(array,dtype='<f4').tofile(folder/f'{name}.f32')
    if (folder/'input.f32').read_bytes()!=source_path.read_bytes():
        raise ValueError(f'block{block} device handoff bytes changed')
    report=dict(block=block,source_device=str(source_path),
                source_device_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                expansion_tensor=record['index'],expansion_sha256=hashlib.sha256(raw).hexdigest(),
                contraction_tensor=contract_record['index'],
                contraction_sha256=hashlib.sha256(contract_raw).hexdigest(),
                qkv_tensor=qkv_record['index'],qkv_sha256=hashlib.sha256(qkv_raw).hexdigest(),
                bridge=('prior candidate ViT device projection; original physical bridge unverified'
                        if image_source is not None and block>31 else
                        'candidate 4x4 logical C512/ViT map; original physical bridge unverified'
                        if image_source is not None else
                        'upstream capture-derived logical map composed with original PTX physical source; no original-kernel execution'
                        if derived else 'PTX source-linear map on logical-HWC control; C512 split-view composition missing; no original-kernel execution'),
                oracle='unchanged native_vit_linear_reference, native_vit_qkv_reference, native_vit_attention_reference, and native_c64_reference.multiply',
                comparison='exact')
    if tokens==16:
        report['attention_contract']='16 valid keys; float64 sum then half denominator, FP8 exponent-value product; candidate only, original reduction unverified'
        report['oracle']='native ViT linear/QKV scalar references plus vit_attention16_candidate; exact candidate comparison, not original parity'
    if block==31 and head_path is not None:report['head_sha256']=hashlib.sha256(head_path.read_bytes()).hexdigest()
    if tokens in (16,64):
        report['projection_tensor']=projection_record['index']
        report['projection_sha256']=hashlib.sha256(projection_raw).hexdigest()
    subprocess.run([str(exe),str(folder),str(tokens)],cwd=ROOT,check=True)
    report['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in folder.glob('*.f32')}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return folder


def run(width=8,height=4,last_block=31,derived=False):
    if last_block not in range(31,39):raise ValueError('last block must be 31..38')
    if last_block>31 and width!=16:raise ValueError('ViT chain requires 16x4/64 tokens')
    for block in range(31,last_block+1):run_block(block,width,height,derived)
    if last_block>31:print(f'ViT block31 -> block{last_block}: PASS ({last_block-31} byte-exact device handoffs)')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=8)
    p.add_argument('--derived',action='store_true',help='use the source-composed C512 logical bridge')
    p.add_argument('--last-block',type=int,default=31);a=p.parse_args()
    raise SystemExit(run(a.width,last_block=a.last_block,derived=a.derived))
