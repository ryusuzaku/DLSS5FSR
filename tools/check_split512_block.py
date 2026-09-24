#!/usr/bin/env python3
"""Check a full unshifted C512 block against unchanged upstream arithmetic."""
from pathlib import Path
import hashlib
import json
import argparse
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
import native_split_reference as R
from decode_tinlayout_global import e4m3fn


def attention_trace(feature,qkv_weights,bias,scales):
    if feature.ndim!=2 or feature.shape[1]!=512 or len(feature)%64:
        raise ValueError('expected complete 8x8 windows')
    windows=len(feature)//64
    qkv=np.stack([R.multiply(feature,m) for m in qkv_weights],axis=1)
    normalized=np.empty_like(qkv)
    scores=np.empty((windows,16,64,64),np.float32)
    exponents=np.empty_like(scores)
    probabilities=np.empty_like(scores)
    context=np.empty((windows*64,512),np.float32)
    for win in range(windows):
        rows=slice(win*64,(win+1)*64)
        for head in range(16):
            sl=slice(head*32,(head+1)*32)
            q=R.F(R.H(R.normalize(qkv[rows,0,sl])*R.H(scales[head])))
            k=R.F(R.normalize(qkv[rows,1,sl]))
            v=R.F(qkv[rows,2,sl])
            normalized[rows,0,sl]=q;normalized[rows,1,sl]=k;normalized[rows,2,sl]=v
            score=R.H(q@k.T+bias[head]);scores[win,head]=score
            aexp=np.clip(R.H(score*np.float32(.044921875)+np.float32(1.30078125)),
                         1.03125,1.5693359375)
            bits=aexp.astype(np.float16).view(np.uint16).astype(np.uint32)
            exp=(((bits<<5)+0x8000)&65535).astype(np.uint16).view(np.float16).astype(np.float32)
            exponents[win,head]=exp
            prob=R.F(R.H(exp*R.H(1/R.denominator(exp))))
            probabilities[win,head]=prob
            context[rows,sl]=R.F(R.H(R.H(prob[:,:32]@v[:32])+prob[:,32:]@v[32:]))
    np.testing.assert_array_equal(context,R.attention(feature.reshape(windows,64,512),list(qkv_weights),bias,scales).reshape(-1,512))
    return dict(qkv=qkv,normalized=normalized,scores=scores,exponents=exponents,
                probabilities=probabilities,context=context)


def run(decoder=False):
    exe=ROOT/'build/split512_block_test.exe'
    if not exe.is_file():raise FileNotFoundError(f'build with bash tools/build_split512_block.sh: {exe}')
    records={v['name']:v for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    blocks=range(40,48) if decoder else (*range(23,31),40)
    for block in blocks:
        seed=51200+block
        record=records[f'block{block}.layer0.layer']
        raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin").read_bytes()
        if len(raw)!=524288:raise ValueError(f'block{block} wrong raw size')
        matrix=R.matrix(np.frombuffer(raw[:262144],np.uint8))
        if matrix.shape!=(512,512) or not np.isfinite(matrix).all():raise ValueError('bad decoded matrix')
        u=np.frombuffer(raw,np.uint8)
        expand=np.empty((8,256,64),np.float32)
        contract=np.empty((8,64,256),np.float32)
        for g in range(8):
            expand[g,R.bits(16384,[3,6,7,8,9,10,11,12]),R.bits(16384,[1,0,4,5,2,13])]=e4m3fn(u[0x40000+g*16384:0x40000+(g+1)*16384])
            contract[g,R.bits(16384,[3,6,7,8,9,10]),R.bits(16384,[1,0,4,5,2,11,12,13])]=e4m3fn(u[0x60000+g*16384:0x60000+(g+1)*16384])
        # FP8 input mirrors the quantized boundary before this kernel.
        rng=np.random.default_rng(seed)
        windows=1 if block==23 else 2
        x=R.F(rng.normal(0,.25,(windows*64,512)).astype(np.float32))
        expected=R.multiply(x,matrix)
        branch=R.ffwd(x,{'pre':matrix,'expand':expand,'contract':contract})
        proj_record=records[f'block{block}.layer1.layer']
        proj_raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{proj_record['index']:03d}.bin").read_bytes()
        if len(proj_raw)!=263168:raise ValueError(f'block{block} wrong projection size')
        ffn_projection=R.matrix(np.frombuffer(proj_raw[:262144],np.uint8))
        channels=np.arange(512)
        order=(channels//16)*16+(channels%8)*2+(channels%16//8)
        skip=np.empty(512,np.float32)
        skip[order]=np.frombuffer(proj_raw[262144:],'<f2').astype(np.float32)
        feature=R.F(R.multiply(branch,ffn_projection,R.H(x*skip)))
        attention_record=records[f'block{block}.layer2.layer']
        attention_raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{attention_record['index']:03d}.bin").read_bytes()
        if len(attention_raw)!=917568:raise ValueError(f'block{block} wrong attention size')
        a=np.frombuffer(attention_raw,np.uint8)
        i=np.arange(262144)
        offsets=(i//1024)*3072+2048+i%1024
        qkv_weights=np.stack([R.matrix(a[offsets+delta]) for delta in (-2048,-1024,0)])
        bias=np.empty((16,64,64),np.float32)
        bias[R.bits(65536,[12,13,14,15]),R.bits(65536,[5,6,10,7,1,11]),
             R.bits(65536,[0,3,8,4,2,9])]=np.frombuffer(attention_raw[0xc0000:0xe0000],'<f2')
        scales=np.frombuffer(attention_raw[0xe0000:],'<f4')
        trace=attention_trace(feature,qkv_weights,bias,scales)
        context=trace['context']
        final_record=records[f'block{block}.layer3.layer']
        final_record_bytes=(ROOT/'dlss5-analysis/tensors'/f"tensor_{final_record['index']:03d}.bin").read_bytes()
        if len(final_record_bytes)!=263168:raise ValueError(f'block{block} wrong final projection size')
        final_weights=R.matrix(np.frombuffer(final_record_bytes[:262144],np.uint8))
        final_skip=np.empty(512,np.float32)
        final_skip[order]=np.frombuffer(final_record_bytes[262144:],'<f2').astype(np.float32)
        final_raw=R.multiply(context,final_weights,R.H(feature*final_skip))
        final=R.F(final_raw)
        folder=ROOT/'build/split512_block'/f'block{block}'
        folder.mkdir(parents=True,exist_ok=True)
        for name,arr in (('input',x),('matrix',matrix),('expected',expected),
                         ('expand',expand),('contract',contract),('branch',branch),
                         ('ffn_projection',ffn_projection),('ffn_skip',skip),('feature',feature),
                         ('qkv_weights',qkv_weights),('scales',scales),('bias',bias),
                         ('final_weights',final_weights),('final_skip',final_skip),
                         ('final_raw',final_raw),('final',final),
                         *trace.items()):
            if not np.isfinite(arr).all():raise ValueError(f'block{block} nonfinite {name}')
            np.asarray(arr,dtype='<f4').tofile(folder/f'{name}.f32')
        manifest={'block':block,'tensor':record['index'],'raw_sha256':hashlib.sha256(raw).hexdigest(),
                  'projection_tensor':proj_record['index'],'projection_sha256':hashlib.sha256(proj_raw).hexdigest(),
                  'attention_tensor':attention_record['index'],'attention_sha256':hashlib.sha256(attention_raw).hexdigest(),
                  'final_tensor':final_record['index'],'final_sha256':hashlib.sha256(final_record_bytes).hexdigest(),
                  'oracle':'unchanged native_split_reference and native_c64_reference.multiply',
                  'tokens':len(x),'windows':windows,'comparison':'exact numeric equality',
                  'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.f32')}}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result=subprocess.run([str(exe),str(folder),str(len(x))],cwd=ROOT)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--decoder',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.decoder))
