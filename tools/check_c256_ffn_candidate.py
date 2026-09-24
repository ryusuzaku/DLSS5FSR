#!/usr/bin/env python3
"""Check an explicitly provisional C256 FFN map on block48 device windows.

The maps extend upstream's measured C64/C128 address-bit rules to C256.
Their C256 original-kernel arithmetic has not been independently measured.
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
from native_c64_reference import multiply
from native_c32_reference import F,H
from decode_tinlayout_global import e4m3fn


def bits(count,positions):
    source=np.arange(count,dtype=np.int32)
    result=np.zeros(count,np.int32)
    for target,position in enumerate(positions):result|=((source>>position)&1)<<target
    return result


def candidate_ffn_maps():
    # Literal C=256 instance of upstream derive_native_ffn_layout.py.
    c=256;d=8;group=list(range(12,d+7))
    maps=dict(
        w1_input=bits(4*c*c,[1,0,4,5,2]+group),
        w1_hidden=bits(4*c*c,[3,6,7,8,9,10,11]+list(range(d+7,2*d+2))),
        w2_hidden=bits(128*c,[1,0,4,5,2,10,11]+group),
        w2_output=bits(128*c,[3,6,7,8,9]+group),
        w3_input=bits(c*c,[1,0,4,5,2]+list(range(d+5,2*d))),
        w3_output=bits(c*c,[3,6,7,8,9]+list(range(10,d+5))))
    for label,rows,cols,width,count in (
            ('W1',maps['w1_hidden'],maps['w1_input'],c,4*c*c),
            ('W2',maps['w2_output'],maps['w2_hidden'],4*c,128*c),
            ('W3',maps['w3_output'],maps['w3_input'],c,c*c)):
        if len(np.unique(rows*width+cols))!=count:
            raise ValueError(f'{label} candidate map collides')
    if not np.array_equal(np.bincount(maps['w2_output'],minlength=c),np.full(c,128)):
        raise ValueError('W2 candidate row capacity is wrong')
    return maps


def decode_ffn(raw,maps):
    if len(raw)==820784:skip_offset=0x78000  # block48 upsample
    elif len(raw) in (689232,820288):skip_offset=0x58010  # ordinary or encoder22 DS block
    else:raise ValueError('wrong C256 tensor size')
    result=[]
    for begin,end,shape,row_key,col_key in (
            (0,262144,(1024,256),'w1_hidden','w1_input'),
            (262144,294912,(256,1024),'w2_output','w2_hidden'),
            (294912,360448,(256,256),'w3_output','w3_input')):
        matrix=np.zeros(shape,np.float32)
        matrix[maps[row_key],maps[col_key]]=e4m3fn(raw[begin:end])
        result.append(matrix)
    order=(np.arange(256)//16)*16+(np.arange(256)%8)*2+(np.arange(256)%16//8)
    skip=np.empty(256,np.float32)
    skip[order]=raw[skip_offset:skip_offset+512].view('<f2').astype(np.float32)
    return (*result,skip)


def reference(x,w1,w2,w3,skip):
    expanded=multiply(x,w1)
    gate=np.clip(expanded,-4,4)
    poly=H(gate*H(np.abs(gate)*np.float32(-.055908203125)+np.float32(.447265625))+
           np.float32(.89453125))
    hidden=F(H(expanded*poly))
    middle=F(multiply(hidden,w2))
    feature=F(multiply(middle,w3,H(x*skip)))
    return dict(expanded=expanded,hidden=hidden,middle=middle,feature=feature)


def run(derived=False,all_windows=False):
    exe=ROOT/'build/c256_ffn_candidate_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with Git Bash tools/build_split512_block.sh')
    source_script=UPSTREAM/'derive_native_ffn_layout.py'
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_124.bin'
    raw=np.frombuffer(raw_path.read_bytes(),np.uint8)
    maps=candidate_ffn_maps()
    w1,w2,w3,skip=decode_ffn(raw,maps)
    input_root=ROOT/'build'/('spatial256_window_derived' if derived else 'spatial256_window')
    output_root=ROOT/'build'/('c256_ffn_candidate_derived' if derived else 'c256_ffn_candidate')
    cases=(('seeded',0),) if all_windows else (('zero',0),('seeded',0),('seeded',3))
    for case,shift in cases:
        source=input_root/f'{case}-s{shift}'/'windows_device.f32'
        if not source.is_file():raise FileNotFoundError('run python tools/check_spatial256_window.py first')
        if source.read_bytes()!=(source.parent/'windows.f32').read_bytes():
            raise ValueError('C256 device window differs from spatial oracle')
        values=np.fromfile(source,'<f4')
        x=(values if all_windows else values[:64*256]).reshape(-1,256)
        stages=reference(x,w1,w2,w3,skip)
        folder=output_root/(f'{case}-s{shift}'+('-all' if all_windows else ''))
        folder.mkdir(parents=True,exist_ok=True)
        for name,array in dict(input=x,w1=w1,w2=w2,w3=w3,skip=skip,**stages).items():
            if not np.isfinite(array).all():raise ValueError(f'nonfinite {case}-s{shift} {name}')
            np.asarray(array,dtype='<f4').tofile(folder/f'{name}.f32')
        subprocess.run([str(exe),str(folder),str(len(x))],cwd=ROOT,check=True)
        if (folder/'feature_device.f32').read_bytes()!=(folder/'feature.f32').read_bytes():
            raise AssertionError('candidate FFN feature device bytes differ')
        report=dict(case=case,shift=shift,tokens=len(x),channels=256,
                    tensor_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    input_device_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    candidate_map_source=str(source_script.relative_to(ROOT)),
                    candidate_map_source_sha256=hashlib.sha256(source_script.read_bytes()).hexdigest(),
                    comparison='exact against upstream multiply/F/H under candidate map',
                    original_kernel_executed=False,original_runtime_validation=False,
                    map_status='candidate C256 extension of measured C64/C128 bit rules')
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    p.add_argument('--all-windows',action='store_true')
    a=p.parse_args();raise SystemExit(run(a.derived,a.all_windows))
