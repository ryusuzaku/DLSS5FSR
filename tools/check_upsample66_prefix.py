#!/usr/bin/env python3
"""Check block66 C64->C32 projection/half merge on block65 device output."""
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


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    source=ROOT/'build/decoder65_candidate_derived/seeded/output/output_device.f32'
    if source.read_bytes()!=source.with_name('output.f32').read_bytes():
        raise ValueError('block65 device output differs from reference')
    x=np.fromfile(source,'<f4').reshape(64,256,64)
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_144.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=22784:raise ValueError('wrong block66 tensor size')
    rows=bits(2048,[3,6,7,8,9])
    cols=bits(2048,[1,0,4,5,2,10])
    if np.unique(rows*64+cols).size!=2048:
        raise ValueError('block66 matrix map collision')
    matrix=np.empty((32,64),np.float32)
    matrix[rows,cols]=e4m3fn(raw[0x2000:0x2800])
    order=np.array([0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15,
                    16,17,20,21,24,25,28,29,18,19,22,23,26,27,30,31])
    c=np.arange(32);other=(c//16)*16+(c%8)*2+(c%16//8)
    weights=np.empty_like(matrix);weights[order]=matrix[other]
    scale=np.empty(32,np.float32)
    scale[order]=raw[0x2860:0x28a0].view('<f2').astype(np.float32)
    low=multiply(x,weights)
    up=np.repeat(np.repeat(low,2,axis=0),2,axis=1)
    root=ROOT/'build/upsample66_prefix_derived'
    cases=(('zero',np.zeros((128,512,32),np.float32)),
           ('seeded',F(np.random.default_rng(6601).normal(0,.03125,(128,512,32)).astype(np.float32))))
    for name,skip in cases:
        merged=H(up+skip*scale)
        folder=root/name;folder.mkdir(parents=True,exist_ok=True)
        for label,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
            array=np.asarray(array,dtype='<f4')
            if not np.isfinite(array).all():raise ValueError(f'nonfinite {name}/{label}')
            array.tofile(folder/f'{label}.f32')
        if (folder/'input.f32').read_bytes()!=source.read_bytes():
            raise ValueError('block65 device handoff changed')
        subprocess.run([str(ROOT/'build/upsample66_prefix_test.exe'),str(folder),'256','64'],
                       cwd=ROOT,check=True)
        device=folder/'merged_device.f32'
        if device.read_bytes()!=(folder/'merged.f32').read_bytes():
            raise AssertionError(f'block66 {name} device merge differs')
        report=dict(block=66,case=name,input_device_sha256=digest(source),
                    tensor_sha256=digest(raw_path),output_device_sha256=digest(device),
                    output_extent=[512,128,32],skip='synthetic encoder4 control',
                    scope='C64->C32 projection and half merge; no C32 body',
                    original_kernel_executed=False,original_runtime_validation=False)
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':run()
