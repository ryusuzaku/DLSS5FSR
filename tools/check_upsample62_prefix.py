#!/usr/bin/env python3
"""Check block62 C128->C64 projection/merge on the block61 device output."""
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
    source=ROOT/'build/decoder61_candidate_derived/seeded/output/output_device.f32'
    if source.read_bytes()!=source.with_name('output.f32').read_bytes():
        raise ValueError('block61 device output differs from reference')
    x=np.fromfile(source,'<f4').reshape(32,128,128)
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_140.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=70048:raise ValueError('wrong block62 tensor size')
    c=64;ob=6;count=2*c*c
    rows=bits(count,[3]+list(range(6,ob+5)))
    cols=bits(count,[1,0,4,5,2]+list(range(ob+5,2*ob+1)))
    if np.unique(rows*128+cols).size!=count:
        raise ValueError('block62 matrix map collision')
    weights=np.empty((64,128),np.float32)
    weights[rows,cols]=e4m3fn(raw[0x7000:0x9000])
    order=(np.arange(c)//16)*16+(np.arange(c)%8)*2+(np.arange(c)%16//8)
    scale=np.empty(c,np.float32)
    scale[order]=raw[0x9080:0x9100].view('<f2').astype(np.float32)
    low=multiply(x,weights)
    up=np.repeat(np.repeat(low,2,axis=0),2,axis=1)
    root=ROOT/'build/upsample62_prefix_derived'
    cases=(('zero',np.zeros((64,256,64),np.float32)),
           ('seeded',F(np.random.default_rng(6201).normal(0,.03125,(64,256,64)).astype(np.float32))))
    for name,skip in cases:
        merged=F(H(up+skip*scale))
        folder=root/name;folder.mkdir(parents=True,exist_ok=True)
        for label,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
            array=np.asarray(array,dtype='<f4')
            if not np.isfinite(array).all():raise ValueError(f'nonfinite {name}/{label}')
            array.tofile(folder/f'{label}.f32')
        if (folder/'input.f32').read_bytes()!=source.read_bytes():
            raise ValueError('block61 device handoff changed')
        subprocess.run([str(ROOT/'build/upsample62_prefix_test.exe'),str(folder),'128','32'],
                       cwd=ROOT,check=True)
        device=folder/'merged_device.f32'
        if device.read_bytes()!=(folder/'merged.f32').read_bytes():
            raise AssertionError(f'block62 {name} device merge differs')
        report=dict(block=62,case=name,input_device_sha256=digest(source),
                    tensor_sha256=digest(raw_path),output_device_sha256=digest(device),
                    output_extent=[256,64,64],skip='synthetic encoder8 control',
                    scope='C128->C64 projection and merge; no C64 body',
                    original_kernel_executed=False,original_runtime_validation=False)
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':run()
