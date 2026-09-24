#!/usr/bin/env python3
"""Check C256->C128 block56 projection/skip merge on block55 device output.

The encoder14 C128 skip is synthetic. The C128 Swin body follows later.
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


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    source=ROOT/'build/decoder55_candidate_derived/output/output_device.f32'
    if not source.is_file():raise FileNotFoundError('run check_decoder49_55_candidate.py first')
    if source.read_bytes()!=source.with_name('output.f32').read_bytes():
        raise ValueError('block55 device output differs from reference')
    x=np.fromfile(source,'<f4').reshape(16,64,256)
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_133.bin'
    raw=raw_path.read_bytes()
    if len(raw)!=230176:raise ValueError('wrong block56 record size')
    values=np.frombuffer(raw,np.uint8)
    c=128;ob=7;count=2*c*c
    rows=bits(count,[3]+list(range(6,ob+5)))
    cols=bits(count,[1,0,4,5,2]+list(range(ob+5,2*ob+1)))
    if np.unique(rows*256+cols).size!=count:raise ValueError('block56 matrix map is not bijective')
    weights=np.empty((128,256),np.float32)
    weights[rows,cols]=e4m3fn(values[0x18000:0x20000])
    order=(np.arange(c)//16)*16+(np.arange(c)%8)*2+(np.arange(c)%16//8)
    scale=np.empty(c,np.float32)
    scale[order]=np.frombuffer(raw[0x20100:0x20200],'<f2').astype(np.float32)
    low=multiply(x,weights)
    up=np.repeat(np.repeat(low,2,axis=0),2,axis=1)
    if not all(np.isfinite(a).all() for a in (weights,scale,low,up)):
        raise ValueError('nonfinite block56 prefix coefficients/input')
    root=ROOT/'build/upsample56_prefix_derived'
    for name,skip in (('zero',np.zeros((32,128,128),np.float32)),
                      ('seeded',F(np.random.default_rng(5601).normal(0,.03125,(32,128,128)).astype(np.float32)))):
        merged=F(H(up+skip*scale))
        if not np.isfinite(merged).all():raise ValueError(f'nonfinite block56 {name} merge')
        folder=root/name;folder.mkdir(parents=True,exist_ok=True)
        for label,array in dict(input=x,weights=weights,scale=scale,skip=skip,low=low,merged=merged).items():
            np.asarray(array,dtype='<f4').tofile(folder/f'{label}.f32')
        if (folder/'input.f32').read_bytes()!=source.read_bytes():
            raise ValueError('block55 device handoff changed')
        subprocess.run([str(ROOT/'build/upsample56_prefix_test.exe'),str(folder),'64','16'],
                       cwd=ROOT,check=True)
        device=folder/'merged_device.f32'
        if device.read_bytes()!=(folder/'merged.f32').read_bytes():
            raise AssertionError(f'block56 {name} device merge differs')
        report=dict(case=name,input_device_sha256=digest(source),tensor_sha256=digest(raw_path),
                    output_device_sha256=digest(device),output_extent=[128,32,128],
                    map_source='native_upsample48_reference.unpack C128 prefix bit rules',
                    skip='synthetic zero or seeded C128 encoder14 control',
                    scope='C256->C128 projection and upsample/skip merge only; no C128 body',
                    original_kernel_executed=False,original_runtime_validation=False)
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':run()
