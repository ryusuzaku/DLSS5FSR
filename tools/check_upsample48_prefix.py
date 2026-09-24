#!/usr/bin/env python3
"""Check the block48 projection/skip prefix on block47 device output.

The C256 skip is synthetic because the matching encoder22 device output and
C256 view map are not in this standalone chain. This does not check the later
fused C256 Swin body or original input/output physical layouts.
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


def run(derived=False):
    exe=ROOT/'build/upsample48_prefix_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    source=ROOT/'build'/('split512_decoder_derived' if derived else 'split512_spatial')/'block47-32x8-s2/final_device.f32'
    if not source.is_file():raise FileNotFoundError('run python tools/check_split512_decoder_chain.py first')
    x=np.fromfile(source,'<f4').reshape(8,32,512)
    records={v['name']:v for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    record=records['block48.layer0.layer']
    path=ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin"
    raw=path.read_bytes()
    if len(raw)!=820784:raise ValueError('wrong block48 record size')
    b=np.frombuffer(raw,np.uint8)
    count=2*256*256
    rows=bits(count,[3]+list(range(6,13)))
    cols=bits(count,[1,0,4,5,2]+list(range(13,17)))
    if np.unique(rows*512+cols).size!=count:raise ValueError('block48 matrix map not bijective')
    weights=np.empty((256,512),np.float32)
    weights[rows,cols]=e4m3fn(b[0x58000:0x78000])
    channels=np.arange(256)
    order=(channels//16)*16+(channels%8)*2+(channels%16//8)
    scale=np.empty(256,np.float32)
    scale[order]=np.frombuffer(raw[0x78200:0x78400],'<f2').astype(np.float32)
    low=multiply(x,weights)
    up=np.repeat(np.repeat(low,2,0),2,1)
    if not all(np.isfinite(a).all() for a in (x,weights,scale,low)):
        raise ValueError('nonfinite block48 projection fixture')
    for name,skip in (('zero',np.zeros((16,64,256),np.float32)),
                      ('seeded',F(np.random.default_rng(4801).normal(0,.03125,(16,64,256)).astype(np.float32)))):
        merged=F(H(up+skip*scale))
        if not np.isfinite(merged).all():raise ValueError(f'nonfinite {name} merge')
        folder=ROOT/'build'/('upsample48_prefix_derived' if derived else 'upsample48_prefix')/name
        folder.mkdir(parents=True,exist_ok=True)
        for label,array in (('input',x),('weights',weights),('scale',scale),('skip',skip),
                            ('low',low),('merged',merged)):
            np.asarray(array,dtype='<f4').tofile(folder/f'{label}.f32')
        if (folder/'input.f32').read_bytes()!=source.read_bytes():
            raise ValueError('block47 device handoff changed')
        subprocess.run([str(exe),str(folder),'32','8'],cwd=ROOT,check=True)
        report=dict(case=name,source_device_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    tensor=record['index'],tensor_sha256=hashlib.sha256(raw).hexdigest(),
                    scope=('block48 C512->C256 projection from source-composed ViT bridge; synthetic C256 skip'
                           if derived else 'block48 C512->C256 projection/skip prefix only; synthetic C256 skip'),
                    oracle='unchanged native_upsample48_reference projection map and merge arithmetic',
                    comparison='exact',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in folder.glob('*.f32')})
        (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    raise SystemExit(run(p.parse_args().derived))
