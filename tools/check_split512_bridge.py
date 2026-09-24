#!/usr/bin/env python3
"""Check block30 raw-device pool and C1024 head against upstream arithmetic."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from check_split512_block import ROOT,R
from decode_tinlayout_global import e4m3fn


def run(width=16,height=8):
    exe=ROOT/'build/split512_bridge_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    if width not in (16,32) or height!=8:raise ValueError('covered extents are 16x8 and 32x8')
    block=ROOT/'build/split512_spatial'/f'block30-{width}x{height}-s2'
    source=block/'final_raw_device.f32'
    if not source.is_file():raise FileNotFoundError('run python tools/check_split512_chain.py first')
    raw=np.fromfile(source,'<f4').reshape(height,width,512)
    np.testing.assert_array_equal(raw.ravel(),np.fromfile(block/'final_raw.f32','<f4'))
    top=R.H(raw[::2,::2]+raw[::2,1::2])
    bottom=R.H(raw[1::2,::2]+raw[1::2,1::2])
    pooled=R.F(R.H(R.H(top+bottom)*np.float32(.25)))
    records={v['name']:v for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    record=records['block30.layer4.layer']
    weight_bytes=(ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin").read_bytes()
    if len(weight_bytes)!=524304:raise ValueError('wrong block30 head weight size')
    order=np.arange(524288)
    rows=R.bits(524288,[3,6,7,8,9,10,11,12,13,14])
    cols=R.bits(524288,[1,0,4,5,2,15,16,17,18])
    if len(np.unique(rows*512+cols))!=len(order):raise ValueError('head map not bijective')
    matrix=np.empty((1024,512),np.float32)
    matrix[rows,cols]=e4m3fn(np.frombuffer(weight_bytes[:524288],np.uint8))
    head=R.F(R.multiply(pooled.reshape(-1,512),matrix)).reshape(height//2,width//2,1024)
    folder=ROOT/'build'/('split512_bridge' if width==16 else f'split512_bridge_{width}x{height}')
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in (('raw',raw),('weights',matrix),('pool',pooled),('head',head)):
        if not np.isfinite(array).all():raise ValueError(f'nonfinite {name}')
        np.asarray(array,dtype='<f4').tofile(folder/f'{name}.f32')
    report=dict(tensor=record['index'],tensor_sha256=hashlib.sha256(weight_bytes).hexdigest(),
                source_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='block30 logical raw projection -> 2x2 pool -> C1024 head',
                oracle='upstream validate_native_split_pool.py and check_native_split_head.py',
                comparison='exact',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in folder.glob('*.f32')})
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([str(exe),str(folder),str(width),str(height)],cwd=ROOT,check=True)
    np.testing.assert_array_equal(np.fromfile(folder/'head_device.f32','<f4'),head.ravel())
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=16);a=p.parse_args()
    raise SystemExit(run(a.width))
