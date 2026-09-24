#!/usr/bin/env python3
"""Feed actual HIP split-block output into each next shifted block."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from check_split512_spatial_block import run_case
from check_split512_block import R


def run(width=16,height=8):
    # Keep the continuation inside the reference's finite half-norm range.
    if width not in (16,32) or height!=8:
        raise ValueError('covered chain extents are 16x8 and 32x8')
    source=R.F(np.random.default_rng(23024).normal(0,.03125,(width*height,512)).astype(np.float32))
    first=run_case(23,width,height,0,source)
    device_path=first/'final_device.f32'
    raw=device_path.read_bytes()
    output=np.frombuffer(raw,'<f4').reshape(width*height,512)
    np.testing.assert_array_equal(output,np.fromfile(first/'final.f32','<f4').reshape(output.shape))
    handoffs=[];last=first
    for block,shift in ((24,3),(25,1),(26,2),(27,0),(28,3),(29,1),(30,2)):
        next_folder=run_case(block,width,height,shift,output)
        input_bytes=(next_folder/'input.f32').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==hashlib.sha256(input_bytes).hexdigest()
        qkv=np.fromfile(next_folder/'qkv.f32','<f4').reshape(-1,3,16,32)
        for matrix in (0,1):
            sums=np.sum(qkv[:,matrix].astype(np.float64)**2,axis=-1)
            if np.any(sums>=65504):raise ValueError(f'block{block} half norm overflow in continuation fixture')
        handoffs.append(dict(source=last.name,destination=next_folder.name,
                             sha256=hashlib.sha256(raw).hexdigest()))
        last=next_folder
        raw=(last/'final_device.f32').read_bytes()
        output=np.frombuffer(raw,'<f4').reshape(width*height,512)
        np.testing.assert_array_equal(output,np.fromfile(last/'final.f32','<f4').reshape(output.shape))
    report={'chain':'block23..30 body; shifts 0,3,1,2,0,3,1,2',
            'extent':f'{width}x{height} logical HWC, 512 channels',
            'handoff':'exact device FP8 output, then next device input',
            'handoffs':handoffs,
            'verdict':'every block exact at all 12 spatial stage comparisons'}
    folder=Path(__file__).resolve().parents[1]/'build'/('split512_chain' if width==16 else f'split512_chain_{width}x{height}')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('block23 -> block30 body: PASS (seven device output/input SHA256 handoffs exact)')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=16);a=p.parse_args()
    raise SystemExit(run(a.width))
