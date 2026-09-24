#!/usr/bin/env python3
"""Feed decoder39 device output through C512 decoder blocks40..47."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from check_split512_spatial_block import ROOT,run_case


def run(derived=False):
    source=ROOT/'build'/('decoder39_entry_derived' if derived else 'decoder39_entry')/'output_device.f32'
    if not source.is_file():raise FileNotFoundError('run python tools/check_decoder39_entry.py first')
    width,height=32,8
    raw=source.read_bytes()
    x=np.frombuffer(raw,'<f4').reshape(width*height,512)
    handoffs=[];last='decoder39 device'
    for block,shift in zip(range(40,48),(0,3,1,2,0,3,1,2)):
        folder=run_case(block,width,height,shift,x,
                        ROOT/'build/split512_decoder_derived' if derived else None)
        if (folder/'input.f32').read_bytes()!=raw:
            raise ValueError(f'block{block} device input handoff changed')
        qkv=np.fromfile(folder/'qkv.f32','<f4').reshape(-1,3,16,32)
        sums=np.sum(qkv[:,:2].astype(np.float64)**2,axis=-1)
        if np.any(sums>=65504):
            raise ValueError(f'block{block} half Q/K norm overflow in decoder control')
        handoffs.append(dict(source=last,destination=folder.name,
                             sha256=hashlib.sha256(raw).hexdigest()))
        last=folder.name
        raw=(folder/'final_device.f32').read_bytes()
        x=np.frombuffer(raw,'<f4').reshape(width*height,512)
        np.testing.assert_array_equal(x,np.fromfile(folder/'final.f32','<f4').reshape(x.shape))
    report=dict(chain='decoder39 device -> block40..47',extent='32x8 logical HWC, C512',
                shifts=[0,3,1,2,0,3,1,2],
                source_layout=('upstream capture-derived logical ViT bridge composed with original PTX source'
                               if derived else 'source-linear ViT control; C512 split-view not recovered'),
                handoffs=handoffs,final_sha256=hashlib.sha256(raw).hexdigest(),
                comparison='exact device stages and byte-exact output/input handoffs')
    out=ROOT/'build'/('split512_decoder_chain_derived' if derived else 'split512_decoder_chain')
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('decoder39 -> block47: PASS (eight byte-exact device handoffs)')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    raise SystemExit(run(p.parse_args().derived))
