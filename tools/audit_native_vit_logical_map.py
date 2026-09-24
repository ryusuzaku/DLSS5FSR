#!/usr/bin/env python3
"""Compare upstream's captured logical ViT map with our static PTX map.

This reads source and evaluates address formulas only. No original kernel runs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from recover_vit_bridge_ptx import ROOT,recover

UPSTREAM=ROOT/'ref/dlss5-port/src/native_vit_gather.h'


def logical_map(tokens):
    t=np.arange(tokens,dtype=np.int32)[:,None]
    c=np.arange(1024,dtype=np.int32)[None,:]
    raster=(t&~15)|((t&1)<<3)|((t&14)>>1)
    channel=(c&~31)|((c&1)<<1)|((c&2)>>1)|((c&4)<<2)|((c&24)>>1)
    return (raster*1024+channel).reshape(-1)


def audit(width,height):
    source=UPSTREAM.read_text(encoding='utf-8')
    for formula in ('((t&1u)<<3)|((t&14u)>>1)',
                    '((c&1u)<<1)|((c&2u)>>1)|((c&4u)<<2)|((c&24u)>>1)'):
        if formula not in source:raise ValueError(f'upstream logical map changed: {formula}')
    n=width*height*1024
    pt_source=recover(width,height)[0]
    logical=logical_map(width*height)
    physical_to_hwc=np.empty(n,np.int32)
    physical_to_hwc[pt_source]=logical
    if not np.array_equal(np.sort(physical_to_hwc),np.arange(n)):
        raise ValueError('physical/logical map not bijective')
    y=physical_to_hwc//(width*1024)
    x=physical_to_hwc//1024%width
    channel=physical_to_hwc%1024
    cell=(y//4)*(width//4)+x//4
    bank=channel//512
    local=((y%4)*4+x%4)*512+channel%512
    blocks=n//8192
    expected_cell=np.arange(blocks,dtype=np.int32)//2
    expected_bank=np.arange(blocks,dtype=np.int32)%2
    cell_ok=np.array_equal(cell.reshape(blocks,8192),np.broadcast_to(expected_cell[:,None],(blocks,8192)))
    bank_ok=np.array_equal(bank.reshape(blocks,8192),np.broadcast_to(expected_bank[:,None],(blocks,8192)))
    repeat_ok=np.array_equal(local.reshape(blocks,8192),np.broadcast_to(local[:8192],(blocks,8192)))
    permutation_ok=np.array_equal(np.sort(local[:8192]),np.arange(8192))
    report=dict(width=width,height=height,entries=n,cell_ok=cell_ok,bank_ok=bank_ok,
                repeat_ok=repeat_ok,permutation_ok=permutation_ok,
                mismatched_cells=int(np.count_nonzero(cell.reshape(blocks,8192)!=expected_cell[:,None])),
                mismatched_banks=int(np.count_nonzero(bank.reshape(blocks,8192)!=expected_bank[:,None])),
                repeated_local_differences=int(np.count_nonzero(local.reshape(blocks,8192)!=local[:8192])),
                logical_source_sha256=hashlib.sha256(UPSTREAM.read_bytes()).hexdigest(),
                scope='upstream captured-logical source formula vs static PTX physical-source address',
                original_kernel_executed=False)
    return report,local[:8192]


def main():
    for width,height in ((8,4),(16,4),(8,8)):
        report,_=audit(width,height)
        print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
