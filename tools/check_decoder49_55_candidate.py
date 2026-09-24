#!/usr/bin/env python3
"""Connected candidate C256 decoder chain, blocks 49 through 55."""
from pathlib import Path
import argparse
import hashlib
import json

from check_decoder49_candidate import ROOT,run as run_block


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(last_block=55):
    if last_block not in range(49,56):raise ValueError('last block must be 49..55')
    previous=ROOT/'build/block48_candidate_output_derived/output_device.f32'
    first_hash=digest(previous)
    blocks=[]
    for block in range(49,last_block+1):
        input_hash=digest(previous)
        result=run_block(block,previous)
        report=json.loads((result.parents[1]/'manifest.json').read_text())
        if report['input_device_sha256']!=input_hash or report['output_device_sha256']!=digest(result):
            raise AssertionError(f'block{block} device handoff hash mismatch')
        blocks.append(dict(block=block,shift=report['shift'],windows=report['windows'],
                           input_sha256=input_hash,output_sha256=digest(result)))
        previous=result
    summary=dict(blocks=blocks,source_sha256=first_hash,final_device_sha256=digest(previous),
                 exact_device_handoffs=max(0,len(blocks)-1),
                 map_status='candidate C256 matrix/bias/residual maps; synthetic encoder22 skip',
                 original_kernel_executed=False,original_runtime_validation=False)
    out=ROOT/'build/decoder49_55_candidate_derived/manifest.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--last-block',type=int,choices=range(49,56),default=55)
    run(p.parse_args().last_block)
