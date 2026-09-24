#!/usr/bin/env python3
"""Chain block62 C64 candidate output through ordinary blocks63..65."""
from pathlib import Path
import hashlib
import json
from check_block62_candidate import run

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    chain=[]
    previous=ROOT/'build/block62_candidate/seeded/output/output_device.f32'
    if not previous.is_file():raise FileNotFoundError('run check_block62_candidate.py first')
    for block in range(63,66):
        run('seeded',False,block)
        root=ROOT/'build'/f'decoder{block}_candidate_derived'/'seeded'
        report=json.loads((root/'manifest.json').read_text())
        output=root/'output/output_device.f32'
        if report['input_device_sha256']!=digest(previous):
            raise AssertionError(f'block{block} device handoff differs')
        if report['output_device_sha256']!=digest(output):
            raise AssertionError(f'block{block} output hash differs')
        chain.append(report)
        previous=output
    summary=dict(blocks=[entry['block'] for entry in chain],
                 shifts=[entry['shift'] for entry in chain],
                 input_device_sha256=chain[0]['input_device_sha256'],
                 output_device_sha256=chain[-1]['output_device_sha256'],
                 exact_interblock_handoffs=2,
                 encoder8_skip='synthetic control inherited from block62',
                 original_kernel_executed=False,original_runtime_validation=False)
    destination=ROOT/'build/decoder63_65_candidate_derived'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
