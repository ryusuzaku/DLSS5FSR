#!/usr/bin/env python3
"""Chain the block66 peer-basis C32 candidate through blocks67..69."""
from pathlib import Path
import hashlib
import json
from check_block66_peer_candidate import run

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    chain=[]
    previous=ROOT/'build/block66_peer_candidate/seeded/output/output_device.f32'
    if not previous.is_file():raise FileNotFoundError('run check_block66_peer_candidate.py first')
    for block in range(67,70):
        run(block,'seeded')
        root=ROOT/'build'/f'decoder{block}_peer_candidate'/'seeded'
        report=json.loads((root/'manifest.json').read_text())
        output=root/'output/output_device.f32'
        if report['input_device_sha256']!=digest(previous):
            raise AssertionError(f'block{block} device handoff differs')
        if report['output_device_sha256']!=digest(output):
            raise AssertionError(f'block{block} output hash differs')
        chain.append(report)
        previous=output
    summary=dict(blocks=[item['block'] for item in chain],
                 shifts=[item['shift'] for item in chain],
                 input_device_sha256=chain[0]['input_device_sha256'],
                 output_device_sha256=chain[-1]['output_device_sha256'],
                 exact_interblock_handoffs=2,
                 basis='public QMMA C32 candidate, block66 transition basis audited',
                 encoder_skips='synthetic controls inherited from earlier decoder stages',
                 original_kernel_executed=False,original_runtime_validation=False)
    destination=ROOT/'build/decoder67_69_peer_candidate'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
