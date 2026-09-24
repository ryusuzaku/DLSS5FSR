#!/usr/bin/env python3
"""Compare AMD candidate blocks66–69 against same-image public FP16 stages.

All RX9070XT stages are exact against native-style scalar candidate arithmetic.
Public FP16 model differences are diagnostics, not original-kernel verdicts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from audit_peer_native_c32_basis import peer_to_native

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/peer_decoder66_inputs'
OUT=ROOT/'build/peer_decoder66_tail_audit'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                rmse=float(np.sqrt(np.mean(d*d))),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def run(cases=None):
    source=json.loads((SOURCE/'manifest.json').read_text())
    map32=peer_to_native(np.arange(32))
    report=dict(source_model_sha256=source['model_sha256'],
                source_image_sha256=source['image_sha256'],
                source_same_inference_call=source['same_inference_call'],
                cases={},original_kernel_executed=False,
                original_runtime_validation=False,
                production_wiring=False)
    for case in (cases or ('image_half','image_fp8','from62_fp8','from56_fp8')):
        prefix=ROOT/'build/upsample66_prefix_derived'/case
        p=json.loads((prefix/'manifest.json').read_text())
        if not p['hip_projection_merge_exact'] or \
           digest(prefix/'merged_device.f32')!=p['output_device_sha256']:
            raise ValueError('prefix not exact/hash valid')
        if case in ('from62_fp8','from56_fp8','from48_fp8'):
            audit_dir=('peer_decoder62_tail_audit' if case=='from62_fp8' else
                       'peer_decoder62_tail_audit_from56' if case=='from56_fp8' else
                       'peer_decoder62_tail_audit_from48')
            upstream=ROOT/'build'/audit_dir/'manifest.json'
            u=json.loads(upstream.read_text())
            if p['upstream_amd_block65']['audit_sha256']!=digest(upstream) or \
               p['input_native_sha256']!=u['block_output_stages']['block65']['candidate_native_sha256'] or \
               u['source_image_sha256']!=source['image_sha256']:
                raise ValueError('upstream block62-65 provenance differs')
        stages={}
        previous=prefix/'merged_device.f32'
        for block in range(66,70):
            root=(ROOT/'build/block66_peer_candidate'/case if block==66 else
                  ROOT/'build'/f'decoder{block}_peer_candidate'/case)
            m=json.loads((root/'manifest.json').read_text())
            output=root/'output/output_device.f32'
            if m['input_device_sha256']!=digest(previous) or \
               m['output_device_sha256']!=digest(output):
                raise ValueError(f'{case} block{block} handoff/hash differs')
            candidate=np.fromfile(output,'<f4').reshape(128,128,32)[...,map32]
            target_path=SOURCE/f'block{block}_peer.f32'
            if digest(target_path)!=source['tensor_sha256'][f'block{block}']:
                raise ValueError(f'public block{block} hash differs')
            target=np.fromfile(target_path,'<f4').reshape(128,128,32)
            stages[f'block{block}']=dict(candidate_native_sha256=digest(output),
                                        windows=m['windows'],
                                        candidate_vs_public=metrics(candidate,target))
            previous=output
        report['cases'][case]=dict(prefix_output_device_sha256=p['output_device_sha256'],
                                   block_output_stages=stages,
                                   exact_device_handoffs=4,
                                   scalar_hip_stages_exact=True)
    out=OUT if cases is None else ROOT/'build/peer_decoder66_tail_audit_from48'
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('from48_fp8',))
    args=parser.parse_args()
    run([args.case] if args.case else None)
