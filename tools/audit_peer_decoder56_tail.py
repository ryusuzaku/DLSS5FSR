#!/usr/bin/env python3
"""Audit same-image AMD candidate blocks56–61 against public FP16 stages."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/peer_decoder56_inputs'
OUT=ROOT/'build/peer_decoder56_tail_audit'
FROM39=Path.home()/'DLSS5FSR-build-offload'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                rmse=float(np.sqrt(np.mean(d*d))),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def run(case='image_fp8'):
    if case not in ('image_fp8','image_from_block48','image_from39'):raise ValueError('bad case')
    source=json.loads((SOURCE/'manifest.json').read_text())
    order=peer_to_native_multihead(np.arange(128))
    prefix=(FROM39/'upsample56_from39' if case=='image_from39' else
            ROOT/'build/upsample56_prefix_derived'/case)
    p=json.loads((prefix/'manifest.json').read_text())
    if not p['hip_projection_merge_exact'] or \
       digest(prefix/'merged_device.f32')!=p['output_device_sha256'] or \
       p['source_model_sha256']!=source['model_sha256']:
        raise ValueError('prefix provenance/exactness differs')
    previous=prefix/'merged_device.f32'
    stages={}
    for block in range(56,62):
        root=(FROM39/'peer_decoder56_from39'/f'block{block}' if case=='image_from39' else
              ROOT/'build/block56_candidate'/case if block==56 else
              ROOT/'build'/f'decoder{block}_candidate_derived'/case)
        m=json.loads((root/'manifest.json').read_text())
        output=root/'output/output_device.f32'
        if m['input_device_sha256']!=digest(previous) or \
           m['output_device_sha256']!=digest(output):
            raise ValueError(f'block{block} handoff/hash differs')
        target_path=SOURCE/f'block{block}_peer.f32'
        if digest(target_path)!=source['tensor_sha256'][f'block{block}']:
            raise ValueError(f'public block{block} hash differs')
        candidate=np.fromfile(output,'<f4').reshape(32,32,128)[...,order]
        target=np.fromfile(target_path,'<f4').reshape(32,32,128)
        stages[f'block{block}']=dict(candidate_native_sha256=digest(output),
                                    windows=m['windows'],
                                    candidate_vs_public=metrics(candidate,target))
        previous=output
    report=dict(source_model_sha256=source['model_sha256'],
                source_image_sha256=source['image_sha256'],
                source_same_inference_call=source['same_inference_call'],
                case=case,prefix_output_device_sha256=p['output_device_sha256'],
                block_output_stages=stages,exact_device_handoffs=6,
                scalar_hip_stages_exact=True,original_kernel_executed=False,
                original_runtime_validation=False,production_wiring=False)
    out=(ROOT/'build/peer_decoder56_tail_audit_from39' if case=='image_from39' else
         OUT if case=='image_fp8' else ROOT/'build/peer_decoder56_tail_audit_from48')
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('image_fp8','image_from_block48','image_from39'),default='image_fp8')
    run(parser.parse_args().case)
