#!/usr/bin/env python3
"""Audit same-image AMD candidate blocks62–65 against public FP16 stages."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from audit_peer_native_c32_basis import peer_to_native_multihead

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/peer_decoder62_inputs'
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
    if case not in ('image_half','image_fp8','from56_fp8','from48_fp8','from39_fp8'):raise ValueError('bad case')
    source=json.loads((SOURCE/'manifest.json').read_text())
    order=peer_to_native_multihead(np.arange(64))
    prefix=(FROM39/'upsample62_from39' if case=='from39_fp8' else
            ROOT/'build/upsample62_prefix_derived'/case)
    p=json.loads((prefix/'manifest.json').read_text())
    if not p['hip_projection_merge_exact'] or \
       digest(prefix/'merged_device.f32')!=p['output_device_sha256'] or \
       p['source_model_sha256']!=source['model_sha256']:
        raise ValueError('prefix provenance/exactness differs')
    if case in ('from56_fp8','from48_fp8','from39_fp8'):
        upstream=ROOT/'build'/('peer_decoder56_tail_audit' if case=='from56_fp8' else
                               'peer_decoder56_tail_audit_from48' if case=='from48_fp8' else
                               'peer_decoder56_tail_audit_from39')/'manifest.json'
        u=json.loads(upstream.read_text())
        if p['upstream_amd_block61']['audit_sha256']!=digest(upstream) or \
           p['input_native_sha256']!=u['block_output_stages']['block61']['candidate_native_sha256'] or \
           u['source_image_sha256']!=source['image_sha256']:
            raise ValueError('upstream block56-61 provenance differs')
    previous=prefix/'merged_device.f32'
    stages={}
    for block in range(62,66):
        root=(FROM39/'peer_decoder62_from39'/f'block{block}' if case=='from39_fp8' else
              ROOT/'build/block62_candidate'/case if block==62 else
              ROOT/'build'/f'decoder{block}_candidate_derived'/case)
        m=json.loads((root/'manifest.json').read_text())
        output=root/'output/output_device.f32'
        if m['input_device_sha256']!=digest(previous) or \
           m['output_device_sha256']!=digest(output):
            raise ValueError(f'block{block} handoff/hash differs')
        target_path=SOURCE/f'block{block}_peer.f32'
        if digest(target_path)!=source['tensor_sha256'][f'block{block}']:
            raise ValueError(f'public block{block} hash differs')
        candidate=np.fromfile(output,'<f4').reshape(64,64,64)[...,order]
        target=np.fromfile(target_path,'<f4').reshape(64,64,64)
        stages[f'block{block}']=dict(candidate_native_sha256=digest(output),
                                    windows=m['windows'],
                                    candidate_vs_public=metrics(candidate,target))
        previous=output
    report=dict(source_model_sha256=source['model_sha256'],
                source_image_sha256=source['image_sha256'],
                source_same_inference_call=source['same_inference_call'],
                case=case,prefix_output_device_sha256=p['output_device_sha256'],
                block_output_stages=stages,exact_device_handoffs=4,
                scalar_hip_stages_exact=True,original_kernel_executed=False,
                original_runtime_validation=False,production_wiring=False)
    out=ROOT/'build'/('peer_decoder62_tail_audit_from39' if case=='from39_fp8' else
                      'peer_decoder62_tail_audit' if case=='image_fp8' else
                      'peer_decoder62_tail_audit_half' if case=='image_half' else
                      'peer_decoder62_tail_audit_from56' if case=='from56_fp8' else
                      'peer_decoder62_tail_audit_from48')
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('image_half','image_fp8','from56_fp8','from48_fp8','from39_fp8'),default='image_fp8')
    a=p.parse_args();run(a.case)
