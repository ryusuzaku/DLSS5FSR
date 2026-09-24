#!/usr/bin/env python3
"""Test the explicitly named shipped-mode2 head-body candidate, not original DLSS.

Runs actual merge -> body -> finish on HIP with intermediate comparisons.
The legacy head oracle is retained as a contrasting hypothesis, not overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import head70_reference as H
import head70_weights as W
import head70_shipped_body as B
import swin1h_ref as R

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', type=Path, default=ROOT / 'build/head70_body_test.exe')
    ap.add_argument('--out', type=Path, default=ROOT / 'build/head70_body_fixtures')
    ap.add_argument('--emit-only', action='store_true')
    a = ap.parse_args()
    tensor = ROOT / 'dlss5-analysis/tensors/tensor_150.bin'
    stage = ROOT / 'dlss5-analysis/tensors/tensor_001.bin'
    # Calibrate this trace recipe against the existing, device-checked C32
    # golden, before testing a new tensor or accepting candidate results.
    xb = R.cw_staged_ffnin(32)
    x = np.asarray([R.e4m3_decode(v) for v in xb], np.float32).reshape(1,64,32)
    calibration = B.forward(x, stage.read_bytes())['body']
    calibration_fnv = R.fnv1a_hex(calibration.ravel().tolist())
    assert calibration_fnv.upper() == 'A34B2CB6', calibration_fnv
    print(f'C32 golden calibration: {calibration_fnv} PASS', flush=True)
    params, legacy, temp = H.load(str(tensor), str(stage))
    body, sm, ss, coeff = params
    # Independently check extraction before either oracle or HIP sees the data.
    raw = tensor.read_bytes()
    ext = W.extract(raw, stage.read_bytes()[W.PAD_AT:W.PAD_AT + 16])
    assert W.repack(*ext) == raw
    mirror = B.rewrite(body)
    # S242 independent C++ loader measured these exact hashes on tensor_150.
    assert B.fnv(body) == 0x87316D4F, f'{B.fnv(body):08X}'
    assert B.fnv(mirror) == 0x8241E5BB, f'{B.fnv(mirror):08X}'
    print(f'head body FNV {B.fnv(body):08X} -> {B.fnv(mirror):08X}; QK temperature {temp:.7g}', flush=True)
    rng = np.random.default_rng(0x24270)
    failures = 0
    cases = [('single',8,8), ('rect',16,24), ('zero',8,16)]
    for name,h,w in cases:
        main_in = rng.uniform(-1,1,(h//2,w//2,32)).astype(np.float32)
        skip = rng.uniform(-1,1,(h,w,32)).astype(np.float32)
        color = rng.uniform(0,1,(h,w,3)).astype(np.float32)
        if name == 'zero':
            main_in.fill(0); skip.fill(0)
        merged = H.windowise(H.merge(main_in,skip,sm,ss))
        trace = B.forward(merged,body)
        rgb = H.finish(H.dewindowise(trace['body'],h,w),color,coeff)
        if name == 'single':
            old = H.body_forward(merged,legacy,temp)
            oldrgb = H.finish(H.dewindowise(old,h,w),color,coeff)
            print('legacy-vs-shipped hypothesis delta (not a pass/fail): '
                  f"body maxabs={np.max(np.abs(old-trace['body'])):.7g}, "
                  f'RGB maxabs={np.max(np.abs(oldrgb-rgb)):.7g}', flush=True)
        folder = a.out / name
        folder.mkdir(parents=True,exist_ok=True)
        arrays = dict(main=main_in,skip=skip,color=color,sm=sm,ss=ss,coeff=coeff,
                      merged=merged,rgb=rgb,**trace)
        for key,v in arrays.items():
            assert np.isfinite(v).all(), (name,key)
            np.asarray(v,dtype='<f4').tofile(folder/(key+'.f32'))
        (folder/'body.bin').write_bytes(body)
        manifest = dict(recipe='shipped-mode2',width=w,height=h,input_scale=.03125,
                        tensor_sha256=hashlib.sha256(raw).hexdigest(),
                        body_raw_fnv=f'{B.fnv(body):08X}',body_rewritten_fnv=f'{B.fnv(mirror):08X}',
                        limits=dict(merge=0,body_stages=.1,rgb=.001),
                        limitations='No QK normalization; not upstream/native head semantics.',
                        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in folder.iterdir() if p.suffix in ('.f32','.bin')})
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'{name}: {w}x{h}, {len(merged)} windows',flush=True)
        if not a.emit_only:
            run = subprocess.run([str(a.exe.resolve()),str(folder.resolve()),str(w),str(h),
                                  '.03125',f'{B.fnv(mirror):08X}'],cwd=ROOT/'build')
            failures += run.returncode != 0
    if a.emit_only:
        print('Fixtures emitted; no device verdict.')
    else:
        print(f'{len(cases)-failures}/{len(cases)} shipped-mode2 head candidate cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
