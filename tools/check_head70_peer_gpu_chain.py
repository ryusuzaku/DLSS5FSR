#!/usr/bin/env python3
"""Check direct GPU buffer handoff through the coherent candidate head.

Requires check_head70_peer_frame.py output and the standalone HIP executable.
The reference is still public-FP16-fed candidate arithmetic, not an
original NVIDIA kernel or production path.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(size,latent_case='public',output_dir=None,frame_dir=None):
    frame=Path(frame_dir) if frame_dir is not None else (
        ROOT/f'build/peer_head_frame_{size}' if latent_case=='public' else
        ROOT/f'build/peer_head_frame_{size}_{latent_case}')
    original=json.loads((frame/'manifest.json').read_text())
    if original['size']!=[size,size] or original['windows']!=size*size//64:
        raise ValueError('frame fixture size differs')
    out=Path(output_dir) if output_dir else frame/'connected_gpu'
    out.mkdir(parents=True,exist_ok=True)
    native=frame/'native_gain';public=frame/'public_gain'
    for name in ('main','skip','sm','ss','coeff','color','merged'):
        shutil.copyfile(native/f'{name}.f32',out/f'{name}.f32')
    shutil.copyfile(native/'rgb.f32',out/'rgb_native.f32')
    shutil.copyfile(public/'rgb.f32',out/'rgb_public.f32')
    chunks=original['body_chunks']
    if not chunks or chunks[0]['first_window']!=0 or \
       chunks[-1]['last_window_exclusive']!=original['windows']:
        raise ValueError('incomplete body chunks')
    body=np.empty((original['windows'],64,32),np.float32)
    for chunk in chunks:
        start=chunk['first_window'];stop=chunk['last_window_exclusive']
        device=frame/'body_chunks'/f'{start:04d}_{stop:04d}'/'body_device.f32'
        if digest(device)!=chunk['body_device_sha256']:
            raise ValueError('body chunk hash differs')
        body[start:stop]=np.fromfile(device,'<f4').reshape(stop-start,64,32)
    body.astype('<f4').tofile(out/'body.f32')
    if digest(out/'body.f32')!=original['body_windows_sha256']:
        raise ValueError('assembled body hash differs')
    shutil.copyfile(frame/'body_chunks'/f"{chunks[0]['first_window']:04d}_{chunks[0]['last_window_exclusive']:04d}"/'weights.f32',out/'weights.f32')
    command=[str(ROOT/'build/head70_peer_frame_chain.exe'),str(out),str(size),str(size)]
    result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (out/'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS')!=4 or 'FAIL' in result.stdout:
        raise RuntimeError(f'connected HIP chain failed:\n{result.stdout[-3000:]}')
    report=dict(size=[size,size],windows=original['windows'],
                latent_case=latent_case,
                source_model_sha256=original['source_model_sha256'],
                source_frame_manifest_sha256=digest(frame/'manifest.json'),
                input_sha256={name:digest(out/f'{name}.f32') for name in
                              ('main','skip','sm','ss','weights','coeff','color')},
                scalar_body_sha256=digest(out/'body.f32'),
                connected_gpu_exact=['merge','body','native-gain-rgb','public-gain-rgb'],
                direct_gpu_buffer_handoff=True,
                original_kernel_executed=False,original_runtime_validation=False,
                production_wiring=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,end='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--latent-case',choices=('public','image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8'),default='public')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--frame-dir',type=Path)
    a=p.parse_args();run(a.size,a.latent_case,a.output_dir,a.frame_dir)
