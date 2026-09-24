#!/usr/bin/env python3
"""Check full-grid block48 candidate output window-to-HWC handoff.

Inputs carry a synthetic encoder22 skip and provisional C256 weight maps.
This verifies the AMD device handoff, not NVIDIA-original output parity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(derived=False):
    exe=ROOT/'build/spatial256_output_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    parent=ROOT/'build'/('c256_attention_candidate_derived' if derived else 'c256_attention_candidate')
    source=parent/'seeded-s0-all'/'projection_residual_device.f32'
    reference=source.with_name('projection_residual.f32')
    if not source.is_file():raise FileNotFoundError('run check_c256_attention_candidate.py --all-windows first')
    if source.read_bytes()!=reference.read_bytes():raise ValueError('attention device output differs from arithmetic reference')
    width,height,channels=64,16,256
    windows=np.fromfile(source,'<f4').reshape(height//8,width//8,8,8,channels)
    output=windows.transpose(0,2,1,3,4).reshape(height,width,channels)
    if not np.isfinite(output).all():raise ValueError('nonfinite block48 candidate output')
    folder=ROOT/'build'/('block48_candidate_output_derived' if derived else 'block48_candidate_output')
    folder.mkdir(parents=True,exist_ok=True)
    source_bytes=source.read_bytes()
    (folder/'windows.f32').write_bytes(source_bytes)
    output.astype('<f4').tofile(folder/'output.f32')
    subprocess.run([str(exe),str(folder),str(width),str(height),'0'],cwd=ROOT,check=True)
    device=folder/'output_device.f32'
    if device.read_bytes()!=(folder/'output.f32').read_bytes():
        raise AssertionError('C256 device output scatter differs from reference')
    report={
        'extent':[width,height,channels], 'windows':int(windows.shape[0]*windows.shape[1]),
        'source_device_sha256':digest(source), 'output_device_sha256':digest(device),
        'mean':float(output.mean()),'std':float(output.std()),
        'comparison':'exact HIP output scatter against NumPy window inverse',
        'map_status':'candidate C256 matrices and FFN-transferred attention residual channel order',
        'encoder22_skip':'synthetic seeded control',
        'original_kernel_executed':False,'original_runtime_validation':False,
    }
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    run(p.parse_args().derived)
