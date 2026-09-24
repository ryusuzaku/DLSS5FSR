#!/usr/bin/env python3
"""Check static-PTX source-linear->ViT mapping on a logical-HWC control."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from recover_vit_bridge_ptx import ROOT,recover,checked_source,checked_inverse_source


def run(width=8,height=4):
    exe=ROOT/'build/vit_bridge_ptx_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    if (width,height) not in ((8,4),(16,4)):raise ValueError('covered bridge extents are 8x4 and 16x4')
    mapdir=ROOT/'build'/f'vit_bridge_ptx_{width}x{height}'
    mapfile=mapdir/'hwc-to-vit.i32'
    if not mapfile.is_file():raise FileNotFoundError(f'run python tools/recover_vit_bridge_ptx.py {width} {height} --out {mapdir} first')
    manifest=json.loads((mapdir/'PROVENANCE.json').read_text())
    if manifest.get('source_index_semantics')!='PTX ld.global physical byte index':
        raise ValueError('stale thread-linear bridge artifact; recover the physical-source map')
    if 'C512 split-view' not in manifest.get('missing_composition',''):
        raise ValueError('bridge provenance does not disclose missing physical-view composition')
    gather=np.fromfile(mapfile,'<i4')
    np.testing.assert_array_equal(gather,recover(width,height)[0])
    if hashlib.sha256(mapfile.read_bytes()).hexdigest()!=manifest['output_sha256']:
        raise ValueError('map hash changed')
    if checked_source()!=(manifest['ptx_sha256'],manifest['kernel_sha256']):
        raise ValueError('PTX changed')
    if checked_inverse_source()!=manifest.get('inverse_kernel_sha256'):
        raise ValueError('inverse PTX provenance missing or changed')
    original=ROOT/'build'/('split512_bridge' if width==8 else 'split512_bridge_32x8')/'head_device.f32'
    if not original.is_file():raise FileNotFoundError('run python tools/check_split512_bridge.py first')
    head=np.fromfile(original,'<f4')
    if len(head)!=width*height*1024:raise ValueError('wrong block30 head extent')
    cases=dict(monotonic=np.arange(len(gather),dtype=np.float32)*np.float32(.0078125),
               block30_head=head)
    for name,x in cases.items():
        folder=ROOT/'build/vit_bridge_ptx_cases'/f'{name}_{width}x{height}';folder.mkdir(parents=True,exist_ok=True)
        x.tofile(folder/'input.f32');x[gather].tofile(folder/'expected.f32')
        subprocess.run([str(exe),str(folder),str(mapdir),str(len(gather))],cwd=ROOT,check=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=8);a=p.parse_args()
    raise SystemExit(run(a.width))
