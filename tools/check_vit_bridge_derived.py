#!/usr/bin/env python3
"""Exercise the source-composed logical C512/ViT bridge on AMD HIP."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from audit_native_vit_logical_map import audit, logical_map
from recover_vit_bridge_ptx import ROOT, checked_source


def run(width=16,height=4):
    if (width,height) not in ((8,4),(16,4)):raise ValueError('covered extents are 8x4 and 16x4')
    exe=ROOT/'build/vit_bridge_ptx_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    source=ROOT/'build/vit_bridge_logical_derived'
    manifest=json.loads((source/'PROVENANCE.json').read_text())
    if manifest.get('original_kernel_executed') or manifest.get('original_split_view_identity_probe_available'):
        raise ValueError('unexpected provenance claims')
    ptx_sha,kernel_sha=checked_source()
    if (ptx_sha,kernel_sha)!=(manifest['ptx_sha256'],manifest['ptx_kernel_sha256']):
        raise ValueError('original PTX changed')
    name=f'{width}x{height}-hwc-to-vit.i32'
    mapbytes=(source/name).read_bytes()
    if hashlib.sha256(mapbytes).hexdigest()!=manifest['outputs'][name]:raise ValueError('logical bridge changed')
    gather=np.frombuffer(mapbytes,'<i4')
    np.testing.assert_array_equal(gather,logical_map(width*height))
    report,_=audit(width,height)
    if not all(report[k] for k in ('cell_ok','bank_ok','repeat_ok','permutation_ok')):
        raise ValueError('C512 geometry no longer valid')
    head_path=ROOT/'build'/('split512_bridge' if width==8 else 'split512_bridge_32x8')/'head_device.f32'
    head=np.fromfile(head_path,'<f4')
    if head.size!=gather.size:raise ValueError('C512 head extent mismatch')
    mapdir=ROOT/'build'/f'vit_bridge_derived_{width}x{height}'
    mapdir.mkdir(parents=True,exist_ok=True)
    (mapdir/'hwc-to-vit.i32').write_bytes(mapbytes)
    cases=dict(monotonic=np.arange(len(gather),dtype=np.float32)*np.float32(.0078125),block30_head=head)
    for case,x in cases.items():
        folder=ROOT/'build/vit_bridge_derived_cases'/f'{case}_{width}x{height}'
        folder.mkdir(parents=True,exist_ok=True)
        x.tofile(folder/'input.f32');x[gather].tofile(folder/'expected.f32')
        subprocess.run([str(exe),str(folder),str(mapdir),str(len(gather))],cwd=ROOT,check=True)
        np.testing.assert_array_equal(np.fromfile(folder/'device.f32','<f4'),x[gather])
    (mapdir/'PROVENANCE.json').write_text(json.dumps(dict(source=str(source.resolve()),
        map_sha256=hashlib.sha256(mapbytes).hexdigest(),head_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest(),
        comparisons=list(cases),original_kernel_executed=False),indent=2)+'\n')
    print(f'{width}x{height} source-composed logical bridge: PASS (monotonic and block30 head)')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=16)
    a=p.parse_args();raise SystemExit(run(a.width))
