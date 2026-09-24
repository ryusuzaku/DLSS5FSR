#!/usr/bin/env python3
"""Check block48 C256 device-prefix output at its shifted 8x8 window boundary."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def run(derived=False):
    exe=ROOT/'build/spatial256_window_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    prefix=ROOT/'build'/('upsample48_prefix_derived' if derived else 'upsample48_prefix')
    output=ROOT/'build'/('spatial256_window_derived' if derived else 'spatial256_window')
    for case in ('zero','seeded'):
        source=prefix/case/'merged_device.f32'
        if not source.is_file():
            raise FileNotFoundError('run python tools/check_upsample48_prefix.py first')
        expected_source=prefix/case/'merged.f32'
        if source.read_bytes()!=expected_source.read_bytes():
            raise ValueError(f'block48 {case} device prefix differs from oracle')
        x=np.fromfile(source,'<f4').reshape(16,64,256)
        if not np.isfinite(x).all():raise ValueError(f'nonfinite {case} prefix')
        for shift in range(4):
            px=4 if shift&1 else 0;py=4 if shift&2 else 0
            ww=((64+px+7)//8)*8;hh=((16+py+7)//8)*8
            padded=np.pad(x,((py,hh-16-py),(px,ww-64-px),(0,0)))
            windows=padded.reshape(hh//8,8,ww//8,8,256)
            windows=windows.transpose(0,2,1,3,4).reshape(-1,64,256)
            folder=output/f'{case}-s{shift}'
            folder.mkdir(parents=True,exist_ok=True)
            x.astype('<f4').tofile(folder/'input.f32')
            windows.astype('<f4').tofile(folder/'windows.f32')
            subprocess.run([str(exe),str(folder),'64','16',str(shift)],cwd=ROOT,check=True)
            if (folder/'windows_device.f32').read_bytes()!=(folder/'windows.f32').read_bytes():
                raise AssertionError('C256 window device bytes differ from reference')
            report=dict(case=case,shift=shift,extent=[64,16,256],
                        padded_extent=[ww,hh,256],windows=len(windows),
                        source_device_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        window_device_sha256=hashlib.sha256((folder/'windows_device.f32').read_bytes()).hexdigest(),
                        oracle='unchanged native_upsample48_reference.upsample spatial pad/window geometry',
                        comparison='exact',original_kernel_executed=False,
                        original_runtime_validation=False)
            (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    raise SystemExit(run(p.parse_args().derived))
