#!/usr/bin/env python3
"""Check C512 HWC/window conversion against upstream's spatial contract."""
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'build/split512_window_test.exe'


def expected_windows(feature,shift):
    height,width,channels=feature.shape
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=(width+px+7)//8*8;hh=(height+py+7)//8*8
    xs=np.arange(ww)-px;ys=np.arange(hh)-py
    valid_x=np.ones(ww,bool) if width==4 else (xs>=0)&(xs<width)
    valid_y=np.ones(hh,bool) if height==4 else (ys>=0)&(ys<height)
    xs=xs%4 if width==4 else np.clip(xs,0,width-1)
    ys=ys%4 if height==4 else np.clip(ys,0,height-1)
    padded=feature[ys[:,None],xs[None,:]].copy()
    padded[~(valid_y[:,None]&valid_x[None,:])]=0
    result=padded.reshape(hh//8,8,ww//8,8,channels).transpose(0,2,1,3,4).reshape(-1,64,channels)
    return result


def run():
    if not EXE.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    cases=((8,8,0),(16,8,1),(12,8,3),(4,8,2),(4,4,3))
    for width,height,shift in cases:
        x=np.arange(width*height*512,dtype=np.float32).reshape(height,width,512)*np.float32(.0078125)
        windows=expected_windows(x,shift)
        folder=ROOT/'build/split512_window'/f'{width}x{height}-s{shift}'
        folder.mkdir(parents=True,exist_ok=True)
        x.tofile(folder/'input.f32');windows.tofile(folder/'windows.f32')
        result=subprocess.run([str(EXE),str(folder),str(width),str(height),str(shift),str(len(windows))],cwd=ROOT)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(run())
