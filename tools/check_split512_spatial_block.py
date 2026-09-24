#!/usr/bin/env python3
"""Check a connected spatial C512 block, including shifted/padded attention."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from check_split512_block import ROOT,R,attention_trace
from check_split512_window import expected_windows


def load_base(block):
    folder=ROOT/'build/split512_block'/f'block{block}'
    manifest_file=folder/'manifest.json'
    if not manifest_file.is_file():
        raise FileNotFoundError(f'run python tools/check_split512_block.py first: {manifest_file}')
    manifest=json.loads(manifest_file.read_text())
    for name,sha in manifest['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'base fixture changed: {name}')
    for key,sha_key in (('tensor','raw_sha256'),('projection_tensor','projection_sha256'),
                        ('attention_tensor','attention_sha256'),('final_tensor','final_sha256')):
        raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{manifest[key]:03d}.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest[sha_key]:
            raise ValueError(f'original tensor changed: {manifest[key]}')
    shapes=dict(matrix=(512,512),expand=(8,256,64),contract=(8,64,256),
                ffn_projection=(512,512),ffn_skip=(512,),qkv_weights=(3,512,512),
                scales=(16,),bias=(16,64,64),final_weights=(512,512),final_skip=(512,))
    return {name:np.fromfile(folder/f'{name}.f32','<f4').reshape(shape)
            for name,shape in shapes.items()},manifest


def run_case(block,width,height,shift,x=None,output_root=None):
    exe=ROOT/'build/split512_block_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    w,source=load_base(block)
    if x is None:
        rng=np.random.default_rng(block*100000+width*1000+height*10+shift)
        x=R.F(rng.normal(0,.25,(width*height,512)).astype(np.float32))
    else:
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(width*height,512):raise ValueError('wrong supplied input shape')
        np.testing.assert_array_equal(x,R.F(x))
    expected=R.multiply(x,w['matrix'])
    branch=R.ffwd(x,{'pre':w['matrix'],'expand':w['expand'],'contract':w['contract']})
    feature=R.F(R.multiply(branch,w['ffn_projection'],R.H(x*w['ffn_skip'])))
    feature_window=expected_windows(feature.reshape(height,width,512),shift)
    trace=attention_trace(feature_window.reshape(-1,512),w['qkv_weights'],w['bias'],w['scales'])
    context=trace['context']
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=(width+px+7)//8*8;hh=(height+py+7)//8*8
    context_hwc=context.reshape(hh//8,ww//8,8,8,512).transpose(0,2,1,3,4).reshape(hh,ww,512)[py:py+height,px:px+width].reshape(-1,512)
    # Independent spatial consumer from unchanged upstream code.
    np.testing.assert_array_equal(context_hwc,
        R.attention_window(feature.reshape(height,width,512),list(w['qkv_weights']),
                           w['bias'],w['scales'],shift).reshape(-1,512))
    final_raw=R.multiply(context_hwc,w['final_weights'],R.H(feature*w['final_skip']))
    final=R.F(final_raw)
    folder=(Path(output_root) if output_root is not None else ROOT/'build/split512_spatial')/f'block{block}-{width}x{height}-s{shift}'
    folder.mkdir(parents=True,exist_ok=True)
    data=dict(input=x,expected=expected,branch=branch,feature=feature,
              feature_window=feature_window,context_hwc=context_hwc,
              final_raw=final_raw,final=final,
              **{name:w[name] for name in w},**trace)
    for name,array in data.items():
        if not np.isfinite(array).all():raise ValueError(f'{folder.name}: nonfinite {name}')
        np.asarray(array,dtype='<f4').tofile(folder/f'{name}.f32')
    report=dict(block=block,width=width,height=height,shift=shift,
                windows=len(feature_window),original_tensor_hashes={k:source[k] for k in
                ('raw_sha256','projection_sha256','attention_sha256','final_sha256')},
                oracle='native_split_reference.attention_window and attention; exact',
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.f32')})
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([str(exe),str(folder),str(len(x)),str(width),str(height),str(shift)],cwd=ROOT,check=True)
    np.testing.assert_array_equal(np.fromfile(folder/'final_device.f32','<f4'),final.ravel())
    np.testing.assert_array_equal(np.fromfile(folder/'final_raw_device.f32','<f4'),final_raw.ravel())
    return folder


def run():
    cases=((23,8,8,0),(23,12,8,3),(40,4,8,2),(40,16,8,1),(40,4,4,3))
    for case in cases:run_case(*case)
    return 0


if __name__=='__main__':raise SystemExit(run())
