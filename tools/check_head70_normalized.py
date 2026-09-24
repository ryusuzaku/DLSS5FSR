#!/usr/bin/env python3
"""S243 exact device checks against the unchanged upstream logical-weight block.

Without --layout-dir, head weights use the explicitly labelled shipped logical
reading for arithmetic smoke tests. This is NOT recovered model connectivity.
With --layout-dir, require upstream's three recovered .npz files; never guess.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import head70_reference as H
import head70_weights as HW
import head70_normalized_reference as R

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',type=Path,default=ROOT/'build/head70_normalized_test.exe')
    ap.add_argument('--out',type=Path,default=ROOT/'build/head70_normalized_fixtures')
    ap.add_argument('--layout-dir',type=Path)
    ap.add_argument('--emit-only',action='store_true')
    a=ap.parse_args()
    raw=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    stage=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    ext=HW.extract(raw,stage[HW.PAD_AT:HW.PAD_AT+16])
    assert HW.repack(*ext)==raw
    body,sm,ss,coeff,_=ext;coeff=coeff[[0,2,4]]
    logical=R.load_recovered(body,a.layout_dir) if a.layout_dir else R.shipped_logical_smoke(body)
    reading='recovered' if a.layout_dir else 'shipped-logical-smoke (maps unverified)'
    print('Head weight reading: '+reading,flush=True)
    rng=np.random.default_rng(0x24370)
    Hh,F=R.N.H,R.N.F
    def fp8(shape,lo=-.25,hi=.25):return F(Hh(rng.uniform(lo,hi,shape)))
    synth=(fp8((128,32)),fp8((32,128)),fp8((32,32)),fp8((32,32)),fp8((32,32)),
           fp8((32,32)),Hh(rng.uniform(-3,3,(64,64))),np.float32(.75),
           Hh(rng.uniform(.2,1,(32,))),Hh(rng.uniform(.2,1,(32,))))
    cases=[('synthetic',8,8,synth),('head_single',8,8,logical),('head_rect',16,24,logical),
           ('head_zero',8,16,logical)]
    failures=0
    for name,h,w,weights in cases:
        main_in=rng.uniform(-1,1,(h//2,w//2,32)).astype(np.float32)
        skip=rng.uniform(-1,1,(h,w,32)).astype(np.float32)
        color=rng.uniform(0,1,(h,w,3)).astype(np.float32)
        if name=='head_zero':main_in.fill(0);skip.fill(0)
        merged=H.windowise(H.merge(main_in,skip,sm,ss))
        trace=R.trace(merged,weights)
        rgb=H.finish(H.dewindowise(trace['body'],h,w),color,coeff)
        norm_in=Hh(np.ldexp(rng.uniform(-1,1,(64,96)),rng.integers(-14,4,(64,96),dtype=np.int32)))
        norm_in[0]=0;norm_in[1]=np.float32(2**-24)
        norm_in[2]=1;norm_in[3,::2]=0
        q,k,v=np.split(norm_in,3,axis=-1)
        norm_expected=np.concatenate((F(Hh(R.N.normalize(q)*Hh(weights[7]))),
                                      F(R.N.normalize(k)),F(v)),axis=-1)
        den_in=Hh(np.exp2(rng.uniform(-8,5,(64,64))))
        den_in[0]=1;den_in[1]=np.float32(2**-14)
        den_expected=R.N.denominator(den_in)
        arrays=dict(main=main_in,skip=skip,color=color,sm=sm,ss=ss,coeff=coeff,
                    weights=R.packed(weights),merged=merged,rgb=rgb,
                    norm_input=norm_in,norm_expected=norm_expected,
                    den_input=den_in,den_expected=den_expected,**trace)
        folder=a.out/name;folder.mkdir(parents=True,exist_ok=True)
        for key,value in arrays.items():
            assert np.isfinite(value).all(),(name,key)
            np.asarray(value,dtype='<f4').tofile(folder/(key+'.f32'))
        manifest=dict(recipe='normalized-head-logical',reading='synthetic' if name=='synthetic' else reading,
                      width=w,height=h,tolerance=0,oracle='unchanged native_c32_reference.block(raw_output=True)',
                      tensor_sha256=hashlib.sha256(raw).hexdigest(),
                      files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.f32')})
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'{name}: {w}x{h}; upstream oracle agreement exact',flush=True)
        if not a.emit_only:
            run=subprocess.run([str(a.exe.resolve()),str(folder.resolve()),str(w),str(h)],cwd=ROOT/'build')
            failures+=run.returncode!=0
    print('Fixtures emitted; no device verdict.' if a.emit_only else
          f'{len(cases)-failures}/{len(cases)} normalized-head cases passed (60 exact comparisons)')
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
