#!/usr/bin/env python3
"""Validate loader mechanics on SYNTHETIC bijections, not recovered model maps.

Compares to unchanged upstream unpack_bytes; rejects absent/corrupt maps.
Temporary fixture maps never enter the reference clone or production paths.
"""
from pathlib import Path
import os
import tempfile
import numpy as np
import head70_weights as W
import head70_normalized_reference as R

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    stage=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    body=W.extract(raw,stage[W.PAD_AT:W.PAD_AT+16])[0]
    checks=0
    with tempfile.TemporaryDirectory(prefix='head70-layout-test-',dir=ROOT/'build') as tmp:
        root=Path(tmp);release=root/'release'
        fdir=release/'preblock-ffn-byte-layout';adir=release/'preblock-attention-layout'
        fdir.mkdir(parents=True);adir.mkdir(parents=True)
        try:R.load_recovered(body,release)
        except FileNotFoundError:checks+=1
        else:raise AssertionError('missing maps accepted')
        i=np.arange(4096,dtype=np.int32);j=np.arange(1024,dtype=np.int32)
        f=dict(w1_hidden=((i//32)*5+3)%128,w1_input=((i%32)*7+1)%32,
               w2_output=((i//128)*7+1)%32,w2_hidden=((i%128)*5+3)%128)
        a=dict(v_output=((j//32)*3+1)%32,v_input=((j%32)*5+7)%32,
               projection_output=((j//32)*7+5)%32,projection_input=((j%32)*3+9)%32,
               skip_channel=(np.arange(32,dtype=np.int32)*5+1)%32)
        b=dict(query=((i//64)*3+9)%64,key=((i%64)*5+7)%64)
        np.savez(fdir/'layout.npz',**f);np.savez(adir/'matrix-layout.npz',**a)
        np.savez(adir/'bias-layout.npz',**b)
        ours=R.load_recovered(body,release)
        old=Path.cwd()
        try:
            os.chdir(root)
            upstream=R.N.unpack_bytes(np.frombuffer(body,np.uint8))
        finally:os.chdir(old)
        np.testing.assert_array_equal(R.packed(ours),R.packed(upstream));checks+=1
        def reject(path,values):
            nonlocal checks
            np.savez(path,**values)
            try:R.load_recovered(body,release)
            except (ValueError,KeyError):checks+=1
            else:raise AssertionError('invalid recovered map accepted')
        bad=dict(a);bad['skip_channel']=np.zeros(32,dtype=np.int32)
        reject(adir/'matrix-layout.npz',bad)
        np.savez(adir/'matrix-layout.npz',**a)
        bad=dict(b);bad['query']=np.zeros(4096,dtype=np.int32)
        reject(adir/'bias-layout.npz',bad)
        np.savez(adir/'bias-layout.npz',**b)
        bad=dict(f);bad['w1_input']=np.full(4096,32,dtype=np.int32)
        reject(fdir/'layout.npz',bad)
        bad=dict(f);bad['w1_input']=f['w1_input'].astype(np.float32)
        reject(fdir/'layout.npz',bad)
        bad=dict(f);bad['w2_hidden']=bad['w2_hidden'][:-1]
        reject(fdir/'layout.npz',bad)
    print(f'{checks}/7 loader checks passed (synthetic maps only; no model-map recovery claimed)')
    return 0


if __name__=='__main__':raise SystemExit(main())
