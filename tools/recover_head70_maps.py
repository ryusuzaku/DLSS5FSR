#!/usr/bin/env python3
"""Recover C32 bit maps from published logical weights and original raw bytes.

Input: upstream 0.22 ZIP or native-game-tiled-assets directory. No binary is run.
Fit NO numeric coefficients: infer a bit permutation from uniquely identifying
coefficient vectors across blocks 1,2,3,67,68; verify ALL coefficients, then
hold out block69 and block70. Ambiguous/non-permutation evidence is an error.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import tempfile
import numpy as np
import head70_weights as HW
import head70_normalized_reference as N

ROOT=Path(__file__).resolve().parents[1]
TRAIN=(1,2,3,67,68)
HOLDOUT=(69,70)


def digest(data):return hashlib.sha256(data).hexdigest()


def infer_permutation(raw,logical):
    """source[:,i] -> logical[:,dest[i]], inferred and then exhaustively checked."""
    a=np.asarray(raw,dtype='<f4').copy();b=np.asarray(logical,dtype='<f4').copy()
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid coefficient tables')
    n=a.shape[1];nbits=n.bit_length()-1
    if 2**nbits!=n:raise ValueError('region is not a power of two')
    # Treat IEEE signed zero as the same coefficient, as upstream matrix unpack does.
    a[a==0]=0;b[b==0]=0
    def fingerprints(x):
        result={}
        for i,col in enumerate(x.T):result.setdefault(col.tobytes(),[]).append(i)
        return result
    src,dst=fingerprints(a),fingerprints(b)
    pairs=[(v[0],dst[k][0]) for k,v in src.items() if len(v)==1 and k in dst and len(dst[k])==1]
    if len(pairs)<nbits:raise ValueError(f'not enough unique coefficient vectors ({len(pairs)})')
    si,di=np.asarray(pairs,np.int64).T;bits=[]
    for d in range(nbits):
        choices=[s for s in range(nbits) if np.array_equal((si>>s)&1,(di>>d)&1)]
        if len(choices)!=1:raise ValueError(f'destination bit {d} ambiguous/non-permutation: {choices}')
        bits.append(choices[0])
    if len(set(bits))!=nbits:raise ValueError('reused source bit')
    index=np.arange(n,dtype=np.int32);mapping=np.zeros(n,np.int32)
    for d,s in enumerate(bits):mapping|=((index>>s)&1)<<d
    if len(np.unique(mapping))!=n:raise ValueError('not a bijection')
    if not np.array_equal(a,b[:,mapping]):raise ValueError('permutation fails full coefficient comparison')
    return mapping,dict(source_bits_for_destination=bits,unique_vectors=len(pairs),
                        checked_coefficients=int(a.size))


class Assets:
    def __init__(self,path):
        self.path=Path(path);self.provenance={};self.archive=None
        if self.path.is_file():
            if not zipfile.is_zipfile(self.path):raise ValueError('assets file must be a ZIP')
            self.archive=zipfile.ZipFile(self.path)
            h=hashlib.sha256()
            with self.path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            self.provenance['archive_sha256']=h.hexdigest()
        elif not self.path.is_dir():raise FileNotFoundError(self.path)

    def read(self,stem,count):
        found=[]
        for ext,dtype in (('.f16','<f2'),('.f32','<f4')):
            name=stem+ext
            if self.archive:
                entries=[v for v in self.archive.infolist()
                         if v.filename.replace('\\','/').split('/')[-1]==name
                         and 'native-game-tiled-assets/' in v.filename.replace('\\','/')]
                for entry in entries:
                    if entry.file_size!=count*np.dtype(dtype).itemsize:raise ValueError(f'wrong size {entry.filename}')
                    found.append((entry.filename,self.archive.read(entry),dtype))
            else:
                p=self.path/name
                if p.is_file():found.append((name,p.read_bytes(),dtype))
        if len(found)!=1:raise ValueError(f'expected exactly one {stem}.f16/.f32, found {len(found)}')
        name,data,dtype=found[0];v=np.frombuffer(data,dtype).astype(np.float32)
        if len(v)!=count or not np.isfinite(v).all():raise ValueError(f'invalid weight file {name}')
        self.provenance[name]=dict(bytes=len(data),sha256=digest(data))
        return v

    def close(self):
        if self.archive:self.archive.close()


def raw_body(block,records):
    entry=records[f'block{block}.layer0.layer']
    raw=(ROOT/'dlss5-analysis/tensors'/f"tensor_{entry['index']:03d}.bin").read_bytes()
    if block==70:
        stage=raw_body(1,records)[0]
        ext=HW.extract(raw,stage[HW.PAD_AT:HW.PAD_AT+16])
        if HW.repack(*ext)!=raw:raise ValueError('head extraction roundtrip')
        return ext[0],raw
    if len(raw) not in (20672,22720):raise ValueError(f'block{block} is not a C32 stage')
    return raw[:20672],raw


def regions(body):
    raw=np.frombuffer(body,np.uint8);h=raw.view('<f2').astype(np.float32)
    return dict(w1=N.N.e4m3fn(raw[:4096]),w2=N.N.e4m3fn(raw[4096:8192]),
                q=N.N.e4m3fn(raw[8288:9312]),k=N.N.e4m3fn(raw[9312:10336]),
                v=N.N.e4m3fn(raw[10336:11360]),p=N.N.e4m3fn(raw[19568:20592]),
                bias=h[5680:9776],g1=h[4104:4136],g2=h[10296:10328])


def logical(assets,block):
    stem='post70' if block==70 else f'block{block}'
    f=assets.read(stem+'-ffn',8736);a=assets.read(stem+'-attention',8225)
    return dict(w1=f[512:4608],w2=f[4608:8704],g1=f[8704:8736],
                q=a[:1024],k=a[1024:2048],v=a[2048:3072],p=a[3072:4096],
                bias=a[4096:8192],g2=a[8193:]),a[8192]


def recover(assets,out):
    records={x['name']:x for x in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    raws={};values={};inputs={}
    for block in TRAIN+HOLDOUT:
        body,original=raw_body(block,records);r=regions(body);v,scale=logical(assets,block)
        if scale!=np.frombuffer(body,'<f4',1,19552)[0]:raise ValueError(f'block{block} temperature mismatch')
        raws[block]=r;values[block]=v;inputs[block]=dict(raw_sha256=digest(original))
    maps={};evidence={}
    for key in raws[1]:
        maps[key],evidence[key]=infer_permutation(np.stack([raws[b][key] for b in TRAIN]),
                                                np.stack([values[b][key] for b in TRAIN]))
        for block in HOLDOUT:
            if not np.array_equal(raws[block][key],values[block][key][maps[key]]):
                raise ValueError(f'held-out block{block} {key} mismatch')
    if not np.array_equal(maps['g1'],HW.ORDER):raise ValueError('recovered FFN gate differs from documented upstream order')
    if not (np.array_equal(maps['q'],maps['v']) and np.array_equal(maps['k'],maps['v'])):
        raise ValueError('Q/K do not share the recovered V map')
    head_raw=raw_body(70,records)[1];stage=raw_body(1,records)[0]
    _,sm,ss,coeff,_=HW.extract(head_raw,stage[HW.PAD_AT:HW.PAD_AT+16])
    np.testing.assert_array_equal(assets.read('post70-scales',64),np.concatenate((sm,ss)))
    np.testing.assert_array_equal(assets.read('post70-head',96),coeff[[0,2,4]].ravel())
    out=Path(out)
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='head70-map-validation-',dir=out.parent) as scratch:
        staging=Path(scratch)/'validated'
        (staging/'preblock-ffn-byte-layout').mkdir(parents=True)
        (staging/'preblock-attention-layout').mkdir()
        np.savez(staging/'preblock-ffn-byte-layout/layout.npz',w1_input=maps['w1']%32,
                 w1_hidden=maps['w1']//32,w2_output=maps['w2']//128,w2_hidden=maps['w2']%128)
        np.savez(staging/'preblock-attention-layout/matrix-layout.npz',v_input=maps['v']%32,
                 v_output=maps['v']//32,projection_input=maps['p']%32,
                 projection_output=maps['p']//32,skip_channel=maps['g2'])
        np.savez(staging/'preblock-attention-layout/bias-layout.npz',query=maps['bias']//64,key=maps['bias']%64)
        # Final independent consumer check: exported maps must reconstruct supplied
        # logical weight arrays for EVERY block, not just satisfy our map arithmetic.
        for block in TRAIN+HOLDOUT:
            body,_=raw_body(block,records);v=values[block]
            expected=np.concatenate([v[k] for k in ('w1','w2','q','k','v','p','bias')]+
                                     [np.frombuffer(body,'<f4',1,19552),v['g1'],v['g2']])
            np.testing.assert_array_equal(N.packed(N.load_recovered(body,staging)),expected)
        report=dict(method='unique cross-block coefficient vectors -> bit permutation; no coefficient fitting',
                    train=TRAIN,holdout=HOLDOUT,regions=evidence,original_tensors=inputs,
                    supplied_assets=str(assets.path.resolve()),asset_hashes=assets.provenance,
                    output_hashes={str(p.relative_to(staging)):digest(p.read_bytes()) for p in staging.rglob('*.npz')},
                    status='all coefficients, held-outs, and independent loader exact')
        (staging/'PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
        staging.rename(out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True,help='0.22 ZIP or extracted native-game-tiled-assets folder')
    p.add_argument('--out',type=Path,default=ROOT/'build/head70_recovered_layouts')
    a=p.parse_args();assets=Assets(a.assets)
    try:r=recover(assets,a.out)
    finally:assets.close()
    print(json.dumps(r,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
