"""S243 head arithmetic and strict recovered-layout loading.

Reference contract: Kien's MIT native_c32_reference.block, imported unchanged.
No original NVIDIA code is executed. The logical-weight API is independent of
archive decoding. Missing/invalid recovered maps are errors, never a fallback.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ref/dlss5-port/Development'))
import native_c32_reference as N
import head70_weights as HW
import head70_shipped_body as S


def trace(tiles, weights):
    w1,w2,qw,kw,vw,pw,bias,scale,fs,ats = weights
    H,F=N.H,N.F
    expanded=H(F(tiles)@w1.T)
    gate=np.clip(expanded,-4,4)
    poly=H(gate*H(np.abs(gate)*np.float32(-.055908203125)+np.float32(.447265625))+np.float32(.89453125))
    hidden=F(H(expanded*poly))
    feature=H(tiles*fs)
    for k in range(0,128,32):
        feature=H(feature+hidden[...,k:k+32]@w2[:,k:k+32].T)
    q,k,v=[H(F(feature)@m.T) for m in (qw,kw,vw)]
    qkv=np.concatenate((q,k,v),axis=-1)
    q=F(H(N.normalize(q)*H(scale))); k=F(N.normalize(k)); v=F(v)
    norm=np.concatenate((q,k,v),axis=-1)
    scores=H(q@k.transpose(0,2,1)+bias)
    bits=np.clip(H(scores*np.float32(.044921875)+np.float32(1.30078125)),1.03125,1.5693359375).astype(np.float16).view(np.uint16).astype(np.uint32)
    ex=(((bits<<5)+0x8000)&65535).astype(np.uint16).view(np.float16).astype(np.float32)
    den=N.denominator(ex)
    prob=F(H(ex*H(1/den)))
    av=np.zeros_like(tiles)
    for k in (0,32): av=H(av+prob[:,:,k:k+32]@v[:,k:k+32])
    projection=F(av)@pw.T
    body=H(projection.astype(np.float64)+H(feature*ats).astype(np.float64))
    # Do not silently evolve the trace into a different oracle.
    np.testing.assert_array_equal(body,N.block(tiles,weights,raw_output=True))
    return dict(expanded=expanded,hidden=hidden,ffn=feature,qkv=qkv,qknorm=norm,
                scores=scores,exp=ex,den=den,prob=prob,context=av,body=body)


def packed(weights):
    w1,w2,q,k,v,p,bias,scale,g1,g2=weights
    flat=np.concatenate([a.ravel() for a in (w1,w2,q,k,v,p,bias,np.array([scale]),g1,g2)]).astype('<f4')
    if flat.size!=16449 or not np.isfinite(flat).all():
        raise ValueError('invalid logical head weights')
    return flat


def shipped_logical_smoke(body):
    """Actual head bytes in SHIPPED maps: arithmetic smoke, NOT recovered maps."""
    w=S.weights(body)
    qkv=np.asarray(w['qkv'],np.float32).reshape(32,96).T
    return (np.asarray(w['w1'],np.float32)[0].T,
            np.asarray(w['w2'],np.float32)[0].T,
            *np.split(qkv,3),np.asarray(w['proj'],np.float32).T,
            np.asarray(w['bias'],np.float32)[0],np.float32(w['temp'][0]),
            np.asarray(w['g1'],np.float32),np.asarray(w['g2'],np.float32))


def _map(data,key,count,extent):
    v=data[key]
    if v.shape!=(count,) or v.dtype.kind not in 'iu' or np.any(v<0) or np.any(v>=extent):
        raise ValueError(f'invalid recovered map {key}')
    return v.astype(np.int64)


def _matrix(raw,rows,cols,shape,fp8=True):
    index=rows*shape[1]+cols
    if np.unique(index).size!=np.prod(shape):
        raise ValueError('recovered matrix map is not a bijection')
    out=np.empty(shape,np.float32)
    out[rows,cols]=N.e4m3fn(raw) if fp8 else raw
    return out


def load_recovered(body, release):
    """Load the exact three upstream recovery files; no inferred C32 maps."""
    if len(body)!=20672: raise ValueError('head body size')
    release=Path(release)
    raw=np.frombuffer(body,np.uint8); half=raw.view('<f2').astype(np.float32)
    with np.load(release/'preblock-ffn-byte-layout/layout.npz',allow_pickle=False) as f:
        w1=_matrix(raw[:4096],_map(f,'w1_hidden',4096,128),_map(f,'w1_input',4096,32),(128,32))
        w2=_matrix(raw[4096:8192],_map(f,'w2_output',4096,32),_map(f,'w2_hidden',4096,128),(32,128))
    mats=[]
    with np.load(release/'preblock-attention-layout/matrix-layout.npz',allow_pickle=False) as a:
        for offset,kind in ((8288,'v'),(9312,'v'),(10336,'v'),(19568,'projection')):
            mats.append(_matrix(raw[offset:offset+1024],_map(a,kind+'_output',1024,32),
                                _map(a,kind+'_input',1024,32),(32,32)))
        skip=_map(a,'skip_channel',32,32)
        if np.unique(skip).size!=32: raise ValueError('skip map is not bijective')
        ats=np.empty(32,np.float32); ats[skip]=half[10296:10328]
    with np.load(release/'preblock-attention-layout/bias-layout.npz',allow_pickle=False) as b:
        bias=_matrix(half[5680:9776],_map(b,'query',4096,64),_map(b,'key',4096,64),(64,64),False)
    fs=np.empty(32,np.float32); fs[HW.ORDER]=half[4104:4136]
    weights=(w1,w2,*mats,bias,np.frombuffer(body,'<f4',1,19552)[0],fs,ats)
    packed(weights)  # validate finite values and counts before use
    return weights
