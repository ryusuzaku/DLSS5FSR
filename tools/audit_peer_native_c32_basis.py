#!/usr/bin/env python3
"""Audit the public QMMA decoder's basis against measured native C64/C128.

This establishes an exact C32 transition basis for a candidate path; it does
not recover the unpublished original C32 FFN/attention maps.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
from native_c64_reference import multiply
from native_c32_reference import H
from compare_peer_weight_layouts import bits,peer_qmma


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peer_to_native(c):
    c=np.asarray(c)
    return (c&~15)|(c&1)|((c&8)>>2)|((c&2)<<1)|((c&4)<<1)


def peer_to_native_multihead(c):
    c=np.asarray(c)
    return (c//16)*16+(c%8)*2+(c%16//8)


def peer_index(n,k,block):
    values=np.arange(n*k,dtype=np.int32);pieces=[]
    for first in range(0,n,block):
        width=min(block,n-first)
        panel=values[first*k:(first+width)*k].reshape(k//32,width//16,8,4,2,2,4)
        pieces.append(panel.transpose(1,4,2,0,5,3,6).reshape(width,k))
    return np.concatenate(pieces).reshape(n,k//16,4,2,2).transpose(0,1,3,2,4).reshape(n,k)


def run(verbose=True):
    report=dict(method='raw-index permutations and original tensor coefficients; no original CUDA execution',
                peer='taowen/dlss5-onnx decode_qmma_matrix',
                native='Kien measured C64/C128 FFN/attention bit maps and block66 C32 prefix reorder',
                c32_layout_status='transition basis proven; C32 body maps still candidate')
    matrix_checks={}
    for c in (64,128):
        d=c.bit_length()-1;n=4*c;count=n*c
        rows=bits(count,[3,6,7,8,9,10,11]+list(range(d+7,2*d+2)))
        cols=bits(count,[1,0,4,5,2]+list(range(12,d+7)))
        native=np.empty((n,c),np.int32);native[rows,cols]=np.arange(count)
        peer=peer_index(n,c,128)
        rowperm=peer_to_native_multihead(np.arange(n))
        colperm=peer_to_native_multihead(np.arange(c))
        if not np.array_equal(peer,native[np.ix_(rowperm,colperm)]):
            raise AssertionError(f'C{c} peer/native FFN map relation differs')
        matrix_checks[f'C{c}_W1']=dict(raw_coefficients=count,exact=True)
        # W2 is 32-output-channel headwise: native stores each 32x128 panel
        # in a sparse C x 4C matrix, while the peer removes the zero panels.
        count=128*c
        group=list(range(12,d+7))
        rows=bits(count,[3,6,7,8,9]+group)
        cols=bits(count,[1,0,4,5,2,10,11]+group)
        native=np.full((c,4*c),-1,np.int32)
        native[rows,cols]=np.arange(count)
        peer=peer_index(c,128,32)
        rowperm=peer_to_native_multihead(np.arange(c))
        hidden=(rowperm//32)[:,None]*128+peer_to_native_multihead(np.arange(128))[None,:]
        if not np.array_equal(peer,native[rowperm[:,None],hidden]):
            raise AssertionError(f'C{c} peer/native W2 headwise map differs')
        matrix_checks[f'C{c}_W2_headwise']=dict(raw_coefficients=count,exact=True)
        # W3, Q/K/V and projection share the measured CxC bit map. QKV's
        # physical 1024-byte groups are head-major, then Q/K/V within head.
        count=c*c
        rows=bits(count,[3,6,7,8,9]+list(range(10,d+5)))
        cols=bits(count,[1,0,4,5,2]+list(range(d+5,2*d)))
        rowperm=peer_to_native_multihead(np.arange(c))
        colperm=peer_to_native_multihead(np.arange(c))
        native=np.empty((c,c),np.int32);native[rows,cols]=np.arange(count)
        peer=peer_index(c,c,c)
        if not np.array_equal(peer,native[np.ix_(rowperm,colperm)]):
            raise AssertionError(f'C{c} peer/native W3 or projection map differs')
        for label in ('W3_mix','attention_projection'):
            matrix_checks[f'C{c}_{label}']=dict(raw_coefficients=count,exact=True)
        qkv=peer_index(3*c,c,3*c).reshape(c//32,3,32,c).transpose(1,0,2,3).reshape(3*c,c)
        raw_index=np.arange(count,dtype=np.int32)
        for operator,label in enumerate('QKV'):
            native=np.empty((c,c),np.int32)
            native[rows,cols]=(raw_index//1024)*3072+operator*1024+raw_index%1024
            if not np.array_equal(qkv[operator*c:(operator+1)*c],
                                  native[np.ix_(rowperm,colperm)]):
                raise AssertionError(f'C{c} peer/native {label} map differs')
            matrix_checks[f'C{c}_{label}']=dict(raw_coefficients=count,exact=True)
    report['measured_matrix_checks']=matrix_checks
    native_bias=bits(4096,[5,6,10,7,1,11])*64+bits(4096,[0,3,8,4,2,9])
    logical=np.arange(4096,dtype=np.int32)
    physical=bits(4096,[0,10,4,1,3,6,7,9,2,5,8,11])
    peer_bias=np.empty(4096,np.int32);peer_bias[physical]=logical
    if not np.array_equal(peer_bias,native_bias):
        raise AssertionError('peer bias map differs from measured C64 map')
    report['C64_bias_positions_exact']=4096
    bias_checks={}
    for c in (64,128):
        count=(c//32)*4096;d=c.bit_length()-1
        head=bits(count,list(range(12,12+d-5)))
        query=bits(count,[5,6,10,7,1,11])
        key=bits(count,[0,3,8,4,2,9])
        native=head*4096+query*64+key
        peer=np.arange(c//32,dtype=np.int32).repeat(4096)*4096+np.tile(peer_bias, c//32)
        if not np.array_equal(peer,native):
            raise AssertionError(f'C{c} peer/native multihead bias map differs')
        bias_checks[f'C{c}']=count
    report['measured_bias_positions_exact']=bias_checks
    report['measured_FFN_skip_channel_basis']='P(c)=(c//16)*16+(c%8)*2+(c%16//8)'
    folder=ROOT/'build/upsample66_prefix_derived/seeded'
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_144.bin'
    raw=np.fromfile(raw_path,np.uint8)
    peer_weight=peer_qmma(raw,0x2000,32,64)
    native_weight=np.fromfile(folder/'weights.f32','<f4').reshape(32,64)
    to_native32=peer_to_native(np.arange(32))
    to_native64=peer_to_native_multihead(np.arange(64))
    if not np.array_equal(peer_weight,native_weight[np.ix_(to_native32,to_native64)]):
        raise AssertionError('block66 transition matrix basis differs')
    x=np.fromfile(folder/'input.f32','<f4').reshape(-1,64)
    peer_low=multiply(x[:,to_native64],peer_weight)
    native_low=np.fromfile(folder/'low.f32','<f4').reshape(-1,32)
    if not np.array_equal(peer_low,native_low[:,to_native32]):
        raise AssertionError('block66 transition projection basis differs')
    peer_skip=np.fromfile(folder/'skip.f32','<f4').reshape(-1,32)[:,to_native32]
    native_merged=np.fromfile(folder/'merged_device.f32','<f4').reshape(-1,32)
    native_scale=np.fromfile(folder/'scale.f32','<f4')
    peer_up=np.repeat(np.repeat(peer_low.reshape(64,256,32),2,axis=0),2,axis=1).reshape(-1,32)
    peer_merge=H(peer_up+peer_skip*native_scale[to_native32])
    if not np.array_equal(peer_merge,native_merged[:,to_native32]):
        raise AssertionError('block66 merged C32 basis differs')
    report['block66']=dict(tensor_sha256=digest(raw_path),
                           native_output_sha256=digest(folder/'merged_device.f32'),
                           peer_to_native_channels=to_native32.tolist(),
                           transition_coefficients_exact=int(peer_weight.size),
                           projection_values_exact=int(peer_low.size),
                           merged_values_exact=int(peer_merge.size))
    out=ROOT/'build/peer_native_c32_basis_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    if verbose:print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':run()
