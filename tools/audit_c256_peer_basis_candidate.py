#!/usr/bin/env python3
"""Cross-check extrapolated C256 raw-index maps against public QMMA layout.

Both decoders start from the same original tensor bytes. Exact agreement is
an offline consistency check, not original-kernel execution or map recovery.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from audit_peer_native_c32_basis import peer_index,peer_to_native_multihead
from check_c256_ffn_candidate import candidate_ffn_maps
from check_c256_attention_candidate import candidate_attention_maps
from compare_peer_weight_layouts import bits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/c256_peer_basis_candidate_audit.json'


def check(label,peer,native):
    if peer.shape!=native.shape or np.any(native<0) or np.unique(native).size!=native.size:
        raise AssertionError(f'{label}: incomplete or colliding candidate map')
    if not np.array_equal(peer,native):
        bad=np.flatnonzero((peer!=native).ravel())
        raise AssertionError(f'{label}: {bad.size} raw-index mismatches; first {bad[0] if bad.size else "none"}')
    return int(peer.size)


def run():
    c=256;d=8;order=peer_to_native_multihead(np.arange(c))
    ffn=candidate_ffn_maps()
    inputs,outputs,heads,queries,keys,offsets=candidate_attention_maps()
    checks={}
    native=np.full((4*c,c),-1,np.int32)
    native[ffn['w1_hidden'],ffn['w1_input']]=np.arange(4*c*c)
    peer=peer_index(4*c,c,128)
    checks['W1']=check('W1',peer,native[np.ix_(peer_to_native_multihead(np.arange(4*c)),order)])
    native=np.full((c,4*c),-1,np.int32)
    native[ffn['w2_output'],ffn['w2_hidden']]=np.arange(128*c)
    peer=peer_index(c,128,32)
    hidden=(order//32)[:,None]*128+peer_to_native_multihead(np.arange(128))[None,:]
    checks['W2_headwise']=check('W2',peer,native[order[:,None],hidden])
    native=np.full((c,c),-1,np.int32)
    native[ffn['w3_output'],ffn['w3_input']]=np.arange(c*c)
    peer=peer_index(c,c,c)
    checks['W3']=check('W3',peer,native[np.ix_(order,order)])
    native=np.full((c,c),-1,np.int32)
    native[outputs,inputs]=np.arange(c*c)
    checks['attention_projection']=check('projection',peer,native[np.ix_(order,order)])
    qkv=peer_index(3*c,c,3*c).reshape(c//32,3,32,c).transpose(1,0,2,3).reshape(3*c,c)
    raw_index=np.arange(c*c,dtype=np.int32)
    for operator,label in enumerate('QKV'):
        native=np.full((c,c),-1,np.int32)
        native[outputs,inputs]=(raw_index//1024)*3072+operator*1024+raw_index%1024
        checks[label]=check(label,qkv[operator*c:(operator+1)*c],native[np.ix_(order,order)])
    n=(c//32)*4096
    native=heads*4096+queries*64+keys
    physical=bits(4096,[0,10,4,1,3,6,7,9,2,5,8,11])
    peer_single=np.empty(4096,np.int32);peer_single[physical]=np.arange(4096)
    peer=np.arange(c//32,dtype=np.int32).repeat(4096)*4096+np.tile(peer_single,c//32)
    checks['attention_bias']=check('bias',peer,native)
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_124.bin'
    report=dict(scope='extrapolated C256 maps versus independently written public QMMA raw-index decoder',
                source_tensor_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                channel_basis='P(c)=(c//16)*16+(c%8)*2+(c%16//8)',
                exact_raw_index_checks=checks,
                total_checked_coefficients=sum(checks.values()),
                original_kernel_executed=False,original_map_recovered=False,
                independent_asset_validation=False)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
