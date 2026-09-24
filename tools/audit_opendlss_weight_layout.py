#!/usr/bin/env python3
"""Compare two published decoders' raw indices for C32/C256 matrices.

OpenDLSS-NR's packedWeightIndex/inversePackedInputIndex are transcribed from
src/nr_model.cpp. This corroborates packing independently of our formulas;
it does not replay that project's unpublished original-kernel captures.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from audit_peer_native_c32_basis import peer_index
from compare_peer_weight_layouts import bits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/opendlss_weight_layout_audit.json'
SOURCE='https://github.com/maanHimself/OpenDLSS-NR/blob/main/src/nr_model.cpp'


def inverse_input(k):
    return (k&~31)+(k&17)+((k&2)<<1)+((k&4)<<1)+((k&8)>>2)


def packed_weight(k,n,outputs):
    ki=k&31;ni=n&127;group=ni&15
    return ((k>>5)*outputs*32+(n>>7)*4096+(ni>>6)*2048+
            ((ni&63)>>4)*512+((((group&7)<<2)|((ki&15)>>2))*16)+
            (((group>>3)<<3)|((ki>>4)<<2)|(ki&3)))


def check(label,expected,actual):
    if expected.shape!=actual.shape or not np.array_equal(expected,actual):
        unequal=np.flatnonzero((expected!=actual).ravel())
        raise AssertionError(f'{label}: {unequal.size} raw-index mismatches')
    if len(np.unique(actual))!=actual.size:
        raise AssertionError(f'{label}: raw-index collision')
    return int(actual.size)


def qkv_peer(c):
    return peer_index(3*c,c,3*c).reshape(c//32,3,32,c).transpose(1,0,2,3).reshape(3*c,c)


def tiled_token(t):
    x=t&7;y=t>>3
    return (y>>2)*32+(x>>2)*16+(y&3)*4+(x&3)


def opendlss_bias_index():
    query,key=np.indices((64,64),dtype=np.int32)
    q=tiled_token(query);k=tiled_token(key)
    m=q&15;n=k&15
    lane=((m&7)<<2)|((n&7)>>1)
    fragment=np.where(m>=8,2,0)+(n&1)
    return (q>>4)*1024+(k>>4)*256+lane*8+(n>>3)*4+fragment


def run():
    checks={}
    for c in (32,256):
        if c==32:
            r,k=np.indices((128,32),dtype=np.int32)
            checks['C32_W1']=check('C32 W1',peer_index(128,32,128),packed_weight(inverse_input(k),r,128))
            r,k=np.indices((32,128),dtype=np.int32)
            checks['C32_W2']=check('C32 W2',peer_index(32,128,32),packed_weight(inverse_input(k),r,32))
        else:
            r,k=np.indices((4*c,c),dtype=np.int32)
            checks['C256_W1_experts']=check('C256 W1',peer_index(4*c,c,128),
                packed_weight((r//128)*c+inverse_input(k),r%128,128))
            r,k=np.indices((c,128),dtype=np.int32)
            checks['C256_W2_experts']=check('C256 W2',peer_index(c,128,32),
                packed_weight((r//32)*128+inverse_input(k),r%32,32))
            r,k=np.indices((c,c),dtype=np.int32)
            checks['C256_W3']=check('C256 W3',peer_index(c,c,c),packed_weight(inverse_input(k),r,c))
        r,k=np.indices((c,c),dtype=np.int32)
        checks[f'C{c}_projection']=check(f'C{c} projection',peer_index(c,c,c),
                                          packed_weight(inverse_input(k),r,c))
        r,k=np.indices((3*c,c),dtype=np.int32)
        native_row=(r%32)+(r//c)*32+((r%c)//32)*96
        checks[f'C{c}_QKV']=check(f'C{c} QKV',qkv_peer(c),
                                  packed_weight(inverse_input(k),native_row,3*c))
        heads=c//32
        expected=np.arange(heads,dtype=np.int32)[:,None,None]*4096+bits(
            4096,[0,10,4,1,3,6,7,9,2,5,8,11]).reshape(1,64,64)
        actual=np.arange(heads,dtype=np.int32)[:,None,None]*4096+opendlss_bias_index()
        checks[f'C{c}_attention_bias']=check(f'C{c} bias',expected,actual)
    records={entry['name']:entry['index'] for entry in
             json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    hashes={}
    for block in (48,67):
        path=ROOT/'dlss5-analysis/tensors'/f'tensor_{records[f"block{block}.layer0.layer"]:03d}.bin'
        hashes[str(block)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(method='exact raw-index comparison of public QMMA decoder and OpenDLSS-NR host decoder',
                opendlss_source=SOURCE,source_tensor_sha256=hashes,
                exact_raw_index_checks=checks,total_checked_coefficients=sum(checks.values()),
                independent_decoder_agreement=True,original_kernel_executed_here=False,
                original_capture_replayed_here=False,
                native_C256_C32_map_proven=False)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
