#!/usr/bin/env python3
"""Derive an SM120 PTX source-linear->ViT map; original code is never run.

The original PTX source is a physical C512-buffer order. Applying this map
directly to logical HWC is a control fixture, not the real model bridge: the
unpublished C512 split-view cell permutation must be composed first.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PTX=ROOT/'dlss5-analysis/cubins/cubin_05_sm_120_ptx.ptx'
ENTRY='.visible .entry cc_vit_1d_repack_2d_to_1d_fp8('
NEXT='.visible .entry cc_vit_1d_repack_1d_to_2d('
INVERSE='.visible .entry cc_vit_1d_repack_1d_to_2d_fp8('
INVERSE_NEXT='.visible .entry cc_vit_1d_ffn_expand('


def checked_source():
    content=PTX.read_bytes();s=content.decode('utf-8')
    body=s[s.index(ENTRY):s.index(NEXT,s.index(ENTRY))]
    # These are the address-forming instructions transcribed below. Fail on a
    # different cubin/PTX instead of silently reusing the mapping.
    required=('ld.param.v2.b32 {%r1, %r2}', 'mul.lo.s32 %r3, %r1, %r2',
              'sub.s32 %r9, %r72, %r27', 'shr.s32 %r11, %r26, 8',
              'div.s32 %r28, %r11, %r2', 'sub.s32 %r30, %r11, %r29',
              'mad.lo.s32 %r49, %r44, %r8, %r45',
              'shl.b32 %r50, %r49, 12', 'add.s32 %r55, %r54, %r48',
              'and.b32 %r65, %r57, -4096', 'add.s32 %r70, %r69, %r64',
              'ld.global.b32 %r73, [%rd6]', 'st.global.b32 [%rd8], %r73')
    for instruction in required:
        if instruction not in body:raise ValueError(f'PTX address contract changed: {instruction}')
    return hashlib.sha256(content).hexdigest(),hashlib.sha256(body.encode()).hexdigest()


def checked_inverse_source():
    s=PTX.read_text(encoding='utf-8')
    body=s[s.index(INVERSE):s.index(INVERSE_NEXT,s.index(INVERSE))]
    required=('ld.param.v2.b32 {%r6, %r1}', 'shr.s32 %r19, %r16, 8',
              'div.s32 %r20, %r19, %r1', 'sub.s32 %r22, %r19, %r21',
              'and.b32 %r46, %r38, -4096', 'add.s32 %r51, %r50, %r45',
              'mad.lo.s32 %r55, %r26, %r5, %r33', 'add.s32 %r60, %r59, %r54',
              'ld.global.b32 %r52, [%rd6]', 'st.global.b32 [%rd8], %r52')
    for instruction in required:
        if instruction not in body:raise ValueError(f'inverse PTX address contract changed: {instruction}')
    return hashlib.sha256(body.encode()).hexdigest()


def inverse_addresses(width,height):
    """Static inverse PTX r51 source/r60 destination, preserving b32 lanes."""
    checked_inverse_source()
    token=np.arange(width*height,dtype=np.int32)[:,None]
    word=np.arange(256,dtype=np.int32)[None,:]
    y=token//width;x=token%width
    cell=(y//4)*(width//4)+x//4
    within=(y%4)*4+x%4
    low4=(word*4)%16
    common=(word//8)*128+((word%8)//4)*2+low4
    source_word=(token//16)*4096+common+((token%16)//8)+(token%8)*16
    destination_word=cell*4096+common+(within//8)+(within%8)*16
    lane=np.arange(4,dtype=np.int32)
    return (source_word[:,:,None]*4+lane).reshape(-1),(destination_word[:,:,None]*4+lane).reshape(-1)


def bit_gather(index,positions):
    result=np.zeros_like(index,dtype=np.int32)
    for dest,source in enumerate(positions):result|=((index>>source)&1)<<dest
    return result


def recover(width,height):
    if width<4 or height<4 or width%4 or height%4 or width*height>64:
        raise ValueError('only complete 4x4 cells up to 64 ViT tokens are covered')
    checked_source()
    token=np.arange(width*height,dtype=np.int32)[:,None]
    word=np.arange(256,dtype=np.int32)[None,:]
    y=token//width;x=token%width
    cell=(y//4)*(width//4)+x//4
    within=(y%4)*4+x%4
    low4=(word*4)%16
    common=(word//8)*128+((word%8)//4)*2+low4
    # PTX r55: physical source word. r49 is the 4x4 cell address,
    # r43 is the pixel within that cell; the other terms come from r9/r10.
    source_word=cell*4096+common+(within//8)+(within%8)*16
    # PTX r70: physical destination word. r61 is raster token mod 16;
    # r65 selects the 16-token group. ld/st are b32, preserving byte lanes.
    dest_word=(token//16)*4096+common+((token%16)//8)+(token%8)*16
    n=width*height*1024
    source_index=(source_word[:,:,None]*4+np.arange(4)).reshape(-1)
    dest_index=(dest_word[:,:,None]*4+np.arange(4)).reshape(-1)
    if (len(np.unique(source_index))!=n or source_index.min()!=0 or source_index.max()!=n-1 or
        len(np.unique(dest_index))!=n or dest_index.min()!=0 or dest_index.max()!=n-1):
        raise ValueError('PTX source/destination addresses are not bijective')
    # Original ViT bytes are unswizzled using these output-position bit lists
    # (check_native_vit_expand.py). Result maps [token,channel] -> PTX source
    # physical byte index. Logical HWC needs the C512 split-view permutation.
    vit_token=bit_gather(dest_index,[2,6,7,8,14,15])
    vit_channel=bit_gather(dest_index,[0,1,3,4,5,9,10,11,12,13])
    logical_dest=vit_token*1024+vit_channel
    if len(np.unique(logical_dest))!=n or logical_dest.min()!=0 or logical_dest.max()!=n-1:
        raise ValueError('PTX destination is not a complete ViT token/channel map')
    # The gather source is the PTX physical ld.global address, not the
    # thread's linear word number. The latter silently made a bijective but
    # incorrect map (S251 audit).
    gather=np.empty(n,dtype=np.int32);gather[logical_dest]=source_index
    # Independent composition: fill source physical with its own address,
    # simulate the PTX b32 copy to r70, then apply the ViT unswizzle.
    physical_source=np.arange(n,dtype=np.int32)
    physical_dest=np.empty(n,dtype=np.int32);physical_dest[dest_index]=physical_source[source_index]
    composed=np.empty(n,dtype=np.int32);composed[logical_dest]=physical_dest[dest_index]
    np.testing.assert_array_equal(gather,composed)
    inverse_source,inverse_dest=inverse_addresses(width,height)
    np.testing.assert_array_equal(inverse_source,dest_index)
    np.testing.assert_array_equal(inverse_dest,source_index)
    inverse_logical=bit_gather(inverse_source,[2,6,7,8,14,15])*1024
    inverse_logical+=bit_gather(inverse_source,[0,1,3,4,5,9,10,11,12,13])
    inverse_gather=np.empty(n,dtype=np.int32)
    inverse_gather[inverse_dest]=inverse_logical
    # This catches the earlier thread-number-vs-physical-source error: an
    # address-level inverse must undo the logical gather itself.
    np.testing.assert_array_equal(inverse_gather,np.argsort(gather))
    return gather,source_index,dest_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('--out',type=Path,default=ROOT/'build/vit_bridge_ptx')
    a=p.parse_args();gather,src,dst=recover(a.width,a.height)
    if a.out.exists():raise FileExistsError(f'refusing to overwrite {a.out}')
    a.out.mkdir(parents=True)
    # Historical filename retained for existing standalone fixtures. Its
    # physical-source precondition is explicit in PROVENANCE, never implicit.
    gather.astype('<i4').tofile(a.out/'hwc-to-vit.i32')
    ptx_sha,kernel_sha=checked_source()
    report=dict(source=str(PTX.resolve()),ptx_sha256=ptx_sha,kernel_sha256=kernel_sha,
                inverse_kernel_sha256=checked_inverse_source(),
                derivation='static SM120 PTX r55/r70 word addresses plus published ViT output bit layout',
                inverse_crosscheck='independent FP8 inverse PTX r51/r60 addresses reverse every forward b32 word',
                source_layout='original C512 physical buffer; fixture supplies source-linear logical HWC',
                source_index_semantics='PTX ld.global physical byte index',
                missing_composition='C512 split-view cell_output_to_hwc; do not use map as recovered real HWC-to-ViT',
                width=a.width,height=a.height,entries=len(gather),
                source_bijective=True,destination_bijective=True,
                output_sha256=hashlib.sha256((a.out/'hwc-to-vit.i32').read_bytes()).hexdigest(),
                original_kernel_executed=False)
    (a.out/'PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
