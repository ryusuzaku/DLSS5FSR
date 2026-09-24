#!/usr/bin/env python3
"""Compose a validated upstream C512 identity-probe map with PTX ViT repack.

Without the original C512 `cell_output_to_hwc` mapping this script cannot
produce a logical-HWC->ViT bridge. Synthetic maps are for tests only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from recover_vit_bridge_ptx import ROOT,recover,checked_source


def load_cell_map(path):
    with np.load(path,allow_pickle=False) as archive:
        required={'output_to_input','cell_output_to_hwc','width','height','channels'}
        if not required.issubset(archive.files):
            raise ValueError(f'C512 split-view archive lacks {sorted(required-set(archive.files))}')
        extent=tuple(int(archive[key]) for key in ('width','height','channels'))
        if extent!=(16,8,512):
            raise ValueError(f'C512 split-view archive has wrong recovery extent: {extent}')
        mapping=np.asarray(archive['output_to_input'])
        cell=np.asarray(archive['cell_output_to_hwc'])
    count=16*8*512
    if mapping.shape!=(count,) or mapping.dtype.kind not in 'iu':
        raise ValueError('C512 output_to_input must contain 65536 integer positions')
    mapping=mapping.astype(np.int64)
    if not np.array_equal(np.sort(mapping),np.arange(count)):
        raise ValueError('C512 output_to_input is not a complete permutation')
    if cell.shape!=(8192,) or cell.dtype.kind not in 'iu':
        raise ValueError('C512 cell map must contain 8192 integer positions')
    cell=cell.astype(np.int64)
    if not np.array_equal(np.sort(cell),np.arange(8192)):
        raise ValueError('C512 cell map is not a complete permutation')
    # Reconstruct the exact coordinate contract used by the upstream
    # recover_native_split_view.py identity probe. A valid cell permutation
    # alone does not establish that the full output mapping uses it.
    pixel=(mapping%(16*8*16))//16
    y,x=pixel//16,pixel%16
    within=mapping%16
    channel=mapping//(16*8*16)*16+(within&12)+((within&1)<<1)+((within&2)>>1)
    local=((y%4)*4+x%4)*512+channel
    if not np.array_equal(local.reshape(-1,8192),np.broadcast_to(cell,(8,8192))):
        raise ValueError('C512 cell map disagrees with output_to_input coordinates')
    output_cell=np.arange(count)//8192
    if not (np.array_equal(x//4,output_cell%4) and
            np.array_equal(y//4,output_cell//4)):
        raise ValueError('C512 output_to_input has wrong 4x4 cell ordering')
    return cell.astype(np.int32)


def compose(width,height,cell):
    if cell.shape!=(8192,) or not np.array_equal(np.sort(cell),np.arange(8192)):
        raise ValueError('invalid C512 cell map')
    source_linear,_,_=recover(width,height)
    n=width*height*1024
    # Adapt the unchanged upstream prepare_native_vit_bridge.py composition:
    # logical HWC -> 4x4 cells, with two C512 banks per pixel; then physical
    # cell output order using the recovered split-view map.
    canonical=np.arange(n,dtype=np.int32).reshape(height//4,4,width//4,4,2,512)
    canonical=canonical.transpose(0,2,4,1,3,5).reshape(-1,2,8192)
    physical=np.empty_like(canonical)
    physical[:,:,np.argsort(cell)]=canonical
    physical_to_hwc=physical.reshape(-1)
    gather=physical_to_hwc[source_linear]
    if not np.array_equal(np.sort(gather),np.arange(n)):
        raise ValueError('composed HWC->ViT map is not bijective')
    return gather.astype(np.int32),physical_to_hwc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell-map',type=Path,required=True,
                   help='upstream native-c512/split-view/mapping.npz')
    p.add_argument('--width',type=int,choices=(8,16),default=16)
    p.add_argument('--height',type=int,choices=(4,8),default=4)
    p.add_argument('--out',type=Path,default=ROOT/'build/vit_bridge_composed')
    a=p.parse_args()
    if a.width*a.height>64:p.error('static PTX fixture covers at most 64 tokens')
    if not a.cell_map.is_file():raise FileNotFoundError(a.cell_map)
    if a.out.exists():raise FileExistsError(f'refusing to overwrite {a.out}')
    cell=load_cell_map(a.cell_map)
    gather,physical=compose(a.width,a.height,cell)
    ptx_sha,kernel_sha=checked_source()
    a.out.mkdir(parents=True)
    gather.astype('<i4').tofile(a.out/'hwc-to-vit.i32')
    np.argsort(gather).astype('<i4').tofile(a.out/'vit-to-hwc.i32')
    report=dict(width=a.width,height=a.height,entries=len(gather),
                cell_map_source=str(a.cell_map.resolve()),
                cell_map_sha256=hashlib.sha256(a.cell_map.read_bytes()).hexdigest(),
                cell_map_permutation=True,full_identity_map_consistent=True,
                ptx_sha256=ptx_sha,kernel_sha256=kernel_sha,
                composition='PTX source-linear map + supplied C512 split-view cell_output_to_hwc; upstream prepare_native_vit_bridge.py geometry',
                forward_sha256=hashlib.sha256((a.out/'hwc-to-vit.i32').read_bytes()).hexdigest(),
                inverse_sha256=hashlib.sha256((a.out/'vit-to-hwc.i32').read_bytes()).hexdigest(),
                original_kernel_executed=False,
                original_runtime_validation=False)
    (a.out/'PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
