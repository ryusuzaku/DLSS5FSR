#!/usr/bin/env python3
"""Derive C512 cell and logical ViT bridge maps from two independent sources.

The upstream logical map reports a match to captured 64- and 640-token
repack data. The original SM120 PTX supplies physical source addresses.
Their composition determines the C512 cell view without executing CUDA.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from audit_native_vit_logical_map import UPSTREAM, audit, logical_map
from compose_vit_bridge import compose
from recover_vit_bridge_ptx import ROOT, checked_source, checked_inverse_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(out):
    out=Path(out)
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    reports=[];cells=[]
    for width,height in ((8,4),(16,4),(8,8)):
        report,cell=audit(width,height)
        if not all(report[k] for k in ('cell_ok','bank_ok','repeat_ok','permutation_ok')):
            raise ValueError(f'{width}x{height} does not have repeated C512 cell geometry')
        gathered,physical=compose(width,height,cell)
        expected=logical_map(width*height)
        np.testing.assert_array_equal(gathered,expected)
        np.testing.assert_array_equal(np.argsort(gathered)[gathered],np.arange(len(gathered)))
        cells.append(cell);reports.append(report)
    for cell in cells[1:]:np.testing.assert_array_equal(cell,cells[0])
    out.mkdir(parents=True)
    (out/'c512-cell-output-to-hwc.i32').write_bytes(cells[0].astype('<i4').tobytes())
    files=['c512-cell-output-to-hwc.i32']
    for (width,height),cell in zip(((8,4),(16,4),(8,8)),cells):
        gathered,_=compose(width,height,cell)
        prefix=f'{width}x{height}'
        (out/f'{prefix}-hwc-to-vit.i32').write_bytes(gathered.astype('<i4').tobytes())
        (out/f'{prefix}-vit-to-hwc.i32').write_bytes(np.argsort(gathered).astype('<i4').tobytes())
        files.extend((f'{prefix}-hwc-to-vit.i32',f'{prefix}-vit-to-hwc.i32'))
    ptx_sha,kernel_sha=checked_source()
    provenance=dict(method='upstream capture-derived logical map composed with original SM120 PTX physical-source addresses',
                    logical_source=str(UPSTREAM.resolve()),logical_source_sha256=digest(UPSTREAM),
                    ptx_sha256=ptx_sha,ptx_kernel_sha256=kernel_sha,
                    inverse_ptx_kernel_sha256=checked_inverse_source(),
                    cell_repeats_across_extents=True,comparisons=reports,
                    original_kernel_executed=False,original_runtime_validation=False,
                    original_split_view_identity_probe_available=False,
                    outputs={name:digest(out/name) for name in files})
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'build/vit_bridge_logical_derived')
    a=p.parse_args()
    print(json.dumps(derive(a.out),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
