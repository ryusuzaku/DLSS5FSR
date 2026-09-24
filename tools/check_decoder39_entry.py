#!/usr/bin/env python3
"""Check block39 arithmetic on source-linear control or derived C512 bridge."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from recover_vit_bridge_ptx import ROOT,recover,inverse_addresses,bit_gather,checked_inverse_source
from audit_native_vit_logical_map import logical_map

sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
import native_decoder_entry_reference as D


def run(derived=False):
    exe=ROOT/'build/decoder39_entry_test.exe'
    if not exe.is_file():raise FileNotFoundError('build with bash tools/build_split512_block.sh')
    vit_path=ROOT/'build'/('vit_block38_16x4_derived' if derived else 'vit_block38_16x4')/'projection_device.f32'
    skip_path=ROOT/'build/split512_spatial/block30-32x8-s2/final_device.f32'
    if not vit_path.is_file() or not skip_path.is_file():
        raise FileNotFoundError('run the 32x8 C512 chain and 64-token ViT31-38 chain first')
    vit=np.fromfile(vit_path,'<f4')
    skip=np.fromfile(skip_path,'<f4')
    if vit.size!=4*16*1024 or skip.size!=8*32*512:raise ValueError('wrong decoder fixture extent')
    inverse_source,inverse_dest=inverse_addresses(16,4)
    logical_vit=bit_gather(inverse_source,[2,6,7,8,14,15])*1024
    logical_vit+=bit_gather(inverse_source,[0,1,3,4,5,9,10,11,12,13])
    inverse=np.empty(len(inverse_dest),dtype=np.int32)
    inverse[inverse_dest]=logical_vit
    np.testing.assert_array_equal(inverse,np.argsort(recover(16,4)[0]))
    if derived:
        logical=logical_map(64)
        bridge_path=ROOT/'build/vit_bridge_logical_derived/16x4-hwc-to-vit.i32'
        np.testing.assert_array_equal(logical,np.fromfile(bridge_path,'<i4'))
        inverse=np.argsort(logical).astype(np.int32)
    inverse=inverse.astype('<i4')
    np.testing.assert_array_equal(np.sort(inverse),np.arange(len(inverse)))
    main=vit[inverse].reshape(4,16,1024)
    skip=skip.reshape(8,32,512)
    records={v['name']:v for v in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    record=records['block39.layer0.layer']
    weight_path=ROOT/'dlss5-analysis/tensors'/f"tensor_{record['index']:03d}.bin"
    raw=weight_path.read_bytes()
    if len(raw)!=525312:raise ValueError('wrong block39 entry tensor size')
    weights,scale=D.unpack(weight_path)
    projected=D.project(main,weights)
    output=D.decoder_entry(main,skip,(weights,scale))
    for name,arr in (('vit',vit),('main',main),('skip',skip),('weights',weights),
                     ('scale',scale),('projected',projected),('output',output)):
        if not np.isfinite(arr).all():raise ValueError(f'nonfinite {name}')
    folder=ROOT/'build'/('decoder39_entry_derived' if derived else 'decoder39_entry')
    folder.mkdir(parents=True,exist_ok=True)
    for name,arr in (('vit',vit),('main',main),('skip',skip),('weights',weights),
                     ('scale',scale),('projected',projected),('output',output)):
        np.asarray(arr,dtype='<f4').tofile(folder/f'{name}.f32')
    inverse.tofile(folder/'inverse.i32')
    if (folder/'vit.f32').read_bytes()!=vit_path.read_bytes():
        raise ValueError('block38 device handoff changed')
    if (folder/'skip.f32').read_bytes()!=skip_path.read_bytes():
        raise ValueError('block30 skip handoff changed')
    subprocess.run([str(exe),str(folder),'16','4'],cwd=ROOT,check=True)
    report=dict(scope=('source-composed logical ViT->decoder HWC; original inverse-kernel runtime unverified'
                       if derived else 'source-linear ViT->decoder HWC control; C512 split-view missing and inverse original-kernel runtime unverified'),
                inverse_ptx_sha256=checked_inverse_source(),
                source_vit_sha256=hashlib.sha256(vit_path.read_bytes()).hexdigest(),
                source_skip_sha256=hashlib.sha256(skip_path.read_bytes()).hexdigest(),
                tensor=record['index'],tensor_sha256=hashlib.sha256(raw).hexdigest(),
                oracle='unchanged native_decoder_entry_reference',comparison='exact',
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in folder.iterdir() if p.suffix in ('.f32','.i32')})
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--derived',action='store_true')
    raise SystemExit(run(p.parse_args().derived))
