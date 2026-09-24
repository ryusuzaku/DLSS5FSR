#!/usr/bin/env python3
"""Synthetic checks for C512 cell-map composition; creates no model map."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np
from compose_vit_bridge import load_cell_map,compose
from recover_vit_bridge_ptx import recover,inverse_addresses,bit_gather


def identity_probe_mapping(cell):
    """Encode logical cell coordinates into the probe's channel-bank input."""
    local_pixel=cell//512
    channel=cell%512
    within=(channel%16&12)+((channel&1)<<1)+((channel&2)>>1)
    output=np.empty(16*8*512,dtype=np.uint32)
    for block in range(8):
        y=(block//4)*4+local_pixel//4
        x=(block%4)*4+local_pixel%4
        output[block*8192:(block+1)*8192]=(
            (channel//16)*(16*8*16)+(y*16+x)*16+within)
    return output


def save_probe(path,cell,mapping=None,**metadata):
    np.savez(path,output_to_input=(identity_probe_mapping(cell) if mapping is None else mapping),
             cell_output_to_hwc=cell,width=16,height=8,channels=512,**metadata)


def run():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'mapping.npz'
        for multiplier in (1,5):
            cell=(np.arange(8192,dtype=np.int32)*multiplier)%8192
            save_probe(path,cell)
            read=load_cell_map(path)
            np.testing.assert_array_equal(read,cell)
            for width,height in ((8,4),(16,4),(8,8)):
                gather,physical=compose(width,height,read)
                physical_input=np.arange(len(gather),dtype=np.int32)[physical]
                source_linear=recover(width,height)[0]
                np.testing.assert_array_equal(physical_input[source_linear],
                                              np.arange(len(gather),dtype=np.int32)[gather])
                np.testing.assert_array_equal(np.argsort(gather)[gather],
                                              np.arange(len(gather)))
                physical_gather,_,dest=recover(width,height)
                inverse_source,inverse_dest=inverse_addresses(width,height)
                inverse_logical=bit_gather(inverse_source,[2,6,7,8,14,15])*1024
                inverse_logical+=bit_gather(inverse_source,[0,1,3,4,5,9,10,11,12,13])
                inverse=np.empty(len(gather),np.int32);inverse[inverse_dest]=inverse_logical
                np.testing.assert_array_equal(inverse,np.argsort(physical_gather))
                thread_linear=np.empty(len(gather),np.int32)
                logical=bit_gather(dest,[2,6,7,8,14,15])*1024
                logical+=bit_gather(dest,[0,1,3,4,5,9,10,11,12,13])
                thread_linear[logical]=np.arange(len(gather))
                if np.array_equal(inverse,np.argsort(thread_linear)):
                    raise AssertionError('inverse crosscheck failed to distinguish thread-linear bug')
                print(f'cell multiplier {multiplier}, {width}x{height}: PASS')
        cell=np.zeros(8192,dtype=np.int32)
        save_probe(path,cell)
        try:load_cell_map(path)
        except ValueError:print('duplicate cell positions: rejected')
        else:raise AssertionError('accepted duplicate C512 positions')
        cell=np.arange(8192,dtype=np.int32)
        np.savez(path,output_to_input=identity_probe_mapping(cell),
                 cell_output_to_hwc=cell,width=16,height=8,channels=256)
        try:load_cell_map(path)
        except ValueError:print('wrong channel count: rejected')
        else:raise AssertionError('accepted wrong C512 channel count')
        mapping=identity_probe_mapping(cell)
        mapping[[0,8192]]=mapping[[8192,0]]
        save_probe(path,cell,mapping=mapping)
        try:load_cell_map(path)
        except ValueError:print('wrong full-map cell geometry: rejected')
        else:raise AssertionError('accepted swapped physical cells')
        changed=cell.copy();changed[[0,1]]=changed[[1,0]]
        save_probe(path,changed,mapping=identity_probe_mapping(cell))
        try:load_cell_map(path)
        except ValueError:print('inconsistent cell/full maps: rejected')
        else:raise AssertionError('accepted inconsistent cell map')
        np.savez(path,cell_output_to_hwc=cell,width=16,height=8,channels=512)
        try:load_cell_map(path)
        except ValueError:print('missing full identity map: rejected')
        else:raise AssertionError('accepted archive without full identity map')
        cell=(np.arange(8192,dtype=np.int32)*5)%8192
        save_probe(path,cell)
        out=Path(tmp)/'composed'
        command=[sys.executable,str(Path(__file__).with_name('compose_vit_bridge.py')),
                 '--cell-map',str(path),'--width','16','--height','4','--out',str(out)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        expected=compose(16,4,cell)[0]
        np.testing.assert_array_equal(np.fromfile(out/'hwc-to-vit.i32','<i4'),expected)
        np.testing.assert_array_equal(np.fromfile(out/'vit-to-hwc.i32','<i4'),np.argsort(expected))
        provenance=json.loads((out/'PROVENANCE.json').read_text())
        if not provenance['full_identity_map_consistent']:
            raise AssertionError('full identity-map validation was not recorded')
        if provenance['original_runtime_validation']:
            raise AssertionError('synthetic map incorrectly labelled runtime-verified')
        again=subprocess.run(command,capture_output=True,text=True)
        if again.returncode==0:raise AssertionError('composer overwrote existing output')
        print('synthetic CLI output/inverse and overwrite refusal: PASS')
    return 0


if __name__=='__main__':raise SystemExit(run())
