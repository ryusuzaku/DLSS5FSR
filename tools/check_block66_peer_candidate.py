#!/usr/bin/env python3
"""Run block66 C32 in the public QMMA basis on the block65 device chain.

The block66 prefix input is converted by a byte-exact audited C32 basis map.
This is a candidate against scalar C32 arithmetic, not original-kernel parity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
from native_c32_reference import H,F
from decode_tinlayout_global import e4m3fn
from head70_normalized_reference import trace,packed
from compare_peer_weight_layouts import bits,peer_qmma
from audit_peer_native_c32_basis import run as audit_basis,peer_to_native

WIDTH,HEIGHT,CHANNELS=512,128,32


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw):
    if raw.size==22784:
        body=np.zeros(20672,np.uint8)
        body[:0x2000]=raw[:0x2000]
        body[0x2000:0x2060]=raw[0x2800:0x2860]
        body[0x2060:]=raw[0x28a0:]
    elif raw.size==20672:body=raw
    else:raise ValueError('C32 record must be 22784 or 20672 bytes')
    w1=peer_qmma(body,0,128,32,output_block=128)
    w2=peer_qmma(body,4096,32,128,output_block=32)
    qkv=peer_qmma(body,8288,96,32)
    projection=peer_qmma(body,19568,32,32)
    half=body.view('<f2').astype(np.float32)
    raw_bias=half[5680:9776]
    logical=np.arange(4096,dtype=np.int32)
    physical=bits(4096,[0,10,4,1,3,6,7,9,2,5,8,11])
    bias=raw_bias[physical].reshape(64,64)
    scale=body[19552:19556].view('<f4')[0]
    ffn_skip=half[4104:4136]
    attention_skip=half[10296:10328]
    values=(w1,w2,*np.split(qkv,3),projection,bias,scale,ffn_skip,attention_skip)
    packed(values)
    return values


def save(folder,values):
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in values.items():
        array=np.asarray(array,dtype='<f4')
        if not np.isfinite(array).all():raise ValueError(f'nonfinite {folder.name}/{name}')
        array.tofile(folder/f'{name}.f32')


def run(block=66,case='seeded',first_window=False,width=WIDTH,height=HEIGHT):
    if block not in range(66,70):raise ValueError('block must be 66..69')
    if case not in ('seeded','zero','image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8','from_candidate_encoder22_fp8','from_candidate_encoder14_fp8'):
        raise ValueError('unknown skip case')
    if first_window and block!=66:raise ValueError('first-window mode only supports block66')
    if width<8 or height<8 or width%8 or height%8:
        raise ValueError('C32 extent must be multiples of 8')
    audit_basis(verbose=False)
    shifts={66:0,67:3,68:1,69:2}
    shift=shifts[block]
    offload=Path.home()/'DLSS5FSR-build-offload'
    suffix='encoder_skip30' if case=='from_encoder_skip30_fp8' else 'candidate_vit16' if case=='from_candidate_vit16_fp8' else 'candidate_encoder22' if case=='from_candidate_encoder22_fp8' else 'candidate_encoder14' if case=='from_candidate_encoder14_fp8' else case.removesuffix('_fp8')
    source=((offload/f'upsample66_{suffix}'/'merged_device.f32' if block==66 else
             offload/f'peer_decoder66_{suffix}'/f'block{block-1}/output/output_device.f32')
            if case in ('from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8','from_candidate_encoder22_fp8','from_candidate_encoder14_fp8') else
            ROOT/'build/upsample66_prefix_derived'/case/'merged_device.f32' if block==66
            else ROOT/'build/block66_peer_candidate'/case/'output'/'output_device.f32' if block==67
            else ROOT/'build'/f'decoder{block-1}_peer_candidate'/case/'output'/'output_device.f32')
    if source.read_bytes()!=source.with_name('merged.f32' if block==66 else 'output.f32').read_bytes():
        raise ValueError(f'block{block} input device handoff differs')
    native=np.fromfile(source,'<f4').reshape(height,width,CHANNELS)
    map32=peer_to_native(np.arange(CHANNELS))
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=((width+px+7)//8)*8;hh=((height+py+7)//8)*8
    peer=native[...,map32]
    padded=np.pad(peer,((py,hh-height-py),(px,ww-width-px),(0,0)))
    windows=padded.reshape(hh//8,8,ww//8,8,CHANNELS).transpose(0,2,1,3,4).reshape(-1,64,CHANNELS)
    root=(offload/f'peer_decoder66_{suffix}'/f'block{block}' if case in ('from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8','from_candidate_encoder22_fp8','from_candidate_encoder14_fp8') else
          ROOT/'build'/('block66_peer_first_window' if first_window else 'block66_peer_candidate')/case
          if block==66 else ROOT/'build'/f'decoder{block}_peer_candidate'/case)
    spatial=root/'spatial';save(spatial,dict(input=native,windows=windows))
    subprocess.run([str(ROOT/'build/spatial32_peer_test.exe'),str(spatial),
                    str(width),str(height),str(shift)],cwd=ROOT,check=True)
    if (spatial/'windows_device.f32').read_bytes()!=(spatial/'windows.f32').read_bytes():
        raise AssertionError('C32 basis/spatial gather differs')
    device_windows=np.fromfile(spatial/'windows_device.f32','<f4').reshape(-1,64,32)
    if first_window:device_windows=device_windows[:1]
    records={item['name']:item['index'] for item in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    raw_path=ROOT/'dlss5-analysis/tensors'/f'tensor_{records[f"block{block}.layer0.layer"]:03d}.bin'
    weights=decode(np.fromfile(raw_path,np.uint8))
    stages=trace(device_windows,weights)
    output=F(stages['body'])
    body=root/'body';save(body,dict(input=device_windows,weights=packed(weights),**stages,output=output))
    subprocess.run([str(ROOT/'build/c32_peer_body_test.exe'),str(body),str(len(device_windows))],cwd=ROOT,check=True)
    output_peer=body/'output_device.f32'
    if output_peer.read_bytes()!=(body/'output.f32').read_bytes():
        raise AssertionError('C32 peer body device output differs')
    if not first_window:
        output_windows=np.fromfile(output_peer,'<f4').reshape(hh//8,ww//8,8,8,CHANNELS)
        peer_image=output_windows.transpose(0,2,1,3,4).reshape(hh,ww,CHANNELS)[py:py+height,px:px+width]
        native_image=np.empty_like(peer_image)
        native_image[...,map32]=peer_image
        out=root/'output';save(out,dict(windows=output_windows,output=native_image))
        subprocess.run([str(ROOT/'build/spatial32_peer_output_test.exe'),str(out),
                        str(width),str(height),str(shift)],cwd=ROOT,check=True)
        output_device=out/'output_device.f32'
        if output_device.read_bytes()!=(out/'output.f32').read_bytes():
            raise AssertionError('C32 peer/native output scatter differs')
    else:output_device=output_peer
    report=dict(block=block,case=case,shift=shift,extent=[width,height,CHANNELS],
                windows=len(device_windows),tensor_sha256=digest(raw_path),
                input_device_sha256=digest(source),output_device_sha256=digest(output_device),
                basis_audit='build/peer_native_c32_basis_audit.json',
                map_status='public QMMA C32 candidate in exact block66 transition basis; no native C32 body map oracle',
                encoder_skips=('public same-image block4 skip' if case.startswith('image_') or case in ('from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8','from_candidate_encoder22_fp8','from_candidate_encoder14_fp8') else
                               'synthetic controls inherited from blocks48/56/62/66'),
                original_kernel_executed=False,original_runtime_validation=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return output_device


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block',type=int,choices=range(66,70),default=66)
    p.add_argument('--case',choices=('seeded','zero','image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8','from_candidate_encoder22_fp8','from_candidate_encoder14_fp8'),default='seeded')
    p.add_argument('--first-window',action='store_true')
    p.add_argument('--width',type=int,default=WIDTH)
    p.add_argument('--height',type=int,default=HEIGHT)
    a=p.parse_args();run(a.block,a.case,a.first_window,a.width,a.height)
