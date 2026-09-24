#!/usr/bin/env python3
"""Continue a derived C256 device output through one decoder block.

The default fixture has a synthetic encoder22 skip. A caller can supply a
same-image block48 prefix and extent. C256 logical coefficient coordinates
remain an extrapolation of measured C64/C128 maps.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

from audit_c256_residual_ptx import run as audit_residual_ptx
from check_c256_ffn_candidate import candidate_ffn_maps,decode_ffn,reference as ffn_reference
from check_c256_attention_candidate import candidate_attention_maps,decode_attention,reference as attention_reference

ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT,CHANNELS=64,16,256
SHIFTS=(3,1,2,0,3,1,2)  # upstream decoder49..55 schedule
ENCODER_SHIFTS=(0,3,1,2,0,3,1,2)  # native encoder15..22 schedule


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(folder,values):
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in values.items():
        array=np.asarray(array,dtype='<f4')
        if not np.isfinite(array).all():raise ValueError(f'nonfinite {folder.name}/{name}')
        array.tofile(folder/f'{name}.f32')


def run(block=49,previous=None,width=WIDTH,height=HEIGHT,output_root=None):
    if block not in (*range(15,23),*range(48,56)):
        raise ValueError('C256 block must be 15..22 or 48..55')
    if width<8 or height<8 or width%8 or height%8:raise ValueError('extent must be divisible by eight')
    shift=ENCODER_SHIFTS[block-15] if block<23 else 0 if block==48 else SHIFTS[block-49]
    evidence=audit_residual_ptx()
    if not evidence['ordinary_c256_block_check']['same_relative_addresses']:
        raise ValueError('ordinary C256 PTX residual transfer failed')
    previous=Path(previous) if previous is not None else ROOT/'build/block48_candidate_output_derived/output_device.f32'
    expected_previous=previous.with_name('merged.f32' if block==48 else 'output.f32')
    if not previous.is_file():raise FileNotFoundError(f'input for block{block} missing: {previous}')
    if previous.read_bytes()!=expected_previous.read_bytes():
        raise ValueError(f'input device output for block{block} differs from reference')
    root=Path(output_root) if output_root is not None else ROOT/'build'/f'decoder{block}_candidate_derived'
    x=np.fromfile(previous,'<f4').reshape(height,width,CHANNELS)
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=((width+px+7)//8)*8;hh=((height+py+7)//8)*8
    padded=np.pad(x,((py,hh-height-py),(px,ww-width-px),(0,0)))
    windows=padded.reshape(hh//8,8,ww//8,8,CHANNELS).transpose(0,2,1,3,4).reshape(-1,64,CHANNELS)
    tokens=len(windows)*64
    spatial=root/'spatial'
    save(spatial,dict(input=x,windows=windows))
    subprocess.run([str(ROOT/'build/spatial256_window_test.exe'),str(spatial),
                    str(width),str(height),str(shift)],cwd=ROOT,check=True)
    if (spatial/'windows_device.f32').read_bytes()!=(spatial/'windows.f32').read_bytes():
        raise AssertionError('block49 spatial device input differs')

    records={r['name']:r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    index=records[f'block{block}.layer0.layer']['index']
    raw_path=ROOT/'dlss5-analysis/tensors'/f'tensor_{index:03d}.bin'
    raw=np.fromfile(raw_path,np.uint8)
    w1,w2,w3,ffn_skip=decode_ffn(raw,candidate_ffn_maps())
    flat=windows.reshape(tokens,CHANNELS)
    ffn=ffn_reference(flat,w1,w2,w3,ffn_skip)
    ffn_folder=root/'ffn'
    save(ffn_folder,dict(input=flat,w1=w1,w2=w2,w3=w3,skip=ffn_skip,**ffn))
    subprocess.run([str(ROOT/'build/c256_ffn_candidate_test.exe'),str(ffn_folder),str(tokens)],cwd=ROOT,check=True)
    feature_path=ffn_folder/'feature_device.f32'
    if feature_path.read_bytes()!=(ffn_folder/'feature.f32').read_bytes():
        raise AssertionError('block49 device FFN output differs')

    qkv,bias,scales,projection,attention_skip=decode_attention(raw,candidate_attention_maps())
    feature=np.fromfile(feature_path,'<f4').reshape(tokens,CHANNELS)
    chunks=[attention_reference(feature[i:i+64],qkv,bias,scales,projection,attention_skip,
                                raw_output=block==22)
            for i in range(0,tokens,64)]
    attention={name:np.concatenate([part[name] for part in chunks],axis=0)
               for name in chunks[0]}
    attn_folder=root/'attention'
    save(attn_folder,dict(feature=feature,qkv_weights=qkv,bias=bias,scales=scales,
                          projection_weights=projection,attention_skip=attention_skip,**attention))
    subprocess.run([str(ROOT/'build/c256_attention_candidate_test.exe'),str(attn_folder),str(len(windows)),
                    *(['raw'] if block==22 else [])],
                   cwd=ROOT,check=True)
    residual_path=attn_folder/'projection_residual_device.f32'
    if residual_path.read_bytes()!=(attn_folder/'projection_residual.f32').read_bytes():
        raise AssertionError('block49 device attention output differs')

    output_windows=np.fromfile(residual_path,'<f4').reshape(hh//8,ww//8,8,8,CHANNELS)
    image=output_windows.transpose(0,2,1,3,4).reshape(hh,ww,CHANNELS)[py:py+height,px:px+width]
    out_folder=root/'output'
    save(out_folder,dict(windows=output_windows,output=image))
    subprocess.run([str(ROOT/'build/spatial256_output_test.exe'),str(out_folder),
                    str(width),str(height),str(shift)],cwd=ROOT,check=True)
    output_device=out_folder/'output_device.f32'
    if output_device.read_bytes()!=(out_folder/'output.f32').read_bytes():
        raise AssertionError('block49 device spatial output differs')
    raw_device=None
    if block==22:
        raw_windows=np.fromfile(attn_folder/'projection_raw_device.f32','<f4').reshape(hh//8,ww//8,8,8,CHANNELS)
        raw_image=raw_windows.transpose(0,2,1,3,4).reshape(hh,ww,CHANNELS)[py:py+height,px:px+width]
        raw_folder=root/'raw_output'
        save(raw_folder,dict(windows=raw_windows,output=raw_image))
        subprocess.run([str(ROOT/'build/spatial256_output_test.exe'),str(raw_folder),
                        str(width),str(height),str(shift)],cwd=ROOT,check=True)
        raw_device=raw_folder/'output_device.f32'
        if raw_device.read_bytes()!=(raw_folder/'output.f32').read_bytes():
            raise AssertionError('block22 raw device spatial output differs')
    report=dict(block=block,shift=shift,extent=[width,height,CHANNELS],windows=len(windows),tokens=tokens,
                tensor_sha256=digest(raw_path),input_device_sha256=digest(previous),
                output_device_sha256=digest(output_device),
                raw_output_device_sha256=digest(raw_device) if raw_device else None,
                exact_stage_checks=['spatial gather/scatter','FFN x4','attention x8','output scatter'],
                input_provenance=str(previous.resolve()),
                input_boundary=('same-image public block14 downsample FP8 boundary'
                                if block==15 else 'connected candidate AMD C256 predecessor'
                                if block<23 else 'connected block48 merge device input'
                                if output_root is not None else 'synthetic control inherited from block48'),
                encoder22_skip=('declared by block48 prefix manifest'
                                if output_root is not None else 'synthetic control inherited from block48')
                               if block>=48 else None,
                map_status='candidate C256 extension and PTX-supported residual order transfer',
                original_kernel_executed=False,original_runtime_validation=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return output_device


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block',type=int,choices=(*range(15,23),*range(48,56)),default=49)
    p.add_argument('--input',type=Path)
    p.add_argument('--width',type=int,default=WIDTH)
    p.add_argument('--height',type=int,default=HEIGHT)
    p.add_argument('--output-root',type=Path)
    a=p.parse_args();run(a.block,a.input,a.width,a.height,a.output_root)
