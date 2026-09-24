#!/usr/bin/env python3
"""Continue a derived C256 device output through one ordinary decoder block.

This is a candidate chain: encoder22 skip is synthetic and C256 logical
coefficient coordinates remain an extrapolation of measured C64/C128 maps.
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


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(folder,values):
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in values.items():
        array=np.asarray(array,dtype='<f4')
        if not np.isfinite(array).all():raise ValueError(f'nonfinite {folder.name}/{name}')
        array.tofile(folder/f'{name}.f32')


def run(block=49,previous=None):
    if block not in range(49,56):raise ValueError('ordinary C256 decoder block must be 49..55')
    shift=SHIFTS[block-49]
    evidence=audit_residual_ptx()
    if not evidence['ordinary_c256_block_check']['same_relative_addresses']:
        raise ValueError('ordinary C256 PTX residual transfer failed')
    previous=Path(previous) if previous is not None else ROOT/'build/block48_candidate_output_derived/output_device.f32'
    expected_previous=previous.with_name('output.f32')
    if not previous.is_file():raise FileNotFoundError(f'input for block{block} missing: {previous}')
    if previous.read_bytes()!=expected_previous.read_bytes():
        raise ValueError(f'input device output for block{block} differs from reference')
    root=ROOT/'build'/f'decoder{block}_candidate_derived'
    x=np.fromfile(previous,'<f4').reshape(HEIGHT,WIDTH,CHANNELS)
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=((WIDTH+px+7)//8)*8;hh=((HEIGHT+py+7)//8)*8
    padded=np.pad(x,((py,hh-HEIGHT-py),(px,ww-WIDTH-px),(0,0)))
    windows=padded.reshape(hh//8,8,ww//8,8,CHANNELS).transpose(0,2,1,3,4).reshape(-1,64,CHANNELS)
    tokens=len(windows)*64
    spatial=root/'spatial'
    save(spatial,dict(input=x,windows=windows))
    subprocess.run([str(ROOT/'build/spatial256_window_test.exe'),str(spatial),
                    str(WIDTH),str(HEIGHT),str(shift)],cwd=ROOT,check=True)
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
    chunks=[attention_reference(feature[i:i+64],qkv,bias,scales,projection,attention_skip)
            for i in range(0,tokens,64)]
    attention={name:np.concatenate([part[name] for part in chunks],axis=0)
               for name in chunks[0]}
    attn_folder=root/'attention'
    save(attn_folder,dict(feature=feature,qkv_weights=qkv,bias=bias,scales=scales,
                          projection_weights=projection,attention_skip=attention_skip,**attention))
    subprocess.run([str(ROOT/'build/c256_attention_candidate_test.exe'),str(attn_folder),str(len(windows))],
                   cwd=ROOT,check=True)
    residual_path=attn_folder/'projection_residual_device.f32'
    if residual_path.read_bytes()!=(attn_folder/'projection_residual.f32').read_bytes():
        raise AssertionError('block49 device attention output differs')

    output_windows=np.fromfile(residual_path,'<f4').reshape(hh//8,ww//8,8,8,CHANNELS)
    image=output_windows.transpose(0,2,1,3,4).reshape(hh,ww,CHANNELS)[py:py+HEIGHT,px:px+WIDTH]
    out_folder=root/'output'
    save(out_folder,dict(windows=output_windows,output=image))
    subprocess.run([str(ROOT/'build/spatial256_output_test.exe'),str(out_folder),
                    str(WIDTH),str(HEIGHT),str(shift)],cwd=ROOT,check=True)
    output_device=out_folder/'output_device.f32'
    if output_device.read_bytes()!=(out_folder/'output.f32').read_bytes():
        raise AssertionError('block49 device spatial output differs')
    report=dict(block=block,shift=shift,extent=[WIDTH,HEIGHT,CHANNELS],windows=len(windows),tokens=tokens,
                tensor_sha256=digest(raw_path),input_device_sha256=digest(previous),
                output_device_sha256=digest(output_device),
                exact_stage_checks=['spatial gather/scatter','FFN x4','attention x8','output scatter'],
                input_provenance=str(previous.relative_to(ROOT)),
                encoder22_skip='synthetic control inherited from block48',
                map_status='candidate C256 extension and PTX-supported residual order transfer',
                original_kernel_executed=False,original_runtime_validation=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return output_device


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block',type=int,choices=range(49,56),default=49)
    p.add_argument('--input',type=Path)
    a=p.parse_args();run(a.block,a.input)
