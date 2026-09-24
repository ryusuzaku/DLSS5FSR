#!/usr/bin/env python3
"""Check the C64 body of block62 from the block61 device output.

The encoder8 skip is synthetic. FFN/matrix/bias maps follow the measured
C64 rules; the attention residual channel order is a PTX-supported candidate.
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
from decode_tinlayout_global import e4m3fn
from check_c256_ffn_candidate import bits,reference as ffn_reference
from check_c256_attention_candidate import reference as attention_reference
from audit_c256_residual_ptx import run as audit_ptx

WIDTH,HEIGHT,CHANNELS,SHIFT=256,64,64,0


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(folder,values):
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in values.items():
        array=np.asarray(array,dtype='<f4')
        if not np.isfinite(array).all():raise ValueError(f'nonfinite {folder.name}/{name}')
        array.tofile(folder/f'{name}.f32')


def decode(raw):
    if raw.size==70048:
        ffn_offset,qkv_offset,bias_offset,scale_offset,projection_offset,attention_offset=(
            0x9000,0x9100,0xc100,0x10100,0x10110,0x11110)
    elif raw.size==61760:
        ffn_offset,qkv_offset,bias_offset,scale_offset,projection_offset,attention_offset=(
            0x7010,0x70a0,0xa0a0,0xe0a0,0xe0b0,0xf0b0)
    else:raise ValueError('C64 record must be 70048 or 61760 bytes')
    c=64;d=6;group=list(range(12,d+7));count=4*c*c
    maps=dict(w1_input=bits(count,[1,0,4,5,2]+group),
              w1_hidden=bits(count,[3,6,7,8,9,10,11]+list(range(d+7,2*d+2))),
              w2_hidden=bits(128*c,[1,0,4,5,2,10,11]+group),
              w2_output=bits(128*c,[3,6,7,8,9]+group),
              w3_input=bits(c*c,[1,0,4,5,2]+list(range(d+5,2*d))),
              w3_output=bits(c*c,[3,6,7,8,9]+list(range(10,d+5))))
    matrices=[]
    for begin,end,shape,row,col in ((0,16384,(256,64),'w1_hidden','w1_input'),
                                    (16384,24576,(64,256),'w2_output','w2_hidden'),
                                    (24576,28672,(64,64),'w3_output','w3_input')):
        matrix=np.zeros(shape,np.float32)
        rows,cols=maps[row],maps[col]
        if np.unique(rows*shape[1]+cols).size!=end-begin:
            raise ValueError(f'collision in {row}/{col}')
        matrix[rows,cols]=e4m3fn(raw[begin:end]);matrices.append(matrix)
    order=(np.arange(c)//16)*16+(np.arange(c)%8)*2+(np.arange(c)%16//8)
    ffn_skip=np.empty(c,np.float32)
    ffn_skip[order]=raw[ffn_offset:ffn_offset+128].view('<f2').astype(np.float32)
    n=c*c;i=np.arange(n)
    inputs=bits(n,[1,0,4,5,2]+list(range(d+5,2*d)))
    outputs=bits(n,[3,6,7,8,9]+list(range(10,d+5)))
    if np.unique(outputs*c+inputs).size!=n:raise ValueError('C64 attention matrix collision')
    offsets=qkv_offset+(i//1024)*3072+2048+i%1024
    qkv=[]
    for delta in (-2048,-1024,0):
        matrix=np.empty((c,c),np.float32)
        matrix[outputs,inputs]=e4m3fn(raw[offsets+delta]);qkv.append(matrix)
    bias_count=2*4096
    heads=bits(bias_count,[12])
    queries=bits(bias_count,[5,6,10,7,1,11])
    keys=bits(bias_count,[0,3,8,4,2,9])
    if np.unique(heads*4096+queries*64+keys).size!=bias_count:
        raise ValueError('C64 attention bias collision')
    bias=np.empty((2,64,64),np.float32)
    bias[heads,queries,keys]=raw[bias_offset:scale_offset].view('<f2').astype(np.float32)
    scales=raw[scale_offset:scale_offset+8].view('<f4').astype(np.float32)
    projection=np.empty((c,c),np.float32)
    projection[outputs,inputs]=e4m3fn(raw[projection_offset:projection_offset+n])
    attention_skip=np.empty(c,np.float32)
    attention_skip[order]=raw[attention_offset:attention_offset+128].view('<f2').astype(np.float32)
    return (*matrices,ffn_skip,np.stack(qkv),bias,scales,projection,attention_skip)


def run(case='seeded',first_window=False,block=62,width=WIDTH,height=HEIGHT):
    if case not in ('seeded','zero','image_half','image_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8'):raise ValueError('unknown skip case')
    if block not in range(62,66):raise ValueError('block must be 62..65')
    if first_window and block!=62:raise ValueError('first-window mode is only for block62')
    if width<8 or height<8 or width%8 or height%8:
        raise ValueError('C64 extent must be multiples of 8')
    shifts={62:0,63:3,64:1,65:2}
    shift=shifts[block]
    evidence=audit_ptx()
    if not next(item for item in evidence['measured_width_cross_checks']
                if '_2h_64_2_' in item['entry'])['same_relative_addresses']:
        raise ValueError('C64 PTX residual address transfer failed')
    offload=Path.home()/'DLSS5FSR-build-offload'
    suffix='encoder_skip30' if case=='from_encoder_skip30_fp8' else 'candidate_vit16' if case=='from_candidate_vit16_fp8' else case.removesuffix('_fp8')
    source=((offload/f'upsample62_{suffix}'/'merged_device.f32' if block==62 else
             offload/f'peer_decoder62_{suffix}'/f'block{block-1}/output/output_device.f32')
            if case in ('from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8') else
            ROOT/'build/upsample62_prefix_derived'/case/'merged_device.f32' if block==62
            else ROOT/'build/block62_candidate'/case/'output'/'output_device.f32' if block==63
            else ROOT/'build'/f'decoder{block-1}_candidate_derived'/case/'output'/'output_device.f32')
    if source.read_bytes()!=source.with_name('merged.f32' if block==62 else 'output.f32').read_bytes():
        raise ValueError(f'block{block} input device handoff differs')
    x=np.fromfile(source,'<f4').reshape(height,width,CHANNELS)
    px=4 if shift&1 else 0;py=4 if shift&2 else 0
    ww=((width+px+7)//8)*8;hh=((height+py+7)//8)*8
    padded=np.pad(x,((py,hh-height-py),(px,ww-width-px),(0,0)))
    windows=padded.reshape(hh//8,8,ww//8,8,CHANNELS).transpose(0,2,1,3,4).reshape(-1,64,CHANNELS)
    root=(offload/f'peer_decoder62_{suffix}'/f'block{block}' if case in ('from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8') else
          ROOT/'build'/('block62_candidate_first_window' if first_window else 'block62_candidate')/case
          if block==62 else ROOT/'build'/f'decoder{block}_candidate_derived'/case)
    spatial=root/'spatial';save(spatial,dict(input=x,windows=windows))
    subprocess.run([str(ROOT/'build/spatial64_test.exe'),str(spatial),str(width),str(height),str(shift)],cwd=ROOT,check=True)
    if (spatial/'windows_device.f32').read_bytes()!=(spatial/'windows.f32').read_bytes():
        raise AssertionError('C64 spatial gather differs')
    device_windows=np.fromfile(spatial/'windows_device.f32','<f4').reshape(-1,CHANNELS)
    if first_window:device_windows=device_windows[:64]
    records={item['name']:item['index'] for item in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    raw_path=ROOT/'dlss5-analysis/tensors'/f'tensor_{records[f"block{block}.layer0.layer"]:03d}.bin'
    w1,w2,w3,ffn_skip,qkv,bias,scales,projection,attention_skip=decode(np.fromfile(raw_path,np.uint8))
    ffn=ffn_reference(device_windows,w1,w2,w3,ffn_skip)
    ffn_folder=root/'ffn'
    save(ffn_folder,dict(input=device_windows,w1=w1,w2=w2,w3=w3,skip=ffn_skip,**ffn))
    subprocess.run([str(ROOT/'build/c64_ffn_candidate_test.exe'),str(ffn_folder),str(len(device_windows))],cwd=ROOT,check=True)
    feature_path=ffn_folder/'feature_device.f32'
    if feature_path.read_bytes()!=(ffn_folder/'feature.f32').read_bytes():
        raise AssertionError('C64 FFN feature differs')
    feature=np.fromfile(feature_path,'<f4').reshape(-1,CHANNELS)
    chunks=[attention_reference(feature[i:i+64],qkv,bias,scales,projection,attention_skip)
            for i in range(0,len(feature),64)]
    stages={name:np.concatenate([part[name] for part in chunks],axis=0) for name in chunks[0]}
    attn_folder=root/'attention'
    save(attn_folder,dict(feature=feature,qkv_weights=qkv,bias=bias,scales=scales,
                          projection_weights=projection,attention_skip=attention_skip,**stages))
    subprocess.run([str(ROOT/'build/c64_attention_candidate_test.exe'),str(attn_folder),str(len(chunks))],cwd=ROOT,check=True)
    residual_path=attn_folder/'projection_residual_device.f32'
    if residual_path.read_bytes()!=(attn_folder/'projection_residual.f32').read_bytes():
        raise AssertionError('C64 attention residual differs')
    if not first_window:
        output_windows=np.fromfile(residual_path,'<f4').reshape(hh//8,ww//8,8,8,CHANNELS)
        image=output_windows.transpose(0,2,1,3,4).reshape(hh,ww,CHANNELS)[py:py+height,px:px+width]
        out=root/'output';save(out,dict(windows=output_windows,output=image))
        subprocess.run([str(ROOT/'build/spatial64_output_test.exe'),str(out),str(width),str(height),str(shift)],cwd=ROOT,check=True)
        output_device=out/'output_device.f32'
        if output_device.read_bytes()!=(out/'output.f32').read_bytes():
            raise AssertionError('C64 scatter differs')
    else:output_device=residual_path
    report=dict(block=block,case=case,shift=shift,extent=[width,height,CHANNELS],
                windows=len(chunks),tensor_sha256=digest(raw_path),input_device_sha256=digest(source),
                output_device_sha256=digest(output_device),
                encoder8_skip=('public same-image half-boundary control' if case=='image_half' else
                               'public same-image FP8-boundary control' if case in ('image_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8') else
                               'synthetic control'),
                map_status='measured C64 FFN/matrix/bias maps; candidate attention residual order',
                original_kernel_executed=False,original_runtime_validation=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('seeded','zero','image_half','image_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8','from_encoder_skip30_fp8','from_candidate_vit16_fp8'),default='seeded')
    p.add_argument('--first-window',action='store_true')
    p.add_argument('--block',type=int,choices=range(62,66),default=62)
    p.add_argument('--width',type=int,default=WIDTH)
    p.add_argument('--height',type=int,default=HEIGHT)
    a=p.parse_args();run(a.case,a.first_window,a.block,a.width,a.height)
