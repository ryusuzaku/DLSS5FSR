#!/usr/bin/env python3
"""Check a coherent image-derived head candidate across a square frame.

The public optimized FP16 model supplies block69 latent, preblock0 skip, and
input RGB from one inference. The HIP body and outer passes are compared
exactly to native-style scalar candidate arithmetic. This does not establish
original NVIDIA C32 kernel or native encoder/decoder parity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image

from audit_peer_native_c32_basis import run as audit_basis, peer_to_native
from check_block66_peer_candidate import decode, save
from head70_normalized_reference import trace, packed, N
from head70_reference import merge, windowise, dewindowise, finish
from head70_weights import extract, repack, PAD_AT

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/peer_coherent_head_inputs'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('bad RGB comparison shape or nonfinite value')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),mae=float(d.mean()),max_abs=float(d.max()),
                rmse=float(np.sqrt(np.mean(d*d))),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def png(path,linear):
    v=np.clip(linear,0,1)
    srgb=np.where(v<=.0031308,12.92*v,1.055*np.power(v,1/2.4)-.055)
    Image.fromarray(np.rint(srgb*255).astype(np.uint8),'RGB').save(path)


def checked_command(command, log):
    result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'device check failed ({result.returncode}): {log}\n{result.stdout[-3000:]}')
    if 'FAIL' in result.stdout:
        raise AssertionError(f'device check reported FAIL: {log}')
    return result.stdout.count('PASS')


def run(size=256,chunk_windows=32,latent_case='public',output_root=None):
    if size<16 or size>256 or size%8 or chunk_windows<1:
        raise ValueError('size must be 16..256 multiple of 8; chunk positive')
    if latent_case not in ('public','image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'):
        raise ValueError('unknown latent source')
    audit_basis(verbose=False)
    source_report=json.loads((SOURCE/'manifest.json').read_text())
    for name,key in [('skip_peer','skip_peer_sha256'),
                     ('latent_peer','latent_peer_sha256'),
                     ('color_linear','color_linear_sha256'),
                     ('enhanced_rgb','enhanced_rgb_sha256'),
                     ('final_rgb','final_rgb_sha256')]:
        if digest(SOURCE/f'{name}.f32')!=source_report[key]:
            raise ValueError(f'public source hash differs: {name}')
    if not source_report['same_inference_call']:
        raise ValueError('latent/skip/color are not coherent')
    out=Path(output_root) if output_root is not None else (
        ROOT/f'build/peer_head_frame_{size}' if latent_case=='public' else
        ROOT/f'build/peer_head_frame_{size}_{latent_case}')
    out.mkdir(parents=True,exist_ok=True)
    channel_map=peer_to_native(np.arange(32))
    if latent_case=='public':
        peer_main=np.fromfile(SOURCE/'latent_peer.f32','<f4').reshape(128,128,32)[:size//2,:size//2]
        main=np.empty_like(peer_main);main[...,channel_map]=peer_main
        latent_hash=source_report['latent_peer_sha256']
        latent_origin='same-image optimized public FP16 block69'
    else:
        tail_path=ROOT/'build'/('peer_decoder66_tail_audit_from38' if latent_case=='from38_fp8' else
                                'peer_decoder66_tail_audit_from39' if latent_case=='from39_fp8' else
                                'peer_decoder66_tail_audit_from48' if latent_case=='from48_fp8' else
                                'peer_decoder66_tail_audit')/'manifest.json'
        tail=json.loads(tail_path.read_text())
        if tail['source_model_sha256']!=source_report['model_sha256'] or \
           tail['source_image_sha256']!=source_report['input_image_sha256']:
            raise ValueError('AMD decoder tail used different public image/model')
        latent_path=(Path.home()/'DLSS5FSR-build-offload'/f'peer_decoder66_{latent_case.removesuffix("_fp8")}'/'block69/output/output_device.f32'
                     if latent_case in ('from39_fp8','from38_fp8') else
                     ROOT/'build/decoder69_peer_candidate'/latent_case/'output/output_device.f32')
        latent_hash=digest(latent_path)
        if latent_hash!=tail['cases'][latent_case]['block_output_stages']['block69']['candidate_native_sha256']:
            raise ValueError('AMD decoder block69 hash differs')
        main=np.fromfile(latent_path,'<f4').reshape(128,128,32)[:size//2,:size//2].copy()
        first_block=39 if latent_case=='from38_fp8' else 40 if latent_case=='from39_fp8' else 48 if latent_case=='from48_fp8' else 56 if latent_case=='from56_fp8' else 62 if latent_case=='from62_fp8' else 66
        latent_origin=f'RX9070XT candidate decoder{first_block}-69, {latent_case} public boundary inputs'
    peer_skip=np.fromfile(SOURCE/'skip_peer.f32','<f4').reshape(256,256,32)[:size,:size]
    skip=np.empty_like(peer_skip);skip[...,channel_map]=peer_skip
    color=np.fromfile(SOURCE/'color_linear.f32','<f4').reshape(256,256,3)[:size,:size].copy()
    public_enhanced=np.fromfile(SOURCE/'enhanced_rgb.f32','<f4').reshape(256,256,3)[:size,:size].copy()
    public_final=np.fromfile(SOURCE/'final_rgb.f32','<f4').reshape(256,256,3)[:size,:size].copy()
    raw=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    stage=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    ordinary,sm,ss,coeff,pad=extract(raw,stage[PAD_AT:PAD_AT+16])
    if repack(ordinary,sm,ss,coeff,pad)!=raw:
        raise AssertionError('head tensor does not round trip')
    coeff=coeff[[0,2,4]]
    merged=merge(main,skip,sm,ss)
    windows=windowise(merged[...,channel_map])
    weights=decode(np.frombuffer(ordinary,np.uint8))
    packed_weights=packed(weights)
    body_windows=np.empty_like(windows)
    chunks=[]
    for start in range(0,len(windows),chunk_windows):
        stop=min(start+chunk_windows,len(windows))
        folder=out/'body_chunks'/f'{start:04d}_{stop:04d}'
        stages=trace(windows[start:stop],weights)
        save(folder,dict(input=windows[start:stop],weights=packed_weights,
                         **stages,output=N.F(stages['body'])))
        passes=checked_command([str(ROOT/'build/c32_peer_body_test.exe'),
                                str(folder),str(stop-start)],folder/'hip_check.log')
        if passes!=12:raise AssertionError(f'expected 12 body-stage passes, got {passes}')
        device=folder/'body_device.f32'
        if device.read_bytes()!=(folder/'body.f32').read_bytes():
            raise AssertionError('HIP/scalar body differs')
        body_windows[start:stop]=np.fromfile(device,'<f4').reshape(stop-start,64,32)
        chunks.append(dict(first_window=start,last_window_exclusive=stop,
                           body_device_sha256=digest(device),stages_exact=12))
        print(f'body windows {stop}/{len(windows)}: 12 HIP stages exact',flush=True)
    peer_image=dewindowise(body_windows,size,size)
    features=np.empty_like(peer_image);features[...,channel_map]=peer_image
    blend=np.float32(.73974609375)
    rgb_results={}
    for name,scale in [('native_gain',.03125),('public_gain',1.0)]:
        enhanced=finish(features,color,coeff,scale)
        fixture=out/name
        save(fixture,dict(main=main,skip=skip,sm=sm,ss=ss,coeff=coeff,
                          color=color,features=windowise(features),
                          merged=windowise(merged),rgb=enhanced))
        passes=checked_command([str(ROOT/'build/head70_test.exe'),str(fixture),
                                str(size),str(size),str(scale)],fixture/'hip_check.log')
        if passes!=2:raise AssertionError(f'expected two outer passes, got {passes}')
        blended=enhanced+blend*(color-enhanced)
        png(out/f'{name}_enhanced.png',enhanced)
        png(out/f'{name}_blended.png',blended)
        rgb_results[name]=dict(input_scale=scale,enhanced_sha256=digest(fixture/'rgb.f32'),
                               hip_outer_stages_exact=2,
                               enhanced_vs_public=metrics(enhanced,public_enhanced),
                               blended_vs_public_final=metrics(blended,public_final))
        print(f'{name}: both HIP outer passes exact',flush=True)
    png(out/'public_final.png',public_final)
    png(out/'input_color.png',color)
    patch_regression_exact=None
    if size==256 and latent_case=='public':
        prior=ROOT/'build/head70_peer_same_image'
        prior_scale1=ROOT/'build/head70_peer_same_image_scale1'
        if (prior/'manifest.json').exists() and (prior_scale1/'manifest.json').exists():
            patch_body=np.fromfile(prior/'body/body_device.f32','<f4').reshape(64,32)
            if not np.array_equal(body_windows[15*32+15],patch_body):
                raise AssertionError('full-frame body disagrees with one-window fixture')
            for name,fixture in [('native_gain',prior),('public_gain',prior_scale1)]:
                full_rgb=np.fromfile(out/name/'rgb.f32','<f4').reshape(256,256,3)
                patch_rgb=np.fromfile(fixture/'outer/rgb.f32','<f4').reshape(8,8,3)
                if not np.array_equal(full_rgb[120:128,120:128],patch_rgb):
                    raise AssertionError(f'full-frame {name} RGB disagrees with one-window fixture')
            patch_regression_exact=True
    report=dict(size=[size,size],windows=len(windows),chunk_windows=chunk_windows,
                source_model_sha256=source_report['model_sha256'],
                source_input_image_sha256=source_report['input_image_sha256'],
                source_latent_sha256=latent_hash,latent_origin=latent_origin,
                latent_case=latent_case,
                source_skip_sha256=source_report['skip_peer_sha256'],
                head_tensor_sha256=hashlib.sha256(raw).hexdigest(),
                body_windows_sha256=hashlib.sha256(body_windows.astype('<f4').tobytes()).hexdigest(),
                body_chunks=chunks,rgb=rgb_results,
                one_window_regression_exact=patch_regression_exact,
                image_sha256={name:digest(out/f'{name}.png') for name in
                              ('native_gain_enhanced','native_gain_blended',
                               'public_gain_enhanced','public_gain_blended',
                               'public_final','input_color')},
                input_color_vs_public_final=metrics(color,public_final),
                original_kernel_executed=False,original_runtime_validation=False,
                native_amd_encoder_decoder_executed=False,
                native_amd_decoder66_69_executed=latent_case!='public',
                native_amd_decoder62_69_executed=latent_case in ('from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'),
                native_amd_decoder56_69_executed=latent_case in ('from56_fp8','from48_fp8','from39_fp8','from38_fp8'),
                native_amd_decoder48_69_executed=latent_case in ('from48_fp8','from39_fp8','from38_fp8'),
                native_amd_decoder40_69_executed=latent_case in ('from39_fp8','from38_fp8'),
                native_amd_decoder39_69_executed=latent_case=='from38_fp8',
                comparison='same-image public FP16 skip/color and selected latent through HIP FP8/half candidate head')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='body_chunks'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--chunk-windows',type=int,default=32)
    parser.add_argument('--latent-case',choices=('public','image_half','image_fp8','from62_fp8','from56_fp8','from48_fp8','from39_fp8','from38_fp8'),default='public')
    parser.add_argument('--output-root',type=Path)
    a=parser.parse_args();run(a.size,a.chunk_windows,a.latent_case,a.output_root)
