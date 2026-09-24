#!/usr/bin/env python3
"""Check block56 C256→C128 prefix on same-image public FP16 inputs.

This is a candidate C256 basis extension with native-style FP8 input
boundaries, not an original NVIDIA-kernel oracle.
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
from native_split_reference import bits
from native_c64_reference import multiply
from native_c32_reference import H,F
from decode_tinlayout_global import e4m3fn
from audit_peer_native_c32_basis import peer_to_native_multihead

SOURCE=ROOT/'build/peer_decoder56_inputs'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('comparison shape/nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(values=int(d.size),exact=int(np.count_nonzero(d==0)),
                mae=float(d.mean()),max_abs=float(d.max()),
                correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))


def run(candidate_block55=False, from39=False, output_root=None, chain_root=None, from38=False):
    if from39 or from38:
        candidate_block55=True
    src=json.loads((SOURCE/'manifest.json').read_text())
    for name in ('block55','skip14','merge56'):
        if digest(SOURCE/f'{name}_peer.f32')!=src['tensor_sha256'][name]:
            raise ValueError(f'source {name} hash differs')
    if candidate_block55:
        candidate_root=(Path(chain_root) if chain_root is not None else
                        Path.home()/'DLSS5FSR-build-offload'/'peer_decoder48_from38' if from38 else
                        Path.home()/'DLSS5FSR-build-offload'/'peer_decoder48_from39' if from39
                        else ROOT/'build/peer_decoder48_candidate')
        native_path=candidate_root/'block55/output/output_device.f32'
        chain=json.loads((candidate_root/'manifest.json').read_text())
        if chain['blocks'][-1]['block']!=55 or digest(native_path)!=chain['final_device_sha256']:
            raise ValueError('same-image candidate block55 handoff differs')
        if (chain['source_model_sha256']!=src['model_sha256'] or
                chain.get('source_image_sha256',src['image_sha256'])!=src['image_sha256']):
            raise ValueError('same-image candidate model/image differs')
        x=np.fromfile(native_path,'<f4').reshape(16,16,256)
    else:
        xpeer=F(np.fromfile(SOURCE/'block55_peer.f32','<f4').reshape(16,16,256))
    speer=F(np.fromfile(SOURCE/'skip14_peer.f32','<f4').reshape(32,32,128))
    p256=peer_to_native_multihead(np.arange(256))
    p128=peer_to_native_multihead(np.arange(128))
    if not candidate_block55:
        x=np.empty_like(xpeer);x[...,p256]=xpeer
    skip=np.empty_like(speer);skip[...,p128]=speer
    raw_path=ROOT/'dlss5-analysis/tensors/tensor_133.bin'
    raw=np.fromfile(raw_path,np.uint8)
    if raw.size!=230176:raise ValueError('wrong block56 tensor size')
    c=128;ob=7;count=2*c*c
    rows=bits(count,[3]+list(range(6,ob+5)))
    cols=bits(count,[1,0,4,5,2]+list(range(ob+5,2*ob+1)))
    if np.unique(rows*256+cols).size!=count:raise ValueError('projection map collision')
    weights=np.empty((128,256),np.float32)
    weights[rows,cols]=e4m3fn(raw[0x18000:0x20000])
    scale=np.empty(c,np.float32)
    scale[p128]=raw[0x20100:0x20200].view('<f2').astype(np.float32)
    low=multiply(x,weights)
    merged_half=H(np.repeat(np.repeat(low,2,axis=0),2,axis=1)+skip*scale)
    merged=F(merged_half)
    out=(Path(output_root) if output_root is not None else
         (Path.home()/'DLSS5FSR-build-offload'/'upsample56_from38' if from38 else
         (Path.home()/'DLSS5FSR-build-offload'/'upsample56_from39' if from39 else
          ROOT/'build/upsample56_prefix_derived'/('image_from_block48' if candidate_block55 else 'image_fp8'))))
    out.mkdir(parents=True,exist_ok=True)
    for name,array in dict(input=x,weights=weights,scale=scale,skip=skip,
                           low=low,merged=merged).items():
        np.asarray(array,dtype='<f4').tofile(out/f'{name}.f32')
    result=subprocess.run([str(ROOT/'build/upsample56_prefix_test.exe'),str(out),'16','16'],
                          cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'hip_check.log').write_text(result.stdout)
    if result.returncode or result.stdout.count('PASS')!=2 or 'FAIL' in result.stdout:
        raise RuntimeError(result.stdout[-3000:])
    if (out/'merged_device.f32').read_bytes()!=(out/'merged.f32').read_bytes():
        raise AssertionError('prefix HIP/scalar merge differs')
    public=np.fromfile(SOURCE/'merge56_peer.f32','<f4').reshape(32,32,128)
    report=dict(case=('image_from38' if from38 else 'image_from39' if from39 else 'image_from_block48' if candidate_block55 else 'image_fp8'),
                input_extent=[16,16,256],output_extent=[32,32,128],
                source_model_sha256=src['model_sha256'],
                source_block55_peer_sha256=src['tensor_sha256']['block55'],
                source_skip14_peer_sha256=src['tensor_sha256']['skip14'],
                block56_tensor_sha256=digest(raw_path),
                input_native_sha256=digest(out/'input.f32'),
                input_from_candidate_block48_chain=candidate_block55,
                skip_native_sha256=digest(out/'skip.f32'),
                output_device_sha256=digest(out/'merged_device.f32'),
                hip_projection_merge_exact=True,
                merged_half_vs_public_fp16=metrics(merged_half[...,p128],public),
                merged_fp8_vs_public_fp16=metrics(merged[...,p128],public),
                basis='C256 P extension candidate; C128 body measured except attention residual',
                original_kernel_executed=False,original_runtime_validation=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,end='')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-block55',action='store_true')
    parser.add_argument('--from39',action='store_true',help='use the C512-derived C256 block55 candidate')
    parser.add_argument('--from38',action='store_true',help='use the AMD block39-derived C256 block55 candidate')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--chain-root',type=Path,help='C256 block48–55 fixture directory')
    args=parser.parse_args()
    run(args.candidate_block55,args.from39,args.output_root,args.chain_root,args.from38)
