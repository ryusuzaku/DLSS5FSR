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


def run(candidate_block55=False, from39=False, output_root=None, chain_root=None,
        from38=False, encoder_skip30=False, candidate_vit16=False,
        candidate_encoder22=False, candidate_skip14=None, candidate_encoder22_down=None):
    if (encoder_skip30 or candidate_vit16 or candidate_encoder22 or candidate_skip14 is not None) and (chain_root is None or output_root is None):
        raise ValueError('custom candidate requires explicit chain/output roots')
    if (candidate_skip14 is None) != (candidate_encoder22_down is None):
        raise ValueError('candidate skip14 needs its matching candidate encoder22 downsample')
    if candidate_skip14 is not None and not candidate_encoder22:
        raise ValueError('candidate skip14 requires a connected candidate encoder22 chain')
    if from39 or from38 or encoder_skip30 or candidate_vit16 or candidate_encoder22:
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
        if encoder_skip30 and chain['case']!='image_encoder_skip30':
            raise ValueError('candidate encoder skip ancestry differs')
        if candidate_vit16 and chain['case']!='image_candidate_vit16':
            raise ValueError('candidate ViT16 ancestry differs')
        if candidate_encoder22 and chain['case']!='image_candidate_encoder22':
            raise ValueError('candidate encoder22 ancestry differs')
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
    candidate_skip14_sha256=None
    candidate_down14_sha256=None
    if candidate_skip14 is None:
        skip=np.empty_like(speer);skip[...,p128]=speer
    else:
        down14_root=Path(candidate_skip14).resolve()
        down14=json.loads((down14_root/'report.json').read_text())
        skip_path=Path(down14['block14_skip_device_path'])
        if (down14['source_model_sha256']!=src['model_sha256'] or
                down14['source_image_sha256']!=src['image_sha256'] or
                not down14['hip_scalar_exact'] or
                digest(skip_path)!=down14['block14_skip_device_sha256'] or
                digest(down14_root/'output_device.f32')!=down14['block14_down_device_sha256']):
            raise ValueError('candidate block14 skip/downsample ancestry differs')
        down22_root=Path(candidate_encoder22_down).resolve()
        down22=json.loads((down22_root/'report.json').read_text())
        c256_root=Path(down22['block22_skip_device_path']).parents[2]
        c256=json.loads((c256_root/'report.json').read_text())
        parent=c256.get('candidate_block14_parent')
        if (down22['source_model_sha256']!=src['model_sha256'] or
                down22['source_image_sha256']!=src['image_sha256'] or
                not down22['hip_scalar_exact'] or
                digest(down22_root/'output_device.f32')!=down22['block22_down_device_sha256'] or
                digest(Path(down22['block22_skip_device_path']))!=down22['block22_skip_device_sha256'] or
                down22['block22_skip_device_sha256']!=chain.get('candidate_skip22_device_sha256') or
                c256['final_device_sha256']!=down22['block21_device_sha256'] or
                parent is None or parent['block14_down_device_sha256']!=down14['block14_down_device_sha256']):
            raise ValueError('candidate block14-to-block22-to-decoder48 ancestry differs')
        skip=np.fromfile(skip_path,'<f4').reshape(32,32,128)
        candidate_skip14_sha256=digest(skip_path)
        candidate_down14_sha256=down14['block14_down_device_sha256']
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
    report=dict(case=('image_candidate_encoder14' if candidate_skip14 is not None else
                      'image_candidate_encoder22' if candidate_encoder22 else
                      'image_candidate_vit16' if candidate_vit16 else
                      'image_encoder_skip30' if encoder_skip30 else
                      'image_from38' if from38 else 'image_from39' if from39 else
                      'image_from_block48' if candidate_block55 else 'image_fp8'),
                input_extent=[16,16,256],output_extent=[32,32,128],
                source_model_sha256=src['model_sha256'],
                source_image_sha256=src['image_sha256'],
                source_block55_peer_sha256=src['tensor_sha256']['block55'],
                source_skip14_peer_sha256=src['tensor_sha256']['skip14'],
                candidate_skip14_device_sha256=candidate_skip14_sha256,
                candidate_block14_down_device_sha256=candidate_down14_sha256,
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
    parser.add_argument('--encoder-skip30',action='store_true',help='use the candidate encoder-skip C256 chain')
    parser.add_argument('--candidate-vit16',action='store_true',help='use the 16-token candidate ViT C256 chain')
    parser.add_argument('--candidate-encoder22',action='store_true',help='use the candidate encoder22 C256 chain')
    parser.add_argument('--candidate-skip14',type=Path,help='candidate encoder14 downsample fixture, including its matching skip')
    parser.add_argument('--candidate-encoder22-down',type=Path,help='matching candidate encoder22 downsample fixture')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--chain-root',type=Path,help='C256 block48–55 fixture directory')
    args=parser.parse_args()
    run(args.candidate_block55,args.from39,args.output_root,args.chain_root,
        args.from38,args.encoder_skip30,args.candidate_vit16,args.candidate_encoder22,
        args.candidate_skip14,args.candidate_encoder22_down)
