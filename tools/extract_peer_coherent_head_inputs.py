#!/usr/bin/env python3
"""Extract same-image preblock0 skip and block69 latent from public FP16 ONNX.

Run with build/peer_onnx_venv/Scripts/python.exe. The temporary full-network
branch is removed after inference; outputs remain public-basis FP16 candidates,
not original NVIDIA FP8 kernel captures.
"""
from pathlib import Path
import json
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest

OUT = ROOT/'build/peer_coherent_head_inputs'
SKIP = '/graph/Cast_9_output_0'
LATENT = '/graph/dec32.2/Cast_51_output_0'
FUSED = '/graph/Transpose_2_output_0'
BODY = '/graph/post_body/Add_2_output_0'
ENHANCED = '/Clip_output_0'


def run():
    if digest(MODEL)!=MODEL_SHA256:
        raise ValueError('unrecognized public ONNX model')
    reference=ROOT/'build/peer_preblock0_skip'
    reference_report=json.loads((reference/'manifest.json').read_text())
    if digest(IMAGE)!=reference_report['input_image_sha256']:
        raise ValueError('preblock branch used a different input image')
    rgb=np.asarray(Image.open(IMAGE).convert('RGB'),np.float32)/255
    if rgb.shape!=(256,256,3):raise ValueError('expected 256x256 image')
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4).astype(np.float32)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        branch=Path(tmp)/'head_inputs.onnx'
        onnx.utils.extract_model(str(MODEL),str(branch),['rgb'],[SKIP,LATENT,FUSED,BODY,ENHANCED,'output'])
        branch_hash=digest(branch)
        session=ort.InferenceSession(str(branch),providers=['CPUExecutionProvider'])
        skip,latent,fused,body,enhanced,final=session.run(None,{'rgb':linear.transpose(2,0,1)[None]})
    if (skip.shape!=(1,256,256,32) or latent.shape!=(1,128,128,32) or
        fused.shape!=(1,256,256,32) or body.shape!=(1,256,256,32)):
        raise ValueError('unexpected public intermediate shapes')
    if enhanced.shape!=(1,3,256,256) or final.shape!=(1,3,256,256):
        raise ValueError('unexpected public RGB output shapes')
    if not all(np.isfinite(x).all() for x in (skip,latent,fused,body,enhanced,final)):
        raise ValueError('nonfinite public intermediate')
    if not np.array_equal(skip,skip.astype('<f2').astype(np.float32)) or \
       not np.array_equal(latent,latent.astype('<f2').astype(np.float32)):
        raise ValueError('optimized branch output is not half-rounded')
    skip_path=OUT/'skip_peer.f32';skip[0].astype('<f4').tofile(skip_path)
    latent_path=OUT/'latent_peer.f32';latent[0].astype('<f4').tofile(latent_path)
    fused_path=OUT/'fused_peer.f32';fused[0].astype('<f4').tofile(fused_path)
    body_path=OUT/'body_peer.f32';body[0].astype('<f4').tofile(body_path)
    enhanced_path=OUT/'enhanced_rgb.f32';enhanced[0].transpose(1,2,0).astype('<f4').tofile(enhanced_path)
    final_path=OUT/'final_rgb.f32';final[0].transpose(1,2,0).astype('<f4').tofile(final_path)
    color_path=OUT/'color_linear.f32';linear.astype('<f4').tofile(color_path)
    if digest(skip_path)!=reference_report['skip_peer_sha256']:
        raise AssertionError('same-image preblock0 skip differs from isolated branch')
    report=dict(input_image_sha256=digest(IMAGE),model_sha256=MODEL_SHA256,
                temporary_branch_sha256=branch_hash,
                output_nodes={'skip':SKIP,'latent':LATENT,'fused':FUSED,'body':BODY,
                              'enhanced':ENHANCED,'final':'output'},
                skip_peer_sha256=digest(skip_path),latent_peer_sha256=digest(latent_path),
                fused_peer_sha256=digest(fused_path),body_peer_sha256=digest(body_path),
                enhanced_rgb_sha256=digest(enhanced_path),final_rgb_sha256=digest(final_path),
                color_linear_sha256=digest(color_path),
                skip_extent=[256,256,32],latent_extent=[128,128,32],
                same_inference_call=True,isolated_skip_exact=True,
                model_variant='optimized AMD FP16',native_fp8_equivalence=False,
                original_kernel_executed=False,original_runtime_validation=False)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
