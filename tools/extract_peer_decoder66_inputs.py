#!/usr/bin/env python3
"""Extract same-image public FP16 inputs to decoder block66.

Run with build/peer_onnx_venv/Scripts/python.exe. This public graph is an
optimized FP16 control, not a native NVIDIA FP8 capture.
"""
from pathlib import Path
import json
import tempfile
import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest

OUT=ROOT/'build/peer_decoder66_inputs'
BLOCK65='/graph/dec64.2/Cast_61_output_0'
SKIP4='/graph/down32_64/Cast_9_output_0'
MERGE66='/graph/up64_32/upsample/Add_output_0'
BLOCK66='/graph/up64_32/body/Cast_51_output_0'
BLOCK67='/graph/dec32.0/Cast_51_output_0'
BLOCK68='/graph/dec32.1/Cast_51_output_0'
BLOCK69='/graph/dec32.2/Cast_51_output_0'


def run():
    if digest(MODEL)!=MODEL_SHA256:raise ValueError('public model hash differs')
    source=ROOT/'build/peer_coherent_head_inputs'
    source_report=json.loads((source/'manifest.json').read_text())
    if source_report['model_sha256']!=MODEL_SHA256 or \
       source_report['input_image_sha256']!=digest(IMAGE):
        raise ValueError('previous public inference used a different model/image')
    color=source/'color_linear.f32'
    if digest(color)!=source_report['color_linear_sha256']:
        raise ValueError('linear RGB source hash differs')
    rgb=np.fromfile(color,'<f4').reshape(256,256,3)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        branch=Path(tmp)/'decoder66_inputs.onnx'
        names={'block65':BLOCK65,'skip4':SKIP4,'merge66':MERGE66,
               'block66':BLOCK66,'block67':BLOCK67,'block68':BLOCK68,
               'block69':BLOCK69}
        onnx.utils.extract_model(str(MODEL),str(branch),['rgb'],list(names.values()))
        branch_hash=digest(branch)
        session=ort.InferenceSession(str(branch),providers=['CPUExecutionProvider'])
        arrays=session.run(None,{'rgb':rgb.transpose(2,0,1)[None]})
    shapes={'block65':(1,64,64,64),'skip4':(1,128,128,32),
            **{name:(1,128,128,32) for name in
               ('merge66','block66','block67','block68','block69')}}
    for (name,_),array in zip(names.items(),arrays):
        shape=shapes[name]
        if array.shape!=shape or not np.isfinite(array).all():
            raise ValueError(f'{name} bad shape/nonfinite: {array.shape}')
        if name!='merge66' and not np.array_equal(array,array.astype('<f2').astype(np.float32)):
            raise ValueError(f'{name} is not half-rounded')
        array[0].astype('<f4').tofile(OUT/f'{name}_peer.f32')
    if digest(OUT/'block69_peer.f32')!=source_report['latent_peer_sha256']:
        raise AssertionError('new same-image block69 differs from prior full inference')
    report=dict(model_sha256=MODEL_SHA256,image_sha256=digest(IMAGE),
                temporary_branch_sha256=branch_hash,
                nodes=names,
                shapes={name:list(shape) for name,shape in shapes.items()},
                tensor_sha256={name:digest(OUT/f'{name}_peer.f32') for name in names},
                prior_block69_exact=True,same_inference_call=True,
                model_variant='optimized AMD FP16',
                native_fp8_equivalence=False,original_kernel_executed=False)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
