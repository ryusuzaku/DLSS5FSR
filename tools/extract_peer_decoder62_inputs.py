#!/usr/bin/env python3
"""Extract same-image optimized public FP16 boundaries around blocks62–65."""
from pathlib import Path
import json
import tempfile
import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest

OUT=ROOT/'build/peer_decoder62_inputs'
NODES={
    'block61':'/graph/dec128.4/Cast_61_output_0',
    'skip8':'/graph/down64_128/Cast_9_output_0',
    'merge62':'/graph/up128_64/upsample/Add_output_0',
    'block62':'/graph/up128_64/body/Cast_61_output_0',
    'block63':'/graph/dec64.0/Cast_61_output_0',
    'block64':'/graph/dec64.1/Cast_61_output_0',
    'block65':'/graph/dec64.2/Cast_61_output_0',
}


def run():
    if digest(MODEL)!=MODEL_SHA256:raise ValueError('public model hash differs')
    prior=ROOT/'build/peer_coherent_head_inputs'
    prior_report=json.loads((prior/'manifest.json').read_text())
    if prior_report['model_sha256']!=MODEL_SHA256 or \
       prior_report['input_image_sha256']!=digest(IMAGE):
        raise ValueError('previous inference model/image differs')
    color=prior/'color_linear.f32'
    if digest(color)!=prior_report['color_linear_sha256']:
        raise ValueError('color source hash differs')
    rgb=np.fromfile(color,'<f4').reshape(256,256,3)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        branch=Path(tmp)/'decoder62_inputs.onnx'
        onnx.utils.extract_model(str(MODEL),str(branch),['rgb'],list(NODES.values()))
        branch_hash=digest(branch)
        session=ort.InferenceSession(str(branch),providers=['CPUExecutionProvider'])
        arrays=session.run(None,{'rgb':rgb.transpose(2,0,1)[None]})
    shapes={'block61':(1,32,32,128),
            **{name:(1,64,64,64) for name in NODES if name!='block61'}}
    for (name,_),array in zip(NODES.items(),arrays):
        if array.shape!=shapes[name] or not np.isfinite(array).all():
            raise ValueError(f'{name} shape/nonfinite: {array.shape}')
        if name!='merge62' and not np.array_equal(array,array.astype('<f2').astype(np.float32)):
            raise ValueError(f'{name} is not half-rounded')
        array[0].astype('<f4').tofile(OUT/f'{name}_peer.f32')
    parent=ROOT/'build/peer_decoder66_inputs'
    parent_report=json.loads((parent/'manifest.json').read_text())
    if digest(OUT/'block65_peer.f32')!=parent_report['tensor_sha256']['block65']:
        raise AssertionError('new block65 differs from prior branch')
    report=dict(model_sha256=MODEL_SHA256,image_sha256=digest(IMAGE),
                temporary_branch_sha256=branch_hash,nodes=NODES,
                shapes={name:list(shape) for name,shape in shapes.items()},
                tensor_sha256={name:digest(OUT/f'{name}_peer.f32') for name in NODES},
                prior_block65_exact=True,same_inference_call=True,
                model_variant='optimized AMD FP16',native_fp8_equivalence=False,
                original_kernel_executed=False)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
