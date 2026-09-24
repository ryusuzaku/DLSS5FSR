#!/usr/bin/env python3
"""Extract same-image optimized public FP16 boundaries around blocks56–61."""
from pathlib import Path
import json
import tempfile
import numpy as np
import onnx
import onnxruntime as ort

from extract_peer_preblock0_skip import ROOT,MODEL,IMAGE,MODEL_SHA256,digest

OUT=ROOT/'build/peer_decoder56_inputs'
NODES={
    'block55':'/graph/dec256.6/Cast_61_output_0',
    'skip14':'/graph/down128_256/Cast_9_output_0',
    'merge56':'/graph/up256_128/upsample/Add_output_0',
    'block56':'/graph/up256_128/body/Cast_61_output_0',
    **{f'block{57+i}':f'/graph/dec128.{i}/Cast_61_output_0' for i in range(5)},
}


def run():
    if digest(MODEL)!=MODEL_SHA256:raise ValueError('public model hash differs')
    prior=ROOT/'build/peer_coherent_head_inputs'
    old=json.loads((prior/'manifest.json').read_text())
    if old['model_sha256']!=MODEL_SHA256 or old['input_image_sha256']!=digest(IMAGE):
        raise ValueError('previous inference model/image differs')
    color=prior/'color_linear.f32'
    if digest(color)!=old['color_linear_sha256']:
        raise ValueError('color source hash differs')
    rgb=np.fromfile(color,'<f4').reshape(256,256,3)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        branch=Path(tmp)/'decoder56_inputs.onnx'
        onnx.utils.extract_model(str(MODEL),str(branch),['rgb'],list(NODES.values()))
        branch_hash=digest(branch)
        session=ort.InferenceSession(str(branch),providers=['CPUExecutionProvider'])
        arrays=session.run(None,{'rgb':rgb.transpose(2,0,1)[None]})
    shapes={'block55':(1,16,16,256),
            **{name:(1,32,32,128) for name in NODES if name!='block55'}}
    for (name,_),array in zip(NODES.items(),arrays):
        if array.shape!=shapes[name] or not np.isfinite(array).all():
            raise ValueError(f'{name} shape/nonfinite: {array.shape}')
        if name!='merge56' and not np.array_equal(array,array.astype('<f2').astype(np.float32)):
            raise ValueError(f'{name} is not half-rounded')
        array[0].astype('<f4').tofile(OUT/f'{name}_peer.f32')
    parent=ROOT/'build/peer_decoder62_inputs'
    parent_report=json.loads((parent/'manifest.json').read_text())
    if digest(OUT/'block61_peer.f32')!=parent_report['tensor_sha256']['block61']:
        raise AssertionError('new block61 differs from prior branch')
    report=dict(model_sha256=MODEL_SHA256,image_sha256=digest(IMAGE),
                temporary_branch_sha256=branch_hash,nodes=NODES,
                shapes={name:list(shape) for name,shape in shapes.items()},
                tensor_sha256={name:digest(OUT/f'{name}_peer.f32') for name in NODES},
                prior_block61_exact=True,same_inference_call=True,
                model_variant='optimized AMD FP16',native_fp8_equivalence=False,
                original_kernel_executed=False)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
