#!/usr/bin/env python3
"""Locate first head-body divergence from the public optimized FP16 graph.

Run with build/peer_onnx_venv/Scripts/python.exe. This is a same-image
diagnostic, not an original NVIDIA-kernel oracle.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image

from extract_peer_preblock0_skip import ROOT, MODEL, IMAGE, MODEL_SHA256, digest
from check_block66_peer_candidate import decode
from head70_weights import extract, PAD_AT

sys.path.insert(0,str(ROOT/'ref/dlss5-port/Development'))
from native_c32_reference import F,H

OUT=ROOT/'build/peer_head_stage_audit'
NODES={
    'ffn_input':'/graph/post_body/Cast_11_output_0',
    'expanded':'/graph/post_body/mlp.0/Cast_1_output_0',
    'hidden':'/graph/post_body/Cast_29_output_0',
    'ffn':'/graph/post_body/Add_1_output_0',
    'qkv':'/graph/post_body/attn/qkv/Cast_1_output_0',
    'body':'/graph/post_body/Add_2_output_0',
}


def metrics(a,b):
    if a.shape!=b.shape:raise ValueError(f'shape {a.shape} != {b.shape}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'values':int(d.size),'exact':int(np.count_nonzero(d==0)),
            'mae':float(d.mean()),'max_abs':float(d.max()),
            'rmse':float(np.sqrt(np.mean(d*d))),
            'correlation':float(np.corrcoef(a.ravel(),b.ravel())[0,1])}


def run():
    if digest(MODEL)!=MODEL_SHA256:raise ValueError('public model hash differs')
    source=ROOT/'build/peer_coherent_head_inputs'
    source_report=json.loads((source/'manifest.json').read_text())
    if source_report['model_sha256']!=MODEL_SHA256 or \
       source_report['input_image_sha256']!=digest(IMAGE):
        raise ValueError('public source mismatch')
    rgb=np.asarray(Image.open(IMAGE).convert('RGB'),np.float32)/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4).astype(np.float32)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        branch=Path(tmp)/'stages.onnx'
        onnx.utils.extract_model(str(MODEL),str(branch),['rgb'],list(NODES.values()))
        branch_hash=digest(branch)
        session=ort.InferenceSession(str(branch),providers=['CPUExecutionProvider'])
        values=session.run(None,{'rgb':linear.transpose(2,0,1)[None]})
    base=ROOT/'build/head70_peer_same_image/body'
    same_input=ROOT/'build/peer_head_intermediate_audit/public_fused_body'
    report={'public_model_sha256':MODEL_SHA256,
            'temporary_branch_sha256':branch_hash,'crop_xy':[120,120],
            'comparison':'optimized public FP16 graph vs candidate HIP FP8/half on same image',
            'original_kernel_executed':False}
    for (name,node),arr in zip(NODES.items(),values):
        if arr.ndim!=4 or arr.shape[0]!=1 or arr.shape[1:3]!=(256,256):
            raise ValueError(f'{name} unexpected shape {arr.shape}')
        base_name='input' if name=='ffn_input' else name
        candidate=np.fromfile(base/f'{base_name}.f32','<f4')
        crop=arr[0,120:128,120:128,:].astype(np.float32).reshape(-1)
        if candidate.size!=crop.size:raise ValueError(f'{name} size differs')
        crop.astype('<f4').tofile(OUT/f'{name}_public_crop.f32')
        report[name]={'node':node,'shape':list(arr.shape),
                      'public_crop_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                      'candidate_vs_public':metrics(candidate,crop)}
        if (same_input/f'{base_name}.f32').exists():
            second=np.fromfile(same_input/f'{base_name}.f32','<f4')
            report[name]['half_public_fused_candidate_vs_public']=metrics(second,crop)
    model=onnx.load(str(MODEL))
    onnx_w=numpy_helper.to_array(next(x for x in model.graph.initializer if
        x.name=='/graph/post_body/mlp.0/MatMul_output_0_half_b')).astype(np.float32)
    raw=(ROOT/'dlss5-analysis/tensors/tensor_150.bin').read_bytes()
    stage=(ROOT/'dlss5-analysis/tensors/tensor_001.bin').read_bytes()
    ordinary,*_=extract(raw,stage[PAD_AT:PAD_AT+16])
    candidate_w=decode(np.frombuffer(ordinary,np.uint8))[0].T
    if not np.array_equal(onnx_w,candidate_w):
        raise AssertionError('public and candidate FFN W1 differ')
    x=np.fromfile(OUT/'ffn_input_public_crop.f32','<f4').reshape(64,32)
    public_y=np.fromfile(OUT/'expanded_public_crop.f32','<f4').reshape(64,128)
    candidate_y=np.fromfile(same_input/'expanded.f32','<f4').reshape(64,128)
    unquantized=x@onnx_w
    quantized=H(F(x)@onnx_w)
    same_entry=np.array_equal(x,np.fromfile(same_input/'input.f32','<f4').reshape(64,32))
    if not same_entry:
        raise AssertionError('public and candidate FFN W1 inputs differ')
    if not np.array_equal(quantized,candidate_y):
        raise AssertionError('FP8-input matmul does not reproduce candidate expansion')
    report['first_divergence']={
        'ffn_w1_exact':True,
        'half_public_input_equals_candidate':True,
        'fp8_input_vs_public_half':metrics(F(x),x),
        'fp8_product_half_vs_candidate':metrics(quantized,candidate_y),
        'unquantized_product_vs_public':metrics(unquantized,public_y),
        'fp8_product_half_vs_public':metrics(quantized,public_y),
        'explanation':'optimized public FP16 graph omits native-style FP8 activation quantization before FFN W1'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
