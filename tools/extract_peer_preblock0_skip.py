#!/usr/bin/env python3
"""Extract and run the public static ONNX preblock0 skip branch on one RGB image.

Invoke with build/peer_onnx_venv/Scripts/python.exe. The output is in the
public model's C32 basis. It is a static-image candidate, not a native
NVIDIA capture or a full encoder/decoder validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'ref/dlss5-onnx/dlss5_real_static_256_amd_fp16.onnx'
IMAGE = ROOT / 'ref/dlss5-onnx/examples/assets/normalized/blue_marble.png'
OUT = ROOT / 'build/peer_preblock0_skip'
OUTPUT_NODE = '/graph/Cast_9_output_0'
MODEL_SHA256 = '7aa891c46f90f3d0a4539701ba009131ac333602634a62ad8675da90d0f8a173'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(image=IMAGE):
    image = Path(image)
    if not MODEL.is_file() or not image.is_file():
        raise FileNotFoundError('public ONNX model or image missing')
    model_hash=digest(MODEL)
    if model_hash!=MODEL_SHA256:
        raise ValueError(f'unrecognized public ONNX model hash {model_hash}')
    OUT.mkdir(parents=True, exist_ok=True)
    branch = OUT / 'preblock0_branch.onnx'
    onnx.utils.extract_model(str(MODEL), str(branch), ['rgb'], [OUTPUT_NODE])
    rgb = np.asarray(Image.open(image).convert('RGB'), np.float32) / 255
    if rgb.shape != (256,256,3):
        raise ValueError(f'expected 256x256 RGB, got {rgb.shape}')
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4).astype(np.float32)
    session = ort.InferenceSession(str(branch), providers=['CPUExecutionProvider'])
    result = session.run(None, {'rgb':linear.transpose(2,0,1)[None]})[0]
    if result.shape != (1,256,256,32) or not np.isfinite(result).all():
        raise ValueError('preblock0 skip result shape/nonfinite')
    if not np.array_equal(result,result.astype('<f2').astype(np.float32)):
        raise ValueError('optimized FP16 branch output is not half-rounded')
    output = OUT / 'skip_peer.f32'; result[0].astype('<f4').tofile(output)
    color = OUT / 'color_linear.f32'; linear.astype('<f4').tofile(color)
    report = dict(input_image_sha256=digest(image),model_sha256=model_hash,
                  branch_sha256=digest(branch),branch_output=OUTPUT_NODE,
                  skip_peer_sha256=digest(output),color_linear_sha256=digest(color),
                  extent=[256,256,32],finite=True,half_values_exact=True,
                  skip_basis='public ONNX peer C32',model_variant='optimized AMD FP16',
                  native_fp8_skip_validated=False,
                  original_kernel_executed=False,original_runtime_validation=False)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return output,color


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,default=IMAGE)
    args=parser.parse_args();run(args.image)
