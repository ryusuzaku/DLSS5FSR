#!/usr/bin/env python3
"""Run public block-4 entry and candidate AMD encoder blocks 5-14 on a capture.

The public optimized FP16 ONNX graph supplies only the upstream block-4
boundary. Blocks 5-14 run through the existing HIP/scalar candidate checks.
This is offline, not a live game network or an original-kernel oracle.
"""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import onnx
import onnxruntime as ort

from audit_peer_native_c32_basis import peer_to_native_multihead
from check_block62_candidate import ROOT, run as run_block
from check_block56_candidate import run as run_c128
from check_c256_ffn_candidate import H, bits
from check_split512_peer_image import metrics
from decode_tinlayout_global import e4m3fn
from extract_peer_encoder64_inputs import NODES as C64_NODES
from extract_peer_encoder128_inputs import NODES as C128_NODES
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256
from native_split_reference import F
from native_c64_reference import multiply


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, output_root, last_block=8, through_block14=False):
    prepared_dir = Path(prepared_dir).resolve()
    output_root = Path(output_root).resolve()
    if last_block not in range(5, 9):
        raise ValueError('last block must be 5..8')
    if through_block14 and last_block != 8:
        raise ValueError('C128 continuation needs candidate block8 downsample')
    prepared_path = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_path.read_text())
    color_path = prepared_dir / 'color_linear.f32'
    if (prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color_path) or
            color_path.stat().st_size != 256 * 256 * 3 * 4 or
            digest(MODEL) != MODEL_SHA256):
        raise ValueError('prepared input or pinned public model differs')
    rgb = np.fromfile(color_path, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB has nonfinite values')
    output_root.mkdir(parents=True, exist_ok=True)
    public_dir = output_root / 'public_boundary'
    public_dir.mkdir(exist_ok=True)
    nodes = dict(C64_NODES)
    if through_block14:
        nodes.update({k: v for k, v in C128_NODES.items() if k != 'block8_down'})
    branch_path = public_dir / ('encoder4_14.onnx' if through_block14 else 'encoder4_8.onnx')
    onnx.utils.extract_model(str(MODEL), str(branch_path), ['rgb'], list(nodes.values()))
    session = ort.InferenceSession(str(branch_path), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    public = {}
    for (name, _), array in zip(nodes.items(), arrays):
        shape = ((1, 16, 16, 256) if name == 'block14_down' else
                 (1, 32, 32, 128) if name in C128_NODES else
                 (1, 64, 64, 64))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        public[name] = array[0].astype('<f4')
        public[name].tofile(public_dir / f'{name}_peer.f32')
    p64 = peer_to_native_multihead(np.arange(64))
    native = np.empty_like(public['block4_down'])
    native[..., p64] = public['block4_down']
    native = F(native)
    boundary = output_root / 'boundary4'
    boundary.mkdir(exist_ok=True)
    previous = boundary / 'output_device.f32'
    native.astype('<f4').tofile(previous)
    native.astype('<f4').tofile(boundary / 'output.f32')
    blocks = []
    for block in range(5, last_block + 1):
        input_sha = digest(previous)
        result = run_block(block=block, previous=previous, width=64, height=64,
                           output_root=output_root / f'block{block}')
        stage = json.loads((result.parents[1] / 'manifest.json').read_text())
        if (stage['input_device_sha256'] != input_sha or
                stage['output_device_sha256'] != digest(result)):
            raise AssertionError(f'candidate block{block} handoff differs')
        answer = np.fromfile(result, '<f4').reshape(64, 64, 64)
        reference_name = 'block8_skip' if block == 8 else f'block{block}'
        comparison = metrics(answer[..., p64], public[reference_name])
        blocks.append(dict(block=block, input_device_sha256=input_sha,
                           output_device_sha256=digest(result),
                           candidate_vs_public_fp16=comparison,
                           hip_scalar_exact=True))
        previous = result
        print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}",
              flush=True)
    downsample8 = None
    if last_block == 8:
        stage = json.loads((output_root / 'block8/manifest.json').read_text())
        raw_path = output_root / 'block8/raw_output/output_device.f32'
        if (digest(raw_path) != stage['raw_output_device_sha256'] or
                digest(previous) != stage['output_device_sha256']):
            raise AssertionError('block8 raw/skip device ancestry differs')
        records = {r['name']: r for r in
                   json.loads((ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
        index = records['block8.layer0.layer']['index']
        tensor = ROOT / 'dlss5-analysis/tensors' / f'tensor_{index:03d}.bin'
        data = np.fromfile(tensor, np.uint8)
        if data.size != 69936 or data.size - 0xf130 != 8192:
            raise ValueError('wrong block8 downsample tensor extent')
        positions = np.arange(2 * 64 * 64, dtype=np.int32)
        inputs = bits(len(positions), [1, 0, 4, 5, 2, 12])
        outputs = bits(len(positions), [3, 6, 7, 8, 9, 10, 11])
        if np.unique(outputs * 64 + inputs).size != len(positions):
            raise ValueError('measured block8 downsample map collides')
        matrix = np.empty((128, 64), np.float32)
        matrix[outputs, inputs] = e4m3fn(data[0xf130:])
        raw = np.fromfile(raw_path, '<f4').reshape(64, 64, 64)
        rows_top = H(raw[::2, ::2] + raw[::2, 1::2])
        rows_bottom = H(raw[1::2, ::2] + raw[1::2, 1::2])
        pool = F(H(H(rows_top + rows_bottom) * np.float32(.25)))
        output = F(multiply(pool.reshape(1024, 64), matrix)).reshape(32, 32, 128)
        if not np.isfinite(output).all():
            raise ValueError('nonfinite candidate block8 downsample')
        down = output_root / 'downsample8'
        down.mkdir(exist_ok=True)
        for name, array in dict(raw=raw, pool=pool, matrix=matrix, output=output).items():
            np.asarray(array, '<f4').tofile(down / f'{name}.f32')
        subprocess.run([str(ROOT / 'build/encoder64_downsample_test.exe'),
                        str(down), '64', '64'], cwd=ROOT, check=True)
        if ((down / 'pool_device.f32').read_bytes() != (down / 'pool.f32').read_bytes() or
                (down / 'output_device.f32').read_bytes() != (down / 'output.f32').read_bytes()):
            raise AssertionError('candidate block8 downsample HIP/scalar differs')
        p128 = peer_to_native_multihead(np.arange(128))
        downsample8 = dict(raw_device_sha256=digest(raw_path),
                           output_device_sha256=digest(down / 'output_device.f32'),
                           tensor_sha256=digest(tensor),
                           candidate_vs_public_fp16=metrics(output[..., p128],
                                                              public['block8_down']),
                           hip_scalar_exact=True,
                           map='measured C64 downsample address bits')
    c64_final = previous
    c128_blocks = []
    if through_block14:
        if downsample8 is None:
            raise AssertionError('candidate block8 downsample missing')
        boundary8 = output_root / 'boundary8'
        boundary8.mkdir(exist_ok=True)
        previous = boundary8 / 'output_device.f32'
        previous.write_bytes((output_root / 'downsample8/output_device.f32').read_bytes())
        (boundary8 / 'output.f32').write_bytes(previous.read_bytes())
        p128 = peer_to_native_multihead(np.arange(128))
        for block in range(9, 15):
            input_sha = digest(previous)
            result = run_c128(block=block, previous=previous, width=32, height=32,
                              output_root=output_root / f'block{block}')
            stage = json.loads((result.parents[1] / 'manifest.json').read_text())
            if (stage['input_device_sha256'] != input_sha or
                    stage['output_device_sha256'] != digest(result)):
                raise AssertionError(f'candidate block{block} handoff differs')
            answer = np.fromfile(result, '<f4').reshape(32, 32, 128)
            reference_name = 'block14_skip' if block == 14 else f'block{block}'
            comparison = metrics(answer[..., p128], public[reference_name])
            c128_blocks.append(dict(block=block, input_device_sha256=input_sha,
                                    output_device_sha256=digest(result),
                                    candidate_vs_public_fp16=comparison,
                                    hip_scalar_exact=True))
            previous = result
            print(f"block{block}: MAE {comparison['mae']:.7g}, corr {comparison['correlation']:.7g}",
                  flush=True)
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_path),
                  color_linear_sha256=digest(color_path),
                  source_model_sha256=MODEL_SHA256,
                  public_branch_sha256=digest(branch_path),
                  public_boundary_sha256={name: digest(public_dir / f'{name}_peer.f32')
                                          for name in nodes},
                  candidate_boundary4_sha256=digest(boundary / 'output_device.f32'),
                  blocks=blocks, final_device_sha256=digest(c64_final),
                  downsample8=downsample8,
                  c128_blocks=c128_blocks,
                  final_c128_device_sha256=digest(previous) if c128_blocks else None,
                  original_kernel_executed=False, full_candidate_inference=False,
                  production_wiring=False)
    (output_root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_encoder64')
    parser.add_argument('--last-block', type=int, default=8)
    parser.add_argument('--through-block14', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.output_root, args.last_block, args.through_block14)
