#!/usr/bin/env python3
"""Extract same-frame public head controls for a captured-input candidate."""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import onnxruntime as ort

from extract_peer_coherent_head_inputs import SKIP, LATENT, FUSED, BODY, ENHANCED
from cached_public_branch import extract_cached
from extract_peer_preblock0_skip import MODEL, MODEL_SHA256


NODES = dict(skip=SKIP, latent=LATENT, fused=FUSED, body=BODY,
             enhanced=ENHANCED, final='output')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, decoder69_dir, output_root):
    prepared_dir = Path(prepared_dir).resolve()
    decoder69_dir = Path(decoder69_dir).resolve()
    out = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    decoder_file = decoder69_dir / 'report.json'
    decoder = json.loads(decoder_file.read_text())
    color_file = prepared_dir / 'color_linear.f32'
    public69_file = decoder69_dir / 'public_boundary/block69_peer.f32'
    candidate69_file = decoder69_dir / 'block69/output/output_device.f32'
    if (digest(MODEL) != MODEL_SHA256 or
            prepared['output_size'] != [256, 256] or
            color_file.stat().st_size != 256 * 256 * 3 * 4 or
            prepared['color_linear_sha256'] != digest(color_file) or
            decoder['source_model_sha256'] != MODEL_SHA256 or
            decoder['source_capture_sha256'] != prepared['source_capture_sha256'] or
            decoder['prepared_manifest_sha256'] != digest(prepared_file) or
            decoder['color_linear_sha256'] != digest(color_file) or
            decoder['public_boundary_sha256']['block69'] != digest(public69_file) or
            decoder['final_device_sha256'] != digest(candidate69_file) or
            not decoder['hip_scalar_exact']):
        raise ValueError('prepared image or candidate decoder69 ancestry differs')
    rgb = np.fromfile(color_file, '<f4').reshape(256, 256, 3)
    if not np.isfinite(rgb).all():
        raise ValueError('prepared RGB contains nonfinite values')
    out.mkdir(parents=True, exist_ok=True)
    branch_file = out / 'head_controls.onnx'
    extract_cached(MODEL, branch_file, ['rgb'], list(NODES.values()), MODEL_SHA256)
    session = ort.InferenceSession(str(branch_file), providers=['CPUExecutionProvider'])
    arrays = session.run(None, {'rgb': rgb.transpose(2, 0, 1)[None]})
    outputs = {}
    for (name, _), array in zip(NODES.items(), arrays):
        shape = ((1, 128, 128, 32) if name == 'latent' else
                 (1, 3, 256, 256) if name in ('enhanced', 'final') else
                 (1, 256, 256, 32))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'public {name} shape/nonfinite: {array.shape}')
        outputs[name] = array
    if not np.array_equal(outputs['skip'], outputs['skip'].astype('<f2').astype(np.float32)):
        raise ValueError('public preblock skip is not half-rounded')
    outputs['skip'][0].astype('<f4').tofile(out / 'skip_peer.f32')
    outputs['latent'][0].astype('<f4').tofile(out / 'latent_peer.f32')
    outputs['fused'][0].astype('<f4').tofile(out / 'fused_peer.f32')
    outputs['body'][0].astype('<f4').tofile(out / 'body_peer.f32')
    outputs['enhanced'][0].transpose(1, 2, 0).astype('<f4').tofile(out / 'enhanced_rgb.f32')
    outputs['final'][0].transpose(1, 2, 0).astype('<f4').tofile(out / 'final_rgb.f32')
    (out / 'color_linear.f32').write_bytes(color_file.read_bytes())
    if digest(out / 'latent_peer.f32') != digest(public69_file):
        raise AssertionError('same-frame public block69 cross-extraction differs')
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  input_image_sha256=digest(color_file),
                  color_linear_sha256=digest(color_file),
                  prepared_manifest_sha256=digest(prepared_file),
                  decoder69_report_sha256=digest(decoder_file),
                  candidate_block69_device_sha256=digest(candidate69_file),
                  model_sha256=MODEL_SHA256,
                  branch_sha256=digest(branch_file), output_nodes=NODES,
                  skip_peer_sha256=digest(out / 'skip_peer.f32'),
                  latent_peer_sha256=digest(out / 'latent_peer.f32'),
                  fused_peer_sha256=digest(out / 'fused_peer.f32'),
                  body_peer_sha256=digest(out / 'body_peer.f32'),
                  enhanced_rgb_sha256=digest(out / 'enhanced_rgb.f32'),
                  final_rgb_sha256=digest(out / 'final_rgb.f32'),
                  same_inference_call=True, same_frame_block69_public_exact=True,
                  original_kernel_executed=False)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('decoder69_dir', type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path.home() / 'DLSS5FSR-build-offload' /
                                'candidate_capture_head_inputs')
    args = parser.parse_args()
    run(args.prepared_dir, args.decoder69_dir, args.output_root)
