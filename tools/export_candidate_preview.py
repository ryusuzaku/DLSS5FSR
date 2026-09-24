#!/usr/bin/env python3
"""Export the pinned offline candidate RGB frame for a fixed in-game preview.

The result only tests the game's model-texture bridge. It does not run the
candidate network on the game's current scene.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(frame_dir, connected_dir, output):
    frame_dir = Path(frame_dir).resolve()
    connected_dir = Path(connected_dir).resolve()
    output = Path(output).resolve()
    frame_path = frame_dir / 'public_gain_enhanced.png'
    frame = json.loads((frame_dir / 'manifest.json').read_text())
    connected = json.loads((connected_dir / 'manifest.json').read_text())
    if (frame['size'] != [256, 256] or
            frame['latent_case'] != 'from_candidate_encoder8_fp8' or
            frame['source_model_sha256'] != connected['source_model_sha256'] or
            connected['latent_case'] != frame['latent_case'] or
            connected['source_frame_manifest_sha256'] != digest(frame_dir / 'manifest.json') or
            set(connected['connected_gpu_exact']) !=
            {'merge', 'body', 'native-gain-rgb', 'public-gain-rgb'} or
            frame['image_sha256']['public_gain_enhanced'] != digest(frame_path)):
        raise ValueError('candidate frame/GPU provenance differs')
    rgb = np.asarray(Image.open(frame_path).convert('RGB'))
    if rgb.shape != (256, 256, 3):
        raise ValueError('expected a 256x256 RGB candidate preview')
    rgba8 = np.empty((256, 256, 4), dtype=np.uint8)
    rgba8[..., :3] = rgb
    rgba8[..., 3] = 255
    rgba16 = (rgba8.astype(np.float32) / np.float32(255)).astype('<f2')
    rgba16[..., 3] = np.float16(1)
    payload = b'D5PREV01' + struct.pack('<II', 256, 256) + rgba8.tobytes() + rgba16.tobytes()
    if len(payload) != 16 + 256 * 256 * (4 + 8):
        raise AssertionError('preview payload extent differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.tmp')
    temp.write_bytes(payload)
    temp.replace(output)
    report = dict(preview_path=str(output), preview_sha256=digest(output),
                  source_frame_manifest_sha256=digest(frame_dir / 'manifest.json'),
                  source_connected_manifest_sha256=digest(connected_dir / 'manifest.json'),
                  source_image_sha256=digest(frame_path), size=[256, 256],
                  purpose='fixed-image game model-texture diagnostic; not live inference')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    offload = Path.home() / 'DLSS5FSR-build-offload'
    parser.add_argument('--frame-dir', type=Path,
                        default=offload / 'peer_head_frame_256_from_candidate_encoder8_fp8')
    parser.add_argument('--connected-dir', type=Path,
                        default=offload / 'peer_head_frame_256_from_candidate_encoder8_fp8_connected_gpu')
    parser.add_argument('--output', type=Path,
                        default=offload / 'candidate_encoder8_fixed_preview.bin')
    args = parser.parse_args()
    run(args.frame_dir, args.connected_dir, args.output)
