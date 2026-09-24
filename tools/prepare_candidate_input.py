#!/usr/bin/env python3
"""Turn one staged NGX proxy capture into a 256x256 candidate RGB input.

The .f32 output is linear RGB in HWC order. Its PNG is a visual aid only;
the PNG's 8-bit quantization must not be fed back as an exact model boundary.
"""

from pathlib import Path
import argparse
import hashlib
import json
import struct

import numpy as np
from PIL import Image


HEADER = struct.Struct('<8sIIIIIIf')
MAGIC = b'D5INP001'
SIZE = 256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_capture(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError('candidate input capture is truncated')
    magic, width, height, bpp, pitch, passthrough, proxy_mode, white = HEADER.unpack_from(data)
    if (magic != MAGIC or not 1 <= width <= 4096 or not 1 <= height <= 4096 or
            bpp not in (4, 8) or pitch < width * bpp or pitch > 4096 * 8 or
            len(data) != HEADER.size + pitch * height or passthrough not in (0, 1) or
            proxy_mode not in (0, 1, 2) or not np.isfinite(white) or white <= 0):
        raise ValueError('candidate input capture header/extent is invalid')
    rows = np.frombuffer(data, dtype=np.uint8, offset=HEADER.size).reshape(height, pitch)
    active = np.ascontiguousarray(rows[:, :width * bpp])
    if bpp == 8:
        proxy = active.view('<f2').reshape(height, width, 4)[..., :3].astype(np.float32)
        fmt = 'R16G16B16A16_FLOAT'
    else:
        proxy = active.reshape(height, width, 4)[..., :3].astype(np.float32) / 255.0
        fmt = 'R8G8B8A8_UNORM'
    if not np.isfinite(proxy).all():
        raise ValueError('captured proxy contains nonfinite RGB')
    meta = dict(width=width, height=height, bpp=bpp, pitch=pitch,
                passthrough=passthrough, proxy_mode=proxy_mode,
                white_point=white, format=fmt)
    return proxy, meta


def center_square_bilinear(source):
    height, width, channels = source.shape
    if channels != 3:
        raise ValueError('expected HWC RGB')
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    # D3D-style bilinear sample at pixel centers, clamped to the crop edge.
    axis = (np.arange(SIZE, dtype=np.float32) + np.float32(.5)) * np.float32(side / SIZE) - np.float32(.5)
    axis = np.clip(axis, 0, side - 1)
    x0 = np.floor(axis).astype(np.int32) + left
    y0 = np.floor(axis).astype(np.int32) + top
    x1 = np.minimum(x0 + 1, left + side - 1)
    y1 = np.minimum(y0 + 1, top + side - 1)
    tx = (axis - np.floor(axis))[None, :, None]
    ty = (axis - np.floor(axis))[:, None, None]
    a = source[y0[:, None], x0[None, :]]
    b = source[y0[:, None], x1[None, :]]
    c = source[y1[:, None], x0[None, :]]
    d = source[y1[:, None], x1[None, :]]
    return ((a * (1 - tx) + b * tx) * (1 - ty) +
            (c * (1 - tx) + d * tx) * ty).astype(np.float32), [left, top, side, side]


def run(capture, output_dir):
    capture = Path(capture).resolve()
    output_dir = Path(output_dir).resolve()
    proxy, meta = read_capture(capture)
    srgb, crop = center_square_bilinear(proxy)
    clipped = float(np.mean((srgb < 0) | (srgb > 1)))
    srgb = np.clip(srgb, 0, 1)
    linear = np.where(srgb <= .04045, srgb / 12.92,
                      np.power((srgb + .055) / 1.055, 2.4)).astype('<f4')
    if not np.isfinite(linear).all():
        raise ValueError('prepared linear RGB is nonfinite')
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_path = output_dir / 'color_linear.f32'
    png_path = output_dir / 'proxy_256.png'
    linear.tofile(linear_path)
    Image.fromarray(np.rint(srgb * 255).astype(np.uint8), 'RGB').save(png_path)
    report = dict(source_capture_sha256=digest(capture), source=meta,
                  crop_xywh=crop, output_size=[SIZE, SIZE],
                  resample='center-square pixel-center bilinear, clamped crop edge',
                  color='staged proxy sRGB decoded to linear RGB; PNG is visual only',
                  clipped_channel_fraction=clipped,
                  proxy_rgb_min=[float(v) for v in proxy.min(axis=(0, 1))],
                  proxy_rgb_max=[float(v) for v in proxy.max(axis=(0, 1))],
                  linear_rgb_mean=[float(v) for v in linear.mean(axis=(0, 1))],
                  color_linear_sha256=digest(linear_path),
                  preview_png_sha256=digest(png_path),
                  native_frontend_parity=False, full_candidate_inference=False)
    (output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.capture, args.output_dir or args.capture.with_suffix(''))
