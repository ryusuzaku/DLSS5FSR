#!/usr/bin/env python3
"""Validate a same-frame raw proxy and GPU-prepared candidate input tensor.

This checks our diagnostic crop/color implementation, not original NVIDIA
frontend parity or live full-network inference.
"""

from pathlib import Path
import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import shutil
import subprocess

import numpy as np

import prepare_candidate_input


COUNT = 256 * 256 * 3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(capture, gpu_tensor, output_dir, standalone_exe=None):
    capture = Path(capture).resolve()
    gpu_tensor = Path(gpu_tensor).resolve()
    output_dir = Path(output_dir).resolve()
    if not capture.is_file() or not gpu_tensor.is_file():
        raise FileNotFoundError('paired raw/GPU capture is incomplete')
    if gpu_tensor.stat().st_size != COUNT * 4:
        raise ValueError('GPU RGB tensor is not 256x256x3 float32')
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / 'cpu_prepared'
    with redirect_stdout(StringIO()):
        prepared = prepare_candidate_input.run(capture, prepared_dir)
    cpu = np.fromfile(prepared_dir / 'color_linear.f32', dtype='<f4')
    gpu = np.fromfile(gpu_tensor, dtype='<f4')
    if cpu.size != COUNT or gpu.size != COUNT:
        raise ValueError('prepared RGB tensor extent differs')
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise ValueError('prepared RGB contains nonfinite values')
    delta = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
    report = {
        'raw_capture_sha256': digest(capture),
        'gpu_tensor_sha256': digest(gpu_tensor),
        'cpu_tensor_sha256': digest(prepared_dir / 'color_linear.f32'),
        'source': prepared['source'],
        'crop_xywh': prepared['crop_xywh'],
        'clipped_channel_fraction': prepared['clipped_channel_fraction'],
        'values': COUNT,
        'gpu_vs_cpu_mae': float(delta.mean()),
        'gpu_vs_cpu_max_abs': float(delta.max()),
        'gpu_vs_cpu_above_1e-6': int(np.count_nonzero(delta > 1e-6)),
        'standalone_gpu_exact': None,
        'native_frontend_parity': False,
        'live_full_candidate_inference': False,
    }
    if standalone_exe is not None:
        standalone_exe = Path(standalone_exe).resolve()
        standalone_out = output_dir / 'standalone_gpu.f32'
        result = subprocess.run(
            [str(standalone_exe), str(capture),
             str(prepared_dir / 'color_linear.f32'), str(standalone_out)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output_dir / 'standalone.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'standalone HIP check failed: {result.stdout[-2000:]}')
        report['standalone_gpu_sha256'] = digest(standalone_out)
        report['standalone_gpu_exact'] = (
            report['gpu_tensor_sha256'] == report['standalone_gpu_sha256'])
    if report['gpu_vs_cpu_max_abs'] > 5e-6:
        (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        raise AssertionError('GPU/CPU diagnostic input differs above tolerance')
    if report['standalone_gpu_exact'] is False:
        (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        raise AssertionError('live GPU and standalone HIP tensors differ')
    gpu_prepared = output_dir / 'gpu_prepared'
    gpu_prepared.mkdir(exist_ok=True)
    shutil.copyfile(gpu_tensor, gpu_prepared / 'color_linear.f32')
    shutil.copyfile(prepared_dir / 'proxy_256.png', gpu_prepared / 'proxy_256.png')
    gpu_manifest = dict(prepared)
    gpu_manifest['color_linear_sha256'] = report['gpu_tensor_sha256']
    gpu_manifest['input_tensor_source'] = 'same-frame HIP GPU diagnostic'
    (gpu_prepared / 'manifest.json').write_text(
        json.dumps(gpu_manifest, indent=2) + '\n')
    report['gpu_prepared_manifest_sha256'] = digest(gpu_prepared / 'manifest.json')
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('gpu_tensor', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--standalone-exe', type=Path)
    args = parser.parse_args()
    run(args.capture, args.gpu_tensor, args.output_dir, args.standalone_exe)
