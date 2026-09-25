#!/usr/bin/env python3
"""Run the offline candidate through the head from one prepared 256x256 input.

Each stage checks the previous candidate device hash and runs its public FP16
comparison boundaries on the same input. Graph extraction is cached by source
model and nodes; comparison outputs are recomputed for every frame. This is
not a live game path.
"""

from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STAGES = ('encoder22', 'encoder30', 'vit39', 'decoder47', 'decoder55',
          'decoder61', 'decoder65', 'decoder69', 'head_inputs', 'head70',
          'head70_connected_gpu')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(prepared_dir, output_root, start_at='encoder22'):
    prepared_dir = Path(prepared_dir).resolve()
    output_root = Path(output_root).resolve()
    prepared_file = prepared_dir / 'manifest.json'
    prepared = json.loads(prepared_file.read_text())
    color = prepared_dir / 'color_linear.f32'
    if (prepared['output_size'] != [256, 256] or
            prepared['color_linear_sha256'] != digest(color) or
            color.stat().st_size != 256*256*3*4):
        raise ValueError('prepared input or manifest differs')
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / 'logs'
    logs.mkdir(exist_ok=True)
    p = str(prepared_dir)
    d = lambda stage: str(output_root / stage)
    commands = {
        'encoder22': ['run_candidate_encoder64_from_capture.py', p,
                      '--through-block22', '--output-root', d('encoder22')],
        'encoder30': ['run_candidate_encoder512_from_capture.py', p, d('encoder22'),
                      '--output-root', d('encoder30')],
        'vit39': ['run_candidate_vit39_from_capture.py', p, d('encoder30'),
                  '--output-root', d('vit39')],
        'decoder47': ['run_candidate_decoder512_from_capture.py', p, d('vit39'),
                      '--output-root', d('decoder47')],
        'decoder55': ['run_candidate_decoder256_from_capture.py', p, d('encoder22'),
                      d('decoder47'), '--output-root', d('decoder55')],
        'decoder61': ['run_candidate_decoder128_from_capture.py', p, d('encoder22'),
                      d('decoder55'), '--output-root', d('decoder61')],
        'decoder65': ['run_candidate_decoder64_from_capture.py', p, d('encoder22'),
                      d('decoder61'), '--output-root', d('decoder65')],
        'decoder69': ['run_candidate_decoder32_from_capture.py', p, d('decoder65'),
                      '--output-root', d('decoder69')],
        'head_inputs': ['extract_candidate_head_inputs_from_capture.py', p, d('decoder69'),
                        '--output-root', d('head_inputs')],
        'head70': ['check_head70_peer_frame.py', '--latent-case', 'capture_candidate',
                   '--source-dir', d('head_inputs'),
                   '--candidate-latent', str(output_root / 'decoder69/block69/output/output_device.f32'),
                   '--candidate-report', str(output_root / 'decoder69/report.json'),
                   '--chunk-windows', '64', '--output-root', d('head70')],
        'head70_connected_gpu': ['check_head70_peer_gpu_chain.py',
                                 '--latent-case', 'capture_candidate',
                                 '--frame-dir', d('head70'),
                                 '--output-dir', d('head70_connected_gpu')],
    }
    if start_at not in STAGES:
        raise ValueError(f'unknown start stage: {start_at}')
    times = {}
    for stage in STAGES[STAGES.index(start_at):]:
        command = [sys.executable, str(ROOT / 'tools' / commands[stage][0]),
                   *commands[stage][1:]]
        print(f'{stage}: running', flush=True)
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.monotonic()-start
        (logs / f'{stage}.log').write_text(result.stdout)
        if result.returncode or 'FAIL' in result.stdout:
            raise RuntimeError(f'{stage} failed after {elapsed:.1f}s; '
                               f'see {logs / (stage + ".log")}\n{result.stdout[-3000:]}')
        times[stage] = elapsed
        print(f'{stage}: complete in {elapsed:.1f}s', flush=True)
    head = json.loads((output_root / 'head70/manifest.json').read_text())
    connected = json.loads((output_root / 'head70_connected_gpu/manifest.json').read_text())
    if (not head['captured_input_candidate_executed'] or
            head['source_model_sha256'] != connected['source_model_sha256'] or
            connected['source_frame_manifest_sha256'] != digest(output_root / 'head70/manifest.json') or
            set(connected['connected_gpu_exact']) !=
            {'merge', 'body', 'native-gain-rgb', 'public-gain-rgb'}):
        raise AssertionError('full-frame head/GPU provenance differs')
    report = dict(source_capture_sha256=prepared['source_capture_sha256'],
                  prepared_manifest_sha256=digest(prepared_file),
                  color_linear_sha256=digest(color),
                  stage_seconds=times,
                  head_manifest_sha256=digest(output_root / 'head70/manifest.json'),
                  connected_manifest_sha256=digest(output_root / 'head70_connected_gpu/manifest.json'),
                  public_gain_blended_vs_public_final=head['rgb']['public_gain']['blended_vs_public_final'],
                  input_vs_public_final=head['input_color_vs_public_final'],
                  offline_only=True, original_kernel_executed=False,
                  production_wiring=False)
    (output_root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--start-at', choices=STAGES, default='encoder22',
                        help='resume after verifying prior stage files still exist')
    args = parser.parse_args()
    run(args.prepared_dir, args.output_root, args.start_at)
