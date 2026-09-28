"""Check changing frames and GPU output consumers with one resident C512 chain."""

from pathlib import Path
import argparse
import json
import shutil
import subprocess

import numpy as np

from check_split512_resident import ROOT, SHIFTS, digest, run as check_fixed
from check_split512_spatial_block import run_case
from check_split512_bridge import run_case as run_head
from audit_native_vit_logical_map import logical_map
from gpu_test_runner import close_worker


def run(fixtures, output, first=23, width=8, height=8, head_fixture=None):
    fixtures, output = Path(fixtures).resolve(), Path(output).resolve()
    executable = ROOT / 'build/split512_frames_test.exe'
    if not executable.is_file():
        raise FileNotFoundError('build with bash tools/build_split512_resident.sh')
    output.mkdir(parents=True, exist_ok=True)
    # Also exercises the original fixed-seed entry after adding device-input support.
    baseline = check_fixed(fixtures, output / 'fixed_control', first, width, height,
                           repeats=2, head_fixture=head_fixture)
    source = fixtures / f'block{first}-{width}x{height}-s0/input.f32'
    a = np.fromfile(source, '<f4').reshape(height, width, 512)
    inputs = dict(a=a, b=np.roll(a, 1, axis=1).copy(), zero=np.zeros_like(a))
    if len({v.tobytes() for v in inputs.values()}) != 3:
        raise ValueError('source does not produce three distinct control inputs')
    frames = {}
    gather = logical_map(width*height//4) if first == 23 else None
    try:
        for name, image in inputs.items():
            frame = output / 'references' / name
            frame.mkdir(parents=True, exist_ok=True)
            image.astype('<f4').tofile(frame / 'input.f32')
            if name == 'a':
                last = fixtures / f'block{first+7}-{width}x{height}-s2'
                head = (Path(head_fixture).resolve() if head_fixture is not None else
                        fixtures / f'block30_head_{width//2}x{height//2}') if first == 23 else None
            else:
                x = image.reshape(-1, 512)
                for block, shift in zip(range(first, first+8), SHIFTS):
                    last = run_case(block, width, height, shift, x, frame / 'standalone')
                    expected = x.astype('<f4').tobytes()
                    if (last / 'input.f32').read_bytes() != expected:
                        raise AssertionError('standalone reference handoff changed')
                    if (last / 'final_device.f32').read_bytes() != (last / 'final.f32').read_bytes():
                        raise AssertionError('standalone/scalar reference differs')
                    x = np.fromfile(last / 'final_device.f32', '<f4').reshape(-1, 512)
                head = run_head(width, height, last / 'final_raw_device.f32', frame / 'head_control') if first == 23 else None
            for tensor in ('final', 'final_raw'):
                shutil.copyfile(last / f'{tensor}_device.f32', frame / f'{tensor}.f32')
            if head is not None:
                shutil.copyfile(head / 'head_device.f32', frame / 'head.f32')
                head_values = np.fromfile(frame / 'head.f32', '<f4')
                if head_values.size != gather.size:
                    raise ValueError('head/bridge extent differs')
                head_values[gather].astype('<f4').tofile(frame / 'bridge.f32')
            frames[name] = frame
    finally:
        close_worker()
    if len({digest(p / 'final.f32') for p in frames.values()}) != 3:
        raise AssertionError('control frames do not produce distinct final outputs')
    sequence = ('a', 'b', 'zero', 'b', 'a')
    frame_list = output / 'frames.txt'
    frame_list.write_bytes(''.join(str(frames[name])+'\n' for name in sequence).encode('mbcs'))
    for index in range(len(sequence)):
        (output / f'frame_{index}').mkdir(exist_ok=True)
    command = [str(executable),
               str(output / 'fixed_control/blocks.txt'), str(width), str(height),
               str(frame_list), str(output)]
    if first == 23:
        map_path = output / 'bridge_map.i32'
        gather.astype('<i4').tofile(map_path)
        head = Path(head_fixture).resolve() if head_fixture is not None else fixtures / f'block30_head_{width//2}x{height//2}'
        command.extend([str(head), str(map_path)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (output / 'frames_device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:
        raise RuntimeError(f'changing-frame check failed: {output / "frames_device.log"}\n{result.stdout[-2000:]}{result.stderr}')
    metrics = json.loads((output / 'frames_metrics.json').read_text())
    names = ('final', 'final_raw', 'head', 'bridge') if first == 23 else ('final', 'final_raw')
    if (metrics['frames'] != len(sequence) or metrics['invalid_views_rejected'] != 2 or
            any(metrics[k] for k in ('post_setup_allocations', 'compute_h2d_bytes', 'compute_d2h_bytes')) or
            metrics['frame_upload_bytes'] != len(sequence)*width*height*512*4 or
            metrics['compute_d2d_bytes'] != len(sequence)*(3*width*height*512 +
                                                         (width*height//4*1024 if first == 23 else 0))*4 or
            not metrics['weights_loaded_once'] or not metrics['input_preserved'] or
            metrics['adjacent_vit_bridge'] != (first == 23) or
            result.stdout.count('PASS') != len(sequence)*len(names)):
        raise AssertionError('changing-frame transfer/ownership contract differs')
    outputs = []
    for index, name in enumerate(sequence):
        hashes = {}
        for tensor in names:
            observed = output / f'frame_{index}' / f'{tensor}.f32'
            expected = frames[name] / f'{tensor}.f32'
            if observed.read_bytes() != expected.read_bytes():
                raise AssertionError(f'frame{index}/{tensor} differs')
            hashes[tensor] = digest(observed)
        outputs.append(dict(frame=name, input_sha256=digest(frames[name] / 'input.f32'), outputs=hashes))
    report = dict(first_block=first, extent=[width, height, 512], sequence=list(sequence),
                  metrics=metrics, exact_outputs=outputs,
                  bridge_map_sha256=digest(output / 'bridge_map.i32') if first == 23 else None,
                  base_sources=baseline['sources'], original_kernel_executed=False,
                  original_bridge_validated=False, production_wiring=False,
                  scope='changing-frame C512 component with resident weights and device output consumers')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(first_block=first, extent=report['extent'], sequence=sequence,
                         exact_arrays=len(sequence)*len(names), metrics=metrics), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--first-block', type=int, choices=(23, 40), default=23)
    parser.add_argument('--width', type=int, choices=(8, 16, 32), default=8)
    parser.add_argument('--height', type=int, default=8)
    parser.add_argument('--head-fixture', type=Path)
    args = parser.parse_args()
    run(args.fixtures, args.output_root, args.first_block, args.width, args.height, args.head_fixture)
