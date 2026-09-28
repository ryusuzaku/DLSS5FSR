"""Validate changing C512 -> head -> gather -> ViT38 GPU execution.

Consumes a completed check_split512_frames encoder run. Revalidates its
sources and hashes; creates independent scalar/standalone ViT references.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

import numpy as np

from check_split512_resident import ROOT, SHIFTS, digest, run as check_fixed
from check_vit_expand_chain import run_block
from audit_native_vit_logical_map import logical_map
from gpu_test_runner import close_worker


def run(source, output, head_fixture=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('use separate source and output directories')
    prior = json.loads((source / 'report.json').read_text())
    width, height, channels = prior['extent']
    sequence = ['a', 'b', 'zero', 'b', 'a']
    if (prior['first_block'] != 23 or width not in (8, 32) or height != 8 or
            channels != 512 or prior['sequence'] != sequence):
        raise ValueError('requires a completed 8x8 or 32x8 encoder changing-frame check')
    lines = (source / 'fixed_control/blocks.txt').read_text(encoding='mbcs').splitlines()
    if len(lines) != 8 or [int(line.split('\t')[0]) for line in lines] != list(SHIFTS):
        raise ValueError('invalid saved C512 block list')
    fixtures = Path(lines[0].split('\t')[1]).parent
    head = (Path(head_fixture).resolve() if head_fixture else
            fixtures / f'block30_head_{width//2}x{height//2}')
    output.mkdir(parents=True, exist_ok=True)
    baseline = check_fixed(fixtures, output / 'fixed_control', width=width, height=height,
                           repeats=1, head_fixture=head)
    if baseline['sources'] != prior['base_sources']:
        raise ValueError('C512 source manifests changed since the changing-frame check')
    names = ('final', 'final_raw', 'head', 'bridge')
    if len(prior['exact_outputs']) != len(sequence):
        raise ValueError('incomplete changing-frame report')
    for name, record in zip(sequence, prior['exact_outputs']):
        frame = source / 'references' / name
        if record['frame'] != name or digest(frame / 'input.f32') != record['input_sha256']:
            raise ValueError('saved frame input changed')
        for tensor in names:
            if digest(frame / f'{tensor}.f32') != record['outputs'][tensor]:
                raise ValueError(f'saved reference changed: {name}/{tensor}')
    gather = logical_map(width*height//4).astype('<i4')
    if (source / 'bridge_map.i32').read_bytes() != gather.tobytes():
        raise ValueError('saved logical gather changed')
    # Early S299 reports predate the map hash; exact reconstruction above is
    # required for those too. If recorded, its historical hash must also match.
    if ('bridge_map_sha256' in prior and
            digest(source / 'bridge_map.i32') != prior['bridge_map_sha256']):
        raise ValueError('saved map hash changed')
    shutil.copyfile(source / 'bridge_map.i32', output / 'bridge_map.i32')
    frames, manifests, weight_hashes = {}, {}, {}
    weight_names = ('weights', 'contract_weights', 'contract_skip', 'qkv_weights',
                    'qkv_scales', 'projection_weights', 'projection_skip')
    try:
        for name in ('a', 'b', 'zero'):
            frame = output / 'references' / name
            frame.mkdir(parents=True, exist_ok=True)
            for tensor in ('input', *names):
                shutil.copyfile(source / 'references' / name / f'{tensor}.f32', frame / f'{tensor}.f32')
            bridge = np.fromfile(frame / 'head.f32', '<f4')[gather]
            if bridge.astype('<f4').tobytes() != (frame / 'bridge.f32').read_bytes():
                raise ValueError('reference gather is not byte-exact')
            previous = frame / 'bridge.f32'
            blocks = []
            for block in range(31, 39):
                folder = run_block(block, width=width//2, height=4, derived=True,
                                   image_source=previous, output_root=frame / f'vit_block{block}')
                manifest = json.loads((folder / 'manifest.json').read_text())
                hashes = {key: manifest['files'][key+'.f32'] for key in weight_names}
                if name == 'a':
                    weight_hashes[block] = hashes
                elif hashes != weight_hashes[block]:
                    raise ValueError('reference cases use different ViT weights')
                previous = folder / 'projection_device.f32'
                if previous.read_bytes() != (folder / 'projection.f32').read_bytes():
                    raise AssertionError('standalone/scalar ViT output differs')
                blocks.append(folder)
            shutil.copyfile(previous, frame / 'vit.f32')
            frames[name] = frame
            manifests[name] = [dict(path=str(p / 'manifest.json'), sha256=digest(p / 'manifest.json'))
                               for p in blocks]
    finally:
        close_worker()
    if len({digest(p / 'vit.f32') for p in frames.values()}) != 3:
        raise AssertionError('three inputs did not produce three distinct ViT outputs')
    vit_list = output / 'vit_blocks.txt'
    vit_list.write_bytes(''.join(str(frames['a'] / f'vit_block{b}')+'\n' for b in range(31, 39)).encode('mbcs'))
    frame_list = output / 'frames.txt'
    frame_list.write_bytes(''.join(str(frames[n])+'\n' for n in sequence).encode('mbcs'))
    for index in range(len(sequence)):
        (output / f'frame_{index}').mkdir(exist_ok=True)
    command = [str(ROOT / 'build/split512_frames_test.exe'),
               str(output / 'fixed_control/blocks.txt'), str(width), str(height),
               str(frame_list), str(output), str(head), str(output / 'bridge_map.i32'), str(vit_list)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (output / 'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:
        raise RuntimeError(f'resident encoder/ViT failed: {output / "device.log"}\n{result.stdout[-2000:]}{result.stderr}')
    metrics = json.loads((output / 'frames_metrics.json').read_text())
    n, hn = width*height*512, width*height//4*1024
    if (metrics['frames'] != 5 or metrics['invalid_views_rejected'] != 4 or
            metrics['vit_stage_comparisons'] != 80 or not metrics['resident_vit'] or
            not metrics['adjacent_vit_bridge'] or not metrics['weights_loaded_once'] or
            not metrics['input_preserved'] or result.stdout.count('PASS') != 105 or
            metrics['frame_upload_bytes'] != 5*n*4 or metrics['compute_d2d_bytes'] != 5*(3*n+3*hn)*4 or
            any(metrics[k] for k in ('post_setup_allocations', 'compute_h2d_bytes', 'compute_d2h_bytes'))):
        raise AssertionError('resident stage/ownership/transfer contract differs')
    exact = []
    for index, name in enumerate(sequence):
        hashes = {}
        for tensor in (*names, 'vit'):
            observed = output / f'frame_{index}/{tensor}.f32'
            if observed.read_bytes() != (frames[name] / f'{tensor}.f32').read_bytes():
                raise AssertionError(f'frame {index}/{tensor} differs')
            hashes[tensor] = digest(observed)
        exact.append(dict(frame=name, outputs=hashes))
    # Prove diagnostic fixtures are read as expectations, never uploaded as later inputs.
    with tempfile.TemporaryDirectory(prefix='vit controls ', dir=output) as temporary:
        tmp = Path(temporary)
        bad = tmp / 'bad_block32'
        shutil.copytree(frames['a'] / 'vit_block32', bad)
        values = np.fromfile(bad / 'input.f32', '<f4')
        values[0] = 123456
        values.tofile(bad / 'input.f32')
        folders = vit_list.read_text(encoding='mbcs').splitlines()
        folders[1] = str(bad)
        bad_list = tmp / 'vit.txt'
        bad_list.write_bytes(('\n'.join(folders)+'\n').encode('mbcs'))
        altered = command.copy()
        altered[5], altered[8] = str(tmp), str(bad_list)
        failure = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (output / 'rejected_handoff.log').write_text(failure.stdout+failure.stderr)
        if failure.returncode != 1 or 'bad_block32: input: FAIL' not in failure.stdout or (tmp / 'frames_metrics.json').exists():
            raise AssertionError('altered ViT second input was accepted')
        altered[2] = '16'
        failure = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True, timeout=30)
        if failure.returncode != 2:
            raise AssertionError('unsupported 32-token attention extent was accepted')
    report = dict(extent=[width, height, 512], vit_tokens=width*height//4, sequence=sequence,
                  metrics=metrics, exact_outputs=exact, source_report_sha256=digest(source / 'report.json'),
                  c512_sources=baseline['sources'], vit_sources=manifests,
                  bridge_map_sha256=digest(output / 'bridge_map.i32'),
                  declared_inputs=prior.get('declared_inputs'),
                  input_provenance=prior.get('input_provenance'),
                  negative_controls=['altered block32 input rejected', '32-token attention extent rejected'],
                  original_kernel_executed=False, original_bridge_validated=False, production_wiring=False,
                  scope='resident C51223-30, pool/head, candidate gather, ViT31-38; default stream')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'], exact_arrays=25, metrics=metrics), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', type=Path, help='completed check_split512_frames output')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--head-fixture', type=Path)
    args = parser.parse_args()
    run(args.frames, args.output_root, args.head_fixture)
