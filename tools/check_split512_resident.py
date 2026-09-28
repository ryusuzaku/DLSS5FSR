"""Validate a resident eight-block C512 chain against saved exact fixtures.

Weights are checked against the existing source-validated base fixtures.
The native runner uploads one activation, preloads weights, checks all stages,
then repeats with zero allocation or host transfers inside the timed region.
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile

from check_split512_spatial_block import ROOT, load_base

SHIFTS = (0, 3, 1, 2, 0, 3, 1, 2)
EXE = ROOT / 'build/split512_resident_test.exe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_file(path, blocks):
    lines = []
    for folder, shift in blocks:
        value = str(folder.resolve())
        if any(c in value for c in '\r\n\t'):
            raise ValueError('fixture path contains a protocol separator')
        lines.append(f'{shift}\t{value}\n')
    path.write_bytes(''.join(lines).encode('mbcs'))


def run(fixtures, output, first=23, width=8, height=8, repeats=10, controls=False,
        head_fixture=None):
    fixtures, output = Path(fixtures).resolve(), Path(output).resolve()
    if first not in (23, 40) or width not in (8, 16, 32) or height != 8 or not 1 <= repeats <= 1000:
        raise ValueError('supported: encoder23 or decoder40, width8/16/32, height8, repeats1..1000')
    if not EXE.is_file():
        raise FileNotFoundError('build with bash tools/build_split512_resident.sh')
    blocks = []
    sources = []
    previous = None
    for block, shift in zip(range(first, first+8), SHIFTS):
        folder = fixtures / f'block{block}-{width}x{height}-s{shift}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        if any(manifest[k] != v for k, v in dict(block=block, width=width, height=height, shift=shift).items()):
            raise ValueError('fixture extent/block/shift differs')
        for name, sha in manifest['files'].items():
            if Path(name).name != name or digest(folder / name) != sha:
                raise ValueError(f'changed fixture: {folder.name}/{name}')
        weights, base = load_base(block)
        for name, values in weights.items():
            if (folder / f'{name}.f32').read_bytes() != values.astype('<f4').tobytes():
                raise ValueError(f'weights differ from source-validated base: {block}/{name}')
        for name in ('raw_sha256', 'projection_sha256', 'attention_sha256', 'final_sha256'):
            if manifest['original_tensor_hashes'][name] != base[name]:
                raise ValueError('source weight ancestry differs')
        if previous is not None and (folder / 'input.f32').read_bytes() != previous.read_bytes():
            raise ValueError('saved inter-block activation handoff differs')
        for name in ('final', 'final_raw'):
            if (folder / f'{name}.f32').read_bytes() != (folder / f'{name}_device.f32').read_bytes():
                raise ValueError('saved scalar/standalone result differs')
        previous = folder / 'final_device.f32'
        blocks.append((folder, shift))
        sources.append(dict(block=block, manifest_sha256=digest(folder / 'manifest.json')))
    # Optional block30 head requires its own existing, checked fixture.
    if head_fixture is not None and first != 23:
        raise ValueError('head fixture is only valid for encoder23..30')
    head = ((Path(head_fixture).resolve() if head_fixture is not None else
             fixtures / f'block30_head_{width//2}x{height//2}') if first == 23 else None)
    if head is not None:
        manifest = json.loads((head / 'manifest.json').read_text())
        for name, sha in manifest['files'].items():
            if Path(name).name != name or digest(head / name) != sha:
                raise ValueError('changed head fixture')
        if (head / 'raw.f32').read_bytes() != (blocks[-1][0] / 'final_raw_device.f32').read_bytes():
            raise ValueError('saved head activation handoff differs')
        raw_path = ROOT / 'dlss5-analysis/tensors' / f"tensor_{manifest['tensor']:03d}.bin"
        if digest(raw_path) != manifest['tensor_sha256']:
            raise ValueError('head source weight ancestry differs')
    output.mkdir(parents=True, exist_ok=True)
    command_file = output / 'blocks.txt'
    list_file(command_file, blocks)
    command = [str(EXE), str(command_file), str(width), str(height), str(repeats), str(output)]
    if head is not None:
        command.append(str(head))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (output / 'device.log').write_text(result.stdout + result.stderr)
    if result.returncode or 'FAIL' in result.stdout:
        raise RuntimeError(f'resident chain failed: {output / "device.log"}\n{result.stdout[-2000:]}{result.stderr}')
    metrics = json.loads((output / 'resident_metrics.json').read_text())
    comparisons = 8*14 + (3 if head is not None else 0)
    if (metrics['diagnostic_comparisons'] != comparisons or result.stdout.count('PASS') != comparisons+1 or
            any(metrics[k] for k in ('repeat_allocations', 'repeat_h2d_bytes', 'repeat_d2h_bytes')) or
            metrics['repeat_d2d_bytes'] != repeats*width*height*512*4 or
            metrics['activation_uploads'] != 1 or metrics['device_handoffs'] != 7+(head is not None)):
        raise AssertionError('resident transfer/stage contract differs')
    exact = {}
    for name in ('final', 'final_raw', *(['head'] if head is not None else [])):
        baseline = (head if name == 'head' else blocks[-1][0]) / f'{name}_device.f32'
        if (output / f'{name}.f32').read_bytes() != baseline.read_bytes():
            raise AssertionError(f'resident {name} is not byte-exact with standalone')
        exact[name] = digest(baseline)
    negative_controls = []
    if controls:
        with tempfile.TemporaryDirectory(prefix='resident controls ', dir=output) as tmp:
            tmp = Path(tmp)
            bad = tmp / 'bad_second_block'
            shutil.copytree(blocks[1][0], bad)
            source = bad / 'input.f32'
            data = source.read_bytes()
            source.write_bytes(struct.pack('<f', 123456.) + data[4:])
            bad_list = tmp / 'bad.txt'
            list_file(bad_list, [blocks[0], (bad, blocks[1][1])])
            bad_command = [str(EXE), str(bad_list), str(width), str(height), '1', str(tmp)]
            failed = subprocess.run(bad_command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            (output / 'rejected_handoff.log').write_text(failed.stdout + failed.stderr)
            if failed.returncode != 1 or ': input: FAIL' not in failed.stdout or (tmp / 'final.f32').exists():
                raise AssertionError('changed second input was uploaded/accepted instead of rejected')
            negative_controls.append('altered second input rejected before publishing outputs')
            bad_command[2] = '6'
            invalid = subprocess.run(bad_command, cwd=ROOT, capture_output=True, text=True, timeout=10)
            if invalid.returncode != 2:
                raise AssertionError('invalid extent accepted')
            negative_controls.append('invalid extent rejected')
    report = dict(first_block=first, extent=[width, height, 512], sources=sources,
                  initial_input_sha256=digest(blocks[0][0] / 'input.f32'), exact_outputs=exact,
                  metrics=metrics, negative_controls=negative_controls,
                  original_kernel_executed=False, production_wiring=False,
                  scope='resident C512 component replay on prevalidated fixtures')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--first-block', type=int, choices=(23, 40), default=23)
    parser.add_argument('--width', type=int, choices=(8, 16, 32), default=8)
    parser.add_argument('--height', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--head-fixture', type=Path, help='separate saved block30 pool/head fixture')
    args = parser.parse_args()
    run(args.fixtures, args.output_root, args.first_block, args.width, args.height, args.repeats,
        args.controls, args.head_fixture)
