"""Connect resident encoder fronts (5-8, 9-14, 15-22) to resident23-70.

With --first-block 15, each frame uploads its C256 boundary14 input instead
of the initial C512 input and skip22. --first-block 9 adds encoder9-14 and
downsample14 (C128 boundary8 upload, device skip14); --first-block 5 adds
encoder5-8 and downsample8 (C64 boundary4 upload, device skip8). All
downstream references are regenerated from independent scalar/standalone
encoder outputs, and every shorter front mode is also run. This validates
candidate arithmetic and GPU residency, not original NVIDIA parity or game
integration.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np
from check_split512_resident import ROOT, digest, list_file
from check_decoder49_candidate import ENCODER_SHIFTS, run as run_c256
from check_block56_candidate import run as run_c128
from check_block62_candidate import run as run_c64
from check_c256_ffn_candidate import H, F, bits, multiply, e4m3fn
from gpu_test_runner import close_worker, run as run_gpu_test
import check_split512_frames, check_resident_encoder_vit, check_resident_decoder47
import check_resident_decoder55, check_resident_decoder61, check_resident_decoder65, check_resident_head70

CAPTURE_BOUNDARY = {15: 'boundary14', 9: 'boundary8', 5: 'boundary4'}
# Stage specs in execution order from the C512 end outwards. `scale` is the
# spatial factor relative to the width x 8 C512 grid; `sigma` is the captured
# boundary standard deviation used for synthetic FP8 inputs.
STAGES = {
    15: dict(blocks=range(15, 23), shifts=ENCODER_SHIFTS, channels=256, scale=2, runner='c256',
             tensor=('block22', 820288, 0xa8440), inputs=[1, 0, 4, 5, 2, 14, 15, 16],
             outputs=[3, 6, 7, 8, 9, 10, 11, 12, 13], exe='encoder256_downsample_test.exe',
             upload='c256_input', result='input', skip='skip22', raw='raw22', pool='pool22',
             comparisons=122, d2d2=15, uploads=(1+2-2), seed=2215, sigma=5.6957),
    9: dict(blocks=range(9, 15), shifts=(0, 3, 1, 2, 0, 3), channels=128, scale=4, runner='c128',
            tensor=('block14', 229936, 0x30230), inputs=[1, 0, 4, 5, 2, 13, 14],
            outputs=[3, 6, 7, 8, 9, 10, 11, 12], exe='encoder128_downsample_test.exe',
            upload='c128_input', result='c256_input', skip='skip14', raw='raw14', pool='pool14',
            comparisons=92, d2d2=30, uploads=(2+4-4), seed=1409, sigma=8.0211),
    5: dict(blocks=range(5, 9), shifts=(0, 3, 1, 2), channels=64, scale=8, runner='c64',
            tensor=('block8', 69936, 0xf130), inputs=[1, 0, 4, 5, 2, 12],
            outputs=[3, 6, 7, 8, 9, 10, 11], exe='encoder64_downsample_test.exe',
            upload='c64_input', result='c128_input', skip='skip8', raw='raw8', pool='pool8',
            comparisons=62, d2d2=60, uploads=(4+8-8), seed=805, sigma=7.0665),
}
HEAD_NAMES = check_resident_head70.NAMES


def downsample_matrix(spec):
    name, size, offset = spec['tensor']
    records = {r['name']: r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    path = ROOT/'dlss5-analysis/tensors'/f"tensor_{records[f'{name}.layer0.layer']['index']:03d}.bin"
    data = np.fromfile(path, np.uint8)
    c = spec['channels']
    if data.size != size or data.size-offset != 2*c*c:
        raise ValueError(f'wrong {name} downsample tensor extent')
    positions = np.arange(2*c*c, dtype=np.int32)
    inputs, outputs = bits(len(positions), spec['inputs']), bits(len(positions), spec['outputs'])
    if np.unique(outputs*c+inputs).size != len(positions):
        raise ValueError(f'candidate {name} downsample map collides')
    matrix = np.empty((2*c, c), np.float32)
    matrix[outputs, inputs] = e4m3fn(data[offset:])
    return matrix, path


def run_block(spec, block, previous, width, height, folder):
    if spec['runner'] == 'c256':
        return run_c256(block, previous, width=width, height=height, output_root=folder)
    runner = run_c128 if spec['runner'] == 'c128' else run_c64
    return runner(block=block, previous=previous, width=width, height=height, output_root=folder)


def stage_reference(frame, image, spec, matrix):
    """Independent encoder blocks and downsample references for one frame."""
    height, width, c = image.shape
    boundary = frame/'boundary'
    boundary.mkdir(parents=True, exist_ok=True)
    for name in ('output', 'output_device'):
        image.astype('<f4').tofile(boundary/f'{name}.f32')
    image.astype('<f4').tofile(frame/f"{spec['upload']}.f32")
    previous, blocks = boundary/'output_device.f32', []
    for block, shift in zip(spec['blocks'], spec['shifts']):
        folder = frame/f'block{block}'
        previous = Path(run_block(spec, block, previous, width, height, folder))
        if json.loads((folder/'manifest.json').read_text())['shift'] != shift:
            raise AssertionError('encoder shift schedule differs')
        blocks.append((folder, shift))
    raw_path = blocks[-1][0]/'raw_output/output_device.f32'
    raw = np.fromfile(raw_path, '<f4').reshape(height, width, c)
    pool = F(H(H(H(raw[::2, ::2]+raw[::2, 1::2])+H(raw[1::2, ::2]+raw[1::2, 1::2]))*np.float32(.25)))
    output = F(multiply(pool.reshape(-1, c), matrix)).reshape(height//2, width//2, 2*c)
    if not np.isfinite(output).all():
        raise ValueError('nonfinite downsample reference')
    down = frame/'downsample'
    down.mkdir(exist_ok=True)
    for name, array in dict(raw=raw, pool=pool, matrix=matrix, output=output).items():
        np.asarray(array, '<f4').tofile(down/f'{name}.f32')
    run_gpu_test([str(ROOT/'build'/spec['exe']), str(down), str(width), str(height)], cwd=ROOT, check=True)
    for name in ('pool', 'output'):
        if (down/f'{name}_device.f32').read_bytes() != (down/f'{name}.f32').read_bytes():
            raise AssertionError(f'downsample HIP/scalar {name} differs')
    shutil.copyfile(raw_path, frame/f"{spec['raw']}.f32")
    shutil.copyfile(down/'pool_device.f32', frame/f"{spec['pool']}.f32")
    shutil.copyfile(down/'output_device.f32', frame/f"{spec['result']}.f32")
    shutil.copyfile(previous, frame/f"{spec['skip']}.f32")
    return blocks, np.fromfile(previous, '<f4').reshape(height, width, c), output


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def expected_metrics(stages, n):
    hn = n//2
    m = dict(frames=5, invalid_views_rejected=40+4*len(stages), vit_stage_comparisons=80,
             decoder39_stage_comparisons=5, decoder_stage_comparisons=112, prefix48_stage_comparisons=4,
             c256_stage_comparisons=120, prefix56_stage_comparisons=4, c128_stage_comparisons=90,
             prefix62_stage_comparisons=4, c64_stage_comparisons=60, prefix66_stage_comparisons=4,
             c32_stage_comparisons=60, head70_stage_comparisons=20, post_setup_allocations=0,
             compute_h2d_bytes=0, compute_d2h_bytes=0)
    uploads, d2d = 101*n, 2*(372*n+n//2+4*hn+hn//2)
    for first, name in ((15, '22'), (9, '14'), (5, '8')):
        active = first in stages
        m[f'encoder{name}_stage_comparisons'] = STAGES[first]['comparisons'] if active else 0
        m[f'downsample{name}_stage_comparisons'] = 3 if active else 0
        if active:
            uploads -= STAGES[first]['uploads']*n
            d2d += STAGES[first]['d2d2']*n
    m.update(frame_upload_bytes=5*uploads*4, compute_d2d_bytes=5*d2d//2*4)
    return m


def run(output, captured_candidate=None, synthetic=False, first=15):
    output = Path(output).resolve()
    if bool(captured_candidate) == bool(synthetic):
        raise ValueError('declare captured or synthetic input')
    if first not in STAGES:
        raise ValueError('first block must be 15, 9 or 5')
    stages = [s for s in (5, 9, 15) if s >= first]  # execution order
    top = STAGES[first]
    if captured_candidate:
        capture = Path(captured_candidate).resolve()
        width, mode = 8, dict(captured_candidate=capture)
        fixtures = capture/'encoder30'
        head = fixtures/'block30_head_4x4'
        boundary = capture/'encoder22'/CAPTURE_BOUNDARY[first]/'output_device.f32'
        image = np.fromfile(boundary, '<f4').reshape(8*top['scale'], 8*top['scale'], top['channels'])
        enc = json.loads((capture/'encoder22/report.json').read_text())
        provenance = dict(kind=f'same-capture candidate AMD {CAPTURE_BOUNDARY[first]} input for A',
                          capture_sha256=enc['source_capture_sha256'], boundary_sha256=digest(boundary))
    else:
        width, mode = 32, dict(synthetic_skip=True)
        fixtures, head = ROOT/'build/split512_spatial', ROOT/'build/split512_bridge_32x8'
        shape = (8*top['scale'], 32*top['scale'], top['channels'])
        image = F(np.random.default_rng(top['seed']).normal(0, top['sigma'], shape).astype(np.float32))
        provenance = dict(kind=f"synthetic FP8 Gaussian C{top['channels']} input; seed{top['seed']}, sigma{top['sigma']}")
    inputs = dict(a=image, b=np.roll(image, top['scale'], axis=1).copy(), zero=np.zeros_like(image))
    matrices = {s: downsample_matrix(STAGES[s]) for s in stages}
    output.mkdir(parents=True, exist_ok=True)
    references, skips, current = {s: {} for s in stages}, {s: {} for s in stages}, dict(inputs)
    try:
        for s in stages:
            spec = STAGES[s]
            for name in ('a', 'b', 'zero'):
                frame = output/f'encoder{s}'/name
                blocks, skips[s][name], current[name] = stage_reference(frame, current[name], spec, matrices[s][0])
                references[s][name] = [dict(block=b, sha256=digest(p/'manifest.json'))
                                       for b, (p, _) in zip(spec['blocks'], blocks)]
                if name == 'a':
                    list_file(output/f'encoder{s}_blocks.txt', blocks)
            print(f'encoder{s}: independent block/downsample references complete', flush=True)
    finally:
        close_worker()
    if captured_candidate:
        canonical = capture/'encoder22'
        for s in stages:
            last, down = STAGES[s]['blocks'][-1], {15: 'downsample22', 9: 'downsample14', 5: 'downsample8'}[s]
            a = output/f'encoder{s}/a'
            for mine, theirs in ((f'block{last}/output/output_device.f32', f'block{last}/output/output_device.f32'),
                                 (f'block{last}/raw_output/output_device.f32', f'block{last}/raw_output/output_device.f32'),
                                 ('downsample/output_device.f32', f'{down}/output_device.f32'),
                                 ('downsample/matrix.f32', f'{down}/matrix.f32')):
                if (a/mine).read_bytes() != (canonical/theirs).read_bytes():
                    raise AssertionError(f'captured A differs from canonical candidate: encoder{s}/{mine}')
    note = 'same-frame resident encoder outputs of declared ' + CAPTURE_BOUNDARY[first] + ' inputs'
    c512 = check_split512_frames.run(fixtures, output/'c512', width=width, height=8, head_fixture=head,
                                     inputs=current, input_provenance=dict(provenance, derived=note))
    check_resident_encoder_vit.run(output/'c512', output/'vit', head)
    check_resident_decoder47.run(output/'vit', output/'decoder47', head)
    check_resident_decoder55.run(output/'decoder47', output/'decoder55', head_fixture=head,
                                 skips=skips[15], skips_provenance=note, **mode)
    check_resident_decoder61.run(output/'decoder55', output/'decoder47', output/'decoder61', head_fixture=head,
                                 **mode, **(dict(skips=skips[9], skips_provenance=note) if 9 in skips else {}))
    check_resident_decoder65.run(output/'decoder61', output/'decoder65', **mode,
                                 **(dict(skips=skips[5], skips_provenance=note) if 5 in skips else {}))
    check_resident_head70.run(output/'decoder65', output/'head70', **mode)
    sequence = ['a', 'b', 'zero', 'b', 'a']
    for name in ('a', 'b', 'zero'):
        frame, source = output/'frames'/name, output/'head70/references'/name
        if frame.exists():
            shutil.rmtree(frame)
        frame.mkdir(parents=True)
        for path in source.glob('*.f32'):
            link(path, frame/path.name)
        for s in stages:
            spec, enc = STAGES[s], output/f'encoder{s}'/name
            for key in ('upload', 'raw', 'pool', 'result', 'skip'):
                target = frame/f'{spec[key]}.f32'
                if target.exists():
                    # Cascade copies of skips and results must be the encoder's own outputs.
                    if target.read_bytes() != (enc/f'{spec[key]}.f32').read_bytes():
                        raise AssertionError(f'cascade frame does not carry encoder{s} {spec[key]}')
                else:
                    link(enc/f'{spec[key]}.f32', target)
    (output/'frames.txt').write_bytes(''.join(str(output/'frames'/v)+'\n' for v in sequence).encode('mbcs'))
    base = json.loads((output/'head70/native_command.json').read_text())
    base[0] = str(ROOT/'build/split512_frames_test.exe')
    base[4] = str(output/'frames.txt')
    n = width*8*512
    modes, command = {}, base
    for depth, s in enumerate(reversed(stages)):  # 15, then 9, then 5
        active = [v for v in (5, 9, 15) if v >= s]
        target = output/f'mode{s}'
        for i in range(5):
            (target/f'frame_{i}').mkdir(parents=True, exist_ok=True)
        command = command+[str(output/f'encoder{s}_blocks.txt'), str(output/f'encoder{s}/a/downsample')]
        command[5] = str(target)
        (target/'native_command.json').write_text(json.dumps(command, indent=2)+'\n')
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (target/'device.log').write_text(result.stdout+result.stderr)
        if result.returncode or 'FAIL' in result.stdout:
            raise RuntimeError(f'resident encoder{s} front failed: {target}/device.log\n{result.stdout[-3000:]}{result.stderr}')
        m = json.loads((target/'frames_metrics.json').read_text())
        expected = expected_metrics(active, n)
        stage_checks = 563+sum(STAGES[v]['comparisons']+3 for v in active)
        names = [*HEAD_NAMES]
        for v in active:
            names += [STAGES[v][k] for k in ('skip', 'raw', 'pool', 'result') if STAGES[v][k] not in names]
        if (any(m[k] != v for k, v in expected.items()) or result.stdout.count('PASS') != stage_checks+5*len(names) or
                not all(m[k] for k in ('resident_head70', 'input_preserved', 'weights_loaded_once')) or
                any(m[f'resident_encoder{x}'] != (v in active) for v, x in ((15, '22'), (9, '14'), (5, '8')))):
            raise AssertionError(f'encoder{s} diagnostic/transfer contract differs: {m}')
        exact = []
        for i, name in enumerate(sequence):
            hashes = {}
            for key in names:
                path = target/f'frame_{i}/{key}.f32'
                if path.read_bytes() != (output/f'frames/{name}/{key}.f32').read_bytes():
                    raise AssertionError(f'endpoint differs: encoder{s} {i}/{key}')
                hashes[key] = digest(path)
            record = dict(frame=name, outputs=hashes)
            for key in (STAGES[s]['upload'], 'skip4', 'skip0', 'color',
                        *(['skip14'] if 9 not in active else []), *(['skip8'] if 5 not in active else [])):
                record[f'{key}_sha256'] = digest(output/f'frames/{name}/{key}.f32')
            exact.append(record)
        modes[s] = dict(command=command, metrics=m, stage_checks=stage_checks, exact_outputs=exact,
                        exact_arrays=5*len(names))
        print(f'encoder{s} front: {stage_checks} stage checks, {5*len(names)} exact arrays, '
              f"{min(m['frame_ms']):.2f}-{max(m['frame_ms']):.2f} ms", flush=True)
    controls = []
    command = modes[first]['command']
    with tempfile.TemporaryDirectory(prefix='encoder controls ', dir=output) as temporary:
        tmp = Path(temporary)
        for s in stages:
            spec = STAGES[s]
            index = 20+2*[15, 9, 5].index(s)
            second, last = spec['blocks'][1], spec['blocks'][-1]
            a = output/f'encoder{s}/a'
            for label, original, relative, expected_text, line in (
                    (f'block{second}', a/f'block{second}', 'spatial/input.f32', f'bad_block{second}/spatial: input: FAIL', 1),
                    (spec['raw'], a/f'block{last}', 'raw_output/output.f32', f"bad_{spec['raw']}/raw_output: output: FAIL", len(spec['blocks'])-1),
                    (spec['pool'], a/'downsample', 'pool.f32', f"bad_{spec['pool']}: pool: FAIL", None)):
                bad = tmp/f'bad_{label}'
                shutil.copytree(original, bad, dirs_exist_ok=True, copy_function=lambda x, y: Path(y).hardlink_to(x))
                p = bad/relative
                values = np.fromfile(p, '<f4')
                values[0] = 123456
                p.unlink()
                values.tofile(p)
                cmd = command.copy()
                cmd[5] = str(tmp)
                if line is None:
                    cmd[index+1] = str(bad)
                else:
                    lines = (output/f'encoder{s}_blocks.txt').read_text(encoding='mbcs').splitlines()
                    lines[line] = lines[line].split('\t')[0]+'\t'+str(bad)
                    (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
                    cmd[index] = str(tmp/'blocks.txt')
                failure = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                (output/f'rejected_{label}.log').write_text(failure.stdout+failure.stderr)
                if (failure.returncode != 1 or expected_text not in failure.stdout or
                        (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):
                    raise AssertionError(f'altered {label} accepted')
                controls.append(f'altered {label} expectation rejected before publishing outputs')
    report = dict(extent=[width, 8, 512], first_block=first,
                  input_extent=[width*top['scale'], 8*top['scale'], top['channels']],
                  output_extent=[width*32, 256, 3], sequence=sequence, input_provenance=provenance,
                  modes={str(s): {k: v for k, v in r.items() if k != 'command'} for s, r in modes.items()},
                  paired_controls=f"B rolls the {top['upload']} tensor {top['scale']} columns; zero is zero. "
                                  'Remaining external skips and RGB stay synthetic pairs from the cascade checkers.',
                  cascade_report_sha256={k: digest(output/k/'report.json') for k in
                                         ('c512', 'vit', 'decoder47', 'decoder55', 'decoder61', 'decoder65', 'head70')},
                  c512_declared_inputs=c512['declared_inputs'],
                  downsample_tensor_sha256={str(s): digest(matrices[s][1]) for s in stages},
                  encoder_references={str(s): r for s, r in references.items()}, negative_controls=controls,
                  original_kernel_executed=False, production_wiring=False,
                  map_status='candidate C64/C128 measured and C256 extended maps; original maps unverified')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'], first_block=first,
                          modes={s: dict(stage_checks=r['stage_checks'], exact_arrays=r['exact_arrays'],
                                         frame_ms=r['metrics']['frame_ms']) for s, r in modes.items()},
                          negative_controls=controls), indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--first-block', type=int, choices=(15, 9, 5), default=15)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--captured-candidate', type=Path)
    g.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    run(a.output_root, a.captured_candidate, a.synthetic, a.first_block)
