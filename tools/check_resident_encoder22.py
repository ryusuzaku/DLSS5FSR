"""Connect resident encoder15-22 and block22 downsample to resident23-70.

Each frame uploads its C256 boundary14 input instead of the initial C512
input and skip22. All downstream references are regenerated from the
independent encoder15-22 scalar/standalone outputs. This validates candidate
arithmetic and GPU residency, not original NVIDIA parity or game integration.
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
from check_c256_ffn_candidate import H, F, bits, multiply, e4m3fn
from gpu_test_runner import close_worker, run as run_gpu_test
import check_split512_frames, check_resident_encoder_vit, check_resident_decoder47
import check_resident_decoder55, check_resident_decoder61, check_resident_decoder65, check_resident_head70

NAMES = (*check_resident_head70.NAMES, 'raw22', 'pool22')
UPSTREAM = ('c256_input', 'skip22', 'input', 'skip14', 'skip8', 'skip4', 'skip0', 'color')
SIGMA = 5.6957  # captured boundary14 standard deviation


def downsample_matrix():
    records = {r['name']: r for r in json.loads((ROOT/'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    path = ROOT/'dlss5-analysis/tensors'/f"tensor_{records['block22.layer0.layer']['index']:03d}.bin"
    data = np.fromfile(path, np.uint8)
    if data.size != 820288 or data.size-0xa8440 != 131072:
        raise ValueError('wrong block22 downsample tensor extent')
    positions = np.arange(2*256*256, dtype=np.int32)
    inputs = bits(len(positions), [1, 0, 4, 5, 2, 14, 15, 16])
    outputs = bits(len(positions), [3, 6, 7, 8, 9, 10, 11, 12, 13])
    if np.unique(outputs*256+inputs).size != len(positions):
        raise ValueError('candidate block22 downsample map collides')
    matrix = np.empty((512, 256), np.float32)
    matrix[outputs, inputs] = e4m3fn(data[0xa8440:])
    return matrix, path


def encoder_reference(frame, c256, matrix):
    """Independent encoder15-22 and downsample22 references for one frame."""
    height, width = c256.shape[:2]
    boundary = frame/'boundary14'
    boundary.mkdir(parents=True, exist_ok=True)
    for name in ('output', 'output_device'):
        c256.astype('<f4').tofile(boundary/f'{name}.f32')
    c256.astype('<f4').tofile(frame/'c256_input.f32')
    previous, blocks = boundary/'output_device.f32', []
    for block, shift in zip(range(15, 23), ENCODER_SHIFTS):
        folder = frame/f'block{block}'
        previous = run_c256(block, previous, width=width, height=height, output_root=folder)
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['shift'] != shift:
            raise AssertionError('encoder shift schedule differs')
        blocks.append((folder, shift))
    raw_path = frame/'block22/raw_output/output_device.f32'
    raw = np.fromfile(raw_path, '<f4').reshape(height, width, 256)
    pool = F(H(H(H(raw[::2, ::2]+raw[::2, 1::2])+H(raw[1::2, ::2]+raw[1::2, 1::2]))*np.float32(.25)))
    output = F(multiply(pool.reshape(-1, 256), matrix)).reshape(height//2, width//2, 512)
    if not np.isfinite(output).all():
        raise ValueError('nonfinite downsample22 reference')
    down = frame/'downsample22'
    down.mkdir(exist_ok=True)
    for name, array in dict(raw=raw, pool=pool, matrix=matrix, output=output).items():
        np.asarray(array, '<f4').tofile(down/f'{name}.f32')
    run_gpu_test([str(ROOT/'build/encoder256_downsample_test.exe'), str(down), str(width), str(height)],
                 cwd=ROOT, check=True)
    for name in ('pool', 'output'):
        if (down/f'{name}_device.f32').read_bytes() != (down/f'{name}.f32').read_bytes():
            raise AssertionError(f'downsample22 HIP/scalar {name} differs')
    shutil.copyfile(raw_path, frame/'raw22.f32')
    shutil.copyfile(down/'pool_device.f32', frame/'pool22.f32')
    return blocks, np.fromfile(previous, '<f4').reshape(height, width, 256), output


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def run(output, captured_candidate=None, synthetic=False):
    output = Path(output).resolve()
    if bool(captured_candidate) == bool(synthetic):
        raise ValueError('declare captured or synthetic input')
    matrix, tensor_path = downsample_matrix()
    if captured_candidate:
        capture = Path(captured_candidate).resolve()
        width, mode = 8, dict(captured_candidate=capture)
        fixtures = capture/'encoder30'
        head = fixtures/'block30_head_4x4'
        c256 = np.fromfile(capture/'encoder22/boundary14/output_device.f32', '<f4').reshape(16, 16, 256)
        enc = json.loads((capture/'encoder22/report.json').read_text())
        provenance = dict(kind='same-capture candidate AMD encoder14 downsample for A',
                          capture_sha256=enc['source_capture_sha256'],
                          boundary14_sha256=digest(capture/'encoder22/boundary14/output_device.f32'))
    else:
        width, mode = 32, dict(synthetic_skip=True)
        fixtures, head = ROOT/'build/split512_spatial', ROOT/'build/split512_bridge_32x8'
        c256 = F(np.random.default_rng(2215).normal(0, SIGMA, (16, 64, 256)).astype(np.float32))
        provenance = dict(kind=f'synthetic FP8 Gaussian C256 input; seed2215, sigma{SIGMA}')
    inputs = dict(a=c256, b=np.roll(c256, 2, axis=1).copy(), zero=np.zeros_like(c256))
    output.mkdir(parents=True, exist_ok=True)
    encoder_refs, downs, skips = {}, {}, {}
    try:
        for name, image in inputs.items():
            frame = output/'encoder22'/name
            blocks, skips[name], downs[name] = encoder_reference(frame, image, matrix)
            encoder_refs[name] = [dict(block=b, sha256=digest(p/'manifest.json'))
                                  for b, (p, _) in zip(range(15, 23), blocks)]
            if name == 'a':
                list_file(output/'encoder_blocks.txt', blocks)
            print(f'{name}: independent encoder15-22/downsample22 references complete', flush=True)
    finally:
        close_worker()
    if captured_candidate:
        canonical = capture/'encoder22'
        a = output/'encoder22/a'
        for mine, theirs in (('block22/output/output_device.f32', 'block22/output/output_device.f32'),
                             ('block22/raw_output/output_device.f32', 'block22/raw_output/output_device.f32'),
                             ('downsample22/output_device.f32', 'downsample22/output_device.f32'),
                             ('downsample22/matrix.f32', 'downsample22/matrix.f32')):
            if (a/mine).read_bytes() != (canonical/theirs).read_bytes():
                raise AssertionError(f'captured A differs from canonical candidate: {mine}')
    note = 'same-frame resident encoder15-22 + downsample22 outputs of declared C256 inputs'
    c512 = check_split512_frames.run(fixtures, output/'c512', width=width, height=8, head_fixture=head,
                                     inputs=downs, input_provenance=dict(provenance, derived=note))
    check_resident_encoder_vit.run(output/'c512', output/'vit', head)
    check_resident_decoder47.run(output/'vit', output/'decoder47', head)
    check_resident_decoder55.run(output/'decoder47', output/'decoder55', head_fixture=head,
                                 skips=skips, skips_provenance=note, **mode)
    check_resident_decoder61.run(output/'decoder55', output/'decoder47', output/'decoder61', head_fixture=head, **mode)
    check_resident_decoder65.run(output/'decoder61', output/'decoder65', **mode)
    check_resident_head70.run(output/'decoder65', output/'head70', **mode)
    sequence = ['a', 'b', 'zero', 'b', 'a']
    for name in ('a', 'b', 'zero'):
        frame, source, enc = output/'frames'/name, output/'head70/references'/name, output/'encoder22'/name
        if frame.exists():
            shutil.rmtree(frame)
        frame.mkdir(parents=True)
        for path in source.glob('*.f32'):
            link(path, frame/path.name)
        for key in ('c256_input', 'raw22', 'pool22'):
            link(enc/f'{key}.f32', frame/f'{key}.f32')
        if ((frame/'skip22.f32').read_bytes() != skips[name].astype('<f4').tobytes() or
                (frame/'input.f32').read_bytes() != downs[name].astype('<f4').tobytes()):
            raise AssertionError('cascade frame does not carry encoder22 outputs')
    (output/'frames.txt').write_bytes(''.join(str(output/'frames'/v)+'\n' for v in sequence).encode('mbcs'))
    for i in range(5):
        (output/f'frame_{i}').mkdir(exist_ok=True)
    command = json.loads((output/'head70/native_command.json').read_text())
    command[0] = str(ROOT/'build/split512_frames_test.exe')
    command[4], command[5] = str(output/'frames.txt'), str(output)
    command.extend([str(output/'encoder_blocks.txt'), str(output/'encoder22/a/downsample22')])
    (output/'native_command.json').write_text(json.dumps(command, indent=2)+'\n')
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (output/'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:
        raise RuntimeError(f'resident encoder22 failed: {output}/device.log\n{result.stdout[-3000:]}{result.stderr}')
    m = json.loads((output/'frames_metrics.json').read_text())
    n = width*8*512
    hn = n//2
    expected = dict(frames=5, invalid_views_rejected=44, vit_stage_comparisons=80, decoder39_stage_comparisons=5,
                    decoder_stage_comparisons=112, prefix48_stage_comparisons=4, c256_stage_comparisons=120,
                    prefix56_stage_comparisons=4, c128_stage_comparisons=90, prefix62_stage_comparisons=4,
                    c64_stage_comparisons=60, prefix66_stage_comparisons=4, c32_stage_comparisons=60,
                    head70_stage_comparisons=20, encoder22_stage_comparisons=122, downsample22_stage_comparisons=3,
                    post_setup_allocations=0, compute_h2d_bytes=0, compute_d2h_bytes=0,
                    frame_upload_bytes=5*100*n*4, compute_d2d_bytes=5*(379*n+n+4*hn+hn//2)*4)
    if (any(m[k] != v for k, v in expected.items()) or result.stdout.count('PASS') != 688+160 or
            not all(m[k] for k in ('resident_encoder22', 'resident_head70', 'resident_c32', 'resident_c64',
                                   'resident_c128', 'resident_c256', 'resident_decoder', 'resident_vit',
                                   'input_preserved', 'weights_loaded_once'))):
        raise AssertionError(f'resident diagnostic/transfer contract differs: {m}')
    exact = []
    for i, name in enumerate(sequence):
        hashes = {}
        for key in ('skip22', 'input', *NAMES):
            path = output/f'frame_{i}/{key}.f32'
            if path.read_bytes() != (output/f'frames/{name}/{key}.f32').read_bytes():
                raise AssertionError(f'endpoint differs: {i}/{key}')
            hashes[key] = digest(path)
        record = dict(frame=name, outputs=hashes)
        for key in UPSTREAM:
            if key not in ('skip22', 'input'):
                record[f'{key}_sha256'] = digest(output/f'frames/{name}/{key}.f32')
        exact.append(record)
    controls = []
    with tempfile.TemporaryDirectory(prefix='encoder22 controls ', dir=output) as temporary:
        tmp = Path(temporary)
        a = output/'encoder22/a'
        for label, original, relative, expected_text in (
                ('block16', a/'block16', 'spatial/input.f32', 'bad_block16/spatial: input: FAIL'),
                ('raw22', a/'block22', 'raw_output/output.f32', 'bad_raw22/raw_output: output: FAIL'),
                ('pool22', a/'downsample22', 'pool.f32', 'bad_pool22: pool: FAIL')):
            bad = tmp/f'bad_{label}'
            shutil.copytree(original, bad, dirs_exist_ok=True, copy_function=lambda s, d: Path(d).hardlink_to(s))
            p = bad/relative
            values = np.fromfile(p, '<f4')
            values[0] = 123456
            p.unlink()
            values.tofile(p)
            cmd = command.copy()
            cmd[5] = str(tmp)
            if label == 'pool22':
                cmd[21] = str(bad)
            else:
                lines = (output/'encoder_blocks.txt').read_text(encoding='mbcs').splitlines()
                index = 1 if label == 'block16' else 7
                lines[index] = lines[index].split('\t')[0]+'\t'+str(bad)
                (tmp/'blocks.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
                cmd[20] = str(tmp/'blocks.txt')
            failure = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            (output/f'rejected_{label}.log').write_text(failure.stdout+failure.stderr)
            if (failure.returncode != 1 or expected_text not in failure.stdout or
                    (tmp/'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):
                raise AssertionError(f'altered {label} accepted')
            controls.append(f'altered {label} expectation rejected before publishing outputs')
    report = dict(extent=[width, 8, 512], input_extent=[2*width, 16, 256], output_extent=[width*32, 256, 3],
                  sequence=sequence, metrics=m, exact_outputs=exact, input_provenance=provenance,
                  paired_controls='B rolls the C256 input two columns; zero is zero. Skip14/8/4/0 and RGB stay '
                                  'external synthetic pairs from the cascade checkers.',
                  cascade_report_sha256={k: digest(output/k/'report.json') for k in
                                         ('c512', 'vit', 'decoder47', 'decoder55', 'decoder61', 'decoder65', 'head70')},
                  c512_declared_inputs=c512['declared_inputs'], downsample_tensor_sha256=digest(tensor_path),
                  encoder_references=encoder_refs, negative_controls=controls, original_kernel_executed=False,
                  production_wiring=False, c256_map_status='candidate C256 extension; original maps unverified')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'], exact_arrays=5*(len(NAMES)+2), metrics=m,
                          negative_controls=controls), indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--captured-candidate', type=Path)
    g.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    run(a.output_root, a.captured_candidate, a.synthetic)
