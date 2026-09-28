"""Extend a validated resident encoder/ViT run through decoder39-47."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import tempfile

import numpy as np

from check_split512_resident import ROOT, SHIFTS, digest, list_file, run as check_fixed
from check_split512_spatial_block import run_case
from check_decoder39_entry import D
from audit_native_vit_logical_map import logical_map
from gpu_test_runner import close_worker, run as run_gpu_test

PRIOR_NAMES = ('final', 'final_raw', 'head', 'bridge', 'vit')
NEW_NAMES = ('main39', 'projected39', 'decoder39', 'decoder47', 'decoder47_raw')


def run(source, output, head_fixture=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('use separate source and output directories')
    prior = json.loads((source / 'report.json').read_text())
    width, height, channels = prior['extent']
    sequence = ['a', 'b', 'zero', 'b', 'a']
    if (width not in (8, 32) or height != 8 or channels != 512 or
            prior['sequence'] != sequence or not prior['metrics']['resident_vit']):
        raise ValueError('requires a completed 16/64-token resident encoder/ViT check')
    block_lines = (source / 'fixed_control/blocks.txt').read_text(encoding='mbcs').splitlines()
    if len(block_lines) != 8 or [int(v.split('\t')[0]) for v in block_lines] != list(SHIFTS):
        raise ValueError('invalid encoder block list')
    fixtures = Path(block_lines[0].split('\t')[1]).parent
    head = Path(head_fixture).resolve() if head_fixture else fixtures / f'block30_head_{width//2}x4'
    output.mkdir(parents=True, exist_ok=True)
    baseline = check_fixed(fixtures, output / 'fixed_control', width=width, height=height,
                           head_fixture=head, repeats=1)
    if baseline['sources'] != prior['c512_sources']:
        raise ValueError('encoder source ancestry changed')
    gather = logical_map(width*height//4).astype('<i4')
    if ((source / 'bridge_map.i32').read_bytes() != gather.tobytes() or
            digest(source / 'bridge_map.i32') != prior['bridge_map_sha256']):
        raise ValueError('forward map changed')
    inverse = np.argsort(gather).astype('<i4')
    if len(prior['exact_outputs']) != 5:
        raise ValueError('incomplete source output report')
    for name, record in zip(sequence, prior['exact_outputs']):
        if record['frame'] != name:
            raise ValueError('source frame ordering changed')
        for tensor in PRIOR_NAMES:
            if digest(source / f'references/{name}/{tensor}.f32') != record['outputs'][tensor]:
                raise ValueError(f'source output changed: {name}/{tensor}')
    declared = prior.get('declared_inputs')
    a = np.fromfile(source / 'references/a/input.f32', '<f4').reshape(height, width, 512)
    expected_inputs = dict(a=a, b=np.roll(a, 1, axis=1).copy(), zero=np.zeros_like(a))
    if not declared and (source / 'references/a/input.f32').read_bytes() != (fixtures / f'block23-{width}x8-s0/input.f32').read_bytes():
        raise ValueError('source input A changed')
    # Check every saved ViT fixture and handoff, not just its final projection.
    for name in ('a', 'b', 'zero'):
        frame = source / 'references' / name
        if declared:
            if digest(frame / 'input.f32') != declared[name]:
                raise ValueError('declared changing input differs')
        elif (frame / 'input.f32').read_bytes() != expected_inputs[name].astype('<f4').tobytes():
            raise ValueError('source changing input differs')
        previous = frame / 'bridge.f32'
        records = prior['vit_sources'][name]
        if len(records) != 8:
            raise ValueError('incomplete ViT source manifests')
        for block, record in zip(range(31, 39), records):
            folder = frame / f'vit_block{block}'
            manifest_path = folder / 'manifest.json'
            if digest(manifest_path) != record['sha256']:
                raise ValueError('ViT source manifest changed')
            manifest = json.loads(manifest_path.read_text())
            if manifest['block'] != block or manifest['source_device_sha256'] != digest(previous):
                raise ValueError('ViT source handoff changed')
            for filename, sha in manifest['files'].items():
                if Path(filename).name != filename or digest(folder / filename) != sha:
                    raise ValueError('ViT source fixture changed')
            for part in ('expansion', 'contraction', 'qkv', 'projection'):
                raw = ROOT / 'dlss5-analysis/tensors' / f"tensor_{manifest[part+'_tensor']:03d}.bin"
                if digest(raw) != manifest[part+'_sha256']:
                    raise ValueError('ViT original tensor changed')
            if (folder / 'input.f32').read_bytes() != previous.read_bytes():
                raise ValueError('ViT reference input differs')
            previous = folder / 'projection_device.f32'
        if previous.read_bytes() != (frame / 'vit.f32').read_bytes():
            raise ValueError('ViT38 reference differs')
    records = {r['name']: r for r in json.loads((ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    tensor_index = records['block39.layer0.layer']['index']
    tensor_path = ROOT / 'dlss5-analysis/tensors' / f'tensor_{tensor_index:03d}.bin'
    if tensor_path.stat().st_size != 525312:
        raise ValueError('wrong decoder39 weight extent')
    weights, scale = D.unpack(tensor_path)
    decoder_sources = {}
    try:
        for name in ('a', 'b', 'zero'):
            frame = output / 'references' / name
            entry = frame / 'entry39'
            entry.mkdir(parents=True, exist_ok=True)
            for tensor in ('input', *PRIOR_NAMES):
                shutil.copyfile(source / f'references/{name}/{tensor}.f32', frame / f'{tensor}.f32')
            vit = np.fromfile(frame / 'vit.f32', '<f4')
            main = vit[inverse].reshape(4, width//2, 1024)
            skip = np.fromfile(frame / 'final.f32', '<f4').reshape(height, width, 512)
            projected = D.project(main, weights)
            merged = D.decoder_entry(main, skip, (weights, scale))
            for tensor, array in dict(vit=vit, main=main, skip=skip, projected=projected,
                                      output=merged, weights=weights, scale=scale).items():
                if not np.isfinite(array).all():
                    raise ValueError(f'nonfinite decoder39 {tensor}')
                np.asarray(array, '<f4').tofile(entry / f'{tensor}.f32')
            inverse.tofile(entry / 'inverse.i32')
            run_gpu_test([str(ROOT / 'build/decoder39_entry_test.exe'), str(entry), str(width//2), '4'],
                         cwd=ROOT, check=True)
            if (entry / 'output_device.f32').read_bytes() != (entry / 'output.f32').read_bytes():
                raise AssertionError('decoder39 scalar/standalone differs')
            for target, original in (('main39', 'main'), ('projected39', 'projected'), ('decoder39', 'output_device')):
                shutil.copyfile(entry / f'{original}.f32', frame / f'{target}.f32')
            entry_report = dict(tensor=tensor_index, tensor_sha256=digest(tensor_path),
                                source_vit_sha256=digest(frame / 'vit.f32'), source_skip_sha256=digest(frame / 'final.f32'),
                                files={p.name: digest(p) for p in entry.iterdir() if p.suffix in ('.f32', '.i32')})
            (entry / 'manifest.json').write_text(json.dumps(entry_report, indent=2)+'\n')
            previous = entry / 'output_device.f32'
            blocks = []
            for block, shift in zip(range(40, 48), SHIFTS):
                x = np.fromfile(previous, '<f4').reshape(-1, 512)
                folder = run_case(block, width, height, shift, x, frame / 'decoder')
                if (folder / 'input.f32').read_bytes() != previous.read_bytes():
                    raise AssertionError('decoder handoff differs')
                for field in ('final', 'final_raw'):
                    if (folder / f'{field}.f32').read_bytes() != (folder / f'{field}_device.f32').read_bytes():
                        raise AssertionError('decoder scalar/standalone differs')
                previous = folder / 'final_device.f32'
                blocks.append((folder, shift))
            shutil.copyfile(previous, frame / 'decoder47.f32')
            shutil.copyfile(folder / 'final_raw_device.f32', frame / 'decoder47_raw.f32')
            decoder_sources[name] = [dict(block=b, manifest_sha256=digest(p / 'manifest.json'))
                                     for b, (p, _) in zip(range(40, 48), blocks)]
            if name == 'a':
                list_file(output / 'decoder_blocks.txt', blocks)
    finally:
        close_worker()
    if len({digest(output / f'references/{name}/decoder47.f32') for name in ('a', 'b', 'zero')}) != 3:
        raise AssertionError('decoder changing inputs did not produce distinct outputs')
    vit_folders = [source / f'references/a/vit_block{b}' for b in range(31, 39)]
    (output / 'vit_blocks.txt').write_bytes(''.join(str(p)+'\n' for p in vit_folders).encode('mbcs'))
    (output / 'frames.txt').write_bytes(''.join(str(output / 'references' / n)+'\n' for n in sequence).encode('mbcs'))
    gather.tofile(output / 'bridge_map.i32')
    for i in range(5):
        (output / f'frame_{i}').mkdir(exist_ok=True)
    command = [str(ROOT / 'build/split512_frames_test.exe'), str(output / 'fixed_control/blocks.txt'),
               str(width), '8', str(output / 'frames.txt'), str(output), str(head),
               str(output / 'bridge_map.i32'), str(output / 'vit_blocks.txt'),
               str(output / 'references/a/entry39'), str(output / 'decoder_blocks.txt')]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    (output / 'device.log').write_text(result.stdout+result.stderr)
    if result.returncode or 'FAIL' in result.stdout:
        raise RuntimeError(f'resident decoder failed: {output / "device.log"}\n{result.stdout[-3000:]}{result.stderr}')
    m = json.loads((output / 'frames_metrics.json').read_text())
    n, hn = width*8*512, width*8//4*1024
    if (m['frames'] != 5 or m['invalid_views_rejected'] != 10 or m['vit_stage_comparisons'] != 80 or
            m['decoder39_stage_comparisons'] != 5 or m['decoder_stage_comparisons'] != 112 or
            not all(m[k] for k in ('resident_vit', 'resident_decoder', 'adjacent_vit_bridge', 'input_preserved', 'weights_loaded_once')) or
            any(m[k] for k in ('post_setup_allocations', 'compute_h2d_bytes', 'compute_d2h_bytes')) or
            m['frame_upload_bytes'] != 5*n*4 or m['compute_d2d_bytes'] != 5*(7*n+4*hn+hn//2)*4 or
            result.stdout.count('PASS') != 247):
        raise AssertionError('resident decoder contract differs')
    exact = []
    for i, name in enumerate(sequence):
        hashes = {}
        for tensor in (*PRIOR_NAMES, *NEW_NAMES):
            observed = output / f'frame_{i}/{tensor}.f32'
            if observed.read_bytes() != (output / f'references/{name}/{tensor}.f32').read_bytes():
                raise AssertionError(f'frame{i}/{tensor} differs')
            hashes[tensor] = digest(observed)
        exact.append(dict(frame=name, input_sha256=digest(output / f'references/{name}/input.f32'), outputs=hashes))
    controls = []
    with tempfile.TemporaryDirectory(prefix='decoder controls ', dir=output) as temporary:
        tmp = Path(temporary)
        bad = tmp / 'bad_entry'
        shutil.copytree(output / 'references/a/entry39', bad)
        values = np.fromfile(bad / 'skip.f32', '<f4')
        values[0] = 123456
        values.tofile(bad / 'skip.f32')
        altered = command.copy()
        altered[5], altered[9] = str(tmp), str(bad)
        failure = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (output / 'rejected_skip.log').write_text(failure.stdout+failure.stderr)
        if failure.returncode != 1 or 'bad_entry: skip: FAIL' not in failure.stdout:
            raise AssertionError('altered encoder skip expectation was accepted')
        controls.append('altered encoder30 skip expectation rejected')
        bad_inverse = inverse.copy()
        bad_inverse[0] = bad_inverse[1]
        bad_inverse.tofile(bad / 'inverse.i32')
        failure = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True, timeout=120)
        if failure.returncode != 2 or 'invalid bridge map' not in failure.stderr:
            raise AssertionError('non-bijective inverse map accepted')
        controls.append('non-bijective inverse rejected before dispatch')
        bad_block = tmp / 'bad_block41'
        shutil.copytree(output / f'references/a/decoder/block41-{width}x8-s3', bad_block)
        values = np.fromfile(bad_block / 'input.f32', '<f4')
        values[0] = 123456
        values.tofile(bad_block / 'input.f32')
        lines = (output / 'decoder_blocks.txt').read_text(encoding='mbcs').splitlines()
        lines[1] = '3\t'+str(bad_block)
        (tmp / 'decoder.txt').write_bytes(('\n'.join(lines)+'\n').encode('mbcs'))
        altered[9], altered[10] = command[9], str(tmp / 'decoder.txt')
        failure = subprocess.run(altered, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (output / 'rejected_decoder_handoff.log').write_text(failure.stdout+failure.stderr)
        if (failure.returncode != 1 or 'bad_block41: input: FAIL' not in failure.stdout or
                (tmp / 'frames_metrics.json').exists() or list(tmp.glob('frame_*'))):
            raise AssertionError('altered decoder input was accepted or output published on failure')
        controls.append('altered block41 input rejected before publishing outputs')
    report = dict(extent=[width, height, 512], sequence=sequence, metrics=m, exact_outputs=exact,
                  decoder39_inputs_preserved=True,
                  source_report_sha256=digest(source / 'report.json'), c512_sources=baseline['sources'],
                  decoder_sources=decoder_sources, decoder39_tensor_sha256=digest(tensor_path),
                  declared_inputs=declared, input_provenance=prior.get('input_provenance'),
                  inverse_map_sha256=digest(output / 'references/a/entry39/inverse.i32'), negative_controls=controls,
                  original_kernel_executed=False, original_bridge_validated=False, production_wiring=False,
                  scope='resident encoder23-30 -> ViT31-38 -> inverse/decoder39 with same-frame skip -> decoder40-47')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(extent=report['extent'], exact_arrays=50, metrics=m, negative_controls=controls), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('encoder_vit', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--head-fixture', type=Path)
    args = parser.parse_args()
    run(args.encoder_vit, args.output_root, args.head_fixture)
