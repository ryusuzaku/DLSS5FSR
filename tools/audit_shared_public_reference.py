"""Compare a shared reference with saved independent-branch outputs.

The baseline must come from the same prepared frame. Negative controls check
that stale input metadata and damaged graph/tensor files cannot be consumed.
The audit never changes the reference or baseline files.
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from shared_public_reference import (
    BRANCH, digest, input_identity, output_groups, run_reference, tensor_name)


def check_rejections(prepared_file, rgb, reference_dir, scratch_dir):
    manifest = json.loads((reference_dir / 'manifest.json').read_text())
    node = next(iter(manifest['outputs']))
    passed = []
    with tempfile.TemporaryDirectory(prefix='reference-rejections-', dir=scratch_dir) as tmp:
        case = Path(tmp)

        def reject(expected):
            try:
                run_reference(prepared_file, rgb, {'probe': node}, case / BRANCH, case)
            except ValueError as exc:
                if expected not in str(exc):
                    raise AssertionError(f'unexpected rejection: {exc}') from exc
                passed.append(expected)
            else:
                raise AssertionError(f'accepted invalid shared reference: {expected}')

        stale = dict(manifest, color_linear_sha256='0' * 64)
        (case / 'manifest.json').write_text(json.dumps(stale))
        reject('different input or model')
        (case / 'manifest.json').write_text(json.dumps(manifest))
        (case / BRANCH).write_bytes(b'damaged graph')
        reject('graph hash differs')
        (case / BRANCH).unlink()
        try:
            os.link(reference_dir / BRANCH, case / BRANCH)
        except OSError:
            shutil.copyfile(reference_dir / BRANCH, case / BRANCH)
        missing = dict(manifest, outputs={})
        (case / 'manifest.json').write_text(json.dumps(missing))
        reject('missing output')
        (case / 'manifest.json').write_text(json.dumps(manifest))
        (case / tensor_name(node)).write_bytes(b'damaged tensor')
        reject('tensor hash differs')
    return passed


def run(prepared_dir, reference_dir, baseline_root, output_report):
    prepared_dir, reference_dir = Path(prepared_dir), Path(reference_dir)
    baseline_root, output_report = Path(baseline_root), Path(output_report)
    prepared_file = prepared_dir / 'manifest.json'
    rgb = np.fromfile(prepared_dir / 'color_linear.f32', '<f4').reshape(256, 256, 3)
    identity = input_identity(prepared_file, rgb)
    checks = []
    for stage, nodes in output_groups().items():
        head = stage == 'head_inputs'
        folder = baseline_root / stage
        report = json.loads((folder / ('manifest.json' if head else 'report.json')).read_text())
        expected_identity = dict(identity)
        if head:
            expected_identity['model_sha256'] = expected_identity.pop('source_model_sha256')
        if any(report.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f'{stage} baseline input/model ancestry differs')
        if 'shared_public_reference_manifest_sha256' in report:
            raise ValueError(f'{stage} baseline must use independent public branches')
        arrays, _, _ = run_reference(prepared_file, rgb, nodes, reference_dir / BRANCH,
                                     reference_dir)
        for (name, _), array in zip(nodes.items(), arrays):
            array = array[0].astype('<f4')
            if head:
                suffix = 'rgb' if name in ('enhanced', 'final') else 'peer'
                if suffix == 'rgb':
                    array = array.transpose(1, 2, 0)
                path = folder / f'{name}_{suffix}.f32'
                expected_hash = report[f'{name}_{suffix}_sha256']
            else:
                path = folder / 'public_boundary' / f'{name}_peer.f32'
                expected_hash = report['public_boundary_sha256'][name]
            if digest(path) != expected_hash:
                raise ValueError(f'{stage}/{name} baseline tensor hash differs')
            raw = path.read_bytes()
            baseline = np.frombuffer(raw, '<f4').reshape(array.shape)
            checks.append(dict(stage=stage, name=name, values=array.size,
                               byte_exact=array.tobytes() == raw,
                               differing=int(np.count_nonzero(array != baseline)),
                               max_abs=float(np.max(np.abs(array - baseline)))))
    output_report.parent.mkdir(parents=True, exist_ok=True)
    rejected = check_rejections(prepared_file, rgb, reference_dir, output_report.parent)
    result = dict(**identity, shared_manifest_sha256=digest(reference_dir / 'manifest.json'),
                  all_exact=all(check['byte_exact'] for check in checks),
                  negative_controls=rejected, comparisons=checks)
    output_report.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_exact']:
        raise AssertionError(f'shared reference differs; see {output_report}')
    print(f'{len(checks)}/{len(checks)} public boundaries byte-exact; '
          f'{len(rejected)} invalid-reference controls rejected')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared_dir', type=Path)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared_dir, args.reference_dir, args.baseline_root, args.output_report)
