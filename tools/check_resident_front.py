"""Check the resident C32 front end (stem, block0-4, block4 downsample).

Builds independent scalar references, runs build/front_resident_test.exe with
all stage checks and repeated byte-exact executions, and reports the candidate
outputs against the public graph's producers they replace. Candidate FP8
rounding between blocks differs from the public graph's FP16 clamp by design.
"""
from pathlib import Path
import argparse
import json
import subprocess

import numpy as np

import front_end_reference as R
from check_split512_resident import ROOT, digest


def metrics(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return dict(mae=float(np.abs(a-b).mean()), max_abs=float(np.abs(a-b).max()),
                correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))


def run_case(rgb, weights, folder, repeats=5):
    height, width, _ = rgb.shape
    result = R.run(rgb, weights, folder)
    command = [str(ROOT/'build/front_resident_test.exe'), str(folder), str(width), str(height), str(repeats)]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    (folder/'device.log').write_text(done.stdout+done.stderr)
    summary = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
    if done.returncode or 'FAIL' in done.stdout or summary['stage_comparisons'] != 78:
        raise RuntimeError(f'front end failed: {folder}/device.log\n{done.stdout[-2000:]}')
    return result, summary


def run(output, captured_prepared):
    output = Path(output).resolve()
    weights = R.load_weights()
    report = dict(tensors={str(b): digest(R.tensor(b)[1]) for b in R.TENSORS}, shifts=R.SHIFTS,
                  stem_offset=hex(R.STEM_AT), downsample_offset=hex(R.DOWN_AT))
    prepared = Path(captured_prepared).resolve()
    rgb = np.fromfile(prepared/'color_linear.f32', '<f4').reshape(256, 256, 3)
    result, summary = run_case(rgb, weights, output/'captured')
    from resident_public_inputs import PublicInputs
    public, info = PublicInputs(output/'public_cache').run(prepared)
    report['captured'] = dict(prepared_manifest_sha256=digest(prepared/'manifest.json'), device=summary,
                              outputs={k: digest(output/'captured'/f'{k}.f32') for k in ('c64_input', 'skip4', 'skip0')},
                              candidate_vs_public={k: metrics(result[k], public[k]) for k in ('c64_input', 'skip4', 'skip0')})
    rng = np.random.default_rng(3107)
    wide = rng.uniform(0, 1, (256, 1024, 3)).astype(np.float32)
    _, summary = run_case(wide, weights, output/'synthetic1024')
    report['synthetic1024'] = dict(kind='uniform RGB seed3107; public 256x256 noise tiled', device=summary)
    report.update(original_kernel_executed=False,
                  map_status='C32 bodies public QMMA; stem and downsample layouts exact against public coefficients')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captured_prepared', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    run(a.output_root, a.captured_prepared)
