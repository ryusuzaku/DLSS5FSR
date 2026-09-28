"""Check the persistent resident server against a validated encoder5 front run.

Replays the A/B/zero/B/A frames of a check_resident_encoder22.py
--first-block 5 output through one server process and requires byte-exact
native- and public-gain RGB, plus rejection of a malformed request.
"""
from pathlib import Path
import argparse
import json
import os
import shutil

import numpy as np

from resident_frame_server import INPUTS, ResidentServer
from check_split512_resident import digest


def link(source, target):
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    report = json.loads((source / 'report.json').read_text())
    if report['first_block'] != 5 or report['sequence'] != ['a', 'b', 'zero', 'b', 'a']:
        raise ValueError('requires a validated encoder5 front report')
    mode = source / 'mode5'
    command = json.loads((mode / 'native_command.json').read_text())
    if report['modes']['5']['exact_arrays'] != 200:
        raise ValueError('encoder5 front report is incomplete')
    output.mkdir(parents=True, exist_ok=True)
    timings, exact = [], []
    with ResidentServer(command) as server:
        for index, name in enumerate(report['sequence']):
            frame = source / 'frames' / name
            request = output / f'request_{index}'
            request.mkdir(exist_ok=True)
            for key in INPUTS:
                link(frame / f'{key}.f32', request / f'{key}.f32')
            timings.append(server.run(request))
            record = dict(frame=name)
            for key in ('rgb_native', 'rgb_public'):
                if (request / f'{key}.f32').read_bytes() != (frame / f'{key}.f32').read_bytes():
                    raise AssertionError(f'server {key} differs for frame {index} ({name})')
                record[key] = digest(request / f'{key}.f32')
            exact.append(record)
            print(f'frame {index} ({name}): RGB exact in {timings[-1]:.2f} ms', flush=True)
        bad = output / 'request_bad'
        bad.mkdir(exist_ok=True)
        for key in INPUTS:
            link(source / 'frames/a' / f'{key}.f32', bad / f'{key}.f32')
        values = np.fromfile(bad / 'skip0.f32', '<f4')[:-1]
        (bad / 'skip0.f32').unlink()
        values.tofile(bad / 'skip0.f32')
        try:
            server.run(bad)
        except RuntimeError as error:
            if 'skip0' not in str(error):
                raise
        else:
            raise AssertionError('truncated skip0 request was accepted')
        # The server must keep serving after a rejected request.
        recovery = server.run(output / 'request_0')
        if (output / 'request_0/rgb_public.f32').read_bytes() != (source / 'frames/a/rgb_public.f32').read_bytes():
            raise AssertionError('server did not recover after a rejected request')
        ready = server.ready
    result = dict(source_report_sha256=digest(source / 'report.json'), ready=ready,
                  requests=len(exact), exact_outputs=exact, frame_ms=timings,
                  recovery_ms=recovery, rejected=['truncated skip0 request'],
                  original_kernel_executed=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'exact_outputs'}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('encoder5_front', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    run(a.encoder5_front, a.output_root)
