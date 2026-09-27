"""Exercise worker IPC, repeated GPU jobs and fail-closed behavior on synthetic data."""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import tempfile
import time
from unittest import mock

import gpu_test_runner as runner


def run():
    listed = subprocess.check_output([str(runner.WORKER), '--list'], text=True)
    assert set(listed.splitlines()) == runner.SUPPORTED
    checks = []
    with tempfile.TemporaryDirectory(prefix='gpu worker spaces ', dir=runner.ROOT / 'build') as tmp:
        folder = Path(tmp)
        values = [float(i % 17) / 16 for i in range(8 * 8 * 64)]
        data = struct.pack(f'<{len(values)}f', *values)
        (folder / 'input.f32').write_bytes(data)
        (folder / 'windows.f32').write_bytes(data)
        args = [str(runner.ROOT / 'build/spatial64_test.exe'), str(folder), '8', '8', '0']
        timings = {}
        for mode in ('process', 'worker'):
            start = time.perf_counter()
            with mock.patch.dict(os.environ, DLSS5_GPU_TEST_MODE=mode):
                for _ in range(6):
                    result = runner.run(args, cwd=runner.ROOT)
                    assert result.returncode == 0
                    assert (folder / 'windows_device.f32').read_bytes() == data
            timings[mode] = time.perf_counter() - start
            if mode == 'worker':
                worker = runner._worker
                assert worker.jobs == 6 and worker.process.poll() is None
        runner.close_worker()
        assert worker.closed and worker.process.returncode == 0
        checks += ['six repeated jobs in one context', 'path containing spaces',
                   'standalone/worker byte equality', 'clean EOF shutdown']

        missing_worker = folder / 'missing-worker.exe'
        with mock.patch.object(runner, 'WORKER', missing_worker):
            with mock.patch.dict(os.environ, DLSS5_GPU_TEST_MODE='auto'):
                assert runner.run(args, cwd=runner.ROOT).returncode == 0
            with mock.patch.dict(os.environ, DLSS5_GPU_TEST_MODE='worker'):
                try:
                    runner.run(args, cwd=runner.ROOT)
                except FileNotFoundError:
                    checks.append('missing worker: auto fallback, forced worker rejects')
                else:
                    raise AssertionError('forced worker accepted missing executable')

        # Corrupt the expected gather only; the GPU still receives valid input.
        (folder / 'windows.f32').write_bytes(struct.pack('<f', 1000) + data[4:])
        with mock.patch.dict(os.environ, DLSS5_GPU_TEST_MODE='worker'), \
                mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('unexpected retry')):
            try:
                runner.run(args, cwd=runner.ROOT)
            except subprocess.CalledProcessError as exc:
                assert exc.returncode == 1 and 'FAIL' in exc.output
                assert runner._worker.closed and runner._worker.process.poll() is not None
                checks.append('comparison failure retires context without retry')
            else:
                raise AssertionError('corrupt expected tensor accepted')
        runner.close_worker()

        worker = runner.Worker()
        try:
            worker.request(['spatial64_test', str(folder / 'absent'), '8', '8', '0'])
        except RuntimeError as exc:
            assert 'exited before completing' in str(exc)
            assert worker.closed and worker.process.poll() is not None
            checks.append('fatal fixture I/O produces EOF and retires context')
        else:
            raise AssertionError('missing fixture accepted')

        worker = runner.Worker()
        try:
            worker._line(time.monotonic() + .02)
        except TimeoutError:
            checks.append('idle response wait is bounded')
        else:
            raise AssertionError('idle response did not time out')
        finally:
            worker.close(force=True)

        worker = runner.Worker()
        with mock.patch.object(worker, '_line', side_effect=TimeoutError('injected response timeout')):
            try:
                worker.request(['spatial64_test', str(folder), '8', '8', '0'])
            except TimeoutError:
                assert worker.closed and worker.process.poll() is not None
                checks.append('request timeout retires context')
            else:
                raise AssertionError('timed-out request accepted')

        for packet in (b'\x01', struct.pack('<II', 1, 9),
                       struct.pack('<III', 1, 1, 32769),
                       struct.pack('<III', 1, 1, 2) + b'x\0'):
            result = subprocess.run([str(runner.WORKER), '--worker'], input=packet,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            assert result.returncode == 2 and b'D5GPU_DONE' not in result.stdout
        checks.append('truncated, excessive and embedded-NUL requests rejected')

        worker = runner.Worker()
        result, _ = worker.request(['unknown-test'])
        assert result == 2 and worker.closed
        checks.append('unknown job rejected and context retired')

        report = dict(checks=checks, jobs_per_mode=6, seconds=timings,
                      output_sha256=hashlib.sha256(data).hexdigest(),
                      original_kernel_executed=False)
    return report


if __name__ == '__main__':
    # Negative controls intentionally print FAIL; keep those in the diagnostic
    # log, so the summary cannot be mistaken for a failed positive test.
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            report = run()
    finally:
        runner.close_worker()
        (runner.ROOT / 'build/gpu_test_runner_check.log').write_text(output.getvalue())
    (runner.ROOT / 'build/gpu_test_runner_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
