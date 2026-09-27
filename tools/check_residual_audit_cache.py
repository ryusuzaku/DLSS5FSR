"""Check PTX audit reuse and rejection after same-size/same-mtime source edits."""

from pathlib import Path
import contextlib
import io
import json
import os
import tempfile
import time
from unittest import mock

import audit_c256_residual_ptx as audit


def run():
    build = audit.ROOT / 'build'
    build.mkdir(exist_ok=True)
    original = audit.PTX.read_bytes()
    marker = f'.visible .entry {audit.ENTRY}('.encode()
    changed = original.replace(marker, marker.replace(b'upsample', b'upsamplx'), 1)
    if changed == original or len(changed) != len(original):
        raise AssertionError('could not make a same-size invalid PTX control')
    with tempfile.TemporaryDirectory(prefix='residual-audit-cache-', dir=build) as tmp:
        path = Path(tmp) / 'source.ptx'
        path.write_bytes(original)
        stat = path.stat()
        with mock.patch.multiple(audit, PTX=path, OUT=Path(tmp) / 'report.json',
                                 _cached_signature=None, _cached_report=None), \
                mock.patch.object(audit, 'run', wraps=audit.run) as full_audit, \
                contextlib.redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            first = audit.checked()
            first_seconds = time.perf_counter() - start
            start = time.perf_counter()
            second = audit.checked()
            reuse_seconds = time.perf_counter() - start
            assert first == second and full_audit.call_count == 1
            second['measured_width_cross_checks'][0]['same_relative_addresses'] = False
            assert audit.checked() == first and full_audit.call_count == 1

            path.write_bytes(changed)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert path.stat().st_size == stat.st_size
            assert path.stat().st_mtime_ns == stat.st_mtime_ns
            try:
                audit.checked()
            except ValueError:
                pass
            else:
                raise AssertionError('cached audit accepted changed PTX')
            assert full_audit.call_count == 2

            path.write_bytes(original)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            assert audit.checked() == first and full_audit.call_count == 2
            assert audit.run() == first and full_audit.call_count == 3
    report = dict(unchanged_source_reused=True, returned_report_isolated=True,
                  same_size_mtime_source_change_rejected=True,
                  restored_source_reused=True, standalone_audit_recomputed=True,
                  first_seconds=first_seconds, reuse_seconds=reuse_seconds)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
