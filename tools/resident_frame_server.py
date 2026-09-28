"""Client for build/resident_frame_server.exe, the persistent resident candidate.

The server loads weights once from the fixture set of a validated 26-argument
split512_frames_test invocation, then runs encoder5-30, ViT31-38,
decoder39-69 and head70 per request with device-only handoffs. Requests supply
the C64 boundary4 input, block4/preblock0 skips and linear RGB; responses give
native- and public-gain enhanced RGB. This is candidate arithmetic, not an
original NVIDIA kernel.
"""
from pathlib import Path
import json
import subprocess
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build/resident_frame_server.exe'
INPUTS = ('c64_input', 'skip4', 'skip0', 'color')


def server_arguments(harness_command):
    """Map a validated 26-argument harness command to the server's arguments."""
    command = [str(v) for v in harness_command]
    if len(command) != 26 or Path(command[0]).name != 'split512_frames_test.exe':
        raise ValueError('expected the full 26-argument resident harness command')
    return [str(EXE), *command[1:4], *command[6:26]]


def extents(width):
    n = width*8*512
    return dict(c64_input=(64, width*8, 64), skip4=(128, width*16, 32),
                skip0=(256, width*32, 32), color=(256, width*32, 3), n=n)


class ResidentServer:
    def __init__(self, harness_command, timeout=120):
        self.arguments = server_arguments(harness_command)
        self.width = int(self.arguments[2])
        self.timeout = timeout
        self.process = subprocess.Popen(self.arguments, cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        line = self._line()
        if not line.startswith('READY '):
            error = self.process.stderr.read() if self.process.poll() is not None else ''
            self.close()
            raise RuntimeError(f'resident server did not start: {line!r} {error}')
        self.ready = line

    def _line(self):
        result = {}

        def read():
            result['line'] = self.process.stdout.readline()
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(self.timeout)
        if reader.is_alive():
            self.close()
            raise TimeoutError('resident server did not answer in time')
        line = result.get('line', '')
        if not line:
            raise RuntimeError('resident server exited')
        return line.strip()

    def run(self, request_dir):
        """Execute one request directory; returns the device interval in ms."""
        request_dir = Path(request_dir).resolve()
        if any(c in str(request_dir) for c in '\r\n'):
            raise ValueError('request path contains a line separator')
        for name in ('rgb_native', 'rgb_public'):
            (request_dir / f'{name}.f32').unlink(missing_ok=True)
        self.process.stdin.write(f'{request_dir}\n')
        self.process.stdin.flush()
        line = self._line()
        if not line.startswith('OK '):
            raise RuntimeError(f'resident server rejected {request_dir}: {line}')
        return float(line[3:])

    def write_request(self, request_dir, arrays):
        shapes = extents(self.width)
        request_dir = Path(request_dir)
        request_dir.mkdir(parents=True, exist_ok=True)
        for name in INPUTS:
            array = np.asarray(arrays[name], '<f4')
            if array.shape != shapes[name] or not np.isfinite(array).all():
                raise ValueError(f'resident input {name} has shape {array.shape}, expected {shapes[name]}')
            array.tofile(request_dir / f'{name}.f32')

    def read_outputs(self, request_dir):
        shape = extents(self.width)['color']
        return {name: np.fromfile(Path(request_dir) / f'{name}.f32', '<f4').reshape(shape)
                for name in ('rgb_native', 'rgb_public')}

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('QUIT\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_command(path):
    return json.loads(Path(path).read_text())
