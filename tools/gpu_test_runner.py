"""Run supported exact HIP tests in one worker context per Python stage.

Auto mode uses a built worker; process mode retains the standalone executable
path. Once a worker starts, any protocol, I/O, timeout or comparison failure
stops that job without retrying it through another execution path.
"""

from pathlib import Path
import atexit
import os
import queue
import struct
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'build/candidate_gpu_worker.exe'
SUPPORTED = frozenset((
    'vit_bridge_ptx_test',
    'vit_expand_chain_test',
    'decoder39_entry_test',
    'spatial32_peer_test',
    'c32_peer_body_test',
    'spatial32_peer_output_test',
    'upsample66_prefix_test',
    'head70_test',
    'spatial64_test',
    'c64_ffn_candidate_test',
    'c64_attention_candidate_test',
    'spatial64_output_test',
    'encoder64_downsample_test',
    'spatial128_test',
    'c128_ffn_candidate_test',
    'c128_attention_candidate_test',
    'spatial128_output_test',
    'encoder128_downsample_test',
    'spatial256_window_test',
    'c256_ffn_candidate_test',
    'c256_attention_candidate_test',
    'spatial256_output_test',
    'encoder256_downsample_test',
))
TIMEOUT_SECONDS = 120
_worker = None
_lock = threading.Lock()
_notified_fallback = False


class Worker:
    def __init__(self, executable=WORKER, cwd=ROOT, timeout=TIMEOUT_SECONDS):
        self.timeout = timeout
        self.sequence = 0
        self.jobs = 0
        self.closed = False
        self.lines = queue.Queue()
        self.process = subprocess.Popen(
            [str(executable), '--worker'], cwd=cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        try:
            deadline = time.monotonic() + timeout
            while True:
                line = self._line(deadline)
                if line == 'D5GPUWORKER 1':
                    break
                if line.startswith('D5GPUWORKER '):
                    raise RuntimeError('unsupported GPU worker protocol; rebuild the worker')
                print(line, flush=True)
        except BaseException:
            self.close(force=True)
            raise
        print(f'GPU worker started (pid {self.process.pid})', flush=True)

    def _read_output(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line.decode('utf-8', errors='replace').rstrip('\r\n'))
        finally:
            self.lines.put(None)

    def _line(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('GPU worker response timed out')
        try:
            line = self.lines.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError('GPU worker response timed out') from exc
        if line is None:
            raise RuntimeError(f'GPU worker exited before completing the request '
                               f'(exit code {self.process.poll()})')
        return line

    def request(self, args, *, capture_output=False):
        if self.closed:
            raise RuntimeError('GPU worker is closed')
        self.sequence += 1
        encoding = 'mbcs' if os.name == 'nt' else 'utf-8'
        encoded = [str(arg).encode(encoding) for arg in args]
        if (not 1 <= len(encoded) <= 8 or self.sequence > 0xffffffff or
                any(len(arg) > 32768 or b'\0' in arg for arg in encoded)):
            raise ValueError('invalid GPU worker arguments')
        packet = struct.pack('<II', self.sequence, len(encoded))
        packet += b''.join(struct.pack('<I', len(arg)) + arg for arg in encoded)
        output = []
        try:
            self.process.stdin.write(packet)
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                line = self._line(deadline)
                if line.startswith('D5GPU_DONE '):
                    fields = line.split()
                    if len(fields) != 3 or int(fields[1]) != self.sequence:
                        raise RuntimeError('GPU worker response does not match the request')
                    result = int(fields[2])
                    if result:
                        self.close(force=True)
                    else:
                        self.jobs += 1
                    return result, '\n'.join(output) + '\n'
                output.append(line)
                if not capture_output:
                    print(line, flush=True)
        except BaseException:
            self.close(force=True)
            raise

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if force and self.process.poll() is None:
            self.process.kill()
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        print(f'GPU worker stopped ({self.jobs} successful jobs)', flush=True)


def close_worker():
    global _worker
    with _lock:
        if _worker is not None:
            _worker.close()
            _worker = None


def run(args, *, cwd, check=True, capture_output=False):
    """Run an exact test; capture_output returns merged stdout/stderr as text."""
    global _worker, _notified_fallback
    args = [str(arg) for arg in args]
    mode = os.environ.get('DLSS5_GPU_TEST_MODE', 'auto')
    output_options = (dict(text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                      if capture_output else {})
    if mode not in ('auto', 'worker', 'process'):
        raise ValueError(f'unknown GPU test mode: {mode}')
    executable = Path(args[0]).resolve()
    supported = (executable.parent == ROOT / 'build' and
                 executable.stem in SUPPORTED and Path(cwd).resolve() == ROOT)
    if mode == 'process' or not supported:
        return subprocess.run(args, cwd=cwd, check=check, **output_options)
    if not WORKER.is_file():
        if mode == 'worker':
            raise FileNotFoundError('build the GPU worker with tools/build_candidate_gpu_worker.sh')
        if not _notified_fallback:
            print('GPU worker not built; using standalone GPU test processes', flush=True)
            _notified_fallback = True
        return subprocess.run(args, cwd=cwd, check=check, **output_options)
    with _lock:
        if _worker is None or _worker.closed:
            _worker = Worker()
        result, output = _worker.request([executable.stem, *args[1:]],
                                         capture_output=capture_output)
    if check and result:
        raise subprocess.CalledProcessError(result, args, output=output)
    return subprocess.CompletedProcess(args, result, stdout=output)


atexit.register(close_worker)
