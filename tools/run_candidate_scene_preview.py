#!/usr/bin/env python3
"""Slow, opt-in game-scene preview via repeated capture and offline AMD candidate.

Run this with the project's ONNX venv while the shim has DebugView=2,
CandidatePreviewReload=1, CandidateInputCaptureTrigger=1 and
CandidateInputCaptureRepeat=1. The preview updates only after an entire
offline pass; it is not real-time inference or an original-kernel result.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time

from export_candidate_preview import run as export_preview
from prepare_candidate_input import run as prepare_input
from run_candidate_frame_from_capture import run as run_frame
from validate_candidate_input_pair import run as validate_input_pair


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_status(path, **fields):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                                    **fields), indent=2) + '\n')
    temp.replace(path)


def capture_next(capture_path, timeout, poll, gpu_input_path=None):
    trigger = Path(str(capture_path) + '.go')
    # A hard console shutdown can leave the trigger behind. It still means
    # "capture the next frame", so a restarted sidecar may adopt it. Never
    # remove a trigger we did not create; the game will consume it on capture.
    created = not trigger.exists()
    before = capture_path.stat().st_mtime_ns if capture_path.exists() else None
    gpu_before = (gpu_input_path.stat().st_mtime_ns
                  if gpu_input_path and gpu_input_path.exists() else None)
    if created:
        trigger.write_text('capture next completed staged proxy frame\n')
    else:
        print(f'adopting pending capture trigger: {trigger}', flush=True)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if not trigger.exists() and capture_path.exists():
                after = capture_path.stat().st_mtime_ns
                if before is None or after != before:
                    if gpu_input_path:
                        if not gpu_input_path.exists():
                            time.sleep(poll)
                            continue
                        gpu_after = gpu_input_path.stat().st_mtime_ns
                        if gpu_before is not None and gpu_after == gpu_before:
                            time.sleep(poll)
                            continue
                    return
            time.sleep(poll)
        raise TimeoutError(f'no new game capture within {timeout:g}s at {capture_path}')
    finally:
        if created:
            trigger.unlink(missing_ok=True)


def run(capture_path, output_root, preview_path, max_updates=0,
        existing_capture=False, capture_timeout=120, poll=0.5,
        gpu_input_path=None, preview_gain='public_gain'):
    capture_path = Path(capture_path).resolve()
    output_root = Path(output_root).resolve()
    preview_path = Path(preview_path).resolve()
    gpu_input_path = Path(gpu_input_path).resolve() if gpu_input_path else None
    input_source = 'paired HIP GPU tensor' if gpu_input_path else 'CPU candidate'
    output_root.mkdir(parents=True, exist_ok=True)
    if not capture_path.parent.is_dir():
        raise ValueError(f'capture parent directory does not exist: {capture_path.parent}')
    if existing_capture and not capture_path.is_file():
        raise ValueError(f'existing capture does not exist: {capture_path}')
    if gpu_input_path and not gpu_input_path.parent.is_dir():
        raise ValueError(f'GPU input parent directory does not exist: {gpu_input_path.parent}')
    if existing_capture and gpu_input_path and not gpu_input_path.is_file():
        raise ValueError(f'existing GPU input does not exist: {gpu_input_path}')
    if existing_capture and max_updates not in (0, 1):
        raise ValueError('--existing-capture supports one update only')
    status_path = output_root / 'status.json'
    last_preview_sha = None
    last_capture_sha = None
    last_preview_completed = None
    preview_manifest = preview_path.with_suffix('.json')
    if preview_path.is_file() and preview_manifest.is_file():
        previous = json.loads(preview_manifest.read_text())
        if previous.get('preview_sha256') == digest(preview_path):
            last_preview_sha = previous['preview_sha256']
            last_preview_completed = datetime.fromtimestamp(
                preview_path.stat().st_mtime, timezone.utc).isoformat()
    iteration = 0
    while not max_updates or iteration < max_updates:
        iteration += 1
        try:
            write_status(status_path, state='waiting_for_capture', iteration=iteration,
                         capture_path=str(capture_path), preview_path=str(preview_path),
                         input_tensor_source=input_source,
                         last_preview_sha256=last_preview_sha,
                         last_preview_capture_sha256=last_capture_sha,
                         last_preview_completed_at_utc=last_preview_completed)
            if not existing_capture:
                capture_next(capture_path, capture_timeout, poll, gpu_input_path)
            # Snapshot the completed atomic capture before another trigger can
            # be armed. Reusing the work directory bounds unattended disk use.
            snapshot = output_root / 'capture.bin'
            if snapshot != capture_path:
                shutil.copyfile(capture_path, snapshot)
            capture_sha = digest(snapshot)
            print(f'capture {iteration}: {capture_sha}', flush=True)
            gpu_snapshot = None
            if gpu_input_path:
                gpu_snapshot = output_root / 'input_gpu.f32'
                if gpu_snapshot != gpu_input_path:
                    shutil.copyfile(gpu_input_path, gpu_snapshot)
            write_status(status_path, state='preparing', iteration=iteration,
                         capture_sha256=capture_sha,
                         input_tensor_source=input_source,
                         last_preview_sha256=last_preview_sha,
                         last_preview_completed_at_utc=last_preview_completed)
            if gpu_snapshot:
                pair = validate_input_pair(snapshot, gpu_snapshot,
                                           output_root / 'input_pair')
                if pair['raw_capture_sha256'] != capture_sha:
                    raise AssertionError('GPU input pair has a different source capture')
                prepared = output_root / 'input_pair' / 'gpu_prepared'
                prep = json.loads((prepared / 'manifest.json').read_text())
            else:
                prepared = output_root / 'prepared'
                prep = prepare_input(snapshot, prepared)
            if prep['source_capture_sha256'] != capture_sha:
                raise AssertionError('capture changed during preparation')
            work = output_root / 'candidate'
            write_status(status_path, state='running_candidate', iteration=iteration,
                         capture_sha256=capture_sha,
                         input_tensor_source=input_source,
                         last_preview_sha256=last_preview_sha,
                         last_preview_completed_at_utc=last_preview_completed)
            report = run_frame(prepared, work)
            if report['source_capture_sha256'] != capture_sha:
                raise AssertionError('candidate output has a different source capture')
            preview = export_preview(work / 'head70', work / 'head70_connected_gpu',
                                     preview_path, preview_gain)
            write_status(status_path, state='preview_ready', iteration=iteration,
                         capture_sha256=capture_sha,
                         preview_sha256=preview['preview_sha256'],
                         preview_path=str(preview_path),
                         preview_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                         stage_seconds=report['stage_seconds'],
                         input_tensor_source=input_source,
                         gpu_tensor_sha256=pair['gpu_tensor_sha256'] if gpu_snapshot else None,
                         preview_gain=preview_gain,
                         original_kernel_executed=False,
                         real_time_inference=False)
            last_preview_sha = preview['preview_sha256']
            last_capture_sha = capture_sha
            last_preview_completed = datetime.fromtimestamp(
                preview_path.stat().st_mtime, timezone.utc).isoformat()
            print(f'preview {iteration}: {preview_path}', flush=True)
            if existing_capture:
                break
        except KeyboardInterrupt:
            write_status(status_path, state='stopped', iteration=iteration,
                         last_preview_sha256=last_preview_sha,
                         last_preview_capture_sha256=last_capture_sha,
                         last_preview_completed_at_utc=last_preview_completed)
            print('sidecar stopped; last completed preview kept', flush=True)
            break
        except TimeoutError as exc:
            if last_preview_sha:
                write_status(status_path, state='capture_timeout', iteration=iteration,
                             last_preview_sha256=last_preview_sha,
                             last_preview_capture_sha256=last_capture_sha,
                             last_preview_completed_at_utc=last_preview_completed,
                             message=str(exc))
                print(f'capture timed out; last completed preview kept: {exc}', flush=True)
                break
            write_status(status_path, state='failed', iteration=iteration,
                         error=f'TimeoutError: {exc}', last_preview_path=None)
            raise
        except Exception as exc:
            write_status(status_path, state='failed', iteration=iteration,
                         error=f'{type(exc).__name__}: {exc}',
                         last_preview_sha256=last_preview_sha,
                         last_preview_path=str(preview_path) if preview_path.exists() else None)
            raise
    return json.loads(status_path.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    offload = Path.home() / 'DLSS5FSR-build-offload'
    parser.add_argument('--capture-path', type=Path,
                        default=offload / 'scene_preview' / 'game_capture.bin')
    parser.add_argument('--output-root', type=Path,
                        default=offload / 'scene_preview')
    parser.add_argument('--preview-path', type=Path,
                        default=offload / 'scene_preview' / 'candidate_preview.bin')
    parser.add_argument('--gpu-input-path', type=Path,
                        help='same-frame 256x256 HIP tensor written by CandidateInputGpuPath')
    parser.add_argument('--preview-gain', choices=('public_gain', 'native_gain'),
                        default='public_gain',
                        help='public graph gain 1 or upstream-reported native launch gain .03125')
    parser.add_argument('--max-updates', type=int, default=0,
                        help='0 loops until stopped; 1 runs one capture and one preview')
    parser.add_argument('--existing-capture', action='store_true',
                        help='process --capture-path once without arming the game trigger')
    parser.add_argument('--capture-timeout', type=float, default=120)
    parser.add_argument('--poll', type=float, default=0.5)
    args = parser.parse_args()
    if args.max_updates < 0 or args.capture_timeout <= 0 or args.poll <= 0:
        parser.error('max-updates must be nonnegative; timeout and poll must be positive')
    run(args.capture_path, args.output_root, args.preview_path,
        args.max_updates, args.existing_capture, args.capture_timeout, args.poll,
        args.gpu_input_path, args.preview_gain)
