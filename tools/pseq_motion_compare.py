"""Motion-aligned temporal edits on stable-input surfaces in a saved capture.

Replay files: PREFIX_NNN_native.bin has three little-endian uint32 values
(width, height, row pitch in bytes), followed by RGBA16F rows. Matching
PREFIX_NNN_in.rgb files (width/height header + RGB8) verify input colour space.
Vectors are current-to-previous UV offsets. Results use current/previous HUD
exclusions, in-bounds samples, depth agreement and shared input-error masks.
This diagnoses stability, not scene correctness or perceived quality.

Example: python tools/pseq_motion_compare.py CAPTURE_DIR
  --replay dynamic=PREFIX --replay locked=PREFIX --json report.json
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from seq_replay import bilinear, read_frame
from pseq_trail_compare import read_rgb8


def read_native(path):
    raw = Path(path).read_bytes()
    if len(raw) < 12:
        raise ValueError(f'truncated native header: {path}')
    w, h, pitch = struct.unpack_from('<3I', raw)
    if not w or not h or pitch < w*8 or pitch % 2 or len(raw) != 12+pitch*h:
        raise ValueError(f'bad native extent/payload: {path}')
    out = np.frombuffer(raw, np.float16, offset=12).reshape(h, pitch//2)[:, :w*4]
    out = out.reshape(h, w, 4)[..., :3].astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f'nonfinite native RGB: {path}')
    return out


def correspondence(current, previous, motion, depth, previous_depth, region, depth_tol=0.05):
    """Find previous pixel centres; invalid vectors never reach the sampler."""
    h, w = current.shape[:2]
    if previous.shape != current.shape or motion.shape != (h, w, 2) or depth.shape != (h, w) or previous_depth.shape != (h, w) or region.shape != (h, w):
        raise ValueError('motion comparison extents differ')
    y, x = np.indices((h, w), dtype=np.float32)
    finite = np.isfinite(motion).all(axis=2)
    with np.errstate(over='ignore', invalid='ignore'):
        mx, my = motion[..., 0]*w, motion[..., 1]*h
    finite &= np.isfinite(mx) & np.isfinite(my)
    mx, my = np.where(finite, mx, 0), np.where(finite, my, 0)
    px, py = x+0.5+mx, y+0.5+my
    inside = (px >= 0.5) & (px <= w-0.5) & (py >= 0.5) & (py <= h-0.5)
    px, py = np.where(finite & inside, px, 0.5), np.where(finite & inside, py, 0.5)
    source = np.concatenate((previous, previous_depth[..., None], region[..., None].astype(np.float32)), axis=2)
    warped = bilinear(source, px, py)
    z = warped[..., 3]
    valid = finite & inside & region & (warped[..., 4] >= 1-1e-6)
    valid &= np.isfinite(depth) & np.isfinite(z) & (depth >= 0) & (z >= 0)
    valid &= np.abs(z-depth) <= depth_tol*np.maximum(np.abs(z), np.abs(depth))+1e-6
    input_error = np.abs(current-warped[..., :3]).mean(axis=2)*255
    return {'valid': valid, 'speed': np.hypot(mx, my), 'input_error': input_error,
            'previous': warped[..., :3], 'px': px, 'py': py}


def compare(directory, replays, count=60, tolerances=(1, 2, 4), depth_tol=0.05,
            max_motion=32, exclude=(), first_pair=2, min_pixels=64):
    files = sorted(Path(directory).glob('*.bin'))[:count]
    if len(files) <= first_pair:
        raise ValueError('not enough capture frames after warm-up')
    bins = ('still', 'moving', 'fast')  # fast is a subset of moving
    def tolerance_key(t):
        value = float(t)
        return str(int(value)) if value.is_integer() else repr(value)
    totals = {tolerance_key(t): {b: {'pixels': 0, 'frame_pairs': 0, 'input2': 0.,
                          'modes': {name: np.zeros(4, np.float64) for name in replays}}
                      for b in bins} for t in tolerances}
    strength = {name: 0. for name in replays}
    previous = None
    previous_outputs = {}
    warm_frames = 0
    for n, path in enumerate(files):
        pic, depth, mv = read_frame(str(path))
        if depth is None or mv is None:
            raise ValueError(f'motion comparison needs motion and depth: {path}')
        pic = np.clip(pic, 0, 1)
        h, w = pic.shape[:2]
        outputs = {name: read_native(f'{prefix}_{n:03d}_native.bin') for name, prefix in replays.items()}
        expected = (pic*255+0.5).astype(np.uint8)
        for name, prefix in replays.items():
            if outputs[name].shape != pic.shape:
                raise ValueError(f'native/capture extent mismatch: {name}, frame {n}')
            if not np.array_equal(read_rgb8(f'{prefix}_{n:03d}_in.rgb'), expected):
                raise ValueError(f'replay input colour differs from capture: {name}, frame {n}')
        if n >= first_pair-1:
            warm_frames += 1
            for name, out in outputs.items():
                strength[name] += float(np.abs(out-pic).mean())*255
        if previous is None:
            region = np.ones((h, w), bool)
            for rect in exclude:
                x0, y0, x1, y1 = rect
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    raise ValueError(f'exclusion outside frame: {rect}')
                region[y0:y1, x0:x1] = False
        if n >= first_pair:
            corr = correspondence(pic, previous[0], mv, depth, previous[1], region, depth_tol)
            speed = corr['speed']
            eligible = corr['valid'] & (speed <= max_motion) & (corr['input_error'] <= max(tolerances))
            if eligible.any():
                input_delta = (pic[eligible]-corr['previous'][eligible])*255
                deltas = {}
                for name, out in outputs.items():
                    old = bilinear(previous_outputs[name], corr['px'][eligible], corr['py'][eligible])
                    result_delta = (out[eligible]-old)*255
                    deltas[name] = (result_delta-input_delta, result_delta)
                groups = {'still': speed[eligible] < 0.5,
                          'moving': speed[eligible] >= 1,
                          'fast': speed[eligible] >= 8}
                for tolerance in tolerances:
                    stable = corr['input_error'][eligible] <= tolerance
                    for group, in_bin in groups.items():
                        mask = stable & in_bin
                        pixels = int(mask.sum())
                        if pixels < min_pixels:
                            continue
                        acc = totals[tolerance_key(tolerance)][group]
                        acc['pixels'] += pixels
                        acc['frame_pairs'] += 1
                        acc['input2'] += float(np.square(input_delta[mask], dtype=np.float64).sum())
                        for name, (edit, result) in deltas.items():
                            e, r = edit[mask], result[mask]
                            acc['modes'][name] += [np.abs(e).sum(dtype=np.float64), np.square(e, dtype=np.float64).sum(),
                                                   np.abs(r).sum(dtype=np.float64), np.square(r, dtype=np.float64).sum()]
        previous = (pic, depth)
        previous_outputs = outputs
    strength = {name: value/warm_frames for name, value in strength.items()}
    report = {'frames': len(files), 'first_pair': first_pair, 'depth_tolerance': depth_tol,
              'max_motion_pixels': max_motion, 'min_pixels_per_pair': min_pixels,
              'excluded_rectangles': list(exclude), 'mean_abs_edit': strength, 'tolerances': {}}
    for tolerance, groups in totals.items():
        report['tolerances'][tolerance] = {}
        for group, acc in groups.items():
            channels = acc['pixels']*3
            scores = {}
            if channels:
                for name, (ea, e2, ra, r2) in acc['modes'].items():
                    rms = float(np.sqrt(e2/channels))
                    scores[name] = {'edit_mae': float(ea/channels), 'edit_rms': rms,
                                    'result_mae': float(ra/channels), 'result_rms': float(np.sqrt(r2/channels)),
                                    'edit_rms_per_mean_edit': rms/strength[name] if strength[name] else None}
            report['tolerances'][tolerance][group] = {'frame_pairs': acc['frame_pairs'], 'pixels': acc['pixels'],
                'input_rms': float(np.sqrt(acc['input2']/channels)) if channels else None, 'scores': scores}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--replay', action='append', required=True, metavar='LABEL=PREFIX')
    ap.add_argument('--count', type=int, default=60)
    ap.add_argument('--input-tol', default='1,2,4', help='mean RGB input mismatch limits, in 1/255 units')
    ap.add_argument('--depth-tol', type=float, default=0.05)
    ap.add_argument('--max-motion', type=float, default=32)
    ap.add_argument('--first-pair', type=int, default=2)
    ap.add_argument('--min-pixels', type=int, default=64)
    ap.add_argument('--exclude-rect', action='append', default=[], metavar='X0,Y0,X1,Y1')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    replays = {}
    for spec in args.replay:
        name, sep, prefix = spec.partition('=')
        if not name or not sep or not prefix or name in replays:
            ap.error('replay needs unique LABEL=PREFIX')
        replays[name] = prefix
    try:
        tolerances = tuple(float(x) for x in args.input_tol.split(','))
        exclude = [tuple(int(x) for x in rect.split(',')) for rect in args.exclude_rect]
    except ValueError:
        ap.error('invalid input tolerances or exclusion rectangle')
    if not tolerances or any(not np.isfinite(t) or t <= 0 for t in tolerances) or len(set(tolerances)) != len(tolerances):
        ap.error('input tolerances must be distinct finite positive values')
    if args.first_pair < 1 or args.count <= args.first_pair or args.min_pixels < 1 or not np.isfinite(args.depth_tol) or not 0 < args.depth_tol <= 1 or not np.isfinite(args.max_motion) or args.max_motion < 1:
        ap.error('invalid warm-up, count, minimum pixels, depth tolerance or maximum motion')
    if any(len(rect) != 4 for rect in exclude):
        ap.error('exclude-rect needs X0,Y0,X1,Y1')
    report = compare(args.directory, replays, args.count, tolerances, args.depth_tol,
                     args.max_motion, exclude, args.first_pair, args.min_pixels)
    text = json.dumps(report, indent=2)+'\n'
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
