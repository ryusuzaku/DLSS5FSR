"""Compare replay trails at several horizons and against one settled target.

Example:
  python tools/pseq_trail_compare.py CAPTURE_DIR --replay p2=BASE_PREFIX
    --replay candidate=TEST_PREFIX --reference p2 --json report.json

The fixed target is the reference replay's mean result 12..16 frames after
uncovering. Only pixels with still motion throughout and stable late input
(RGB span below 2/255) qualify. Every candidate uses the same target. This is
an offline stability comparison, not ground truth for the rendered scene.
Reported colour errors and edit strengths use 1/255 units. The horizon
scores reproduce pseq_trail.py's still-camera/motion masks. Low eligible
counts limit the evidence; an empty scores object means no eligible sample.
Edit strength always covers the full frame; exclusions affect comparisons.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from seq_replay import read_frame


def read_rgb8(path):
    data = Path(path).read_bytes()
    w, h = struct.unpack_from('<2I', data)
    if len(data) != 8 + w*h*3:
        raise ValueError(f'bad RGB replay size: {path}')
    return np.frombuffer(data, np.uint8, offset=8).reshape(h, w, 3)


def compare(directory, replays, reference, count=60, horizons=(1, 4, 8), settled=(12, 16), span=2.0, exclude=()):
    files = sorted(Path(directory).glob('*.bin'))[:count]
    if not files:
        raise ValueError('no capture frames')
    pics, mags, inputs = [], [], []
    outputs = {name: [] for name in replays}
    edits = {name: 0.0 for name in replays}
    for i, path in enumerate(files):
        pic, _, mv = read_frame(str(path))
        if mv is None:
            raise ValueError(f'no motion: {path}')
        h, w = mv.shape[:2]
        pics.append((np.clip(pic, 0, 1)*255).astype(np.float32))
        mags.append(np.hypot(mv[..., 0]*w, mv[..., 1]*h))
        shared = None
        for name, prefix in replays.items():
            inp = read_rgb8(f'{prefix}_{i:03d}_in.rgb')
            out = read_rgb8(f'{prefix}_{i:03d}_out.rgb')
            if inp.shape != (h, w, 3) or out.shape != inp.shape:
                raise ValueError(f'extent mismatch: {name}, frame {i}')
            if shared is None:
                shared = inp
            elif not np.array_equal(inp, shared):
                raise ValueError(f'replay inputs differ: {name}, frame {i}')
            outputs[name].append(out)
            edits[name] += float(np.abs(out.astype(np.float32)-inp).mean())/len(files)
        inputs.append(shared)
    medians = [float(np.median(m)) for m in mags]
    region = np.ones(mags[0].shape, bool)
    height, width = region.shape
    for rect in exclude:
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'exclusion rectangle outside frame: {rect}')
        region[y0:y1, x0:x1] = False

    def change(name, n):
        return outputs[name][n].astype(np.float32)-inputs[n].astype(np.float32)

    def result(name, n):
        return np.clip(pics[n]+change(name, n), 0, 255)

    def masks(n, later):
        if any(medians[k] > 0.3 for k in range(n-1, n+later+1)):
            return None
        still = region.copy()
        for k in range(n, n+later+1):
            still &= mags[k] < 0.5
        return still & (mags[n-1] > 4.0), still & (mags[n-1] < 0.5)

    report = {'frames': len(files), 'edit': edits, 'excluded_rectangles': list(exclude), 'horizons': {}}
    for later in horizons:
        acc = {name: {kind: [0., 0.] for kind in ('result', 'change')} for name in replays}
        nu = nb = used = 0
        for n in range(1, len(files)-later):
            mask = masks(n, later)
            if mask is None or int(mask[0].sum()) < 200:
                continue
            unc, bg = mask
            nu += int(unc.sum()); nb += int(bg.sum()); used += 1
            for name in replays:
                for kind, fn in (('result', result), ('change', change)):
                    d = np.abs(fn(name, n)-fn(name, n+later)).mean(axis=2)
                    acc[name][kind][0] += float(d[unc].sum())
                    acc[name][kind][1] += float(d[bg].sum())
        scores = {name: {kind: {'uncovered': a/nu, 'background': b/nb,
                              'trail': a/nu-b/nb} for kind, (a, b) in values.items()}
                  for name, values in acc.items()} if used and nb else {}
        report['horizons'][str(later)] = {'frame_pairs': used, 'uncovered_pixels': nu,
                                        'background_pixels': nb, 'scores': scores}

    start, end = settled
    offsets = sorted({0, 1, 2, 4, 8} & set(range(start)))
    errors = {name: {str(t): [0., 0.] for t in offsets} for name in replays}
    nu = nb = used = 0
    for n in range(1, len(files)-end):
        mask = masks(n, end)
        if mask is None:
            continue
        tail = pics[n+start:n+end+1]
        stable = (np.maximum.reduce(tail)-np.minimum.reduce(tail)).mean(axis=2) < span
        unc, bg = mask[0] & stable, mask[1] & stable
        if int(unc.sum()) < 200:
            continue
        target = np.zeros_like(pics[n])
        for k in range(n+start, n+end+1):
            target += result(reference, k)/(end-start+1)
        nu += int(unc.sum()); nb += int(bg.sum()); used += 1
        for name in replays:
            for t in offsets:
                d = np.abs(result(name, n+t)-target).mean(axis=2)
                errors[name][str(t)][0] += float(d[unc].sum())
                errors[name][str(t)][1] += float(d[bg].sum())
    scores = {name: {t: {'uncovered': a/nu, 'background': b/nb, 'excess': a/nu-b/nb}
                     for t, (a, b) in values.items()} for name, values in errors.items()} if used and nb else {}
    report['settled_target'] = {'reference': reference, 'start': start, 'end': end,
        'input_span_limit': span, 'frame_pairs': used, 'uncovered_pixels': nu,
        'background_pixels': nb, 'scores': scores}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--replay', action='append', required=True, metavar='LABEL=PREFIX')
    ap.add_argument('--reference', required=True, help='replay label for the common settled target')
    ap.add_argument('--count', type=int, default=60)
    ap.add_argument('--later', default='1,4,8')
    ap.add_argument('--settled', default='12:16')
    ap.add_argument('--input-span', type=float, default=2.0, help='late input RGB span in 1/255 units')
    ap.add_argument('--exclude-rect', action='append', default=[], metavar='X0,Y0,X1,Y1',
                    help='exclude known guide mismatches such as a HUD rectangle; repeatable')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    replays = {}
    for spec in args.replay:
        name, sep, prefix = spec.partition('=')
        if not name or not sep or not prefix or name in replays:
            ap.error('--replay needs a unique nonempty LABEL=PREFIX')
        replays[name] = prefix
    try:
        horizons = tuple(int(x) for x in args.later.split(','))
        settled = tuple(int(x) for x in args.settled.split(':'))
        exclude = [tuple(int(x) for x in rect.split(',')) for rect in args.exclude_rect]
    except ValueError:
        ap.error('later and settled need integer frame offsets')
    if args.reference not in replays or args.count < 1 or any(n < 1 for n in horizons):
        ap.error('invalid reference, count or horizon')
    if len(settled) != 2 or not 1 <= settled[0] <= settled[1] or args.input_span <= 0:
        ap.error('settled needs positive START:END and input-span must be positive')
    if any(len(rect) != 4 for rect in exclude):
        ap.error('exclude-rect needs X0,Y0,X1,Y1')
    report = compare(args.directory, replays, args.reference, args.count, horizons, settled, args.input_span, exclude)
    text = json.dumps(report, indent=2)+'\n'
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
