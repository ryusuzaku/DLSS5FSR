"""Which frame do a sequence capture's motion vectors belong to?

For every consecutive pair of frames the previous picture is carried onto the
current one with the vectors recorded one frame earlier, with the current
frame's, and with the next frame's (previous = current + mv, mv in fractions of
the frame). The vectors that belong to the picture leave the smallest error;
where the three fields differ most (silhouettes of things that move against the
background) the difference is largest, so the errors are also given for those
pixels alone.

Usage: mv_align.py SEQ_DIR [--first N] [--count N] [--linear]
SEQ_DIR: a shim sequence ("<capture>.seq.<tick>") or a present-mode one
("<capture>.pseq.<tick>"). --linear: the pictures are linear light (a shim
sequence captured with Passthrough=1); they are sRGB-encoded first.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_replay import read_frame, srgb_enc, LUMA


def warp(img, px, py):
    h, w = img.shape[:2]
    px = np.clip(px, 0, w - 1); py = np.clip(py, 0, h - 1)
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = px - x0; fy = py - y0
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq')
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--linear', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.seq, '*.bin')))[a.first:a.first + a.count]
    ids = [int(os.path.basename(f)[:-4]) for f in files]
    frames = []
    for f in files:
        rgb, depth, mv = read_frame(f)
        y = (srgb_enc(rgb) if a.linear else np.clip(rgb, 0, 1)) @ LUMA
        frames.append((y.astype(np.float32), mv))
    print('%d frames (%d..%d), %d without vectors' % (len(files), ids[0], ids[-1], sum(1 for _, m in frames if m is None)))
    names = ['none', 'mv[n-1]', 'mv[n]', 'mv[n+1]']
    tot = np.zeros(4); edge = np.zeros(4); used = 0
    print('frame   |mv| px   ' + '  '.join('%8s' % n for n in names) + '   | where the fields differ: ' + '  '.join('%8s' % n for n in names[1:]))
    for k in range(2, len(frames) - 1):
        if ids[k] != ids[k - 1] + 1 or ids[k + 1] != ids[k] + 1 or ids[k - 1] != ids[k - 2] + 1:
            continue
        y, mv = frames[k]
        prev = frames[k - 1][0]
        fields = [frames[k - 1][1], mv, frames[k + 1][1]]
        if any(f is None for f in fields):
            continue
        h, w = y.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        err = [np.abs(y - prev)]
        for f in fields:
            err.append(np.abs(y - warp(prev, xx + f[..., 0] * w, yy + f[..., 1] * h)))
        # Pixels whose vector changes by more than 2 px between neighbouring frames.
        d = np.maximum(np.hypot((fields[0][..., 0] - mv[..., 0]) * w, (fields[0][..., 1] - mv[..., 1]) * h),
                       np.hypot((fields[2][..., 0] - mv[..., 0]) * w, (fields[2][..., 1] - mv[..., 1]) * h))
        sel = d > 2.0
        e = np.array([x.mean() for x in err]) * 1000
        tot += e; used += 1
        line = '%5d  %7.2f   ' % (ids[k], np.hypot(mv[..., 0] * w, mv[..., 1] * h).mean()) + '  '.join('%8.2f' % v for v in e)
        if sel.sum() > 500:
            es = np.array([x[sel].mean() for x in err]) * 1000
            edge += es
            line += '   | %4.1f%%:                 ' % (100 * sel.mean()) + '  '.join('%8.2f' % v for v in es[1:])
        print(line)
    if used:
        print('mean          ' + '  '.join('%8.2f' % v for v in tot / used) + '   (luminance error x1000, lower = the vectors fit)')
        print('best overall: %s; where the fields differ: %s' % (names[int(np.argmin(tot))], names[1 + int(np.argmin(edge[1:]))]))


if __name__ == '__main__':
    main()
