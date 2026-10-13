"""Whose trail is it? Measures, in a present-mode recording, what is left behind
moving things in the game's own picture and in the result after the engine.

A pixel counts as just uncovered in frame n when it is still background now
(its vector under 0.5 px) but belonged to something moving in frame n-1 (vector
over 4 px there). Its value is compared with the same pixel `later` frames on,
when whatever was left has faded, provided it is background in every frame in
between and at the end. The same comparison over ordinary still background
gives the noise floor. Reported for the picture the game presented, for the
result (picture + the engine's change), and for the change alone.

Usage: pseq_trail.py PSEQ_DIR [--later 4] [--replay PREFIX]
--replay: take the result from an offline replay (seq_replay's PREFIX_NNN_out.rgb) instead of the recording.
Only frames where the camera is nearly still (median vector < 0.3 px) count.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseq_compare import read_pseq
from seq_replay import read_frame
from pseq_compare import read_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pseq')
    ap.add_argument('--later', type=int, default=4)
    ap.add_argument('--replay', default='')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.pseq, '*.bin')))
    pics, deltas, mags = [], [], []
    for i, f in enumerate(files):
        pic, delta = read_pseq(f)
        if a.replay:
            o = '%s_%03d_out.rgb' % (a.replay, i)
            if not os.path.exists(o):
                break
            delta = (read_rgb(o) - read_rgb('%s_%03d_in.rgb' % (a.replay, i))) / 255.0
        _, _, mv = read_frame(f)
        h, w = mv.shape[:2]
        pics.append((np.clip(pic, 0, 1) * 255).astype(np.float32))
        deltas.append((delta * 255).astype(np.float32))
        mags.append(np.hypot(mv[..., 0] * w, mv[..., 1] * h))
    L = a.later
    acc = {k: [0.0, 0.0] for k in ('pic', 'res', 'chg')}
    n_unc = n_bg = used = 0
    for n in range(1, len(pics) - L):
        if any(np.median(mags[k]) > 0.3 for k in range(n - 1, n + L + 1)):
            continue
        still = np.ones(mags[n].shape, bool)
        for k in range(n, n + L + 1):
            still &= mags[k] < 0.5
        unc = still & (mags[n - 1] > 4.0)
        bg = still & (mags[n - 1] < 0.5)
        if unc.sum() < 200:
            continue
        used += 1
        n_unc += int(unc.sum()); n_bg += int(bg.sum())
        for key, a0, a1 in (('pic', pics[n], pics[n + L]),
                            ('res', np.clip(pics[n] + deltas[n], 0, 255), np.clip(pics[n + L] + deltas[n + L], 0, 255)),
                            ('chg', deltas[n], deltas[n + L])):
            d = np.abs(a0 - a1).mean(axis=2)
            acc[key][0] += float(d[unc].sum()); acc[key][1] += float(d[bg].sum())
    if not used:
        print('no frame with a still camera and something moving')
        return
    print('%d frames used; %d uncovered pixels, %d still background pixels; difference to %d frames later, 1/255' % (used, n_unc, n_bg, L))
    for key, name in (('pic', "the game's own picture"), ('res', 'the result after the engine'), ('chg', "the engine's change alone")):
        u, b = acc[key][0] / n_unc, acc[key][1] / n_bg
        print('  %-28s just uncovered %6.2f   still background %5.2f   left behind %6.2f' % (name, u, b, u - b))


if __name__ == '__main__':
    main()
