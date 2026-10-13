"""Look at a present-mode recording: picture | result in the game | change x gain.

Usage: pseq_view.py PSEQ_DIR OUT.png [--frames 0,1,2] [--crop x0,y0,x1,y1] [--gain 4] [--mv]
--mv adds a panel with the staged motion (hue = direction, brightness = pixels a frame, 16 px full).
"""
import argparse, colorsys, glob, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseq_compare import read_pseq
from seq_replay import read_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pseq'); ap.add_argument('out')
    ap.add_argument('--frames', default='0,1,2')
    ap.add_argument('--crop', default='')
    ap.add_argument('--gain', type=float, default=4.0)
    ap.add_argument('--mv', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.pseq, '*.bin')))
    rows = []
    for k in [int(v) for v in a.frames.split(',')]:
        pic, delta = read_pseq(files[k])
        pic = np.clip(pic, 0, 1) * 255
        delta = delta * 255
        tiles = [pic, np.clip(pic + delta, 0, 255), np.clip(delta * a.gain + 128, 0, 255)]
        if a.mv:
            _, depth, mv = read_frame(files[k])
            h, w = mv.shape[:2]
            px, py = mv[..., 0] * w, mv[..., 1] * h
            mag = np.clip(np.hypot(px, py) / 16.0, 0, 1)
            hue = (np.arctan2(py, px) / (2 * np.pi)) % 1.0
            i = np.floor(hue * 6).astype(int) % 6
            f = hue * 6 - np.floor(hue * 6)
            q, t = 1 - f, f
            r = np.choose(i, [1, q, 0, 0, t, 1]); g = np.choose(i, [t, 1, 1, q, 0, 0]); b = np.choose(i, [0, 0, t, 1, 1, q])
            tiles.append(np.stack([r, g, b], -1) * mag[..., None] * 255)
        if a.crop:
            x0, y0, x1, y1 = [int(v) for v in a.crop.split(',')]
            tiles = [t[y0:y1, x0:x1] for t in tiles]
        print(os.path.basename(files[k]), 'mean |change| %.2f' % np.abs(delta).mean())
        rows.append(np.concatenate(tiles, axis=1))
    im = Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8))
    if im.width > 2400:
        im = im.resize((2400, int(im.height * 2400 / im.width)), Image.LANCZOS)
    im.save(a.out)


if __name__ == '__main__':
    main()
