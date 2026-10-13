"""Did the engine, in the game, answer the picture it was recorded with?

A present-mode recording (final_picture's sequence capture) holds, per frame,
the working-size picture and the change map the engine returned for it. This
compares that change with an offline replay of the same recording
(build/seq_replay.exe ... count 1 0, which writes PREFIX_NNN_in/out.rgb): with
the replay's change for the same frame, for the frame before and for the frame
after. In a correct run the same frame wins and the difference is small after
the first few frames (the replay starts without a history).

Usage: pseq_compare.py PSEQ_DIR REPLAY_PREFIX [--sheet FRAME --crop x0,y0,x1,y1 --out file.png]
"""
import argparse, glob, os, struct, sys
import numpy as np


def read_pseq(path):
    raw = open(path, 'rb').read()
    assert raw[:8] == b'D5INP001', path
    w, h, bpp, pitch = struct.unpack_from('<4I', raw, 8)
    rows = np.frombuffer(raw, np.uint8, pitch * h, 36).reshape(h, pitch)[:, :w * 8]
    pic = rows.copy().view(np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)
    at, out = 36 + pitch * h, None
    while at + 16 <= len(raw):
        tag = raw[at:at + 8]
        tw, th = struct.unpack_from('<2I', raw, at + 8)
        if tag == b'D5DEP001':
            at += 16 + tw * th * 4
        elif tag == b'D5MOV001':
            at += 16 + tw * th * 8
        elif tag == b'D5OUT001':
            out = np.frombuffer(raw, np.float16, tw * th * 4, at + 16).reshape(th, tw, 4)[..., :3].astype(np.float32)
            at += 16 + tw * th * 8
        else:
            break
    return np.nan_to_num(pic), out


def read_rgb(path):
    b = open(path, 'rb').read()
    w, h = struct.unpack('<II', b[:8])
    return np.frombuffer(b, np.uint8, offset=8).reshape(h, w, 3).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pseq')
    ap.add_argument('replay')
    ap.add_argument('--sheet', type=int, default=-1)
    ap.add_argument('--crop', default='')
    ap.add_argument('--out', default='pseq_compare.png')
    ap.add_argument('--gain', type=float, default=4.0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.pseq, '*.bin')))
    game, offline, pics = [], [], []
    for k, f in enumerate(files):
        o = '%s_%03d_out.rgb' % (a.replay, k)
        if not os.path.exists(o):
            break
        pic, delta = read_pseq(f)
        pics.append(np.clip(pic, 0, 1) * 255)
        game.append(delta * 255)  # the map is out - in, in the picture's encoding
        offline.append(read_rgb(o) - read_rgb('%s_%03d_in.rgb' % (a.replay, k)))
    n = len(game)
    print('%d frames compared; change in 1/255 of the range' % n)
    print('frame  |game|  |offline|   game - offline of frame: n-1     n     n+1')
    tot = np.zeros(3); used = 0
    for k in range(1, n - 1):
        d = [np.abs(game[k] - offline[j]).mean() for j in (k - 1, k, k + 1)]
        print('%5d  %6.2f  %8.2f                           %6.2f %6.2f %6.2f' % (
            k, np.abs(game[k]).mean(), np.abs(offline[k]).mean(), d[0], d[1], d[2]))
        if k >= 8:
            tot += d; used += 1
    if used:
        print('mean from frame 8:                              %6.2f %6.2f %6.2f' % tuple(tot / used))
    if a.sheet >= 0 and a.sheet < n:
        from PIL import Image
        k = a.sheet
        tiles = [pics[k], np.clip(pics[k] + game[k], 0, 255), np.clip(pics[k] + offline[k], 0, 255),
                 np.clip(game[k] * a.gain + 128, 0, 255), np.clip(offline[k] * a.gain + 128, 0, 255)]
        if a.crop:
            x0, y0, x1, y1 = [int(v) for v in a.crop.split(',')]
            tiles = [t[y0:y1, x0:x1] for t in tiles]
        Image.fromarray(np.concatenate(tiles, axis=1).astype(np.uint8)).save(a.out)
        print('sheet (picture | game result | offline result | game change | offline change):', a.out)


if __name__ == '__main__':
    main()
