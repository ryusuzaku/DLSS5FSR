"""Look lab: one capture through the engine with several option sets, side by side.

Usage: look_lab.py capture.bin out_dir [variant ...]
Each variant is NAME=opt;opt;... (engine key=value options added after the
24 path lines of the game config). Writes out_dir/NAME.png per variant,
out_dir/input.png, and out_dir/sheet.png (all of them in a grid, labelled).
Frames are shown as the staging holds them (display-encoded), clamped to 0..1.
"""
import os, struct, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
TOOL = f'{ROOT}/resident_engine_frame.exe'
DLL = os.environ.get('LAB_DLL', f'{ROOT}/resident_engine_dev.dll')
BASE = f'{ROOT}/resident_engine_check/engine_config_game.txt'


def load_capture(path):
    raw = open(path, 'rb').read()
    w, h, bpp, pitch = struct.unpack_from('<4I', raw, 8)
    rows = np.frombuffer(raw[36:36 + pitch * h], np.uint8).reshape(h, pitch)[:, :w * bpp]
    return to_rgb(rows.copy(), w, h, bpp)


def to_rgb(rows, w, h, bpp):
    if bpp == 8:
        a = rows.view(np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)
    else:
        a = rows.reshape(h, w, 4)[..., :3].astype(np.float32) / 255.0
    return np.clip(np.nan_to_num(a), 0.0, 1.0)


def load_out(path):
    raw = open(path, 'rb').read()
    w, h, bpp = struct.unpack_from('<3I', raw, 0)
    rows = np.frombuffer(raw[12:12 + w * h * bpp], np.uint8).reshape(h, w * bpp)
    return to_rgb(rows.copy(), w, h, bpp)


def png(a, path):
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def run(capture, out_dir, name, opts):
    base = open(BASE).read().splitlines()[:24]
    cfg = os.path.join(out_dir, f'{name}.cfg')
    with open(cfg, 'w', newline='\n') as f:
        f.write('\n'.join(base + [o for o in opts if o]) + '\n')
    out = os.path.join(out_dir, f'{name}.bin')
    r = subprocess.run([TOOL, DLL, cfg, capture, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'{name}: {r.stdout}{r.stderr}')
    return load_out(out)


def main():
    capture, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    tiles = [('input', load_capture(capture))]
    png(tiles[0][1], os.path.join(out_dir, 'input.png'))
    for spec in sys.argv[3:]:
        name, _, opts = spec.partition('=')
        img = run(capture, out_dir, name, opts.split(';'))
        png(img, os.path.join(out_dir, f'{name}.png'))
        d = np.abs(img - tiles[0][1]).mean()
        print(f'{name}: mean |change| {d:.4f}')
        tiles.append((name, img))
    h, w = tiles[0][1].shape[:2]
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w, rows * (h + 24)), 'black')
    draw = ImageDraw.Draw(sheet)
    for k, (name, img) in enumerate(tiles):
        x, y = (k % cols) * w, (k // cols) * (h + 24)
        sheet.paste(Image.fromarray((img * 255 + 0.5).astype(np.uint8)), (x, y + 24))
        draw.text((x + 6, y + 5), name, fill='white')
    sheet.save(os.path.join(out_dir, 'sheet.png'))


if __name__ == '__main__':
    main()
