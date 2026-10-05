"""Temporal test of the luma NR path: flicker on a still scene and trails behind a moving object.

Usage: temporal_test.py capture.bin NAME=opt;opt ... [--dll path] [--out dir]
Each NAME is an engine option set (added after the game config's 24 path lines).
Runs build/resident_engine_frame.exe at game cadence (16 ms frames) with per-frame input noise
(RE_NOISE, the game's frame-to-frame variation) and the shim's lag, and reports per config:
  flicker  mean |edit_f - edit_(f-1)| on a still camera (edit = log2 luminance ratio output/input), x1000
  strength mean |edit| on the still camera, x1000 (how much NR changes the picture)
  trail    mean |edit - edit_still| where a moving object has just been (it should be gone), x1000
  object   the same inside the object's current rectangle, x1000 (lag of the edit on the object)
Lower flicker / trail at similar strength is better.
"""
import os, struct, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
TOOL = f'{ROOT}/resident_engine_frame.exe'
BASE = f'{ROOT}/resident_engine_check/engine_config_game.txt'
NOISE = '0.004'      # display units, about one 8-bit level
FRAME_MS = os.environ.get('TT_FRAME_MS', '16')
LAG = os.environ.get('TT_LAG', '2')
OBJ = (200, 200, 160, 220)   # source x, y, w, h of the moving patch in the capture
PAN = 8                      # pixels per frame
FRAMES = 36


def read(path):
    raw = open(path, 'rb').read()
    w, h, bpp = struct.unpack_from('<3I', raw, 0)
    a = np.frombuffer(raw[12:12 + w*h*bpp], np.uint8).reshape(h, w, bpp)
    if bpp == 8:
        v = a.view(np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)
    else:
        v = a[..., :3].astype(np.float32) / 255.0
    v = np.clip(np.nan_to_num(v), 0, 1)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def edit(prefix, i):
    return np.log2((read(f'{prefix}_{i:03d}.bin') + 0.005) / (read(f'{prefix}_in_{i:03d}.bin') + 0.005))


def run(dll, capture, cfg, prefix, extra_env, pan):
    env = dict(os.environ, RE_DUMP_PREFIX=prefix, RE_NOISE=NOISE, RE_FRAME_MS=FRAME_MS, RE_SHIM_LAG=LAG, **extra_env)
    r = subprocess.run([TOOL, dll, cfg, capture, prefix + '_last.bin', str(pan), str(FRAMES), 'uv', '1', '0'],
                       env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'{prefix}: {r.stdout}{r.stderr}')


def main():
    args = sys.argv[1:]
    dll = f'{ROOT}/resident_engine.dll'
    out = tempfile.mkdtemp(prefix='temporal_')
    if '--dll' in args:
        i = args.index('--dll'); dll = args[i+1]; del args[i:i+2]
    if '--out' in args:
        i = args.index('--out'); out = args[i+1]; del args[i:i+2]
    capture, specs = args[0], args[1:]
    os.makedirs(out, exist_ok=True)
    base = open(BASE).read().splitlines()[:24]
    print(f'{"config":<18} {"flicker":>8} {"strength":>9} {"trail":>7} {"object":>7}')
    for spec in specs:
        name, _, opts = spec.partition('=')
        cfg = os.path.join(out, name + '.cfg')
        with open(cfg, 'w', newline='\n') as f:
            f.write('\n'.join(base + [o for o in opts.split(';') if o]) + '\n')
        still, moving = os.path.join(out, name + '_still'), os.path.join(out, name + '_move')
        run(dll, capture, cfg, still, {}, 0)
        sx, sy, ow, oh = OBJ
        run(dll, capture, cfg, moving, {'RE_OBJECT': f'{sx},{sy},{ow},{oh},{40},{300}'}, PAN)
        idx = range(FRAMES // 2, FRAMES)
        e = [edit(still, i) for i in idx]
        flicker = np.mean([np.abs(e[k] - e[k-1]).mean() for k in range(1, len(e))])
        strength = np.mean([np.abs(x).mean() for x in e])
        ref = np.mean(e, axis=0)
        trail, obj = [], []
        lag = int(LAG)
        for i in idx:
            k = max(0, i - lag)       # the staged frame the output belongs to
            x0 = 40 + k*PAN
            em = edit(moving, i)
            cur = np.zeros_like(em, bool); cur[300:300+oh, max(0, x0):max(0, x0+ow)] = True
            behind = np.zeros_like(em, bool); behind[300:300+oh, max(0, x0-6*PAN):max(0, x0)] = True
            if behind.any(): trail.append(np.abs(em - ref)[behind].mean())
            if cur.any(): obj.append(np.abs(em[cur]).mean())
        print(f'{name:<18} {flicker*1000:8.2f} {strength*1000:9.2f} {np.mean(trail)*1000:7.2f} {np.mean(obj)*1000:7.2f}')


if __name__ == '__main__':
    main()
