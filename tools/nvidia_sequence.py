"""Our network with the original's temporal pipeline on the peer DLSSNR-AMD's moving sequences,
measured against NVIDIA's own numbers (docs/ngx-verification/moving-sequence-results.json).

Sequences (made from the verification's Tomb Raider frame, 10 frames):
  still   the frame unchanged (exactly the peer's: NVIDIA's change per frame 0.324/255 at 1080p)
  object  a disc textured with the mirrored frame slides 6 px a frame (the peer's description;
          its exact disc size is not published, so this one is comparable only roughly)
Prints per frame: PSNR against NVIDIA's single-frame output (frame 0 is the single-frame case),
and the change per frame (mean |out_k - out_k-1| over pixels and channels, 1/255, the previous
frame moved by the motion vectors), next to NVIDIA's.

Usage: nvidia_sequence.py [still|object] [--res 1920x1080] [--frames 10] [--exe ...] [--config ...]
"""
import argparse, json, os, struct, subprocess
import numpy as np
from PIL import Image

REF = 'P:/peers/DLSSNR-AMD/docs/ngx-verification'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')


def write_frame(path, rgb, mv, depth):
    h, w = rgb.shape[:2]
    with open(path, 'wb') as f:
        f.write(struct.pack('<2I', w, h)); f.write(rgb.astype(np.uint8).tobytes())
        f.write(mv.astype(np.float32).tobytes()); f.write(depth.astype(np.float32).tobytes())


def make(kind, img, frames, out):
    h, w = img.shape[:2]
    os.makedirs(out, exist_ok=True)
    for k in range(frames):
        mv = np.zeros((h, w, 2), np.float32)
        depth = np.full((h, w), 0.01, np.float32)
        rgb = img.copy()
        if kind == 'object':
            r = h // 6
            cx, cy = w // 4 + 6 * k, h // 2
            yy, xx = np.mgrid[0:h, 0:w]
            disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
            tex = img[:, ::-1]
            rgb[disc] = tex[disc]
            mv[disc] = (-6.0, 0.0)            # previous = current + mv
            depth[disc] = 0.5                 # reversed-Z: the disc is in front
        write_frame(os.path.join(out, f'{k:04d}.seq'), rgb, mv, depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', nargs='?', default='still')
    ap.add_argument('--res', default='1920x1080')
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--exe', default=os.path.join(ROOT, 'network_sequence.exe'))
    ap.add_argument('--config', default=os.path.join(ROOT, 'resident_engine_check', 'engine_config_game.txt'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'sequence'))
    a = ap.parse_args()
    img = np.asarray(Image.open(f'{REF}/single-frame-inputs/{a.res}.png').convert('RGB'))
    ref = np.asarray(Image.open(f'{REF}/single-frame-outputs/{a.res}_nvidia.png').convert('RGB')).astype(np.float64)
    seq_in = os.path.join(a.out, a.kind + '_in'); seq_out = os.path.join(a.out, a.kind + '_out')
    os.makedirs(seq_out, exist_ok=True)
    make(a.kind, img, a.frames, seq_in)
    r = subprocess.run([a.exe, a.config, seq_in, seq_out, str(a.frames)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    nv = json.load(open(f'{REF}/moving-sequence-results.json'))
    key = {'still': 'still_scene', 'object': 'moving_object'}[a.kind] + '_' + a.res
    prev = None
    changes = []
    for k in range(a.frames):
        raw = open(os.path.join(seq_out, f'{k:04d}.f32'), 'rb').read()
        w, h = struct.unpack_from('<2I', raw, 0)
        o = np.round(np.clip(np.frombuffer(raw, np.float32, w * h * 3, 8).reshape(h, w, 3), 0, 1) * 255.0)
        psnr = 10 * np.log10(255.0 ** 2 / np.mean((o - ref) ** 2)) if a.kind == 'still' else float('nan')
        line = f'frame {k}: PSNR vs NVIDIA single frame {psnr:6.2f} dB'
        if prev is not None:
            if a.kind == 'object':
                # move the previous output by the vectors (only the disc moves, 6 px)
                pm = prev.copy()
                inp = open(os.path.join(seq_in, f'{k:04d}.seq'), 'rb').read()
                mv = np.frombuffer(inp, np.float32, w * h * 2, 8 + w * h * 3).reshape(h, w, 2)
                ys, xs = np.nonzero(mv[..., 0] != 0)
                pm[ys, xs] = prev[ys, np.clip(xs - 6, 0, w - 1)]
                ch = np.abs(o - pm).mean()
            else:
                ch = np.abs(o - prev).mean()
            changes.append(ch)
            line += f', change {ch:.3f}/255'
        print(line)
        prev = o
    print(f'mean change per frame: ours {np.mean(changes):.3f}/255, NVIDIA {nv[key]["change_per_frame_255_nvidia"]}/255,'
          f' peer {nv[key]["change_per_frame_255_dlssnr_amd"]}/255;  NVIDIA vs peer PSNR per frame: {nv[key]["psnr_db_per_frame"]}')


if __name__ == '__main__':
    main()
