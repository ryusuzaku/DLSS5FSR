"""Compare the resident network with NVIDIA's own DLSS 5 NR output on a reference frame.

The reference: peer DLSSNR-AMD's NGX verification (docs/ngx-verification), NVIDIA's
nvngx_dlssnr.dll 310.8.0 run through NGX on an RTX 5090 with DLSSNR.Reset (one frame, no
history) and the defaults Intensity 1, Style 0, LocalTone 1, LocalStructure 1, Skin -1,
AutoMask 1. Same measures as the peer's single-frame table: PSNR and correlation of the edits
(output - input) against NVIDIA, the edit size (RMS, 1/255), mean difference per channel.

Usage: nvidia_parity.py [RES ...] [--extent compact|legacy] [--seed N] [--config cfg] [--exe tool] [--out dir]
RES: 1920x1080 (default), 2560x1440, 3840x2160.
"""
import argparse, os, struct, subprocess
import numpy as np
from PIL import Image

REF = 'P:/peers/DLSSNR-AMD/docs/ngx-verification'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def measures(name, out, ref, inp):
    e_out, e_ref = out - inp, ref - inp
    corr = np.corrcoef(e_out.ravel(), e_ref.ravel())[0, 1]
    return (f'{name:<14} PSNR {psnr(out, ref):6.2f} dB  edit corr {corr:.4f}  edit RMS {np.sqrt((e_out ** 2).mean()):6.2f}'
            f' (NVIDIA {np.sqrt((e_ref ** 2).mean()):5.2f})  mean diff ' +
            ' '.join(f'{v:+.2f}' for v in (out - ref).reshape(-1, 3).mean(0)) +
            f'  within 1/255 {np.mean(np.all(np.abs(out - ref) <= 1, -1)) * 100:5.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res', nargs='*', default=['1920x1080'])
    ap.add_argument('--extent', default='compact')
    ap.add_argument('--seed', default='0')
    ap.add_argument('--config', default=os.path.join(ROOT, 'resident_engine_check', 'engine_config_game.txt'))
    ap.add_argument('--exe', default=os.path.join(ROOT, 'network_parity.exe'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'parity'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for res in a.res:
        inp = np.asarray(Image.open(f'{REF}/single-frame-inputs/{res}.png').convert('RGB'))
        ref = np.asarray(Image.open(f'{REF}/single-frame-outputs/{res}_nvidia.png').convert('RGB')).astype(np.float64)
        peer = np.asarray(Image.open(f'{REF}/single-frame-outputs/{res}_dlssnr-amd.png').convert('RGB')).astype(np.float64)
        h, w = inp.shape[:2]
        rgb_path = os.path.join(a.out, f'{res}.rgb')
        with open(rgb_path, 'wb') as f:
            f.write(struct.pack('<2I', w, h)); f.write(inp.tobytes())
        out_path = os.path.join(a.out, f'{res}_ours.f32')
        r = subprocess.run([a.exe, a.config, rgb_path, out_path, '0', '1', '1', '-1', '1', a.seed, a.extent],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout + r.stderr)
        raw = open(out_path, 'rb').read()
        ow, oh = struct.unpack_from('<2I', raw, 0)
        ours = np.frombuffer(raw, np.float32, ow * oh * 3, 8).reshape(oh, ow, 3)
        ours8 = np.round(np.clip(ours, 0, 1) * 255.0)
        Image.fromarray(ours8.astype(np.uint8)).save(os.path.join(a.out, f'{res}_ours.png'))
        x = inp.astype(np.float64)
        print(f'== {res} ({a.extent}, seed {a.seed})')
        print(measures('ours', ours8, ref, x))
        print(measures('peer', peer, ref, x))
        print(measures('input (no NR)', x, ref, x))


if __name__ == '__main__':
    main()
