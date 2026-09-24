#!/usr/bin/env python3
"""tools/staged_model.py -- analytic model of the staged proxy bytes.

The level-3 window comes from the NR proxy (gpu.cpp copies proxy->res):
rolloff + sRGB of the bilinear-blit frame -- NOT raw source pixels.
This module predicts the 256 proxy bytes for any 8x8 window so a run's
window can be confirmed (or found) before the oracle verdict runs.

Pipeline order (all in the repo, all verified to the byte):
  source  harness gradient 320x180 (R=x*255//319, G=y*255//179,
            B=step at x=160, A=255; proven == nr_0000_in.bmp)
  blit    bilinear 4x centre-mapping (blit.hlsl linear path;
            coord=(d+0.5)*n_src/n_out-0.5) + UNORM store (round-half-even)
  encode  white-point scale + soft knee above luma 0.75 (dlssnr.hlsl
            gMode==0), then exact piecewise LinearToSrgb + UNORM store
Transfer/colour strengths never touch staging (they gate the resolve
blend only); debug views don't either (staging copies pre-resolve).

Usage:
  python3 tools/staged_model.py                 self-test (needs the
      build/ dumps + staged paste on the shared tree; all comparisons
      are model-vs-device-data, never model-vs-model)
  python3 tools/staged_model.py --win OX OY [--white W]
      print the predicted win line + oracle checksum/maxabs for a run
      (oracle via tools/emit_yp_golden.py).
"""

import math
import os
import re
import struct
import sys

SRC_W, SRC_H, DST_W, DST_H = 320, 180, 1280, 720
K_LUMA = (0.2126, 0.7152, 0.0722)


def src_pixel(x, y):
    """Harness gradient byte (tests/ngx_harness.cpp source fill)."""
    return (x * 255 // (SRC_W - 1), y * 255 // (SRC_H - 1),
            0 if x < SRC_W // 2 else 255, 255)


def blit_val(d, n_out, vals):
    """One bilinear centre-mapped tap + UNORM RNE store.

    vals: source channel bytes (length n_src).
    """
    n_src = len(vals)
    c = (d + 0.5) * n_src / n_out - 0.5
    if c <= 0:
        v = float(vals[0])
    elif c >= n_src - 1:
        v = float(vals[-1])
    else:
        i = math.floor(c)
        t = c - i
        v = vals[i] * (1.0 - t) + vals[i + 1] * t
    return int(round(v))


def lin2srgb(v):
    v = min(max(v, 0.0), 1.0)
    if v < 0.0031308:
        return v * 12.92
    return 1.055 * (max(v, 1e-8) ** (1.0 / 2.4)) - 0.055


def proxy_window(ox, oy, white=1.0):
    """256 staged-proxy bytes for the 8x8 window at (ox, oy)."""
    r_src = [src_pixel(x, 0)[0] for x in range(SRC_W)]
    g_src = [src_pixel(0, y)[1] for y in range(SRC_H)]
    b_src = [src_pixel(x, 0)[2] for x in range(SRC_W)]
    out = bytearray()
    for y in range(oy, oy + 8):
        g = blit_val(y, DST_H, g_src) / 255.0
        for x in range(ox, ox + 8):
            lin = [blit_val(x, DST_W, r_src) / 255.0, g,
                   blit_val(x, DST_W, b_src) / 255.0]
            disp = [v / white for v in lin]
            lum = sum(K_LUMA[c] * disp[c] for c in range(3))
            if lum > 0.75:
                rolled = 0.75 + 0.25 * (1.0 - math.exp(-(lum - 0.75) / 0.25))
                disp = [v * rolled / lum for v in disp]
            out.extend(int(round(lin2srgb(v) * 255)) for v in disp)
            out.append(255)
    return bytes(out)


def _read_bmp_24(path):
    d = open(path, 'rb').read()
    off = struct.unpack('<I', d[10:14])[0]
    w = struct.unpack('<i', d[18:22])[0]
    h = struct.unpack('<i', d[22:26])[0]
    assert struct.unpack('<H', d[28:30])[0] == 24, path
    stride = w * 3
    return [[(d[off + (h - 1 - y) * stride + x * 3 + 2],
              d[off + (h - 1 - y) * stride + x * 3 + 1],
              d[off + (h - 1 - y) * stride + x * 3])
             for x in range(w)] for y in range(h)]


def selftest(root='.'):
    # 1. analytic source == dumped input, every pixel.
    ib = _read_bmp_24(os.path.join(root, 'build/nr_0000_in.bmp'))
    assert (len(ib), len(ib[0])) == (SRC_H, SRC_W)
    bad = sum(1 for y in range(SRC_H) for x in range(SRC_W)
              if ib[y][x] != src_pixel(x, y)[:3])
    print('source vs in.bmp: %d/57600 differ' % bad)
    assert bad == 0, bad
    # 2. model(0,0) == staged device bytes from the level-3 paste.
    paste = open(os.path.join(root, 'build/staged_paste.txt')).read()
    win = bytes.fromhex(re.search(r'bytes=([0-9A-Fa-f]+)', paste).group(1))
    m00 = proxy_window(0, 0)
    print('model(0,0) vs staged win: %s' %
          ('MATCH' if m00 == win else 'MISMATCH'))
    assert m00 == win, 'model(0,0) mismatch'
    # 3. model(636,0) == sRGB of the dumped (decoded-proxy) region:
    # device data both sides, model never compared to itself.
    ob = _read_bmp_24(os.path.join(root, 'build/nr_0000_out.bmp'))
    assert (len(ob), len(ob[0])) == (DST_H, DST_W)
    m636 = proxy_window(636, 0)
    back = bytes(b for y in range(8) for x in range(636, 644)
                 for p in [ob[y][x]]
                 for b in (int(round(lin2srgb(p[0] / 255) * 255)),
                           int(round(lin2srgb(p[1] / 255) * 255)),
                           int(round(lin2srgb(p[2] / 255) * 255)), 255))
    print('model(636,0) vs sRGB(dump region): %s' %
          ('MATCH' if m636 == back else 'MISMATCH'))
    assert m636 == back, 'model(636,0) mismatch'
    # 4. full-frame characterisation: model vs sRGB(dump). The chain
    # stacks four UNORM roundings (blit, proxy, dump, re-encode), so the
    # bound is <=2 with rationale, not 0; the exact gates are 1-3.
    r_src = [src_pixel(x, 0)[0] for x in range(SRC_W)]
    g_src = [src_pixel(0, y)[1] for y in range(SRC_H)]
    b_src = [src_pixel(x, 0)[2] for x in range(SRC_W)]
    rr = [blit_val(x, DST_W, r_src) for x in range(DST_W)]
    gg = [blit_val(y, DST_H, g_src) for y in range(DST_H)]
    bb = [blit_val(x, DST_W, b_src) for x in range(DST_W)]
    worst, exact, total = 0, 0, 0
    for y in range(DST_H):
        for x in range(DST_W):
            lin = [rr[x] / 255.0, gg[y] / 255.0, bb[x] / 255.0]
            lum = sum(K_LUMA[c] * lin[c] for c in range(3))
            if lum > 0.75:
                rolled = 0.75 + 0.25 * (1.0 - math.exp(-(lum - 0.75) / 0.25))
                lin = [v * rolled / lum for v in lin]
            for c in range(3):
                want = int(round(lin2srgb(lin[c]) * 255))
                back_c = int(round(lin2srgb(ob[y][x][c] / 255) * 255))
                d = abs(want - back_c)
                worst = max(worst, d)
                exact += (d == 0)
                total += 1
    print('full-frame model vs sRGB(dump): worst=%d exact=%.3f' %
          (worst, exact / total))
    assert worst <= 2, worst
    print('SELFTEST-OK')


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--win':
        ox, oy = int(sys.argv[2]), int(sys.argv[3])
        white = 1.0
        if '--white' in sys.argv:
            white = float(sys.argv[sys.argv.index('--white') + 1])
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from emit_yp_golden import fe_block_yp, R
        import math as _m
        win = proxy_window(ox, oy, white)
        print('hip: fe-staged win x=%d y=%d bytes=%s (predicted)' %
              (ox, oy, win.hex().upper()))
        fl = R.flat(fe_block_yp(win))
        fin = [v for v in fl if _m.isfinite(v)]
        print('oracle checksum=%s maxabs=%.4f finite=%d/2048' %
              (R.fnv1a_hex(fin), max(abs(v) for v in fl), len(fin)))
    else:
        selftest(sys.argv[1] if len(sys.argv) > 1 else '.')
