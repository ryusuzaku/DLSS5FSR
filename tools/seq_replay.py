"""Offline replay of the shim's map resolve over an in-game sequence capture.

The shim applies the engine's ratio map L frames late: it walks each pixel back through the
motion history to the map's frame, gates the map entry on luminance (and depth), and scales
the frame by the ratio. This replays that resolve (a numpy port of the map branch of
src/ngx/shaders/dlssnr.hlsl, NR_MODE_RESOLVE) on captured frames, with the engine's per-frame
maps from build/resident_engine_seq.exe, and compares every lagged result with the zero-lag
one (the frame's own map, no walk): the difference is exactly what the lag and the gates cost.

Usage:
  seq_replay.py SEQ_DIR MAP_DIR OUT_DIR [--lags 0,2,4,6] [--first N] [--count N]
                [--depth 3x3|off] [--tol 0.3] [--sheet FRAME] [--crop x0,y0,x1,y1]
SEQ_DIR: the shim's "<capture>.seq.<tick>" folder (D5INP001 + D5DEP001 + D5MOV001 per frame).
MAP_DIR: resident_engine_seq.exe output (map_NNNN.bin, RG16F ratio / Y_in).
Prints per lag: mean |lag error| (log2 luminance, x1000) over all pixels and over moving ones,
the edit strength, and the flicker (mean |edit_n - edit_n-1|) on still pixels.
"""
import argparse, glob, os, struct
import numpy as np

E = 0.005
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb_dec(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_enc(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def read_frame(path):
    raw = open(path, 'rb').read()
    assert raw[:8] == b'D5INP001', path
    w, h, bpp, pitch = struct.unpack_from('<4I', raw, 8)
    rows = np.frombuffer(raw, np.uint8, pitch * h, 36).reshape(h, pitch)[:, :w * bpp]
    rgb = rows.copy().view(np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)
    rgb = np.nan_to_num(rgb)
    at, depth, mv = 36 + pitch * h, None, None
    while at + 16 <= len(raw):
        tag = raw[at:at + 8]
        tw, th = struct.unpack_from('<2I', raw, at + 8)
        if tag == b'D5DEP001':
            depth = np.frombuffer(raw, np.float32, tw * th, at + 16).reshape(th, tw)
            at += 16 + tw * th * 4
        elif tag == b'D5MOV001':
            mv = np.frombuffer(raw, np.float32, tw * th * 2, at + 16).reshape(th, tw, 2)
            at += 16 + tw * th * 8
        else:
            break
    return rgb, depth, mv


def read_map(path):
    raw = open(path, 'rb').read()
    w, h, _ = struct.unpack_from('<3I', raw, 0)
    return np.frombuffer(raw, np.float16, w * h * 2, 12).reshape(h, w, 2).astype(np.float32)


def bilinear(img, px, py):
    """Sample img (h, w, c) at pixel-centre coordinates (px, py) like SampleLevel with clamp."""
    h, w = img.shape[:2]
    fx, fy = px - 0.5, py - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    top = img[y0c, x0c] * (1 - tx) + img[y0c, x1c] * tx
    bot = img[y1c, x0c] * (1 - tx) + img[y1c, x1c] * tx
    return top * (1 - ty) + bot * ty


def neighbourhood(a, fn):
    p = np.pad(a, 1, mode='edge')
    h, w = a.shape
    return fn(np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))


class Replay:
    def __init__(self, seq_dir, map_dir, first, count):
        self.files = sorted(glob.glob(os.path.join(seq_dir, '*.bin')))[first:first + count]
        self.first = first
        self.map_dir = map_dir
        self.cache = {}

    def frame(self, i):
        if i not in self.cache:
            rgb, depth, mv = read_frame(self.files[i])
            m = read_map(os.path.join(self.map_dir, f'map_{i + self.first:04d}.bin'))
            self.cache[i] = (rgb, depth, mv, m)
            for k in [k for k in self.cache if k < i - 10]:
                del self.cache[k]
        return self.cache[i]

    def resolve(self, n, lag, scale, tol=0.3, depth_mode='3x3', fill=False, fill_gate=True, dsample=0.0):
        """The shim's map resolve of frame n with the map of frame n-lag (passthrough input)."""
        proxy, _, _, _ = self.frame(n)
        h, w = proxy.shape[:2]
        px, py = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        depth_w = np.ones((h, w), np.float32)
        for i in range(lag):
            _, d_now, mv, _ = self.frame(n - i)
            _, d_prev, _, _ = self.frame(n - i - 1)
            ax = np.clip(px.astype(int), 0, w - 1)
            ay = np.clip(py.astype(int), 0, h - 1)
            v = np.nan_to_num(mv[ay, ax])
            px = px + v[..., 0] * scale[0]
            py = py + v[..., 1] * scale[1]
            if depth_mode != 'off' and d_now is not None and d_prev is not None:
                d = d_now[ay, ax]
                tx = np.clip(px.astype(int), 0, w - 1)
                ty = np.clip(py.astype(int), 0, h - 1)
                r = 1 if depth_mode == '3x3' else 0
                best = np.full((h, w), 1e30, np.float32)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        dp = d_prev[np.clip(ty + dy, 0, h - 1), np.clip(tx + dx, 0, w - 1)]
                        best = np.minimum(best, np.abs(d - dp) / np.maximum(np.maximum(d, dp), 1e-7))
                depth_w = np.minimum(depth_w, np.clip((0.25 - best) / 0.15, 0, 1))
        _, d_map, _, mp = self.frame(n - lag)
        if dsample and lag > 0 and d_map is not None and self.frame(n)[1] is not None:
            # Depth-aware bilinear: each of the four taps weighted by how
            # close its depth (map frame) is to this pixel's, so a sample on a
            # silhouette does not mix the occluder's edit with the background's.
            d_cur = self.frame(n)[1]
            fx, fy = px - 0.5, py - 0.5
            x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
            tx, ty = fx - x0, fy - y0
            acc = np.zeros((h, w, 2), np.float32)
            wsum = np.zeros((h, w), np.float32)
            for dx, dy, wb in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
                qx, qy = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)
                dq = d_map[qy, qx]
                rel = np.abs(dq - d_cur) / np.maximum(np.maximum(dq, d_cur), 1e-7)
                wt = wb * np.exp(-(rel / dsample) ** 2) + 1e-6 * wb
                acc += mp[qy, qx] * wt[..., None]
                wsum += wt
            m = acc / np.maximum(wsum, 1e-12)[..., None]
        else:
            m = bilinear(mp, px, py)
        inside = (px >= 0) & (px <= w) & (py >= 0) & (py <= h)
        filled = np.zeros((h, w), bool)
        self.hidden = np.zeros((h, w), bool)
        if lag > 0 and d_map is not None and self.frame(n)[1] is not None:
            # pixels hidden on the map's frame (depth at the walked position differs by > 25%)
            d_cur = self.frame(n)[1]
            hx, hy = np.clip(px.astype(int), 0, w - 1), np.clip(py.astype(int), 0, h - 1)
            dm = d_map[hy, hx]
            self.hidden = np.abs(d_cur - dm) / np.maximum(np.maximum(d_cur, dm), 1e-7) > 0.25
        if fill and lag > 0 and d_map is not None:
            # Disocclusion fill: where this pixel's depth differs from the
            # map frame's at the walked position (it was hidden then), take
            # the map entry of the nearest tap around that position that has
            # this pixel's depth (the background beside the occluder) instead
            # of no edit, which reads as a light band behind moving objects.
            d_cur = self.frame(n)[1]
            tx = np.clip(px.astype(int), 0, w - 1)
            ty = np.clip(py.astype(int), 0, h - 1)
            rel = lambda a, b: np.abs(a - b) / np.maximum(np.maximum(a, b), 1e-7)
            hidden = rel(d_cur, d_map[ty, tx]) > 0.25
            need = hidden.copy()
            mf = m.copy()
            for radius in (3, 6, 12, 24, 48):
                best = np.full((h, w), np.inf, np.float32)
                pick = np.zeros((h, w, 2), np.float32)
                for ang in range(8):
                    ox = int(round(radius * np.cos(ang * np.pi / 4)))
                    oy = int(round(radius * np.sin(ang * np.pi / 4)))
                    qx, qy = np.clip(tx + ox, 0, w - 1), np.clip(ty + oy, 0, h - 1)
                    r = rel(d_cur, d_map[qy, qx])
                    cand = mp[qy, qx]
                    ok = need & (r < 0.1) & (cand[..., 1] >= 0) & (r < best)
                    best = np.where(ok, r, best)
                    pick = np.where(ok[..., None], cand, pick)
                got = need & np.isfinite(best)
                mf = np.where(got[..., None], pick, mf)
                filled |= got
                need &= ~got
            m = mf
            depth_w = np.where(hidden, 1.0, depth_w)  # the fill replaces the depth rejection
        peak = proxy.max(-1)
        fade = np.clip((1.0 - peak) / 0.05, 0, 1)
        y = srgb_dec(proxy) @ LUMA
        lo, hi = neighbourhood(y, lambda s: s.min(0)), neighbourhood(y, lambda s: s.max(0))
        nearest = np.clip(m[..., 1], lo, hi)
        weight = fade * np.exp(-np.abs(np.log2((nearest + E) / (np.maximum(m[..., 1], 0) + E))) / tol)
        if not fill_gate:
            weight = np.where(filled, fade, weight)
        weight = np.where(inside & (m[..., 1] >= 0), weight, 0.0) * depth_w
        r = 1.0 + (m[..., 0] - 1.0) * weight
        before = srgb_enc(y)
        after = srgb_enc(np.clip(y * r, 0, 1))
        r = np.where(y > 1e-5, after / np.maximum(before, 1e-5), r)
        r = np.where(np.isfinite(r), r, 1.0)
        out = np.where(proxy >= 1.0, proxy, proxy * r[..., None])
        return out


def lum(rgb):
    return srgb_dec(rgb) @ LUMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq'); ap.add_argument('maps'); ap.add_argument('out')
    ap.add_argument('--lags', default='0,2,4,6')
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--count', type=int, default=100000)
    ap.add_argument('--depth', default='3x3')
    ap.add_argument('--tol', type=float, default=0.3)
    ap.add_argument('--dsample', type=float, default=0.0, help='depth-aware map sampling: relative depth scale (0 = plain bilinear)')
    ap.add_argument('--fill', default='off', help='off | gate | nogate: disocclusion fill (luminance gate on filled pixels or not)')
    ap.add_argument('--scale', default='uv', help='uv (vectors x frame size) or px')
    ap.add_argument('--sheet', type=int, default=-1, help='frame index for an image sheet')
    ap.add_argument('--crop', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rp = Replay(a.seq, a.maps, a.first, a.count)
    lags = [int(x) for x in a.lags.split(',')]
    n_frames = len(rp.files)
    w, h = rp.frame(0)[0].shape[1], rp.frame(0)[0].shape[0]
    scale = (w, h) if a.scale == 'uv' else (1.0, 1.0)
    start = max(lags) + 1
    stats = {L: {'err': [], 'err_mv': [], 'err_hid': [], 'strength': [], 'flicker': []} for L in lags}
    prev_edit = {}
    sheet_tiles = []
    for n in range(start, n_frames):
        proxy, _, mv, _ = rp.frame(n)
        yin = lum(proxy)
        truth = rp.resolve(n, 0, scale, a.tol, a.depth)
        e_truth = np.log2(lum(truth) + E) - np.log2(yin + E)
        moving = np.hypot(mv[..., 0] * scale[0], mv[..., 1] * scale[1]) > 0.5
        moving = neighbourhood(moving.astype(np.float32), lambda s: s.max(0)) > 0
        for L in lags:
            out = truth if L == 0 else rp.resolve(n, L, scale, a.tol, a.depth, a.fill != 'off', a.fill == 'gate', a.dsample)
            e = np.log2(lum(out) + E) - np.log2(yin + E)
            err = np.abs(e - e_truth)
            st = stats[L]
            st['err'].append(err.mean())
            if moving.any():
                st['err_mv'].append(err[moving].mean())
            if L > 0 and rp.hidden.any():
                st['err_hid'].append(err[rp.hidden].mean())
            st['strength'].append(np.abs(e).mean())
            if L in prev_edit:
                still = ~moving
                st['flicker'].append(np.abs(e - prev_edit[L])[still].mean())
            prev_edit[L] = e
            if n == a.sheet:
                sheet_tiles.append((f'lag {L}', out, err))
        if n == a.sheet:
            sheet_tiles.insert(0, ('input', proxy, None))
    print(f'frames {start}..{n_frames - 1} ({w}x{h}), depth test {a.depth}, tol {a.tol}, fill {a.fill}')
    print(f'{"lag":>4} {"err all":>8} {"err moving":>11} {"err uncov":>10} {"strength":>9} {"flicker":>8}   (log2 luminance x1000)')
    for L in lags:
        st = stats[L]
        f = lambda v: f'{np.mean(v) * 1000:8.2f}' if v else '     n/a'
        print(f'{L:>4} {f(st["err"])} {f(st["err_mv"]):>11} {f(st["err_hid"]):>10} {f(st["strength"]):>9} {f(st["flicker"])}')
    if sheet_tiles:
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = [int(v) for v in a.crop.split(',')] if a.crop else (0, 0, w, h)
        tw, th = x1 - x0, y1 - y0
        cols = len(sheet_tiles)
        img = Image.new('RGB', (cols * tw, 2 * th))
        dr = ImageDraw.Draw(img)
        for k, (name, rgb, err) in enumerate(sheet_tiles):
            img.paste(Image.fromarray((np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8)), (k * tw, 0))
            if err is not None:
                heat = np.clip(err[y0:y1, x0:x1] * 4, 0, 1)
                img.paste(Image.fromarray((heat * 255).astype(np.uint8)).convert('RGB'), (k * tw, th))
            dr.text((k * tw + 6, 6), name, fill='yellow')
        img.save(os.path.join(a.out, f'sheet_{a.sheet:04d}.png'))
        print('sheet', os.path.join(a.out, f'sheet_{a.sheet:04d}.png'))


if __name__ == '__main__':
    main()
