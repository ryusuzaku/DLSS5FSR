"""Recover how each resident weight fixture file is gathered from NVIDIA's tensors,
and write fixture files from any tensor set with the same per-block layouts.

The engine reads its weights from fixture folders (one per original block) that
earlier per-stage tools wrote from the WEIGHTS_HT tensors. Every value of such a
file is one element of its block's tensors (the block's payloads concatenated in
name order), decoded as E4M3, FP16 or FP32. This recovers the gather maps from
the data instead of re-running those tools:

1. Kinds with several blocks (the same file in blocks of the same layout): stack
   a file's values over the blocks; each element's signature is (in practice)
   unique among the tensor positions, matching signatures gives the map.
2. Kinds left ambiguous (a single block, or repeated values): fit a structured
   map: a resolved kind's map shifted by a constant offset (same file in a bigger
   payload) or by a few offsets in runs (something inserted in the middle); a
   bit-linear one, position = base + sum_b c_b * bit_b(i), each c_b voted from the
   candidate positions of element pairs; or a bit permutation (c_b powers of two)
   found depth-first.
A map is accepted only when it reproduces the HT file byte for byte for every
block of its kind; the rest is reported, never guessed.

  derive OPENED_LIST MAPS.npz [--only TEXT]
         OPENED_LIST: the files the engine opens, one per line: build the engine
         DLL with -include tools/fopen_log.h, run resident_engine_full_test with
         FOPEN_LOG=list.txt, then sort -u. (S361: 133 maps for all 570 files; the
         HT tensors through them give every file byte for byte.)
  apply  MAPS.npz TENSORS OUT [--role small] [--check] [--config CFG]
         TENSORS: an extract_dlssnr.py output folder (weights.json + tensors/).
         Files go to OUT/<drive>_drive/<original path>. --role small writes only
         the blocks the 310.9.16 small model has, from its role blocks, plus its
         block lists and OUT/engine_config_small.txt (from CFG's options);
         --check compares every written file with the original.
"""
import argparse
import collections
import json
import os
import re

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
HT = os.path.join(ROOT, 'dlss5-analysis')
# 310.9.16 small model block -> the HT block whose role (layer types, payload
# sizes) it has; every small record has exactly the size of its role's record.
SMALL_ROLE = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9, 6: 14, 7: 15, 8: 22, 9: 23, 10: 30,
              **{b: b + 20 for b in range(11, 19)}, 19: 39, 20: 40, 21: 47, 22: 48, 23: 55,
              24: 56, 25: 61, 26: 62, 27: 65, 28: 66, 29: 69, 30: 70}
HT_TO_SMALL = {h: s for s, h in SMALL_ROLE.items()}
F8, F16, F32 = 0, 1, 2


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * m / 8 * 2.0 ** -6
        else:
            t[b] = s * (1 + m / 8) * 2.0 ** (e - 7)
    return t


E4M3 = e4m3_table()


def tensor_index(folder):
    meta = json.load(open(os.path.join(folder, 'weights.json')))['tensors']
    by_block = collections.defaultdict(list)
    for t in meta:
        by_block[int(re.match(r'block(\d+)\.', t['name']).group(1))].append(t)
    for b in by_block:
        by_block[b].sort(key=lambda t: t['name'])
    return by_block


def payload(folder, block_tensors):
    return b''.join(open(os.path.join(folder, 'tensors', f"tensor_{t['index']:03d}.bin"), 'rb').read()
                    for t in block_tensors)


def decode(raw, dtype, offsets):
    """Values at byte offsets (a numpy int array) of one dtype."""
    a = np.frombuffer(raw, np.uint8)
    if dtype == F8:
        return E4M3[a[offsets]]
    width = 2 if dtype == F16 else 4
    idx = offsets[:, None] + np.arange(width)
    b = np.ascontiguousarray(a[idx])
    return b.view(np.float16 if dtype == F16 else np.float32).ravel().astype(np.float32)


def all_positions(raw):
    """Every candidate decoding: (values, codes) with code = dtype << 40 | byte offset."""
    a = np.frombuffer(raw, np.uint8)
    vals, codes = [E4M3[a]], [np.arange(a.size, dtype=np.int64)]
    h = np.frombuffer(raw[:len(raw) // 2 * 2], np.float16).astype(np.float32)
    vals.append(h); codes.append((F16 << 40) | np.arange(h.size, dtype=np.int64) * 2)
    for off in (0, 2):
        n = (len(raw) - off) // 4
        vals.append(np.frombuffer(raw[off:off + n * 4], np.float32).copy())
        codes.append((F32 << 40) | (off + np.arange(n, dtype=np.int64) * 4))
    return np.concatenate(vals), np.concatenate(codes)


def apply_codes(raw, codes):
    out = np.zeros(codes.size, np.float32)
    for d in (F8, F16, F32):
        m = (codes >= 0) & ((codes >> 40) == d)
        if m.any():
            out[m] = decode(raw, d, codes[m] & ((1 << 40) - 1))
    return out


def block_of(path):
    p = path.replace('\\', '/')
    for pat in (r'/block(\d+)-8x8', r'/block(\d+)_head', r'/vit_block(\d+)', r'/entry(\d+)', r'/prefix(\d+)',
                r'/references/a/block(\d+)/', r'/a/block(\d+)/'):
        m = re.search(pat, p)
        if m:
            return int(m.group(1))
    if '/references/a/head70/' in p:
        return 70
    m = re.search(r'front_end_check2/captured/block(\d)/', p)
    if m:
        return int(m.group(1))
    if p.endswith('front_end_check2/captured/stem_weights.f32'):
        return 0
    if p.endswith('front_end_check2/captured/matrix.f32'):
        return 4
    m = re.search(r'/encoder(\d+)/a/downsample/', p)
    if m:
        return {5: 8, 9: 14, 15: 22}[int(m.group(1))]
    return None


def kind_key(path, sizes):
    """Same file name, file size and block payload sizes: candidates for one map."""
    p = path.replace('\\', '/')
    return f'{p.split("/")[-1]}|{os.path.getsize(path)}|{",".join(map(str, sizes))}'


def where(path):
    return re.sub(r'\d+', 'N', '/'.join(path.replace('\\', '/').split('/')[-4:-1]))


def sig(cols):
    a = np.ascontiguousarray(np.stack(cols, 1).astype(np.float32))
    return a.view(np.dtype((np.void, a.shape[1] * 4))).ravel()


def matches(raws, codes, data):
    """Does the map reproduce every member's file bytes?"""
    used = codes[codes >= 0]
    if used.size:
        off, width = used & ((1 << 40) - 1), np.array([1, 2, 4])[used >> 40]
        if off.min() < 0 or (off + width).max() > min(len(r) for r in raws):
            return False
    return all(np.array_equal(apply_codes(r, codes).view(np.uint32), data[:, k].view(np.uint32))
               for k, r in enumerate(raws))


OFF = (1 << 40) - 1


class Candidates:
    """Tensor positions whose values equal each file element's, over all members."""

    def __init__(self, raws, data):
        codes_all, vals = None, []
        for r in raws:  # same payload sizes, so the same codes for every member
            v, c = all_positions(r)
            vals.append(v)
            codes_all = c if codes_all is None else codes_all
        s = sig(vals)
        del vals
        self.order = np.argsort(s, kind='stable')
        ss = s[self.order]
        want = sig([data[:, k] for k in range(data.shape[1])])
        self.lo = np.searchsorted(ss, want, 'left')
        self.hi = np.searchsorted(ss, want, 'right')
        self.codes_all = codes_all
        self.count = self.hi - self.lo
        self.n = data.shape[0]

    def __call__(self, i, dtype=None):
        c = self.codes_all[self.order[self.lo[i]:self.hi[i]]]
        if dtype is not None:
            c = c[(c >> 40) == dtype] & OFF
        return c

    def first(self):
        return self.codes_all[self.order[np.minimum(self.lo, self.order.size - 1)]]


def fit_offset(template, cand):
    """Template map + one constant byte offset: intersect over the rarest elements."""
    ok = np.flatnonzero((template >= 0) & (cand.count > 0))
    anchors = ok[np.argsort(cand.count[ok], kind='stable')][:64]
    deltas = None
    for i in anchors:
        d = cand(i, int(template[i] >> 40)) - (template[i] & OFF)
        deltas = d if deltas is None else np.intersect1d(deltas, d)
        if deltas.size <= 1:
            break
    for delta in ([] if deltas is None else deltas[:4]):
        codes = template.copy()
        m = codes >= 0
        codes[m] = (codes[m] >> 40 << 40) | ((codes[m] & OFF) + int(delta))
        yield codes


def fit_bits(cand):
    """position = base + sum_b c_b * bit_b(i), one dtype; each c_b voted from rare pairs."""
    n = cand.n
    bits = max(1, int(np.ceil(np.log2(n))))
    idx = np.arange(n)
    for dtype in (F8, F16, F32):
        coeff = []
        for b in range(bits):
            i = idx[((idx >> b) & 1) == 0]
            i = i[(i | (1 << b)) < n]
            if not i.size:
                coeff.append(0)
                continue
            j = i | (1 << b)
            cost = np.minimum(cand.count[i], cand.count[j]) * 1e6 + np.maximum(cand.count[i], cand.count[j])
            pick = i[np.argsort(cost, kind='stable')][:48]
            diffs = []
            for a in pick:
                p, q = cand(int(a), dtype), cand(int(a) | (1 << b), dtype)
                if p.size == 0 or q.size == 0 or p.size * q.size > 4_000_000:
                    continue
                diffs.append(np.unique((q[:, None] - p[None, :]).ravel()))
            if not diffs:
                coeff = None
                break
            u, c = np.unique(np.concatenate(diffs), return_counts=True)
            coeff.append(int(u[np.argmax(c)]))
        if coeff is None:
            continue
        part = sum(((idx >> b) & 1) * c for b, c in enumerate(coeff))
        rare = np.argsort(cand.count, kind='stable')[:64]
        bases = None
        for a in rare:
            o = cand(int(a), dtype) - part[a]
            bases = o if bases is None else np.intersect1d(bases, o)
            if bases.size <= 1:
                break
        for base in ([] if bases is None else bases[:4]):
            yield (dtype << 40) | (int(base) + part).astype(np.int64)


def fit_piecewise(template, cand, raw, data, max_segments=4):
    """Template map split into a few runs (in template offset order), each shifted by
    its own constant: a body whose payload has something inserted in the middle."""
    m = np.flatnonzero(template >= 0)
    srt = m[np.argsort(template[m] & OFF, kind='stable')]
    deltas = set()
    for chunk in np.array_split(srt, 32):
        anchors = chunk[np.argsort(cand.count[chunk], kind='stable')][:24]
        d = None
        for i in anchors:
            if cand.count[i] == 0:
                continue
            e = cand(i, int(template[i] >> 40)) - (template[i] & OFF)
            d = e if d is None else np.intersect1d(d, e)
            if d.size <= 1:
                break
        if d is not None:
            deltas.update(int(x) for x in d[:2])
    deltas = sorted(deltas)
    if not deltas or len(deltas) > 16:
        return
    want = data[:, 0].view(np.uint32)
    valid = []
    for d in deltas:
        c = template.copy()
        c[m] = (c[m] >> 40 << 40) | ((c[m] & OFF) + d)
        off = c[m] & OFF
        ok = np.zeros(template.size, bool)
        inside = (off >= 0) & (off + np.array([1, 2, 4])[c[m] >> 40] <= len(raw))
        vals = np.zeros(m.size, np.float32)
        vals[inside] = apply_codes(raw, c[m][inside])
        ok[m] = inside & (vals.view(np.uint32) == want[m])
        valid.append(ok)
    valid = np.array(valid)
    # Greedy runs along the template order: keep a shift while it fits.
    choice = np.empty(srt.size, int)
    cur, segments = -1, 0
    for t, i in enumerate(srt):
        if cur < 0 or not valid[cur, i]:
            ks = np.flatnonzero(valid[:, i])
            if ks.size == 0:
                return
            run = [np.argmin(np.append(valid[k, srt[t:]], False)) for k in ks]
            cur = int(ks[int(np.argmax(run))])
            segments += 1
            if segments > max_segments:
                return
        choice[t] = cur
    codes = template.copy()
    d = np.array(deltas)[choice]
    codes[srt] = (codes[srt] >> 40 << 40) | ((codes[srt] & OFF) + d)
    yield codes


def fit_perm(cand, raw, data):
    """position = pos(a) + sum_b (bit_b(i) - bit_b(a)) * 2^p(b) * width: a bit permutation
    of the index (a gather of a reshaped, transposed array), anchored on the element a
    with the rarest value. Depth-first over the bits, every partial choice checked on
    the elements its bits reach."""
    n = cand.n
    bits = int(np.ceil(np.log2(n)))
    if 1 << bits != n:
        return
    want = data[:, 0].view(np.uint32)
    for dtype, width in ((F8, 1), (F16, 2), (F32, 4)):
        span = int(np.ceil(np.log2(max(len(raw) // width, 2))))
        counts = np.array([cand(int(i), dtype).size for i in np.argsort(cand.count, kind='stable')[:256]])
        order = np.argsort(cand.count, kind='stable')[:256]
        a = int(order[np.argmin(np.where(counts > 0, counts, 1 << 60))])
        anchors = cand(a, dtype)
        if anchors.size == 0 or anchors.size > 4096:
            continue
        for pa in anchors:
            chosen, used = [], set()

            def ok_upto(b):
                s = np.arange(1 << (b + 1)) if b < 16 else np.random.default_rng(b).integers(0, 1 << (b + 1), 65536)
                i = a ^ s
                part = np.zeros(i.size, np.int64)
                for k, c in enumerate(chosen):
                    part += (((i >> k) & 1) - ((a >> k) & 1)) * c
                pos = int(pa) + part
                if pos.min() < 0 or pos.max() + width > len(raw):
                    return False
                v = apply_codes(raw, (dtype << 40) | pos)
                return np.array_equal(v.view(np.uint32), want[i])

            def dfs(b):
                if b == bits:
                    return True
                for k in range(span):
                    if k in used:
                        continue
                    chosen.append((1 << k) * width); used.add(k)
                    if ok_upto(b) and dfs(b + 1):
                        return True
                    chosen.pop(); used.discard(k)
                return False

            if dfs(0):
                idx = np.arange(n)
                part = sum((((idx >> k) & 1) - ((a >> k) & 1)) * c for k, c in enumerate(chosen))
                yield (dtype << 40) | (int(pa) + part).astype(np.int64)
                return


def derive(opened, out, only=None):
    files = [l.strip() for l in open(opened) if l.strip() and (not only or only in l)]
    ht = tensor_index(HT)
    kinds = collections.defaultdict(list)
    for f in files:
        b = block_of(f)
        if b is None or f.endswith('noise.f32'):
            print('not a weight file:', f)
            continue
        kinds[kind_key(f, [t['payload_size'] for t in ht[b]])].append((b, f))
    raws = {}

    def raw(b):
        if b not in raws:
            raws[b] = payload(HT, ht[b])
        return raws[b]

    def by_signature(members):
        blocks = [b for b, _ in members]
        data = np.stack([np.fromfile(f, np.float32) for _, f in members], 1)
        rws = [raw(b) for b in blocks]
        cand = Candidates(rws, data)
        miss = int((cand.count == 0).sum())
        zero = (len(blocks) > 2) & np.all(data == 0, 1) & ~np.signbit(data).any(1)
        codes = np.where(zero, -1, cand.first())
        info = dict(blocks=blocks, files=[f for _, f in members], n=int(data.shape[0]), miss=miss,
                    ambiguous=int(((cand.count != 1) & ~zero).sum()))
        good = not miss and np.all((cand.count == 1) | zero) and matches(rws, codes, data)
        return good, codes, info, data, cand

    resolved, pending = {}, []
    for key, members in sorted(kinds.items(), key=lambda kv: kv[1][0][0]):
        members.sort()
        groups = [members]
        good, codes, info, data, cand = by_signature(members)
        if not good and len({where(f) for _, f in members}) > 1:
            # A shared name and size, but more than one layout: split by folder kind.
            split = collections.defaultdict(list)
            for b, f in members:
                split[where(f)].append((b, f))
            groups = list(split.values())
        for g in groups:
            if g is not members:
                good, codes, info, data, cand = by_signature(g)
            name = f"{where(g[0][1])}|{key}"
            if good:
                resolved[name] = (codes, info, 'signature')
            else:
                pending.append((name, g))
        raws.clear()
    # Structured fits for the rest, against single blocks or ambiguous families.
    for name, members in pending:
        good, codes, info, data, cand = by_signature(members)
        rws = [raw(b) for b, _ in members]
        how = None
        if not info['miss']:
            fname, size = name.split('|')[1:3]
            templates = [(k, v[0]) for k, v in resolved.items() if k.split('|')[1:3] == [fname, size]]
            for tkey, tcodes in templates:
                for c in fit_offset(tcodes, cand):
                    if matches(rws, c, data):
                        how = ('offset from ' + tkey.split('|')[0], c)
                        break
                if how:
                    break
            if not how and len(rws) == 1:
                for tkey, tcodes in templates:
                    for c in fit_piecewise(tcodes, cand, rws[0], data):
                        if matches(rws, c, data):
                            how = ('pieces of ' + tkey.split('|')[0], c)
                            break
                    if how:
                        break
            if not how:
                for c in fit_bits(cand):
                    if matches(rws, c, data):
                        how = ('bit-linear', c)
                        break
            if not how:
                for c in fit_perm(cand, rws[0], data):
                    if matches(rws, c, data):
                        how = ('bit permutation', c)
                        break
        if how:
            resolved[name] = (how[1], info, how[0])
        else:
            print(f"UNRESOLVED {name.split('|')[0]}/{name.split('|')[1]} blocks {info['blocks']} n={info['n']} "
                  f"miss={info['miss']} ambiguous={info['ambiguous']}")
        raws.clear()
    meta, arrays = {}, {}
    for k, (key, (codes, info, how)) in enumerate(sorted(resolved.items())):
        arrays[f'm{k}'] = codes
        meta[f'm{k}'] = dict(key=key, how=how, **info)
        print(f"{how[:44]:44} {key.split('|')[0]}/{key.split('|')[1]} blocks {info['blocks']}")
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    files_done = sum(len(m['files']) for m in meta.values())
    print(f'{len(resolved)} maps covering {files_done} of {sum(len(v) for v in kinds.values())} files')


def mirror(out, path):
    """Where apply writes the copy of an original fixture path."""
    drive, rest = os.path.splitdrive(path.replace('\\', '/'))
    return os.path.normpath(os.path.join(out, drive.rstrip(':').lower() + '_drive', rest.lstrip('/')))


# The small model's block lists: (original list, [(shift, HT block folder)]). Shift
# codes in our convention (1 = x, 2 = y, 3 = both), from NVIDIA's 310.9.16 layer
# table; each level keeps its first block and its last (down / outview) block.
CAPTURED = 'P:/DLSS5FSR/build/resident_encoder8_check/captured'
SMALL_LISTS = [
    ('encoder5_blocks.txt', [(0, 'encoder5/a/block5'), (3, 'encoder5/a/block8')]),
    ('encoder9_blocks.txt', [(0, 'encoder9/a/block9'), (3, 'encoder9/a/block14')]),
    ('encoder15_blocks.txt', [(0, 'encoder15/a/block15'), (3, 'encoder15/a/block22')]),
    ('decoder61/fixed_control/blocks.txt',
     [(0, 'C:/Users/tomik/DLSS5FSR-build-offload/scene_preview_c512_worker_check/candidate/encoder30/block23-8x8-s0'),
      (3, 'C:/Users/tomik/DLSS5FSR-build-offload/scene_preview_c512_worker_check/candidate/encoder30/block30-8x8-s2')]),
    ('decoder61/decoder_control/blocks.txt',
     [(1, 'decoder47/references/a/decoder/block40-8x8-s0'), (2, 'decoder47/references/a/decoder/block47-8x8-s2')]),
    ('decoder61/c256_blocks.txt', [(1, 'decoder55/references/a/block48'), (2, 'decoder55/references/a/block55')]),
    ('decoder61/c128_blocks.txt', [(1, 'decoder61/references/a/block56'), (2, 'decoder61/references/a/block61')]),
    ('decoder65/c64_blocks.txt', [(1, 'decoder65/references/a/block62'), (2, 'decoder65/references/a/block65')]),
    ('head70/c32_blocks.txt', [(1, 'head70/references/a/block66'), (2, 'head70/references/a/block69')]),
]


def small_lists(out, config):
    """Block lists and an engine config pointing at the small fixture tree."""
    for name, entries in SMALL_LISTS:
        dst = mirror(out, f'{CAPTURED}/{name}')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w', newline='\r\n') as fh:
            for shift, folder in entries:
                full = folder if ':' in folder else f'{CAPTURED}/{folder}'
                fh.write(f'{shift}\t{mirror(out, full)}\n')
    vit = mirror(out, f'{CAPTURED}/decoder47/vit_blocks.txt')
    with open(vit, 'w', newline='\r\n') as fh:
        for b in range(31, 39):
            fh.write(mirror(out, f'{CAPTURED}/vit/references/a/vit_block{b}') + '\n')
    # Model-independent inputs the engine also opens: the noise tile, the bridge map.
    for f in ('P:/DLSS5FSR/build/front_end_check2/captured/noise.f32', f'{CAPTURED}/decoder47/bridge_map.i32'):
        os.makedirs(os.path.dirname(mirror(out, f)), exist_ok=True)
        open(mirror(out, f), 'wb').write(open(f, 'rb').read())
    lines = open(config, 'rb').read().decode().splitlines()
    for i in range(24):  # the fixture lines; the options after them stay
        if lines[i].strip() and not lines[i].strip().isdigit():
            lines[i] = mirror(out, lines[i].strip())
    lines = [l for l in lines if not l.startswith('model=')] + ['model=small']
    dst = os.path.join(out, 'engine_config_small.txt')
    with open(dst, 'w', newline='\r\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('lists and', dst, 'written')


def apply(maps, tensors, out, role, check, config=None):
    z = np.load(maps)
    meta = json.loads(str(z['meta']))
    idx = tensor_index(tensors)
    written = same = 0
    for name, m in meta.items():
        codes = z[name]
        for b, f in zip(m['blocks'], m['files']):
            src = HT_TO_SMALL.get(b) if role == 'small' else b
            if src is None:
                continue
            dst = mirror(out, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            values = apply_codes(payload(tensors, idx[src]), codes)
            values.tofile(dst)
            written += 1
            if check:
                ok = open(dst, 'rb').read() == open(f, 'rb').read()
                same += ok
                if not ok:
                    print('DIFFERS', f)
    print(f'{written} files written' + (f', {same} byte-identical to the originals' if check else ''))
    if role == 'small':
        small_lists(out, config)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('derive'); d.add_argument('opened'); d.add_argument('maps')
    d.add_argument('--only', help='only fixture paths containing this text (for testing a fit)')
    a = sub.add_parser('apply'); a.add_argument('maps'); a.add_argument('tensors'); a.add_argument('out')
    a.add_argument('--role', choices=('ht', 'small'), default='ht'); a.add_argument('--check', action='store_true')
    a.add_argument('--config', default='P:/DLSS5FSR/build/resident_engine_check/engine_config_nvidia.txt',
                   help='engine config whose fixture lines --role small rewrites')
    args = ap.parse_args()
    if args.cmd == 'derive':
        derive(args.opened, args.maps, args.only)
    else:
        apply(args.maps, args.tensors, args.out, args.role, args.check, args.config)


if __name__ == '__main__':
    main()
