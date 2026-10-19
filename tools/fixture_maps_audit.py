"""Sanity checks for a fixture tree written by fixture_maps.py apply --role small.

Per role block: are the written values finite, how do their magnitudes compare
with the HT role files, and which payload bytes do the maps never read? In HT
those bytes are padding / unused records; if the small model has data where HT
has none, its layout differs and the maps would miss it.

Usage: fixture_maps_audit.py MAPS.npz SMALL_TENSORS SMALL_FIXTURES
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixture_maps import HT, HT_TO_SMALL, OFF, mirror, payload, tensor_index  # noqa: E402

maps, small_dir, fixtures = sys.argv[1:4]
z = np.load(maps)
meta = json.loads(str(z['meta']))
ht_idx, sm_idx = tensor_index(HT), tensor_index(small_dir)
used = {}
problems = 0
for name, m in meta.items():
    codes = z[name]
    c = codes[codes >= 0]
    width = np.array([1, 2, 4])[c >> 40]
    for b, f in zip(m['blocks'], m['files']):
        if b not in HT_TO_SMALL:
            continue
        mask = used.setdefault(b, np.zeros(sum(t['payload_size'] for t in ht_idx[b]), bool))
        for w in (1, 2, 4):
            o = (c & OFF)[width == w]
            for k in range(w):
                mask[o + k] = True
        ht = np.fromfile(f, np.float32)
        sm = np.fromfile(mirror(fixtures, f), np.float32)
        if not np.all(np.isfinite(sm)):
            print('NONFINITE', f); problems += 1
        a, s = np.abs(ht).mean(), np.abs(sm).mean()
        ratio = s / a if a else float('inf') if s else 1.0
        if not 0.25 < ratio < 4:
            print(f'MAGNITUDE {ratio:6.2f}x {f}'); problems += 1
print('blocks: HT unread bytes nonzero / small unread bytes nonzero (of unread)')
for b in sorted(used):
    mask = used[b]
    ht = np.frombuffer(payload(HT, ht_idx[b]), np.uint8)
    sm = np.frombuffer(payload(small_dir, sm_idx[HT_TO_SMALL[b]]), np.uint8)
    un = ~mask
    hn, sn = int((ht[un] != 0).sum()), int((sm[un] != 0).sum())
    flag = '' if sn <= max(hn * 2, 64) else '   <-- small has data the maps do not read'
    problems += bool(flag)
    print(f'  HT {b:2} -> small {HT_TO_SMALL[b]:2}: unread {int(un.sum()):>8} B, nonzero HT {hn:>8}, small {sn:>8}{flag}')
print('problems:', problems)
