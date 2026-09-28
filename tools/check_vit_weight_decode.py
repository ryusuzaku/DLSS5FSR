"""Compare fast ViT decoding with every address and all 32 source records."""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import vit_weight_decode as fast
import native_vit_linear_reference as original
import native_vit_qkv_reference as qkv_original
from native_split_reference import bits


def exact(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError('decoded arrays differ in shape, dtype or bits')


def run():
    address_count = 0
    for inputs, outputs in ((1024, 4096), (4096, 1024), (1024, 1024)):
        # Unique values prove each address, independently of repeated FP8 codes.
        raw = np.arange(inputs * outputs, dtype=np.uint32)
        ib, ob = inputs.bit_length() - 1, outputs.bit_length() - 1
        reference = np.empty((outputs, inputs), np.uint32)
        oi = bits(raw.size, [6, 3, 9, 7, 8] + list(range(10, ob + 5)))
        ii = bits(raw.size, [0, 1, 2, 4, 5] + list(range(ob + 5, ib + ob)))
        reference[oi, ii] = raw
        exact(fast._logical_codes(raw, inputs, outputs), reference)
        address_count += raw.size
    exact(fast._FP8, fast.e4m3fn(np.arange(256, dtype=np.uint8)))
    assert np.signbit(fast._FP8[128]) and not np.signbit(fast._FP8[0])

    records = {v['name']: v for v in json.loads(
        (fast.ROOT / 'dlss5-analysis/model.resolved.json').read_text())['tensors']}
    timings = dict(original=0., optimized=0.)
    verified = []
    for block in range(31, 39):
        for layer in (0, 1, 2, 4):
            record = records[f'block{block}.layer{layer}.layer']
            path = fast.ROOT / 'dlss5-analysis/tensors' / f"tensor_{record['index']:03d}.bin"
            raw = path.read_bytes()
            start = time.perf_counter()
            if layer == 0:
                want = [original.unpack_expand(path)]
            elif layer == 2:
                matrices, scales = qkv_original.unpack(path)
                want = [*matrices, scales]
            else:
                want = original.unpack_residual(path, 4096 if layer == 1 else 1024)
            timings['original'] += time.perf_counter() - start
            start = time.perf_counter()
            if layer == 0:
                got = [fast.matrix(np.frombuffer(raw[:4194304], np.uint8), 1024, 4096)]
            elif layer == 2:
                matrices, scales = fast.unpack_qkv(raw)
                got = [*matrices, scales]
            else:
                got = fast.unpack_residual(raw, 4096 if layer == 1 else 1024)
            timings['optimized'] += time.perf_counter() - start
            assert len(got) == len(want)
            for a, b in zip(got, want):
                exact(a, b)
            verified.append(dict(block=block, layer=layer,
                                 source_sha256=hashlib.sha256(raw).hexdigest()))
        print(f'block{block}: all four records byte-exact', flush=True)

    rejects = [lambda: fast.matrix(np.zeros(4, np.uint8), 1024, 1024),
               lambda: fast.matrix(np.zeros(4, np.float32), 1024, 1024),
               lambda: fast.matrix(np.zeros(4, np.uint8), 2, 2),
               lambda: fast.unpack_residual(b'', 4096),
               lambda: fast.unpack_qkv(b''),
               lambda: fast.unpack_qkv(np.full(32, np.nan, '<f4').tobytes() + bytes(3145728))]
    for call in rejects:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid decoder input accepted')
    report = dict(exact_addresses=address_count, exact_fp8_codes=256,
                  exact_records=verified, seconds=timings, invalid_inputs_rejected=len(rejects),
                  comparison='unchanged vendored decoder', original_kernel_executed=False)
    (fast.ROOT / 'build/vit_weight_decode_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**report, 'exact_records': len(verified)}, indent=2))
    return report


if __name__ == '__main__':
    run()
