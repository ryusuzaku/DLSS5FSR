#!/usr/bin/env python3
"""Generate fixtures and test HIP head outer passes against the Python reference.

Run tools/build_head70.sh first, then python tools/check_head70.py.
Exact equality is required, including a rectangular multi-window case, a final
partial RGB thread block, zero products, signed cancellation, and clipping.
The body features are supplied inputs. This is NOT a full-head device test.
"""
from pathlib import Path
import argparse
import subprocess
import numpy as np
import head70_reference as R
import head70_weights as W

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "build/head70_test.exe")
    ap.add_argument("--out", type=Path, default=ROOT / "build/head70_fixtures")
    a = ap.parse_args()
    raw = (ROOT / "dlss5-analysis/tensors/tensor_150.bin").read_bytes()
    stage = (ROOT / "dlss5-analysis/tensors/tensor_001.bin").read_bytes()
    body, sm, ss, coeff, pad = W.extract(raw, stage[W.PAD_AT:W.PAD_AT + 16])
    assert W.repack(body, sm, ss, coeff, pad) == raw
    coeff = coeff[[0, 2, 4]]
    rng = np.random.default_rng(0x70C32)
    cases = [("small", 8, 8, .03125), ("rect", 16, 24, .03125),
             ("zero", 8, 16, .03125), ("clip", 16, 16, 1.),
             ("cancellation", 8, 24, .03125), ("scale_zero", 8, 8, 0.)]
    for name, h, w, scale in cases:
        main_in = rng.uniform(-2, 2, (h // 2, w // 2, 32)).astype(np.float32)
        skip = rng.uniform(-2, 2, (h, w, 32)).astype(np.float32)
        features = R.H(rng.uniform(-8, 8, (h, w, 32)))
        color = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
        cf = coeff.copy()
        if name == "zero":
            main_in.fill(0); skip.fill(0); features.fill(0)
        elif name == "clip":
            features *= 128
            color.reshape(-1)[::3] = -1
            color.reshape(-1)[1::3] = 2
        elif name == "cancellation":
            # Exact f16 inputs span exponents and force signed product truncation.
            features = R.H(np.ldexp(rng.choice([-1., 1.], (h, w, 32)),
                                    rng.integers(-12, 8, (h, w, 32), dtype=np.int32)))
            cf = R.H(rng.uniform(-1, 1, (3, 32)))
            features[..., 1::2] = -features[..., ::2]
            cf[:, 1::2] = cf[:, ::2]
        merged = R.windowise(R.merge(main_in, skip, sm, ss))
        rgb = R.finish(features, color, cf, scale)
        if name == "clip":
            assert np.any(rgb == 0) and np.any(rgb == 1)
        fixture = a.out / name
        fixture.mkdir(parents=True, exist_ok=True)
        arrays = dict(main=main_in, skip=skip, sm=sm, ss=ss, coeff=cf, color=color,
                      features=R.windowise(features), merged=merged, rgb=rgb)
        for key, value in arrays.items():
            assert np.isfinite(value).all(), (name, key)
            np.asarray(value, dtype="<f4").tofile(fixture / (key + ".f32"))
        print(f"{name}: {w}x{h}, input_scale={scale}", flush=True)
        subprocess.run([str(a.exe.resolve()), str(fixture.resolve()), str(w), str(h),
                        str(scale)], check=True, cwd=ROOT / "build")
    print(f"{len(cases)}/{len(cases)} head outer-pass cases passed (12 comparisons)")


if __name__ == "__main__":
    main()
