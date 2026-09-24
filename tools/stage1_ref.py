#!/usr/bin/env python3
"""
tools/stage1_ref.py — Ground-truth Python reference for Stage 1 staging & quantisation kernels.

Covers:
1. rgba8_to_f32x4: Unpacks RGBA8 UNORM to linear contiguous float32.
2. h2e4m3: Exact IEEE 754 float16 -> FP8 E4M3FN quantiser with round-to-nearest
   ties-to-even (RN) and saturation-to-finite (satfinite), matching NVIDIA
   cvt.rn.satfinite.e4m3x2.f16x2.
3. mpcubic_silu: Bit-exact MpCubicSilu f16 activation using exact Fractions
   and single rounding per FMA operation (__hfma2).
4. normalise_input: (x - 0.5) * (2.0 * scale) + dither in f32 -> f16.

Can run standalone to generate lookup tables and verify GPU dumps.
"""

import sys
import struct
import math
from fractions import Fraction
import numpy as np


# ---------------------------------------------------------------------------
# 1. RGBA8 UNORM -> float32 staging reference
# ---------------------------------------------------------------------------

def rgba8_to_f32x4_ref(byte_data: bytes, w: int, h: int, pitch_bytes: int) -> list[float]:
    """Unpacks RGBA8 UNORM bytes to linear [h * w * 4] float32 list (val / 255.0)."""
    out = [0.0] * (h * w * 4)
    for y in range(h):
        row_offset = y * pitch_bytes
        for x in range(w):
            px_offset = row_offset + x * 4
            out_offset = (y * w + x) * 4
            for c in range(4):
                b = byte_data[px_offset + c]
                out[out_offset + c] = b / 255.0
    return out


# ---------------------------------------------------------------------------
# 2. IEEE 754 float16 -> FP8 E4M3FN (RN-satfinite)
# ---------------------------------------------------------------------------

def e4m3_to_fraction(byte_val: int) -> Fraction | None:
    """Decodes positive E4M3 byte (0..127) to exact Fraction. Returns None for NaN."""
    e = (byte_val >> 3) & 0xF
    m = byte_val & 7
    if e == 0:
        return Fraction(m, 8) * Fraction(1, 64)  # 2^-6
    if e == 15 and m == 7:
        return None  # NaN
    return Fraction(8 + m, 8) * (Fraction(2, 1) ** (e - 7))


def f16_bits_to_fraction(u16: int) -> tuple[int, Fraction | None, bool, bool]:
    """
    Decodes float16 bits into (sign, fraction_mag, is_nan, is_inf).
    fraction_mag is None for NaN/Inf.
    """
    sign = (u16 >> 15) & 1
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x3FF

    if exp == 0x1F:
        if mant != 0:
            return sign, None, True, False
        return sign, None, False, True

    if exp == 0 and mant == 0:
        return sign, Fraction(0, 1), False, False

    if exp == 0:
        # Subnormal float16: (-1)^sign * 2^-14 * (mant / 1024) = 2^-24 * mant
        mag = Fraction(mant, 1024) * Fraction(1, 16384)
    else:
        # Normal float16: (-1)^sign * 2^(exp - 15) * (1 + mant / 1024)
        mag = Fraction(1024 + mant, 1024) * (Fraction(2, 1) ** (exp - 15))

    return sign, mag, False, False


# Precompute all 127 positive representable E4M3FN values and their midpoints
_E4M3_REPR = [(b, e4m3_to_fraction(b)) for b in range(127)]  # b=0..126
_E4M3_MIDPOINTS = []
for i in range(len(_E4M3_REPR) - 1):
    b0, v0 = _E4M3_REPR[i]
    b1, v1 = _E4M3_REPR[i + 1]
    mid = (v0 + v1) / 2
    _E4M3_MIDPOINTS.append((mid, b0, b1))


def quantise_f16_to_e4m3fn(u16: int) -> int:
    """
    Converts a 16-bit IEEE 754 half float representation to 8-bit E4M3FN.
    Semantics match PTX cvt.rn.satfinite.e4m3x2.f16x2:
    - RN ties-to-even.
    - Saturation to finite (satfinite): clamp |x| >= 448.0 and inf to 448.0 (0x7E / 0xFE).
    - NaN preserved as 0x7F / 0xFF.
    - Zero signed: +0.0 -> 0x00, -0.0 -> 0x80.
    """
    sign, mag, is_nan, is_inf = f16_bits_to_fraction(u16)

    if is_nan:
        return (sign << 7) | 0x7F

    if is_inf:
        # Satfinite: infinities saturate to MAX_NORM (448.0)
        return (sign << 7) | 0x7E

    if mag == 0:
        return (sign << 7) | 0x00

    # Underflow threshold: midpoint between 0 and smallest subnormal (1/512 = 0.001953125)
    # Midpoint is 1/1024.
    if mag < _E4M3_MIDPOINTS[0][0]:
        return (sign << 7) | 0x00
    if mag == _E4M3_MIDPOINTS[0][0]:
        # Exact tie with 0: 0 has even mantissa (0) -> rounds to 0
        return (sign << 7) | 0x00

    # Saturation threshold: last midpoint is between 416 (b=125, mantissa 5)
    # and 448 (b=126, mantissa 6). Midpoint is 432.
    last_mid, b_prev, b_max = _E4M3_MIDPOINTS[-1]
    if mag >= 448:
        return (sign << 7) | 0x7E
    if mag > last_mid:
        return (sign << 7) | 0x7E
    if mag == last_mid:
        # Tie between 416 and 448: 448 has mantissa 6 (even), 416 has 5 (odd).
        # Rounds to 448 (b=126 = 0x7E).
        return (sign << 7) | 0x7E

    # Binary search through midpoints
    low = 0
    high = len(_E4M3_MIDPOINTS) - 1
    # We know mag >= _E4M3_MIDPOINTS[0] and mag < _E4M3_MIDPOINTS[-1]
    while low <= high:
        mid_idx = (low + high) // 2
        mid_val, b0, b1 = _E4M3_MIDPOINTS[mid_idx]
        if mag == mid_val:
            # Exact tie! Pick the one with even mantissa (bit 0 == 0)
            m0 = b0 & 7
            m1 = b1 & 7
            chosen = b0 if (m0 & 1 == 0) else b1
            return (sign << 7) | chosen
        elif mag < mid_val:
            high = mid_idx - 1
        else:
            low = mid_idx + 1

    # low is the index of the first midpoint > mag.
    chosen = _E4M3_MIDPOINTS[low][1]
    return (sign << 7) | chosen


# Build complete 65536-entry reference LUT
def generate_e4m3_lut() -> bytes:
    table = bytearray(65536)
    for u in range(65536):
        table[u] = quantise_f16_to_e4m3fn(u)
    return bytes(table)


# ---------------------------------------------------------------------------
# 3. MpCubicSilu activation reference (Fraction-exact single-rounding FMA)
# ---------------------------------------------------------------------------

def float_to_f16_bits(val: float) -> int:
    """Converts a Python float to 16-bit IEEE 754 half float representation with proper overflow."""
    if math.isnan(val):
        return 0x7E00
    h = np.float16(val)
    return int(h.view(np.uint16))


def f16_bits_to_float(bits: int) -> float:
    return float(np.uint16(bits).view(np.float16))


def round_fraction_to_f16_bits(frac: Fraction) -> int:
    """
    Rounds an exact Fraction to IEEE 754 half-precision float bits (RN ties-to-even).
    """
    if frac == 0:
        return 0x0000
    val_f64 = float(frac)
    return float_to_f16_bits(val_f64)


# Exact MpCubicSilu constants
C0_F16 = float_to_f16_bits(-0.055908203125)  # 0xAB28
C1_F16 = float_to_f16_bits(0.447265625)       # 0x3728
C2_F16 = float_to_f16_bits(0.89453125)        # 0x3B28
FOUR_F16 = float_to_f16_bits(4.0)             # 0x4400
NEGFOUR_F16 = float_to_f16_bits(-4.0)         # 0xC400


def fma_f16(a_bits: int, b_bits: int, c_bits: int) -> int:
    """
    Bit-exact FMA on float16: round_to_f16(a * b + c) with single rounding.
    """
    _, fa, is_nan_a, is_inf_a = f16_bits_to_fraction(a_bits)
    _, fb, is_nan_b, is_inf_b = f16_bits_to_fraction(b_bits)
    _, fc, is_nan_c, is_inf_c = f16_bits_to_fraction(c_bits)

    if is_nan_a or is_nan_b or is_nan_c:
        return 0x7E00  # canonical NaN
    if is_inf_a or is_inf_b or is_inf_c:
        val = f16_bits_to_float(a_bits) * f16_bits_to_float(b_bits) + f16_bits_to_float(c_bits)
        return float_to_f16_bits(val)

    sign_a = -1 if (a_bits & 0x8000) else 1
    sign_b = -1 if (b_bits & 0x8000) else 1
    sign_c = -1 if (c_bits & 0x8000) else 1

    exact_res = (sign_a * fa) * (sign_b * fb) + (sign_c * fc)
    return round_fraction_to_f16_bits(exact_res)


def mpcubic_silu_ref(x_bits: int) -> int:
    """
    Reference MpCubicSilu implementation matching:
      y = clamp(x, -4.0, +4.0)
      t = fma(c0, |y|, c1)
      u = fma(y, t, c2)
      out = x * u
    """
    _, fx, is_nan_x, is_inf_x = f16_bits_to_fraction(x_bits)
    if is_nan_x:
        return 0x7E00  # NaN

    # clamp(x, -4.0, 4.0)
    x_val = f16_bits_to_float(x_bits)
    if x_val > 4.0:
        y_bits = FOUR_F16
    elif x_val < -4.0:
        y_bits = NEGFOUR_F16
    else:
        y_bits = x_bits

    # |y|
    abs_y_bits = y_bits & 0x7FFF

    # t = fma(c0, |y|, c1)
    t_bits = fma_f16(C0_F16, abs_y_bits, C1_F16)

    # u = fma(y, t, c2)
    u_bits = fma_f16(y_bits, t_bits, C2_F16)

    # out = x * u (multiplication rounded to half)
    val = f16_bits_to_float(x_bits) * f16_bits_to_float(u_bits)
    return float_to_f16_bits(val)


# ---------------------------------------------------------------------------
# 4. Normalise input reference
# ---------------------------------------------------------------------------

def normalise_input_ref(x_f32: float, scale: float, dither: float = 0.0) -> int:
    """
    Computes (x - 0.5f) * (2.0f * scale) + dither in float32 and rounds to float16 bits.
    """
    res_f32 = (x_f32 - 0.5) * (2.0 * scale) + dither
    return float_to_f16_bits(res_f32)


# ---------------------------------------------------------------------------
# CLI / Validation routines
# ---------------------------------------------------------------------------

def verify_gpu_dump(dump_path: str, kind: str) -> bool:
    """Verifies an exhaustive 65,536-entry dump from the GPU executable."""
    with open(dump_path, "rb") as f:
        data = f.read()

    if kind == "h2e4m3":
        if len(data) != 65536:
            print(f"[FAIL] {dump_path}: expected 65536 bytes, got {len(data)}")
            return False
        ref = generate_e4m3_lut()
        mismatches = 0
        first_diff = None
        for u in range(65536):
            got = data[u]
            want = ref[u]
            if got != want:
                mismatches += 1
                if first_diff is None:
                    first_diff = (u, got, want)
        if mismatches == 0:
            print(f"[PASS] h2e4m3: 65536/65536 exact bit matches vs Python reference!")
            return True
        else:
            u, got, want = first_diff
            _, mag, _, _ = f16_bits_to_fraction(u)
            print(f"[FAIL] h2e4m3: {mismatches}/65536 mismatches. First at u16=0x{u:04X} ({float(mag) if mag else None}): got 0x{got:02X}, want 0x{want:02X}")
            return False

    elif kind == "mpcubic":
        if len(data) != 65536 * 2:
            print(f"[FAIL] {dump_path}: expected 131072 bytes, got {len(data)}")
            return False
        gpu_words = struct.unpack("<65536H", data)
        mismatches = 0
        nan_cases = 0
        first_diff = None
        def is_nan_bits(bits: int) -> bool:
            return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0

        for u in range(65536):
            got = gpu_words[u]
            want = mpcubic_silu_ref(u)
            if is_nan_bits(got) and is_nan_bits(want):
                nan_cases += 1
                continue
            if got != want:
                mismatches += 1
                if first_diff is None:
                    first_diff = (u, got, want)
        if mismatches == 0:
            print(f"[PASS] mpcubic_silu2: 65536/65536 verified exact vs Python reference ({nan_cases} NaN inputs confirmed)!")
            return True
        else:
            u, got, want = first_diff
            print(f"[FAIL] mpcubic_silu2: {mismatches}/65536 mismatches. First at u16=0x{u:04X}: got 0x{got:04X} ({f16_bits_to_float(got)}), want 0x{want:04X} ({f16_bits_to_float(want)})")
            return False
    else:
        raise ValueError(f"Unknown verification kind: {kind}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if len(sys.argv) > 2 and sys.argv[1] == "--verify":
        kind = sys.argv[2]
        path = sys.argv[3]
        success = verify_gpu_dump(path, kind)
        sys.exit(0 if success else 1)
    else:
        print("Self-testing Python reference functions...")
        # Self-test h2e4m3 sanity
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(0.0)) == 0x00
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(-0.0)) == 0x80
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(1.0)) == 0x38  # 1.0 = 0x38
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(448.0)) == 0x7E
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(500.0)) == 0x7E # satfinite
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(-448.0)) == 0xFE
        assert quantise_f16_to_e4m3fn(float_to_f16_bits(-600.0)) == 0xFE # satfinite
        assert quantise_f16_to_e4m3fn(0x7C00) == 0x7E  # +inf -> 0x7E
        assert quantise_f16_to_e4m3fn(0xFC00) == 0xFE  # -inf -> 0xFE
        assert (quantise_f16_to_e4m3fn(0x7E00) & 0x7F) == 0x7F  # NaN -> 0x7F / 0xFF

        # Self-test mpcubic constants
        print("c0 bit pattern:", hex(C0_F16))
        print("c1 bit pattern:", hex(C1_F16))
        print("c2 bit pattern:", hex(C2_F16))
        silu_0 = f16_bits_to_float(mpcubic_silu_ref(float_to_f16_bits(0.0)))
        silu_1 = f16_bits_to_float(mpcubic_silu_ref(float_to_f16_bits(1.0)))
        print(f"MpCubicSilu(0.0) = {silu_0} (expected 0.0)")
        print(f"MpCubicSilu(1.0) = {silu_1} (expected ~1.286)")
        print("All Python reference self-tests PASSED.")
