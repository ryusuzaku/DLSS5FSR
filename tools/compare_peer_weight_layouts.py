#!/usr/bin/env python3
"""Compare public ONNX decoders with the native port's measured bit layouts.

This is a coordinate audit, not a native-kernel oracle. The ONNX decoder was
transcribed to NumPy so the third-party code and models need not be executed.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref/dlss5-port/Development"))
from decode_tinlayout_global import e4m3fn

OUT = ROOT / "build" / "peer_weight_layout_audit.json"


def bits(count, positions):
    source = np.arange(count, dtype=np.int32)
    target = np.zeros(count, dtype=np.int32)
    for destination_bit, source_bit in enumerate(positions):
        target |= ((source >> source_bit) & 1) << destination_bit
    return target


def peer_qmma(raw, offset, n, k, output_block=None, qkv_heads=None):
    """NumPy translation of taowen/dlss5-onnx weights.decode_qmma_matrix."""
    block = output_block or n
    values = e4m3fn(raw[offset:offset + n * k])
    pieces = []
    start = 0
    for first in range(0, n, block):
        width = min(block, n - first)
        panel = values[start:start + width * k].reshape(k // 32, width // 16, 8, 4, 2, 2, 4)
        pieces.append(panel.transpose(1, 4, 2, 0, 5, 3, 6).reshape(width, k))
        start += width * k
    matrix = np.concatenate(pieces).reshape(n, k // 16, 4, 2, 2).transpose(0, 1, 3, 2, 4).reshape(n, k)
    if qkv_heads is not None:
        matrix = matrix.reshape(qkv_heads, 3, 32, k).transpose(1, 0, 2, 3).reshape(n, k)
    return matrix


def native_matrix(raw, offset, n, k, row, col):
    result = np.empty((n, k), np.float32)
    result[row, col] = e4m3fn(raw[offset:offset + n * k])
    return result


def compare(name, native, peer):
    if native.shape != peer.shape:
        raise ValueError(f"{name}: shape mismatch {native.shape} / {peer.shape}")
    # A multiset check proves that both decoders consumed the same coefficient
    # values before comparing their locations.
    if not np.array_equal(np.sort(native.ravel()), np.sort(peer.ravel())):
        raise ValueError(f"{name}: coefficient multiset differs")
    return {"same_coefficients": True,
            "coordinate_exact": bool(np.array_equal(native, peer)),
            "coordinate_equal_fraction": float(np.mean(native == peer))}


def native_maps(c):
    d = c.bit_length() - 1
    size = c * c
    return {
        "w1_row": bits(4 * size, [3, 6, 7, 8, 9, 10, 11] + list(range(d + 7, 2 * d + 2))),
        "w1_col": bits(4 * size, [1, 0, 4, 5, 2] + list(range(12, d + 7))),
        "attn_row": bits(size, [3, 6, 7, 8, 9] + list(range(10, d + 5))),
        "attn_col": bits(size, [1, 0, 4, 5, 2] + list(range(d + 5, 2 * d))),
    }


def run():
    records = {r["name"]: r for r in json.loads((ROOT / "dlss5-analysis/model.resolved.json").read_text())["tensors"]}
    output = {"peer": "https://huggingface.co/taowen/dlss5-onnx",
              "peer_decoder": "src/dlss5/weights.py decode_qmma_matrix",
              "native_decoder": "Development/derive_native_ffn_layout.py and derive_native_attention_layout.py",
              "blocks": {}}
    for block, c, qkv_offset, projection_offset in ((5, 64, 28832, 57520),
                                                       (9, 128, 98592, 180528),
                                                       (48, 256, 492544, 754720)):
        record = records[f"block{block}.layer0.layer"]
        raw = np.fromfile(ROOT / "dlss5-analysis/tensors" / f"tensor_{record['index']:03d}.bin", np.uint8)
        maps = native_maps(c)
        w1 = native_matrix(raw, 0, 4 * c, c, maps["w1_row"], maps["w1_col"])
        n = native_matrix(raw, projection_offset, c, c, maps["attn_row"], maps["attn_col"])
        output["blocks"][str(block)] = {
            "channels": c,
            "native_map_status": "measured C64/C128" if c < 256 else "extrapolated C256",
            "w1": compare("w1", w1, peer_qmma(raw, 0, 4 * c, c, output_block=128)),
            "projection": compare("projection", n, peer_qmma(raw, projection_offset, c, c)),
        }
        # The QKV physical groups are packed per head; compare V only.
        offsets = qkv_offset + (np.arange(c * c) // 1024) * 3072 + 2048 + np.arange(c * c) % 1024
        native_v = np.empty((c, c), np.float32)
        native_v[maps["attn_row"], maps["attn_col"]] = e4m3fn(raw[offsets])
        peer_v = peer_qmma(raw, qkv_offset, 3 * c, c, qkv_heads=c // 32)[2*c:3*c]
        output["blocks"][str(block)]["v"] = compare("v", native_v, peer_v)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["blocks"], indent=2))
    print(OUT)


if __name__ == "__main__":
    run()
