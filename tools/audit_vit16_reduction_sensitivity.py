#!/usr/bin/env python3
"""Measure 16-key ViT attention sensitivity to plausible half-sum orders.

This is a fixed-input candidate audit. None of the alternate reductions is an
original NVIDIA oracle, and this does not propagate changed outputs to later
blocks. It checks the saved HIP/scalar baseline before comparing alternatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref/dlss5-port/Development"))
from native_c32_reference import F, H  # noqa: E402
from native_c32_softmax_sum import denominator  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def half_sequential(values: np.ndarray) -> np.ndarray:
    total = np.zeros((*values.shape[:-1], 1), np.float32)
    for key in range(values.shape[-1]):
        total = H(total + values[..., key:key + 1])
    return total


def half_pairwise(values: np.ndarray) -> np.ndarray:
    current = values
    while current.shape[-1] > 1:
        current = H(current[..., 0::2] + current[..., 1::2])
    return current


def half_padded64(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (0, 0), (0, 48)))
    key_order = np.zeros(64, np.int32)
    for bit, dest in enumerate((4, 0, 1, 3, 2, 5)):
        key_order |= ((np.arange(64) >> bit) & 1) << dest
    return denominator(padded[..., np.argsort(key_order)])


def output_for_denominator(exponents: np.ndarray, v: np.ndarray,
                           den: np.ndarray) -> np.ndarray:
    vb = v.reshape(16, 32, 32).transpose(1, 0, 2)
    numerator = H(F(exponents).astype(np.float64) @ vb.astype(np.float64))
    output = F(H(numerator * H(1 / den))).transpose(1, 0, 2).reshape(16, 1024)
    if not np.isfinite(output).all():
        raise ValueError("nonfinite alternate attention")
    return output


def audit_block(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if folder.name != f"vit_block{manifest['block']}":
        raise ValueError(f"stage folder and manifest block differ: {folder}")
    required = ("qkv.f32", "exponents.f32", "attention.f32")
    for name in required:
        if digest(folder / name) != manifest["files"][name]:
            raise ValueError(f"stage file changed: {folder / name}")
    qkv = np.fromfile(folder / "qkv.f32", "<f4").reshape(16, 3, 1024)
    exponents = np.fromfile(folder / "exponents.f32", "<f4").reshape(32, 16, 16)
    saved = np.fromfile(folder / "attention.f32", "<f4").reshape(16, 1024)
    if not (np.isfinite(qkv).all() and np.isfinite(exponents).all() and
            np.isfinite(saved).all()):
        raise ValueError(f"nonfinite saved stage: {folder}")
    baseline_den = H(exponents.astype(np.float64).sum(axis=-1, keepdims=True))
    baseline = output_for_denominator(exponents, qkv[:, 2], baseline_den)
    if not np.array_equal(baseline, saved):
        raise AssertionError(f"saved 16-key attention does not match declared baseline: {folder}")
    variants = {
        "sequential_half": half_sequential(exponents),
        "balanced_half": half_pairwise(exponents),
        "zero_padded_64_key_tree": half_padded64(exponents),
    }
    comparison = {}
    for name, den in variants.items():
        if den.shape != baseline_den.shape or not np.isfinite(den).all() or np.any(den <= 0):
            raise ValueError(f"invalid {name} denominator at {folder}")
        alternative = output_for_denominator(exponents, qkv[:, 2], den)
        difference = np.abs(alternative - saved)
        comparison[name] = {
            "denominator_unequal": int(np.count_nonzero(den != baseline_den)),
            "denominator_total": int(den.size),
            "denominator_max_abs": float(np.max(np.abs(den - baseline_den))),
            "attention_unequal": int(np.count_nonzero(alternative != saved)),
            "attention_total": int(saved.size),
            "attention_mae": float(np.mean(difference, dtype=np.float64)),
            "attention_max_abs": float(np.max(difference)),
            "attention_sha256": hashlib.sha256(
                np.asarray(alternative, "<f4").tobytes()).hexdigest(),
        }
    return {
        "block": manifest["block"],
        "qkv_sha256": digest(folder / "qkv.f32"),
        "exponents_sha256": digest(folder / "exponents.f32"),
        "baseline_attention_sha256": digest(folder / "attention.f32"),
        "baseline_exact": True,
        "variants": comparison,
    }


def run(stage_root: Path, out: Path) -> dict:
    stage_root = stage_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    blocks = []
    for block in range(31, 39):
        folder = stage_root / f"vit_block{block}"
        if block > 31:
            prior = stage_root / f"vit_block{block - 1}" / "projection_device.f32"
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if digest(prior) != manifest["source_device_sha256"]:
                raise ValueError(f"ViT block{block} device ancestry differs")
        blocks.append(audit_block(folder))
    result = {
        "stage_root_name": stage_root.name,
        "scope": "fixed-input 16-token denominator order sensitivity; no original kernel and no changed-output propagation",
        "baseline": "float64 16-key sum rounded once to half; existing HIP/scalar candidate",
        "variants": {
            "sequential_half": "16 keys accumulated in raster order with half rounding after each add",
            "balanced_half": "balanced pairwise 16-key tree with half rounding at each level",
            "zero_padded_64_key_tree": "existing 64-key recovered tree after appending 48 zeros; hypothetical at 16 tokens",
        },
        "blocks": blocks,
        "original_16_token_reduction_validated": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in blocks:
        parts = [f"block{item['block']}"]
        for name, metrics in item["variants"].items():
            parts.append(f"{name}={metrics['attention_unequal']}/{metrics['attention_total']}")
        print(" ".join(parts))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path,
                        default=Path.home() / "DLSS5FSR-build-offload" / "candidate_game_vit39")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "build/vit16_reduction_sensitivity_game.json")
    args = parser.parse_args()
    run(args.stage_root, args.out)
