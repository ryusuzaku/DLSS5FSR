#!/usr/bin/env python3
"""Connect original 4x4 repack evidence to a saved candidate logical bridge.

The original kernel proves physical 2D<->1D repacking. The C512 physical-view
to logical-HWC cell map remains an inference from the upstream logical gather.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_native_vit_logical_map import UPSTREAM, audit, logical_map
from compose_vit_bridge import compose
from validate_volunteer_repack import validate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(results_zip: Path, candidate_root: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    original = validate(results_zip)
    if not original["passed"] or "4x4" not in {item["shape"] for item in original["maps"]}:
        raise ValueError("matching original 4x4 repack map missing")
    with ZipFile(results_zip) as archive:
        report = json.loads(archive.read("report.json"))
        forward = np.frombuffer(archive.read("vit_2d_to_1d_4x4.i32"), "<u4")
        reverse = np.frombuffer(archive.read("vit_1d_to_2d_4x4.i32"), "<u4")
        if not (np.array_equal(forward, np.arange(16 * 1024)) and
                np.array_equal(reverse, np.arange(16 * 1024))):
            raise AssertionError("original 4x4 physical repack unexpectedly differs from identity")
        original_hashes = {
            item["kernel"]: item["map_sha256"] for item in report["repack_maps"]
            if (item["width"], item["height"]) == (4, 4)
        }
    candidate_root = candidate_root.resolve()
    candidate = json.loads((candidate_root / "report.json").read_text(encoding="utf-8"))
    map_file = candidate_root / "vit_map/hwc-to-vit.i32"
    if digest(map_file) != candidate["bridge_map_sha256"]:
        raise ValueError("candidate bridge map differs from its saved report")
    saved = np.fromfile(map_file, "<i4")
    audited, cell = audit(4, 4)
    if not all(audited[key] for key in ("cell_ok", "bank_ok", "repeat_ok", "permutation_ok")):
        raise ValueError("source-derived C512 cell geometry failed its checks")
    composed, _ = compose(4, 4, cell)
    if not (np.array_equal(composed, logical_map(16)) and
            np.array_equal(saved, composed)):
        raise AssertionError("saved candidate gather differs from source/PTX composition")
    result = {
        "source_capture_sha256": candidate["source_capture_sha256"],
        "source_model_sha256": candidate["source_model_sha256"],
        "candidate_bridge_map_sha256": digest(map_file),
        "original_dll_sha256": original["dll_sha256"],
        "original_cubin_sha256": original["cubin_sha256"],
        "original_4x4_forward_inverse_map_sha256": original_hashes,
        "upstream_logical_source_sha256": digest(UPSTREAM),
        "inferred_c512_cell_map_sha256": hashlib.sha256(
            np.asarray(cell, "<i4").tobytes()).hexdigest(),
        "candidate_map_matches_source_physical_composition": True,
        "original_4x4_physical_repack_runtime_validated": True,
        "original_c512_split_view_runtime_validated": False,
        "original_16_token_attention_runtime_validated": False,
        "original_full_logical_bridge_validated": False,
        "scope": "exact physical repack plus source-derived logical/C512 cell candidate; not full original connectivity",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_zip", type=Path)
    parser.add_argument("--candidate-root", type=Path,
                        default=Path.home() / "DLSS5FSR-build-offload" / "candidate_game_vit39")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                                "build/vit16_original_repack_component.json")
    args = parser.parse_args()
    run(args.results_zip, args.candidate_root, args.out)
