#!/usr/bin/env python3
"""Validate RTX 5080 original ViT repack maps against the PTX address model.

Standard library only. The ZIP is read in place. No NVIDIA DLL/cubin is needed.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


DLL_SHA256 = "E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E"
CUBIN_SHA256 = "BFB2EBE117B7C4D92A78E1885412ACBB80233F2D9B11AF1E854430EB3CC0A2A1"


def expected_forward(width: int, height: int) -> array:
    """PTX physical destination byte -> source byte for the 2D-to-1D kernel."""
    if width % 4 or height % 4 or width * height > 64 or width <= 0 or height <= 0:
        raise ValueError("unsupported repack shape")
    mapping = array("I", [0]) * (width * height * 1024)
    for token in range(width * height):
        y, x = divmod(token, width)
        cell = (y // 4) * (width // 4) + x // 4
        within = (y % 4) * 4 + x % 4
        for word in range(256):
            common = (word // 8) * 128 + ((word % 8) // 4) * 2 + (word * 4) % 16
            source_word = cell * 4096 + common + within // 8 + (within % 8) * 16
            dest_word = (token // 16) * 4096 + common + (token % 16) // 8 + (token % 8) * 16
            for lane in range(4):
                mapping[dest_word * 4 + lane] = source_word * 4 + lane
    return mapping


def expected_inverse(width: int, height: int) -> array:
    """Inverse PTX physical destination byte -> source byte."""
    if width % 4 or height % 4 or width * height > 64 or width <= 0 or height <= 0:
        raise ValueError("unsupported repack shape")
    mapping = array("I", [0]) * (width * height * 1024)
    for token in range(width * height):
        y, x = divmod(token, width)
        cell = (y // 4) * (width // 4) + x // 4
        within = (y % 4) * 4 + x % 4
        for word in range(256):
            common = (word // 8) * 128 + ((word % 8) // 4) * 2 + (word * 4) % 16
            source_word = (token // 16) * 4096 + common + (token % 16) // 8 + (token % 8) * 16
            dest_word = cell * 4096 + common + within // 8 + (within % 8) * 16
            for lane in range(4):
                mapping[dest_word * 4 + lane] = source_word * 4 + lane
    return mapping


def read_map(archive: ZipFile, item: dict) -> array:
    name = item["map_file"]
    if name not in archive.namelist() or "/" in name or "\\" in name:
        raise ValueError(f"missing or invalid map name {name!r}")
    data = archive.read(name)
    size = item["width"] * item["height"] * 1024
    if len(data) != size * 4 or item["entries"] != size:
        raise ValueError(f"wrong map size: {name}")
    digest = hashlib.sha256(data).hexdigest().upper()
    if digest != item["map_sha256"].upper():
        raise ValueError(f"map hash mismatch: {name}")
    values = array("I")
    values.frombytes(data)
    if values.itemsize != 4:
        raise ValueError("platform uint32 size mismatch")
    import sys
    if sys.byteorder != "little":
        values.byteswap()
    if len(set(values)) != size or min(values) != 0 or max(values) != size - 1:
        raise ValueError(f"map not a full permutation: {name}")
    return values


def validate(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "report.json" not in names:
            raise ValueError("duplicate entries or missing report")
        report = json.loads(archive.read("report.json"))
        if report.get("errors"):
            raise ValueError(f"probe reported errors: {report['errors']}")
        if report.get("dll_sha256", "").upper() != DLL_SHA256:
            raise ValueError("DLL is a different model build")
        if report.get("repack_cubin", {}).get("sha256", "").upper() != CUBIN_SHA256:
            raise ValueError("SM120 cubin hash mismatch")
        if not any("RTX 5080" in line and "12.0" in line
                   for line in report.get("gpu", {}).get("lines", [])):
            raise ValueError("RTX 5080/SM120 GPU provenance missing")
        by_shape = {}
        for item in report["repack_maps"]:
            shape = (item["width"], item["height"])
            direction = "2d_to_1d" if item["kernel"].endswith("2d_to_1d_fp8") else (
                "1d_to_2d" if item["kernel"].endswith("1d_to_2d_fp8") else None)
            if direction is None or direction in by_shape.setdefault(shape, {}):
                raise ValueError("unexpected or duplicate kernel/shape")
            by_shape[shape][direction] = read_map(archive, item)
        if not by_shape:
            raise ValueError("no repack maps")
        details = []
        for (width, height), pair in sorted(by_shape.items()):
            if set(pair) != {"2d_to_1d", "1d_to_2d"}:
                raise ValueError(f"missing forward/inverse pair for {width}x{height}")
            forward, inverse = pair["2d_to_1d"], pair["1d_to_2d"]
            mismatches = sum(a != b for a, b in zip(forward, expected_forward(width, height)))
            inverse_ptx_mismatches = sum(a != b for a, b in zip(inverse, expected_inverse(width, height)))
            inverse_mismatches = sum(inverse[forward[i]] != i for i in range(len(forward)))
            details.append({"shape": f"{width}x{height}", "entries": len(forward),
                            "ptx_forward_mismatches": mismatches,
                            "ptx_inverse_mismatches": inverse_ptx_mismatches,
                            "forward_inverse_mismatches": inverse_mismatches,
                            "forward_identity_entries": sum(i == v for i, v in enumerate(forward))})
        return {"dll_sha256": report["dll_sha256"],
                "cubin_sha256": report["repack_cubin"]["sha256"],
                "gpu": report["gpu"]["lines"],
                "maps": details,
                "passed": all(d["ptx_forward_mismatches"] == 0 and
                              d["ptx_inverse_mismatches"] == 0 and
                              d["forward_inverse_mismatches"] == 0 for d in details)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    result = validate(args.zip)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
