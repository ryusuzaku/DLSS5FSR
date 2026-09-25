#!/usr/bin/env python3
"""Small, source-only RTX 5080 evidence collector. No game injection or installs.

The only GPU action is launching the original DLL's SM120 ViT repack kernels
on synthetic byte patterns. The DLL and extracted cubin stay on this machine.
"""
from __future__ import annotations

import argparse
from array import array
import ctypes as C
import hashlib
import json
import platform
from pathlib import Path
import random
import struct
import subprocess
import sys
import traceback
import zipfile

import pestruct
from inspect_dlss5_logical_assets import inventory


TARGETS = (
    "cc_vit_1d_repack_2d_to_1d_fp8",
    "cc_vit_1d_repack_1d_to_2d_fp8",
)
KNOWN_DLL_SHA256 = "E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E"
ARENA_BYTES = 4 * 1024 * 1024
ONE = 0x38  # FP8 E4M3 1.0


def sha256(data: bytes | Path) -> str:
    h = hashlib.sha256()
    if isinstance(data, Path):
        with data.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    else:
        h.update(data)
    return h.hexdigest().upper()


def find_repack_cubin(dll: Path) -> tuple[bytes, dict]:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Install the sole add-on: py -3.12 -m pip install --user zstandard==0.25.0") from exc
    data = dll.read_bytes()
    candidates = []
    for fatbin_offset in pestruct.find_fatbins(data):
        for entry in pestruct.parse_fatbin(data, fatbin_offset):
            if entry.sm != 120 or entry.kind != 2:
                continue
            payload = data[entry.payload_offset:entry.payload_offset + entry.payload_size]
            if payload.startswith(b"\x28\xb5\x2f\xfd"):
                blob = zstandard.ZstdDecompressor().decompress(
                    payload, max_output_size=entry.uncompressed_size)
            elif payload.startswith(b"\x7fELF"):
                blob = payload
            else:
                continue
            cubin = pestruct.parse_cubin(blob, 0)
            if cubin and all(symbol in cubin.kernels for symbol in TARGETS):
                candidates.append((blob, {
                    "sm": entry.sm,
                    "fatbin_file_offset": fatbin_offset,
                    "entry_file_offset": entry.offset,
                    "sha256": sha256(blob),
                    "bytes": len(blob),
                    "symbols": list(TARGETS),
                }))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one SM120 cubin with both ViT repack kernels; found {len(candidates)}")
    return candidates[0]


class CUDADriver:
    def __init__(self, cubin: bytes):
        if sys.platform != "win32":
            raise RuntimeError("The GPU probe currently supports Windows only")
        self.lib = C.WinDLL("nvcuda.dll")
        u32, u64, voidp = C.c_uint, C.c_uint64, C.c_void_p
        self._bind("cuInit", [u32])
        self._bind("cuDeviceGet", [C.POINTER(C.c_int), C.c_int])
        self._bind("cuDevicePrimaryCtxRetain", [C.POINTER(voidp), C.c_int])
        self._bind("cuCtxSetCurrent", [voidp])
        self._bind("cuModuleLoadData", [C.POINTER(voidp), voidp])
        self._bind("cuModuleGetFunction", [C.POINTER(voidp), voidp, C.c_char_p])
        self._bind("cuMemAlloc_v2", [C.POINTER(u64), C.c_size_t])
        self._bind("cuMemFree_v2", [u64])
        self._bind("cuMemcpyHtoD_v2", [u64, voidp, C.c_size_t])
        self._bind("cuMemcpyDtoH_v2", [voidp, u64, C.c_size_t])
        self._bind("cuMemsetD8_v2", [u64, C.c_ubyte, C.c_size_t])
        self._bind("cuLaunchKernel", [voidp, u32, u32, u32, u32, u32, u32,
                                      u32, voidp, C.POINTER(voidp), voidp])
        self._bind("cuCtxSynchronize", [])
        self._bind("cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)], check=False)
        self._call("cuInit", 0)
        device = C.c_int()
        self._call("cuDeviceGet", C.byref(device), 0)
        self.context = voidp()
        self._call("cuDevicePrimaryCtxRetain", C.byref(self.context), device.value)
        self._call("cuCtxSetCurrent", self.context)
        self.module = voidp()
        # Keep the backing allocation live while CUDA reads the module image.
        self.cubin_buffer = C.create_string_buffer(cubin)
        self._call("cuModuleLoadData", C.byref(self.module), self.cubin_buffer)
        self.functions = {}
        for name in TARGETS:
            function = voidp()
            self._call("cuModuleGetFunction", C.byref(function), self.module, name.encode())
            self.functions[name] = function
        self.source = u64()
        self.destination = u64()
        self._call("cuMemAlloc_v2", C.byref(self.source), ARENA_BYTES)
        self._call("cuMemAlloc_v2", C.byref(self.destination), ARENA_BYTES)

    def _bind(self, name: str, argtypes: list, check: bool = True) -> None:
        function = getattr(self.lib, name)
        function.argtypes = argtypes
        function.restype = C.c_int
        if check:
            function.errcheck = lambda result, func, args: self._checked(result, func.__name__)

    def _checked(self, result: int, name: str) -> int:
        if result:
            message = C.c_char_p()
            try:
                self.lib.cuGetErrorString(result, C.byref(message))
                detail = message.value.decode(errors="replace") if message.value else "unknown"
            except Exception:
                detail = "unknown"
            raise RuntimeError(f"{name}: CUDA error {result} ({detail})")
        return result

    def _call(self, name: str, *args) -> None:
        getattr(self.lib, name)(*args)

    def launch(self, symbol: str, width: int, height: int, input_bytes: bytes) -> bytes:
        if len(input_bytes) != ARENA_BYTES:
            raise ValueError("Input must fill the 4 MiB source arena")
        output_bytes = width * height * 1024
        source_buffer = C.create_string_buffer(input_bytes)
        output_buffer = C.create_string_buffer(output_bytes)
        self._call("cuMemcpyHtoD_v2", self.source.value, source_buffer, ARENA_BYTES)
        self._call("cuMemsetD8_v2", self.destination.value, 0, ARENA_BYTES)
        params = C.create_string_buffer(struct.pack("<QQii", self.source.value,
                                                     self.destination.value, height, width))
        args = (C.c_void_p * 1)(C.addressof(params))
        inverse = symbol == TARGETS[1]
        gx = (output_bytes // 4 + 255) // 256 if inverse else width * height + 16
        bx, by = (256, 1) if inverse else (32, 4)
        self._call("cuLaunchKernel", self.functions[symbol], gx, 1, 1,
                   bx, by, 1, 0, None, args, None)
        self._call("cuCtxSynchronize")
        self._call("cuMemcpyDtoH_v2", output_buffer, self.destination.value, output_bytes)
        return output_buffer.raw

    def close(self) -> None:
        for ptr in (getattr(self, "source", None), getattr(self, "destination", None)):
            if ptr is not None and ptr.value:
                self._call("cuMemFree_v2", ptr.value)


def recover_map(driver: CUDADriver, symbol: str, width: int, height: int,
                output_dir: Path) -> dict:
    count = width * height * 1024
    baseline = driver.launch(symbol, width, height, bytes([ONE]) * ARENA_BYTES)
    if baseline != bytes([ONE]) * count:
        raise RuntimeError(f"{symbol} {width}x{height}: baseline filled {baseline.count(ONE)}/{count} bytes")
    offsets = array("I", [0]) * count
    for bit in range(22):
        run = 1 << bit
        pattern = bytes(run) + bytes([ONE]) * run
        test_input = pattern * (ARENA_BYTES // (2 * run))
        output = driver.launch(symbol, width, height, test_input)
        for index, value in enumerate(output):
            if value == ONE:
                offsets[index] |= 1 << bit
            elif value != 0:
                raise RuntimeError(f"{symbol}: non-binary output at bit {bit}, byte {index}: {value}")
    if max(offsets) >= ARENA_BYTES or len(set(offsets)) != count:
        raise RuntimeError(f"{symbol}: mapping is outside source arena or not injective")
    for seed in (1709, 1721):
        rng = random.Random(seed)
        test_input = bytes((byte % 126) + 1 for byte in rng.randbytes(ARENA_BYTES))
        output = driver.launch(symbol, width, height, test_input)
        if any(value != test_input[offsets[i]] for i, value in enumerate(output)):
            raise RuntimeError(f"{symbol}: held-out byte test failed for seed {seed}")
    direction = "2d_to_1d" if symbol == TARGETS[0] else "1d_to_2d"
    filename = f"vit_{direction}_{width}x{height}.i32"
    path = output_dir / filename
    if sys.byteorder != "little":
        offsets.byteswap()
    path.write_bytes(offsets.tobytes())
    return {
        "kernel": symbol,
        "width": width,
        "height": height,
        "entries": count,
        "source_arena_bytes": ARENA_BYTES,
        "output_to_input_semantics": "each uint32 maps an output byte offset to its source byte offset",
        "launches": 25,
        "held_out_seeds": [1709, 1721],
        "bijection_on_selected_offsets": True,
        "map_file": filename,
        "map_sha256": sha256(path),
    }


def gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
        return {"exit_code": result.returncode,
                "lines": result.stdout.strip().splitlines(),
                "error": result.stderr.strip()[:300]}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True, help="your own nvngx_dlssnr.dll")
    parser.add_argument("--assets", type=Path, help="optional full package ZIP or extracted asset folder")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-gpu", action="store_true", help="metadata/asset report only")
    args = parser.parse_args()
    if not args.dll.is_file():
        parser.error(f"DLL not found: {args.dll}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output folder is not empty; choose a new --out folder: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "kit": "DLSS5FSR RTX 5080 volunteer kit v0.1",
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "gpu": gpu_info(),
        "dll_name": args.dll.name,
        "dll_bytes": args.dll.stat().st_size,
        "dll_sha256": sha256(args.dll),
        "known_model_match": False,
        "repack_maps": [],
        "errors": [],
    }
    report["known_model_match"] = report["dll_sha256"] == KNOWN_DLL_SHA256
    if args.assets:
        try:
            assets = inventory(args.assets, True)
            # Local source path is not needed in the return file.
            assets["source_name"] = args.assets.name
            report["assets"] = assets
        except Exception as exc:
            report["errors"].append(f"asset inventory: {exc}")
    if not args.no_gpu:
        try:
            cubin, details = find_repack_cubin(args.dll)
            report["repack_cubin"] = details
            driver = CUDADriver(cubin)
            try:
                for width, height in ((4, 4), (8, 8)):
                    for symbol in TARGETS:
                        report["repack_maps"].append(recover_map(
                            driver, symbol, width, height, args.out))
            finally:
                driver.close()
        except Exception as exc:
            report["errors"].append(f"GPU probe: {exc}")
            (args.out / "gpu-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    report_file = args.out / "report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    archive = args.out / "dlss5fsr-5080-results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in (report_file, *sorted(args.out.glob("vit_*.i32")), args.out / "gpu-error.txt"):
            if path.is_file():
                output.write(path, path.name)
    print(f"DLL SHA256: {report['dll_sha256']}")
    print(f"Matching model build: {report['known_model_match']}")
    print(f"Recovered maps: {len(report['repack_maps'])}/4")
    print(f"Share this small results archive: {archive.resolve()}")
    for error in report["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
