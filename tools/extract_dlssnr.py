"""
extract_dlssnr.py - build the complete DLSSNR model manifest from nvngx_dlssnr.dll

This is the "Week 1" tool: it turns the opaque NVIDIA DLL into an inspectable
model package. Everything here is read-only and dependency-light (stdlib +
zstandard).

OUTPUT (<out>/)
    report.json       everything, one file
    model.json        tensor manifest: names, offsets, shapes-as-sizes, dtype
    kernels.json      every kernel symbol, mapped to cubin + sm arch
    weights.json      where the weight blob lives and how to carve it
    cubins/*.elf      decompressed CUBINs
    cubins/*.json     per-cubin sections / symbols / kernels
    tensors/*.bin     only with --dump-tensors (147 MB)

FORMATS REVERSED FOR THIS FILE
------------------------------
PE resource "WEIGHTS_HT" (RT_RCDATA) holds the weights:

    u64  total_size            == resource size
    repeat:
      u64 name_len
      char name[name_len]      ASCII, e.g. "block23.layer2.layer"
      u64 chunk_size           == 20 + data_len (excludes this first u64)
        u64 chunk_size2        == chunk_size
        u64 payload_size       == chunk_size - 40
        u32 1
        u8  payload[payload_size]
        u8  trailer[20]        u64(0), u64(1), u32(numel)
      -> next record follows immediately; data_len == chunk_size - 20

    Payload begins at q+28, immediately after u32(1). The former q+29
    interpretation skipped the first weight byte and appended a trailer zero.

Fatbins hold ZSTD-compressed, multi-arch CUBINs:

    container:  u32 magic 0xBA55ED50 | u16 ver | u16 hdr(0x10) | u64 total
    entry:      u16 kind(2=ELF) | u16 ver | u32 hdr(0x40) | u64 entry_size
                | u64 payload_size | u16 ? | u16 ? | u32 sm_arch
                ... u64 uncompressed_size @ +0x38 ... payload @ +0x40
    next entry starts at: entry + header_size + entry_size
    (entry_size == align8(payload_size) -- NOT an RVA, and NOT the full stride)

Usage:  python extract_dlssnr.py <dll> [--out dlssnr-analysis] [--dump-tensors]
"""

from __future__ import annotations

import sys
import os
import json
import struct
import argparse
import hashlib
import re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pestruct as P
from classify_dtype import classify

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

INTERESTING = [
    rb"cc_vit[a-z0-9_]*", rb"cc_swin[a-z0-9_]*", rb"cc_dec[a-z0-9_]*",
    rb"cc_tin[a-z0-9_]*", rb"cc_[a-z0-9_]+", rb"cg2r_[a-z0-9_]+",
    rb"cuda_[a-z0-9_]+", rb"sm_[0-9]+[a-z]?", rb"fp8[a-z0-9_]*",
    rb"e4m3[a-z0-9_]*", rb"e5m2[a-z0-9_]*",
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.2f} {u}" if u != "B" else f"{n} B"
        n /= 1024.0
    return str(n)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def parse_tensor_table(data: bytes, base: int) -> list[dict]:
    """Carve the WEIGHTS_HT tensor directory."""
    total = struct.unpack_from("<Q", data, base)[0]
    out = []
    p = 8
    while p + 8 <= total:
        name_len = struct.unpack_from("<Q", data, base + p)[0]
        if not 1 <= name_len <= 255:
            break
        name = data[base + p + 8:base + p + 8 + name_len].decode(
            "ascii", "replace")
        q = p + 8 + name_len
        if q + 28 > total:
            break
        chunk = struct.unpack_from("<Q", data, base + q)[0]
        chunk2 = struct.unpack_from("<Q", data, base + q + 8)[0]
        payload = struct.unpack_from("<Q", data, base + q + 16)[0]
        flag = struct.unpack_from("<I", data, base + q + 24)[0]
        if not (chunk == chunk2 and payload == chunk - 40 and 0 < chunk < total):
            break
        data_len = chunk - 20
        data_off = q + 28
        if data_off + data_len > total:
            raise ValueError(f"Tensor {name}: record exceeds resource")
        trailer_off = base + data_off + payload
        zero, rank, numel = struct.unpack_from("<QQI", data, trailer_off)
        if (zero, rank, numel * 2, flag) != (0, 1, payload, 1):
            raise ValueError(f"Tensor {name}: unexpected WEIGHTS_HT trailer/header")
        numel = struct.unpack_from("<I", data, base + data_off + data_len - 4)[0]
        out.append({
            "name": name,
            "index": len(out),
            "record_offset": p,
            "data_offset": data_off,             # relative to resource start
            "abs_file_offset": base + data_off,
            "data_len": data_len,
            "payload_size": payload,
            "numel": numel,
            "flag": flag,
            "trailer_size": 20,
        })
        p = data_off + data_len
    return out


# ---------------------------------------------------------------------------
# cubins
# ---------------------------------------------------------------------------

def iter_fatbin_entries(data: bytes):
    dctx = zstandard.ZstdDecompressor() if zstandard else None
    for i, off in enumerate(P.find_fatbins(data)):
        chs = struct.unpack_from("<H", data, off + 6)[0]
        total = struct.unpack_from("<Q", data, off + 8)[0]
        for ent in P.parse_fatbin(data, off):
            raw = data[ent.payload_offset:ent.payload_offset + ent.payload_size]
            blob, mode = None, "raw"
            if raw[:4] == ZSTD_MAGIC:
                if dctx is None:
                    mode = "zstd(no-decompressor)"
                else:
                    blob = bytes(dctx.decompressobj().decompress(raw))
                    mode = "zstd"
            elif raw[:4] == b"\x7fELF":
                blob = raw
            elif ent.is_ptx and raw[:1] == b"\x0a":
                blob = raw          # PTX stored verbatim
                mode = "plain"
            rec = {
                "fatbin": i,
                "entry": ent.index,
                "fatbin_offset": hex(off),
                "entry_offset": hex(ent.offset),
                "container_total": total,
                "container_header_size": chs,
                "kind": ent.kind,
                "entry_version": hex(ent.version),
                "entry_header_size": ent.header_size,
                "entry_size": ent.entry_size,
                "payload_size": ent.payload_size,
                "sm": ent.sm,
                "arch": ent.name,
                "kind": ("ptx" if ent.is_ptx else "elf"),
                "declared_uncompressed_size": ent.uncompressed_size,
                "payload_offset": hex(ent.payload_offset),
                "compression": mode,
            }
            if blob is not None:
                rec["decompressed_size"] = len(blob)
                rec["size_ok"] = (len(blob) == ent.uncompressed_size)
                rec["sha256_16"] = sha(blob)
            yield rec, blob


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("--out", default="dlssnr-analysis")
    ap.add_argument("--dump-tensors", action="store_true",
                    help="write each tensor payload to tensors/ (147 MB)")
    ap.add_argument("--sample", type=int, default=65536,
                    help="bytes per tensor used for dtype classification")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    cub_dir = os.path.join(out, "cubins")
    os.makedirs(cub_dir, exist_ok=True)
    if args.dump_tensors:
        os.makedirs(os.path.join(out, "tensors"), exist_ok=True)

    print(f"[*] {args.dll}  ({human(os.path.getsize(args.dll))})")
    pe = P.parse_pe(args.dll)
    print(f"    {pe.machine_name} | {len(pe.sections)} sections | "
          f"{len(pe.resources)} resources | {len(pe.exports)} exports")

    report = {
        "source": {
            "file": os.path.abspath(args.dll),
            "size": os.path.getsize(args.dll),
            "machine": pe.machine_name,
            "image_base": hex(pe.image_base),
            "exports": [n for _, n in pe.exports],
            "imports": pe.imports,
        },
        "sections": [P.asdict(s) for s in pe.sections],
    }

    # ---------------- weights -------------------------------------------
    print("\n[*] weights")
    rdata = [r for r in pe.resources if r.type_name == "RT_RCDATA"]
    if not rdata:
        print("    !! no RT_RCDATA resource found")
        tensors = []
        blob = None
    else:
        blob = max(rdata, key=lambda r: r.size)
        print(f"    resource {blob.name!r}  {blob.size:,} bytes "
              f"@ file 0x{blob.file_offset:X}")
        tensors = parse_tensor_table(pe.data, blob.file_offset)
        print(f"    {len(tensors)} tensors, last record ends at "
              f"{tensors[-1]['data_offset'] + tensors[-1]['data_len']:,}")

        total_payload = sum(t["payload_size"] for t in tensors)
        print(f"    payload total {total_payload:,} bytes "
              f"({human(total_payload)})")

        print("    classifying dtypes ...")
        hist = {}
        for t in tensors:
            off = blob.file_offset + t["data_offset"]
            sample = pe.data[off:off + min(t["payload_size"], args.sample)]
            dt, detail = classify(sample)
            t["dtype"] = dt
            t["dtype_confidence"] = "inferred"
            t["dtype_stats"] = {k: v for k, v in detail.items()
                                if not k.endswith("_score")}
            hist[dt] = hist.get(dt, 0) + 1
        print("    " + ", ".join(f"{k}={v}" for k, v in
                                 sorted(hist.items(), key=lambda x: -x[1])))

        if args.dump_tensors:
            for t in tensors:
                off = blob.file_offset + t["data_offset"]
                with open(os.path.join(out, "tensors",
                                       f"tensor_{t['index']:03d}.bin"),
                          "wb") as fh:
                    fh.write(pe.data[off:off + t["payload_size"]])
            print(f"    dumped {len(tensors)} tensor payloads")

    report["weights"] = {
        "resource": (P.asdict(blob) if blob else None),
        "tensor_count": len(tensors),
        "payload_layout_version": 2,
        "tensors": tensors,
    }
    with open(os.path.join(out, "weights.json"), "w", encoding="utf-8") as fh:
        json.dump(report["weights"], fh, indent=2)
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "source": report["source"]["file"],
            "format": "NVIDIA DLSSNR / WEIGHTS_HT v1",
            "payload_layout_version": 2,
            "dtype_note": ("dtypes are inferred statistically from payload "
                           "distribution; validate against kernel params"),
            "tensor_count": len(tensors),
            "tensors": tensors,
        }, fh, indent=2)

    # ---------------- cubins --------------------------------------------
    print("\n[*] cubins")
    cubins_meta, kernels_all = [], []
    for rec, blob_data in iter_fatbin_entries(pe.data):
        ext = ".ptx" if rec.get("kind") == "ptx" else ".elf"
        tag = f"cubin_{rec['fatbin']:02d}_{rec['arch']}{ext.replace('.', '_')}"
        if blob_data is None:
            print(f"    {tag:<22} UNDECOMPRESSED")
            cubins_meta.append(rec)
            continue
        print(f"    {tag:<22} {rec['payload_size']:>10,} -> "
              f"{len(blob_data):>10,}  ok={rec['size_ok']}")
        fname = f"cubins/{tag}{ext}"
        with open(os.path.join(cub_dir, tag + ext), "wb") as fh:
            fh.write(blob_data)
        meta = dict(rec)
        meta["file"] = fname
        if ext == ".elf":
            c = P.parse_cubin(blob_data, 0)
            secs = [P.asdict(s) for s in c.sections] if c else []
            kernels = c.kernels if c else []
            meta.update({
                "arch": c.arch if c else "",
                "elf_machine": c.machine if c else None,
                "section_count": len(secs),
                "sections": secs,
                "kernels": kernels,
            })
            for k in kernels:
                kernels_all.append({"kernel": k, "cubin": tag,
                                    "arch": rec["arch"], "sm": rec["sm"]})
        else:
            # PTX: pull the entry-point names straight out of the source
            meta["ptx_entry_points"] = sorted(set(
                m.group(1).decode()
                for m in re.finditer(rb"\.visible\s+\.entry\s+([A-Za-z0-9_]+)",
                                     blob_data)))
            meta["ptx_version"] = (re.search(rb"\.version\s+([0-9.]+)",
                                             blob_data[:128]).group(1).decode()
                                   if re.search(rb"\.version\s+([0-9.]+)",
                                                blob_data[:128]) else "")
        cubins_meta.append(meta)
        with open(os.path.join(cub_dir, tag + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    report["cubins"] = cubins_meta
    report["kernel_count"] = len(kernels_all)
    with open(os.path.join(out, "kernels.json"), "w", encoding="utf-8") as fh:
        json.dump({"kernel_count": len(kernels_all), "kernels": kernels_all},
                  fh, indent=2)
    archs = sorted({c.get("arch", "?") for c in cubins_meta})
    print(f"    {len(cubins_meta)} cubins | {len(kernels_all)} kernel symbols "
          f"| archs: {', '.join(archs)}")

    # ---------------- orientation strings --------------------------------
    strings: dict[str, int] = {}
    for pat in INTERESTING:
        for m in re.finditer(pat, pe.data):
            s = m.group(0).decode("ascii", "replace")
            strings[s] = strings.get(s, 0) + 1
    report["strings"] = strings

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[+] wrote {out}/{{report,model,weights,kernels}}.json "
          f"+ cubins/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
