"""Reproduce the ViT payload and PTX checks; write a compact evidence report.

Run after extract_dlssnr.py --dump-tensors and resolve_tensors.py.
Uses only the standard library and the local DLL/PTX; no GPU is required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct

from extract_dlssnr import parse_tensor_table
from resolve_tensors import vit_layout


def moments(values):
    return {"count": len(values), "finite": sum(map(math.isfinite, values)),
            "min": min(values), "max": max(values), "first_8": list(values[:8])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="dlss5-analysis")
    args = ap.parse_args()
    root = Path(args.root)
    model = json.loads((root / "model.json").read_text())
    weights = json.loads((root / "weights.json").read_text())
    dll = Path(model["source"]).read_bytes()
    base = weights["resource"]["file_offset"]
    tensors = parse_tensor_table(dll, base)
    assert len(tensors) == model["tensor_count"] == 153
    assert tensors[-1]["data_offset"] + tensors[-1]["data_len"] == weights["resource"]["size"]
    verified = []
    for tensor, saved in zip(tensors, model["tensors"]):
        assert tensor["name"] == saved["name"]
        off = tensor["abs_file_offset"]
        raw = dll[off:off + tensor["payload_size"]]
        assert raw == (root / "tensors" / f"tensor_{tensor['index']:03d}.bin").read_bytes()
        assert saved["abs_file_offset"] == off
        # Derive q independently from the directory entry and check the framing.
        q = base + tensor["record_offset"] + 8 + len(tensor["name"])
        assert off == q + 28
        assert dll[off + len(raw):off + len(raw) + 16] == struct.pack("<QQ", 0, 1)
        layout = vit_layout(tensor["name"], raw)
        if layout is None:
            continue
        row = {"name": tensor["name"], "index": tensor["index"],
               "payload_file_offset": off, "payload_bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(), "segments": layout["segments"]}
        for segment in row["segments"]:
            if segment["role"] in ("residual_coefficients", "qkv_head_coefficients", "scalar"):
                code = "f" if segment["dtype"] == "fp32" else "e"
                count = segment["bytes"] // struct.calcsize(code)
                vals = struct.unpack_from(f"<{count}{code}", raw, segment["offset"])
                segment["values"] = moments(vals)
                if segment["role"] == "scalar":
                    segment["values"]["decoded_as"] = "fp16 (candidate only)"
        verified.append(row)
    assert len(verified) == 40

    ptx_path = root / "cubins" / "cubin_05_sm_120_ptx.ptx"
    ptx = ptx_path.read_text(errors="replace")
    entries = list(re.finditer(r"\.visible \.entry ([^(]+)\(", ptx))
    kernels = {}
    for i, match in enumerate(entries):
        name = match[1]
        if name not in ("cc_vit_ffn_expand_fp8", "cc_vit_ffn_contract_fp8", "cc_vit_projection_fp8"):
            continue
        end = entries[i + 1].start() if i + 1 < len(entries) else len(ptx)
        body = ptx[match.start():end]
        dims = re.search(r"Conv2d1x1ConfigILi(\d+)ELi(\d+)E", body).groups()
        expand = "expand" in name
        expected = (4096, 1024) if expand else (1024, 4096) if "contract" in name else (1024, 1024)
        assert tuple(map(int, dims)) == expected
        assert ("MpCubicSiluActivation" in body) == expand
        record = {"entry_line": ptx.count("\n", 0, match.start()) + 1,
                  "config_out_in": list(expected),
                  "activation": "MpCubicSiluActivation" if expand else "NoActivation"}
        if not expand:
            size = expected[0] * expected[1]
            load = f"ld.global.b32 %r265, [%rd68+{size}];"
            # Same compiler register path in both kernels: skip -> half2 ->
            # coefficient multiply -> initial MMA accumulator.
            trace = [
                f"ld.param.b64 %rd1, [{name}_param_0+8];",
                f"ld.param.b64 %rd11, [{name}_param_0+24];",
                "add.s64 %rd36, %rd1, %rd35;",
                "ld.weak.global.cg.v4.u32 { %r1510,%r1511,%r1512,%r1513},[%rd34];",
                "mov.b32 {%rs21, %rs22}, %r1510;",
                "cvt.rn.f16x2.e4m3x2 %r264, %rs21;",
                "cvta.to.global.u64 %rd66, %rd11;", load,
                "{mul.f16x2 %r1605,%r264,%r265;",
                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r1605, %r1604},",
            ]
            evidence = []
            for instruction in trace:
                pos = body.index(instruction)
                evidence.append({"line": ptx.count("\n", 0, match.start() + pos) + 1,
                                 "instruction": instruction})
            record["residual_coefficient_offset"] = size
            record["trace"] = evidence
        kernels[name] = record
    assert len(kernels) == 3
    result = {"source_sha256": hashlib.sha256(dll).hexdigest(),
              "payload_layout_version": 2, "validated_records": len(tensors),
              "validated_vit_tensors": len(verified), "ptx": str(ptx_path),
              "kernels": kernels, "vit_tensors": verified,
              "limitations": ["Layer-number mapping uses unique payload layout compatibility; no host launch capture.",
                              "Logical matrix shapes do not establish physical FP8 tile packing.",
                              "layer0 zero suffix purpose and layer3 scalar semantics remain unconfirmed."]}
    out = root / "vit_verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    coefficients = [s["values"] for t in verified for s in t["segments"] if s["role"] == "residual_coefficients"]
    print(f"Validated {len(tensors)} DLL payloads, 40 ViT layouts, 3 PTX kernels.")
    print(f"Residual coefficients: {sum(x['count'] for x in coefficients)} finite FP16 values, "
          f"range [{min(x['min'] for x in coefficients)}, {max(x['max'] for x in coefficients)}].")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
