#!/usr/bin/env python3
"""Audit block48 FFN/attention residual addresses in original SM120 PTX.

This compares physical paired-half weight loads and their accumulator slots.
It does not establish a physical-to-logical channel map or run NVIDIA code.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PTX = ROOT / "dlss5-analysis/cubins/cubin_03_sm_120_ptx.ptx"
ENTRY = "cc_tinlayout_fused_swin_8h_256_8_upsample_fp8"
OUT = ROOT / "build/c256_residual_ptx_audit.json"
REG = r"%(?:r|rd)\d+"
DEFINE = re.compile(rf"^(\w+(?:\.\w+)*)\s+({REG}),\s*([^;]+);")
LOAD = re.compile(rf"^ld\.global\.b32\s+({REG}),\s*\[({REG})\+(\d+)\];")
MUL = re.compile(rf"^\{{?mul\.f16x2\s+({REG}),({REG}),({REG});")


def entry_lines(ptx=PTX, entry=ENTRY):
    text = ptx.read_text()
    marker = f".visible .entry {entry}("
    start = text.index(marker)
    end = text.index("\n.visible .entry ", start + len(marker))
    return text[start:end].splitlines()


def collect(lines, offsets=(491520, 820256)):
    definitions = {}
    loads = {offset: [] for offset in offsets}
    multiplies = []
    for line_number, line in enumerate(lines):
        s = line.strip()
        m = DEFINE.match(s)
        if m:
            op, target, operands = m.groups()
            definitions.setdefault(target, []).append((line_number, op, [x.strip() for x in operands.split(",")]))
        m = LOAD.match(s)
        if m and int(m.group(3)) in loads:
            target, address, offset = m.groups()
            loads[int(offset)].append((line_number, target, address))
        m = MUL.match(s)
        if m:
            multiplies.append(m.groups())
    return definitions, loads, multiplies


def cross_width_address_check(ptx, entry, offsets, warps):
    definitions, loads, multiplies = collect(entry_lines(ptx, entry), offsets)
    if any(len(entries) != 32 for entries in loads.values()):
        raise ValueError(f"expected 32 residual loads per stream in {entry}")
    rows = []
    all_bases = []
    for offset, entries in loads.items():
        weight_regs = [weight for _, weight, _ in entries]
        used = [weight for _, _, weight in multiplies if weight in weight_regs]
        if used != weight_regs:
            raise ValueError(f"residual multiply order differs in {entry} at {offset}")
        bases = set().union(*(global_bases(reg, definitions, line) for line, _, reg in entries))
        if len(bases) != 1:
            raise ValueError(f"weight loads use mixed pointer bases in {entry}: {bases}")
        all_bases.append(bases)
        addresses = []
        for warp in range(warps):
            for lane in range(32):
                cache = {}
                addresses.append([evaluate(reg, definitions, warp, lane, line, cache)
                                  for line, _, reg in entries])
        rows.append(addresses)
    if rows[0] != rows[1]:
        raise ValueError(f"residual addresses differ in {entry}")
    if all_bases[0] != all_bases[1]:
        raise ValueError(f"residual weight pointer differs in {entry}")
    return {"entry": entry, "warps_checked": warps,
            "lanes_per_warp": 32, "paired_loads_each": 32,
            "same_relative_addresses": True}


def global_bases(register, definitions, before):
    if not register.startswith("%") or register in ("%tid.y", "%laneid"):
        return set()
    prior = [item for item in definitions[register] if item[0] < before]
    if not prior:
        raise ValueError(f"no definition before use for {register}")
    line_number, op, operands = prior[-1]
    if op.startswith("cvta.to.global"):
        return {operands[0]}
    return set().union(*(global_bases(operand, definitions, line_number) for operand in operands))


def evaluate(register, definitions, warp, lane, before, cache):
    key = (register, before)
    if key in cache:
        return cache[key]
    if register == "%tid.y":
        return warp
    if register == "%laneid":
        return lane
    if not register.startswith("%"):
        return int(register, 0)
    prior = [item for item in definitions[register] if item[0] < before]
    if not prior:
        raise ValueError(f"no definition before use for {register}")
    line_number, op, operands = prior[-1]
    if op.startswith("cvta.to.global"):
        # Pointer provenance is checked separately by global_bases().
        value = 0
    elif op.startswith("mov"):
        value = evaluate(operands[0], definitions, warp, lane, line_number, cache)
    else:
        a = evaluate(operands[0], definitions, warp, lane, line_number, cache)
        b = evaluate(operands[1], definitions, warp, lane, line_number, cache)
        if op.startswith("add."):
            value = a + b
        elif op.startswith("mad."):
            c = evaluate(operands[2], definitions, warp, lane, line_number, cache)
            value = a * b + c
        elif op.startswith("sub."):
            value = a - b
        elif op.startswith("shl."):
            value = a << b
        elif op.startswith("shr."):
            value = a >> b
        elif op.startswith("and."):
            value = a & b
        elif op.startswith("or."):
            value = a | b
        elif op.startswith("mul."):
            value = a * b
        else:
            raise ValueError(f"unsupported PTX in residual address: {op} {register}")
    cache[key] = value
    return value


def run():
    lines = entry_lines()
    definitions, loads, multiplies = collect(lines)
    for offset, entries in loads.items():
        if len(entries) != 32:
            raise ValueError(f"expected 32 paired-half loads at {offset}, got {len(entries)}")
    weight_registers = {offset: [r for _, r, _ in entries] for offset, entries in loads.items()}
    multipliers = {}
    for offset, regs in weight_registers.items():
        by_weight = {weight: (out, feature) for out, feature, weight in multiplies if weight in regs}
        if len(by_weight) != 32 or any(weight not in by_weight for weight in regs):
            raise ValueError(f"residual load/multiply mapping incomplete at {offset}")
        multipliers[offset] = [by_weight[weight][0] for weight in regs]
    # The two 32-register streams occupy the same ordered accumulator slots.
    first = [int(r[2:]) for r in multipliers[491520]]
    second = [int(r[2:]) for r in multipliers[820256]]
    if first != list(range(first[0], first[0] + 32)):
        raise ValueError("FFN residual register order is not contiguous")
    if second != list(range(second[0], second[0] + 32)):
        raise ValueError("attention residual register order is not contiguous")
    mma_text = "\n".join(lines)
    for stream in (first, second):
        for left in stream[::2]:
            pair = rf"\{{%r{left}, %r{left+1}\}}"
            pattern = (r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s+"
                       + pair + r",\s*\{[^}]*\},\s*\{[^}]*\},\s*" + pair + r";")
            if not re.search(pattern, mma_text):
                raise ValueError(f"residual pair {left}/{left+1} is not an MMA accumulator")
    addresses = {}
    for offset, entries in loads.items():
        addresses[offset] = []
        for warp in range(8):
            for lane in range(32):
                cache = {}
                row = [evaluate(reg, definitions, warp, lane, line, cache) for line, _, reg in entries]
                addresses[offset].append(row)
    if addresses[491520] != addresses[820256]:
        raise ValueError("FFN/attention residual physical addresses differ")
    bases = [set().union(*(global_bases(reg, definitions, line)
                           for line, _, reg in loads[offset]))
             for offset in (491520, 820256)]
    if len(bases[0]) != 1 or bases[0] != bases[1]:
        raise ValueError(f"FFN/attention residual pointers differ: {bases}")
    # Every warp's 16 unique b32 loads cover 32 FP16 coefficients; two row
    # copies of each load correspond to the MMA m16n8 accumulator geometry.
    per_warp = []
    for warp in range(8):
        found = set()
        for lane in range(32):
            for byte_address in addresses[491520][warp * 32 + lane]:
                if byte_address % 4:
                    raise ValueError("unaligned weight load")
                found.update((byte_address // 2, byte_address // 2 + 1))
        expected = set(range(warp * 32, (warp + 1) * 32))
        if found != expected:
            raise ValueError(f"warp {warp} does not cover its 32 channels")
        per_warp.append(len(found))
    report = {
        "ptx": str(PTX.relative_to(ROOT)), "entry": ENTRY,
        "ffn_skip_offset": 491520, "attention_skip_offset": 820256,
        "paired_loads_each": 32, "same_relative_addresses_all_8_warps_32_lanes": True,
        "same_ordered_accumulator_slots": True,
        "all_16_register_pairs_each_initialize_mma": True,
        "unique_f16_coefficients_per_warp": per_warp,
        "inference": "attention raw skip uses the same physical accumulator order as FFN raw skip",
        "logical_channel_map_status": "candidate transfer; no original-kernel result or decoded logical weight oracle",
    }
    report["measured_width_cross_checks"] = [
        cross_width_address_check(
            ROOT / "dlss5-analysis/cubins/cubin_01_sm_120_ptx.ptx",
            "cc_tinlayout_fused_swin_2h_64_2_upsample_fp8", (36864, 69904), 2),
        cross_width_address_check(
            ROOT / "dlss5-analysis/cubins/cubin_02_sm_120_ptx.ptx",
            "cc_tinlayout_fused_swin_4h_128_4_upsample_fp8", (131072, 229904), 4),
    ]
    report["ordinary_c256_block_check"] = cross_width_address_check(
        PTX, "cc_tinlayout_fused_swin_8h_256_8_fp8", (360464, 688704), 8)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
