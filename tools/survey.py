"""
survey.py - first-pass structural survey of nvngx_dlssnr.dll.

Answers the questions that gate every later step:
  * what sections / resources exist
  * where the weight blob lives (and is it separable)
  * how many fatbins / CUBINs are embedded, and for which sm_ targets
  * which kernel names are present

Usage:  python survey.py <path-to-dll> [--json out.json] [--strings]
"""

from __future__ import annotations

import sys
import os
import json
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pestruct as P


KERNEL_HINTS = re.compile(
    rb"(cc_[a-z0-9_]+|cg2r_[a-z0-9_]+|cuda_[a-z0-9_]+|nvof_[a-z0-9_]+|"
    rb"[a-z0-9_]*kernel[a-z0-9_]*)", re.I)


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024.0
    return str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strings", action="store_true",
                    help="dump candidate kernel-name strings")
    ap.add_argument("--min-str", type=int, default=6)
    args = ap.parse_args()

    print(f"[*] parsing {args.dll} ({human(os.path.getsize(args.dll))})")
    pe = P.parse_pe(args.dll)

    print(f"\n=== PE ===")
    print(f"machine        : {pe.machine_name}")
    print(f"image base     : 0x{pe.image_base:X}")
    print(f"entry rva      : 0x{pe.entry_rva:X}")
    print(f"sections       : {pe.num_sections}")

    print(f"\n=== SECTIONS ({len(pe.sections)}) ===")
    print(f"{'name':<12}{'vaddr':>12}{'vsize':>12}{'rawsize':>12}  flags")
    for s in pe.sections:
        print(f"{s.name:<12}{s.vaddr:>#12x}{s.vsize:>12}"
              f"{s.rawsize:>12}  {s.flags_str}")

    print(f"\n=== RESOURCES ({len(pe.resources)}) ===")
    by_size = sorted(pe.resources, key=lambda r: -r.size)
    print(f"{'type':<16}{'name':<20}{'lang':>6}{'size':>14}  offset")
    for r in by_size[:40]:
        print(f"{r.type_name:<16}{r.name:<20}{r.lang:>6}{r.size:>14,}  "
              f"0x{r.file_offset:X}" if r.file_offset is not None else
              f"{r.type_name:<16}{r.name:<20}{r.lang:>6}{r.size:>14,}  ?")

    # ---- fatbins -------------------------------------------------------
    print(f"\n=== FATBIN SCAN ===")
    fb_offs = P.find_fatbins(pe.data)
    print(f"fatbin magic occurrences: {len(fb_offs)}")
    all_entries: list[P.FatbinEntry] = []
    for o in fb_offs:
        ents = P.parse_fatbin(pe.data, o)
        if ents:
            print(f"  0x{o:X} -> {len(ents)} entries "
                  f"[{', '.join(sorted({e.name for e in ents}))}]")
            all_entries.extend(ents)
    print(f"total fatbin entries: {len(all_entries)}")
    kinds = {}
    for e in all_entries:
        kinds[e.name] = kinds.get(e.name, 0) + 1
    for k in sorted(kinds):
        print(f"   {k:<12} x{kinds[k]}")

    # ---- raw ELF cubins -------------------------------------------------
    print(f"\n=== CUBIN SCAN (ELF64 / e_machine=190) ===")
    elf_offs = P.scan_elf_cubins(pe.data)
    print(f"cubins found: {len(elf_offs)}")
    cubins = []
    for o in elf_offs:
        c = P.parse_cubin(pe.data, o)
        if c:
            cubins.append(c)
            print(f"  0x{o:X}  arch={c.arch or '?':<8} "
                  f"sections={len(c.sections):<3} kernels={len(c.kernels)}")

    # ---- strings --------------------------------------------------------
    kernels: set[str] = set()
    if args.strings:
        print(f"\n=== KERNEL-ISH STRINGS ===")
        for m in KERNEL_HINTS.finditer(pe.data):
            s = m.group(0).decode("ascii", "replace")
            if len(s) >= args.min_str:
                kernels.add(s)
        for k in sorted(kernels):
            print("   " + k)

    print(f"\n=== EXPORTS ({len(pe.exports)}) ===")
    for o, n in pe.exports[:60]:
        print(f"  {o:>5}  {n}")

    print(f"\n=== IMPORTS ({len(pe.imports)}) ===")
    for i in pe.imports:
        print("   " + i)

    if args.json:
        out = {
            "file": os.path.abspath(args.dll),
            "size": os.path.getsize(args.dll),
            "machine": pe.machine_name,
            "image_base": hex(pe.image_base),
            "sections": [P.asdict(s) for s in pe.sections],
            "resources": [P.asdict(r) for r in by_size],
            "fatbin_offsets": [hex(o) for o in fb_offs],
            "fatbin_entries": [P.asdict(e) for e in all_entries],
            "cubins": [{
                "offset": hex(c.offset),
                "size": c.size,
                "arch": c.arch,
                "machine": c.machine,
                "kernels": c.kernels,
                "sections": [P.asdict(s) for s in c.sections],
            } for c in cubins],
            "exports": [{"ord": o, "name": n} for o, n in pe.exports],
            "imports": pe.imports,
            "kernel_strings": sorted(kernels),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[+] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
