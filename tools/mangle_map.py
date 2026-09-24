#!/usr/bin/env python3
"""Map the C++ layer classes instantiated across the DLSSNR cubins.

Why this exists
---------------
The PTX still carries Itanium-mangled names on ``.shared`` storage
declarations (and a few other places). Those names name the production
layer classes, their template configs, their activation and their
dtype. That is an architecture spec read straight out of the shipped
binary -- no guessing required.

Read-only. No demangler needed: we only pull out the length-prefixed
identifiers and the ``Li``/``Lb`` literal template arguments.

Usage
-----
    python tools/mangle_map.py                 # table of instantiations
    python tools/mangle_map.py --files         # also show which cubins
    python tools/mangle_map.py --raw SYMBOL    # decode one symbol

The cubin directory is dlss5-analysis/cubins, relative to the repo root
(the parent of tools/), unless --cubins is given.

Caveat that matters: template-argument *slot semantics are NOT known*.
This tool prints values in order. Do not name the slots from the values.
"""

import argparse
import collections
import glob
import os
import re
import sys

SYM_RE = re.compile(r"_Z[A-Za-z0-9_$.]+")
# Itanium length-prefixed identifier, e.g. "29CrazyCuckooFusedSwin2d2HLayer".
# The length prefix exists BECAUSE identifiers contain digits, so the name
# must be sliced to exactly that many characters -- matching greedily on
# [A-Za-z0-9_$]* swallows the rest of the symbol. (Found the hard way.)
NAME_OK = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.]*$")
LIT_RE = re.compile(r"L([ibj])(\d+)E")
ACT_RE = re.compile(r"\d{1,3}([A-Za-z][A-Za-z0-9]*Activation)E")
DTYPE_RE = re.compile(r"LNS_5DTypeE(\d+)E")
# the OOB policy block marks the end of the numeric config head
OOB_MARKERS = ("NS_10OOBRdByDim", "NS_14LayerOutputOOB", "Lj")
CLASS_SUFFIX = ("Layer", "Ffwd")


def components(sym):
    """Split an Itanium symbol into its length-prefixed identifier parts."""
    out = []
    i, n = 0, len(sym)
    while i < n:
        if sym[i].isdigit():
            j = i
            while j < n and sym[j].isdigit() and (j - i) < 4:
                j += 1
            length = int(sym[i:j])
            name = sym[j:j + length]
            if length and NAME_OK.match(name):
                out.append(name)
                i = j + length
                continue
        i += 1
    return out


def class_of(sym):
    """The layer class: last component ending in Layer/Ffwd, else comps[1]."""
    comps = components(sym)
    hits = [c for c in comps if c.endswith(CLASS_SUFFIX)]
    if hits:
        return hits[-1]
    return comps[1] if len(comps) > 1 else (comps[0] if comps else "?")


def head_literals(sym):
    """Literal template args before the OOB policy block (the real config)."""
    cut = len(sym)
    for m in OOB_MARKERS:
        i = sym.find(m)
        if i != -1:
            cut = min(cut, i)
    head = sym[:cut]
    out = []
    for kind, val in LIT_RE.findall(head):
        v = int(val)
        out.append(v if kind == "j" else (bool(v) if kind == "b" else v))
    return tuple(out)


def tail_literal(sym):
    """The single Li<N> that some configs carry after the OOB block."""
    m = re.search(r"E(Li(\d+)E)ELNS_5DTypeE", sym)
    return int(m.group(2)) if m else None


def decode(sym):
    return {
        "class": class_of(sym),
        "config": head_literals(sym),
        "tail": tail_literal(sym),
        "act": (ACT_RE.search(sym).group(1) if ACT_RE.search(sym) else "?"),
        "dtype": (DTYPE_RE.search(sym).group(1) if DTYPE_RE.search(sym) else "?"),
        "member": components(sym)[-1] if components(sym) else "?",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cubins", default=None, help="directory of *_ptx.ptx")
    ap.add_argument("--files", action="store_true", help="show cubin names")
    ap.add_argument("--raw", default=None, help="decode one mangled symbol")
    ap.add_argument("--members", action="store_true",
                    help="also list the shared-storage member names")
    args = ap.parse_args(argv)

    if args.raw:
        d = decode(args.raw)
        for k in ("class", "config", "tail", "act", "dtype", "member"):
            print("  %-8s %s" % (k, d[k]))
        return 0

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdir = args.cubins or os.path.join(root, "dlss5-analysis", "cubins")
    files = sorted(glob.glob(os.path.join(cdir, "*_ptx.ptx")))
    if not files:
        print("no *_ptx.ptx under %s" % cdir, file=sys.stderr)
        return 1

    rows = collections.OrderedDict()
    nsym = 0
    units = collections.Counter()  # CUDA leaks original .cu names here
    for path in files:
        base = os.path.basename(path)
        data = open(path, "rb").read().decode("utf-8", "replace")
        for m in re.finditer(r"_INTERNAL_[0-9a-f]+_\d+_([A-Za-z0-9_]+)_[0-9a-f]+",
                             data):
            units[m.group(1)] += 1
        for m in SYM_RE.finditer(data):
            sym = m.group(0)
            nsym += 1
            d = decode(sym)
            if d["class"] == "?" or d["class"].startswith("_INTERNAL_"):
                continue
            key = (d["class"], d["config"], d["tail"], d["act"], d["dtype"])
            e = rows.setdefault(key, {"n": 0, "files": set(), "members": set()})
            e["n"] += 1
            e["files"].add(base)
            e["members"].add(d["member"])

    print("scanned %d cubins, %d mangled occurrences, %d distinct "
          "instantiations" % (len(files), nsym, len(rows)))
    if units:
        print()
        print("source units named by _INTERNAL_ symbols "
              "(separators flattened):")
        for u, c in units.most_common():
            print("   %-46s %d" % (u, c))
    print()
    hdr = "%-26s %-6s %-4s %-46s %s" % (
        "class", "dtype", "n", "activation", "config (slot semantics UNKNOWN)")
    print(hdr)
    print("-" * len(hdr))
    for (cls, cfg, tail, act, dt), e in sorted(
            rows.items(), key=lambda kv: (kv[0][0], kv[0][4], kv[0][1])):
        cfgs = ",".join(str(c) for c in cfg)
        if tail is not None:
            cfgs += " | tail=%d" % tail
        print("%-26s %-6s %-4d %-46s %s" % (cls, dt, e["n"], act, cfgs))
        if args.files:
            print("%-26s   files: %s" % ("", ", ".join(sorted(e["files"]))))
        if args.members:
            print("%-26s   shared: %s" % ("", ", ".join(sorted(e["members"]))))

    # The dtype correlation worth checking every time: does one config
    # slot double when dtype goes 1 -> 2?
    print()
    print("dtype-1 vs dtype-2 config diff (same class+activation):")
    byclass = collections.defaultdict(lambda: {})
    for (cls, cfg, tail, act, dt), e in rows.items():
        byclass[(cls, act)][dt] = cfg
    for (cls, act), d in sorted(byclass.items()):
        if "1" in d and "2" in d and d["1"] != d["2"]:
            a, b = d["1"], d["2"]
            diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
            print("  %-26s %s" % (cls, diffs if diffs else "same shape"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
