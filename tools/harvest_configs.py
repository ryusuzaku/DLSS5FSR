"""
harvest_configs.py - pull every `tin` layer-configuration template tuple out of
the sm_120 PTX dumps and demangle it.

NVIDIA's template inference library encodes layer hyperparameters as C++
template arguments, so the model architecture is sitting in the binary as
plain integers. Extracting them gives authoritative shapes -- far better than
inferring them from tensor byte counts.

Usage:
    python harvest_configs.py [--raw]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demangle_tin import Demangler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5-analysis" / "cubins"

# Full mangled symbols in the PTX. `tin` layer types live in namespace tin3_1,
# but the fused Swin blocks (CrazyCuckoo*) are in the global / cc_samples
# namespace, so match on any `_Z` symbol that carries a *Config template.
SYM_RE = re.compile(r"_Z[A-Za-z0-9_$.]*ConfigI[A-Za-z0-9_$.]*")


def demangle(name: str) -> str:
    return Demangler(name).run()


def _balanced_slice(s: str, open_idx: int) -> str | None:
    """Slice s[open_idx:] through the '>' that closes the '<' at open_idx."""
    if open_idx >= len(s) or s[open_idx] != "<":
        return None
    depth = 0
    for k in range(open_idx, len(s)):
        if s[k] == "<":
            depth += 1
        elif s[k] == ">":
            depth -= 1
            if depth == 0:
                return s[open_idx : k + 1]
    return None


def harvest() -> dict:
    """Return {config_kind: {tuple_str: occurrence_count}}."""
    found: dict[str, dict[str, int]] = {}
    n_sym = 0
    for ptx in sorted(CUBINS.glob("*_ptx.ptx")):
        text = ptx.read_text(encoding="utf-8", errors="replace")
        for m in SYM_RE.finditer(text):
            n_sym += 1
            dem = demangle(m.group(0))
            # every *Config<...> occurrence in the demangled output
            for cm in re.finditer(r"[A-Za-z_0-9:]+Config<", dem):
                kind = cm.group(0)[:-1].split("::")[-1]
                body = _balanced_slice(dem, cm.end() - 1)
                if body is None:
                    continue
                cfg = cm.group(0) + body
                found.setdefault(kind, {}).setdefault(cfg, 0)
                found[kind][cfg] += 1
    print(f"scanned symbols: {n_sym}", file=sys.stderr)
    return found


def main() -> None:
    found = harvest()
    out: dict[str, list[dict]] = {}
    for kind in sorted(found):
        rows = [
            {"config": cfg, "count": n}
            for cfg, n in sorted(found[kind].items(), key=lambda kv: -kv[1])
        ]
        out[kind] = rows

    dest = ROOT / "dlss5-analysis" / "configs.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for kind, rows in out.items():
        print(f"\n=== {kind}  ({len(rows)} distinct) ===")
        for r in rows:
            print(f"  [{r['count']:4d}] {r['config']}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
