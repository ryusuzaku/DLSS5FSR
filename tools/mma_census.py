"""
mma_census.py - factor a fused Swin stage's tile geometry (M/N/K) out of its mma
instruction stream. This is HANDOFF §113's method, restored as a tool: that
result came from a scratch script that no longer exists, and the method survived
only as prose in archive/progress.md.

The weights are the mma B operand (§116), so counts of distinct fragments obey

    mma     = M_tiles * N_tiles * K_steps
    B-pairs =           N_tiles * K_steps
    A-sets  = M_tiles            * K_steps
    chains  = M_tiles * N_tiles

which admits exactly one factorisation:

    M = sqrt(A_sets * chains / B_pairs)      N = chains / M      K = B_pairs / N

**The census must be run PER WEIGHT REGION, not over the whole kernel.** Each
region (expand / contract / self-link / ...) is its own GEMM with its own tile
counts; mixing them means no single factorisation exists. Running it whole-kernel
on C=64 gave mma=304 with an implied product of 912 -- the partition is what
makes the arithmetic close (HANDOFF §127 records that miss). Regions are
separated here by which weak weight load a mma's B fragment came from.

The delicate part is `chains`. A chain is a run of mma accumulating into the same
D across K-steps, and its ROOT is a mma whose C nothing produced (a frozen seed,
or a generated value). A naive "the mma that produced this D" map is WRONG -- one
D can be consumed as C by several later mma, so lengths get double-counted; that
mistake gave 368 chain-mma against a 256-mma kernel (HANDOFF §127).

Usage:
    python mma_census.py <kernel-substring> [--whole]
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUBINS = ROOT / "dlss5-analysis" / "cubins"

ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\s*\(")
# mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {D},{A},{B},{C};
MMA_RE = re.compile(
    r"mma\.sync\.aligned\.(m\d+n\d+k\d+)[^;]*?"
    r"\{([^}]*)\},\s*\{([^}]*)\},\s*\{([^}]*)\},\s*\{([^}]*)\}")
LOAD_RE = re.compile(r"ld\.weak\.global[\w\.]*\s*\{([^}]*)\},\s*\[([^\]]*)\]")


def tuples(s):
    return tuple(x.strip() for x in s.split(","))


def iter_kernels(text):
    starts = [(m.start(), m.group(1)) for m in ENTRY_RE.finditer(text)]
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        yield name, text[pos:end]


def factor(mmas):
    """Counts and the one implied (M, N, K) for a set of mma from ONE region."""
    if not mmas:
        return None
    produced = defaultdict(list)
    for i, x in enumerate(mmas):
        produced[x["D"]].append(i)

    def is_root(x):
        c = x["C"]
        return c[0] == c[1] or not produced.get(c)

    consumer = {}
    for i, x in enumerate(mmas):
        for j in range(i + 1, len(mmas)):
            if mmas[j]["C"] == x["D"]:
                consumer[i] = j
                break

    hist = Counter()
    roots = 0
    for i, x in enumerate(mmas):
        if not is_root(x):
            continue
        roots += 1
        ln, cur, seen = 1, i, {i}
        while cur in consumer and consumer[cur] not in seen:
            cur = consumer[cur]
            seen.add(cur)
            ln += 1
        hist[ln] += 1

    A = len({x["A"] for x in mmas})
    B = len({x["B"] for x in mmas})
    D = len({x["D"] for x in mmas})
    C = len({x["C"] for x in mmas})
    out = {"mma": len(mmas), "A": A, "B": B, "D": D, "C": C, "chains": roots,
           "hist": dict(sorted(hist.items()))}
    if A and B and roots:
        M = round((A * roots / B) ** 0.5)
        if M > 0:
            N = roots / M
            K = (B / N) if N else 0
            out["MNK"] = (M, N, K)
            out["fit"] = (M * N * K == len(mmas))
    return out


LD_PARAM_RE = re.compile(r"ld.param.b64 ([^,]+), [[]([^]]*)[]]")
CVTA_RE = re.compile(r"cvta\.to\.global\.u64\s+(%\w+),\s+(%\w+)")
ADD_IMM_RE = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(-?\d+)")
ADD_REG_RE = re.compile(r"add\.s64\s+(%\w+),\s+(%\w+),\s+(%\w+)")


def collect(body):
    """Every mma, tagged with the weight-load site its B fragment came from, and
    every site with the literal byte offset of its address (where the kernel
    uses literals at all -- see HANDOFF §126)."""
    flat = re.sub(r"\s+", " ", body)

    # Walk the stream once, propagating param-derived pointers to a literal
    # offset, so each weight load can be given an address.
    origin = {}
    site_of, site_addr = {}, {}
    n_sites = 0
    for tok in flat.split("; "):
        tok = tok.strip()
        m = LD_PARAM_RE.match(tok)
        if m:
            _sl = m.group(2).split("+")
            origin[m.group(1)] = int(_sl[1]) if len(_sl) > 1 and _sl[1].isdigit() else 0
            continue
        m = CVTA_RE.match(tok)
        if m and m.group(2) in origin:
            origin[m.group(1)] = origin[m.group(2)]
            continue
        m = ADD_REG_RE.match(tok)
        if m:
            dst, a, b = m.group(1), m.group(2), m.group(3)
            if a in origin and b not in origin:
                origin[dst] = origin[a]
            elif b in origin and a not in origin:
                origin[dst] = origin[b]
            continue
        m = ADD_IMM_RE.match(tok)
        if m:
            if m.group(2) in origin:
                origin[m.group(1)] = origin[m.group(2)] + int(m.group(3))
            continue
        m = LOAD_RE.match(tok)
        if m:
            addr = m.group(2).strip()
            for r in tuples(m.group(1)):
                site_of[r] = n_sites
            site_addr[n_sites] = origin.get(addr)
            n_sites += 1

    mmas = []
    for m in MMA_RE.finditer(flat):
        D, A, B, C = (tuples(m.group(k)) for k in (2, 3, 4, 5))
        sites = {site_of[r] for r in B if r in site_of}
        mmas.append({"D": D, "A": A, "B": B, "C": C,
                     "site": sites.pop() if len(sites) == 1 else None})
    return mmas, n_sites, site_addr


def report(title, mmas, n_sites):
    print("\n--- %s ---" % title)
    print("  mma %d   weight-load sites %d" % (len(mmas), n_sites))
    r = factor(mmas)
    if not r:
        print("  (no mma)")
        return
    print("  A-sets %d  B-pairs %d  D-sets %d  C-sets %d  chains %d"
          % (r["A"], r["B"], r["D"], r["C"], r["chains"]))
    print("  chain histogram %s" % r["hist"])
    if "MNK" in r:
        M, N, K = r["MNK"]
        print("  implied M=%g N=%g K=%g -> product %g  %s"
              % (M, N, K, M * N * K,
                 "== mma" if r["fit"] else "!= mma"))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    want = args[0]
    whole = "--whole" in args
    for p in sorted(CUBINS.glob("*_ptx.ptx")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, body in iter_kernels(text):
            if want not in name or not name.startswith(("cc_tinlayout_fused",
                                                         "cc_split_swin",
                                                         "cc_vit")):
                continue
            mmas, n_sites, site_addr = collect(body)
            print("\n########## %s   [%s] ##########" % (name, p.name))
            if whole:
                report("WHOLE KERNEL (mixed regions)", mmas, n_sites)
                continue
            # Merge neighbouring load sites into weight REGIONS: sites whose
            # addresses are within one region stride of each other belong to the
            # same sub-matrix (HANDOFF §111/§113).
            stride = 512
            region_of = {}
            by_addr = sorted((a, s) for s, a in site_addr.items() if a is not None)
            rid = -1
            prev = None
            for a, s in by_addr:
                if prev is None or a - prev > stride * 2:
                    rid += 1
                region_of[s] = rid
                prev = a
            print("  %d addressed sites -> %d regions (stride %d)"
                  % (len(by_addr), (rid + 1) if by_addr else 0, stride))
            groups = defaultdict(list)
            for x in mmas:
                if x["site"] is None:
                    groups[None].append(x)
                else:
                    groups[region_of.get(x["site"], -x["site"] - 100)].append(x)
            for site in sorted(groups, key=lambda s: (s is None, s)):
                label = ("region = weight-load site %d" % site) if site is not None \
                    else "UNTRACED: B not directly from a weight load"
                report(label, groups[site], n_sites)


if __name__ == "__main__":
    main()
