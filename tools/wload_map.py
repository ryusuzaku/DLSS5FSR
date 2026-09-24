#!/usr/bin/env python3
"""tools/wload_map.py -- production weight byte<->(k,n) map (HANDOFF S116).

PROOF ARTIFACT for item 2. The production fused kernel
`cc_tinlayout_fused_swin_2h_64_2_fp8` (cubin_01) loads its fp8 weights with
`ld.weak.global.ca.v4.u32` into the mma **B** operand REGISTER-DIRECT (no
shuffle/cvt), with a linear per-lane address

    addr = blob_base + pass*PASS_SHIFT + laneid*16 + subblock_imm

Because every lane reads a contiguous 16-byte chunk that *is* its B fragment,
the blob is stored in **m16n8k32.e4m3 B-fragment order**, NOT as a dense
matrix. The port's dense `W[k*N+n]` (in-major) therefore does not mirror it.

B fragment (PTX ISA "Matrix Fragments for mma.m16n8k32", 8-bit, Figs 90/91):
    g   = lane >> 2        (0..7)      col = g
    tig = lane & 3         (0..3)
    reg0 = rows 4*tig + {0,1,2,3}          (K rows 0..15)
    reg1 = rows 4*tig+16 + {0,1,2,3}       (K rows 16..31)
Two B registers per thread == one 8-elem B tile; one 16 B/lane load == two
tiles (bytes 0..7 = tile a, bytes 8..15 = tile b, n += 8).

Byte e of lane L inside a 512 B sub-block (o = L*16 + e):
    e 0..3   -> k_local = 4*tig+e          n_local = g
    e 4..7   -> k_local = 4*tig+(e-4)+16   n_local = g
    e 8..11  -> k_local = 4*tig+(e-8)      n_local = g+8
    e 12..15 -> k_local = 4*tig+(e-12)+16  n_local = g+8
Inverse for (k_local, n_local): g=n_local%8, half=n_local//8,
    tig=(k_local%16)//4, h=k_local//16, m=k_local%4
    lane = 4*g+tig,  e = 8*half + 4*h + m,  o = lane*16 + e.

Region tiling (blob-relative, C=64; from the PTX B-source + D->C chain walk).
Within a pass: sub-block s -> K-step = s // (Npass//16), N-pair = s % (Npass//16),
where Npass = N covered per pass and sub-blocks per K-step = Npass/16.

    R1 expand   [0,16384)     2 passes, each 8192 B = w1[64][128]  (K=64,N=128)
    R2 contract [16384,24576)  W2[128][64], pass splits N (32+32)  (K=128,N=64)
    R3 self     [24576,28672)  Wself[64][64] (C==D self-link)      (K=64,N=64)

Usage:  python3 tools/wload_map.py             # self-check + summary
        python3 tools/wload_map.py 16384 16640 # byte offsets -> (k,n)
"""
from __future__ import annotations
import sys

SUB = 512          # bytes per sub-block (32 lanes x 16 B)
LANE = 16          # bytes per lane


def _b2n(o_in_sub: int) -> tuple[int, int]:
    """(k_local, n_local) for a byte inside one 512 B sub-block."""
    lane, e = divmod(o_in_sub, LANE)
    g, tig = lane >> 2, lane & 3
    half, h, m = e >> 3, (e & 7) >> 2, e & 3
    return 4 * tig + m + 16 * h, g + 8 * half


def _n2b(k_local: int, n_local: int) -> int:
    g, half = n_local % 8, n_local // 8
    tig, h, m = (k_local % 16) // 4, k_local // 16, k_local % 4
    return (4 * g + tig) * LANE + 8 * half + 4 * h + m


class Region:
    """One fp8 weight matrix (or a pass-split region) in the blob."""

    def __init__(self, name, base, K, N, passes=1, nsplit=0):
        self.name, self.base, self.K, self.N = name, base, K, N
        self.passes, self.nsplit = passes, nsplit
        self.size = K * N
        self.per_pass = self.size // passes
        self.npass = nsplit if nsplit else N        # N covered per pass
        assert self.per_pass % SUB == 0, self.name
        assert self.per_pass // SUB == (self.npass // 16) * (K // 32), self.name

    def byte_to_kn(self, off):
        r = off - self.base
        if not (0 <= r < self.size):
            return None
        p, rr = divmod(r, self.per_pass)
        s, in_sub = divmod(rr, SUB)
        per_kstep = self.npass // 16                 # sub-blocks per K-step
        kstep, npair = divmod(s, per_kstep)
        kl, nl = _b2n(in_sub)
        return kstep * 32 + kl, p * self.nsplit + npair * 16 + nl

    def kn_to_byte(self, k, n):
        if not (0 <= k < self.K and 0 <= n < self.N):
            return None
        p = n // self.npass if self.nsplit else 0
        npair, nl = divmod(n % self.npass, 16)
        per_kstep = self.npass // 16
        s = (k // 32) * per_kstep + npair
        return self.base + p * self.per_pass + s * SUB + _n2b(k % 32, nl)


R1a = Region("R1expand.p0", 0, 64, 128)
R1b = Region("R1expand.p1", 8192, 64, 128)
R2 = Region("R2contract", 16384, 128, 64, passes=2, nsplit=32)
R3 = Region("R3self", 24576, 64, 64)
REGIONS = [R1a, R1b, R2, R3]


def check():
    ok = True
    for R in REGIONS:
        seen = {}
        for k in range(R.K):
            for n in range(R.N):
                b = R.kn_to_byte(k, n)
                seen.setdefault(b, []).append((k, n))
                if R.byte_to_kn(b) != (k, n):
                    print("ROUNDTRIP FAIL", R.name, (k, n), b, R.byte_to_kn(b))
                    ok = False
        dup = {b: v for b, v in seen.items() if len(v) > 1}
        miss = R.size - len(seen)
        print(f"{R.name:11s} base={R.base:6d} K={R.K:3d} N={R.N:3d} "
              f"covered={len(seen):5d} dup={len(dup)} missing={miss}")
        ok = ok and not dup and miss == 0
    print("BIJECTION+ROUNDTRIP OK" if ok else "FAIL")
    return ok


def main():
    if len(sys.argv) >= 2:
        for a in sys.argv[1:]:
            off = int(a)
            hit = [f"{R.name} (k,n)={R.byte_to_kn(off)}" for R in REGIONS
                   if R.byte_to_kn(off)]
            print(f"byte {off:6d} -> " + ("; ".join(hit) if hit else "(no region)"))
        return
    ok = check()
    print("\nregion2 sub-block 16384 (Kstep0,npair0): (e -> (k,n)) for lanes 0,1,4,5")
    for lane in (0, 1, 4, 5):
        print("  lane", lane,
              [(e, R2.byte_to_kn(16384 + lane * 16 + e)) for e in (0, 4, 8, 12)])
    print("\nin-major vs production byte for W2[k][n]:")
    for (k, n) in [(0, 0), (0, 1), (1, 0), (4, 0), (16, 0), (0, 8)]:
        print(f"  W2[{k:3d}][{n:3d}]  in-major={k*64+n:6d}   production={R2.kn_to_byte(k, n):6d}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# S231 step 3 -- the same rewrite the SHIM applies, for the oracle to read.
#
# TinRewriteStageFFN (hip_backend.cpp) un-permutes W1 (per head, K=C, N=H1P) and
# W2 (K=H1P, N=W*heads, nsplit=W) into the dense form the kernels read, via
# TinRegionToDense -> dst[n*K + k] = src[base + p*perPass + s*512 + TinN2B(k%32, nl)]
# with TinN2B == this file's _n2b.
#
# PROVEN byte-identical to the device, not assumed: the shim logs the rewritten
# tensor's FNV next to the reading it applied, and
#   python -c "...mode2_dense(tensor_007, 256)..."
# returns EA6B987D, exactly what the shim logs for `HipFfnTranspose=2`
# (tensor_007.bin, C=256, H1P=128, W=32, heads=8). The dense tensor is C909B021.
# Regenerate any golden for the tiled reading only after re-checking that.
# ---------------------------------------------------------------------------
def mode2_dense(raw: bytes, C: int, H1P: int = 128, W: int = 32) -> bytes:
    """The shim's mode-2 un-permutation of a stage tensor's W1/W2 regions."""
    heads = C // 32
    R1, R2 = C * H1P * heads, H1P * W * heads
    if len(raw) < R1 + R2:
        raise ValueError("tensor too small for C=%d" % C)
    out = bytearray(raw)

    def region(base, K, N, nsplit, dst_base):
        npass = nsplit if nsplit else N
        perPass, perKstep = K * npass, npass // 16
        for n in range(N):
            p, nn = divmod(n, npass)
            npair, nl = divmod(nn, 16)
            for k in range(K):
                s = (k // 32) * perKstep + npair
                out[dst_base + n * K + k] =                     raw[base + p * perPass + s * 512 + _n2b(k % 32, nl)]

    for h in range(heads):
        region(h * C * H1P, C, H1P, 0, h * C * H1P)      # W1, per head
    region(R1, H1P, W * heads, W, R1)                     # W2
    return bytes(out)
