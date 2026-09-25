# Original RTX 5080 ViT repack validation (2026-09-25)

A volunteer ran the public v0.1 source-only kit against an RTX 5080 (driver
617.14, compute capability 12.0). Its SHA-256-verified `nvngx_dlssnr.dll` is
`E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E`,
the same signed model build used here. The extracted SM120 cubin hash is
`BFB2EBE117B7C4D92A78E1885412ACBB80233F2D9B11AF1E854430EB3CC0A2A1`.
The volunteer's results ZIP is kept privately and excluded from Git.

The probe executed both original `cc_vit_1d_repack_2d_to_1d_fp8` and
`cc_vit_1d_repack_1d_to_2d_fp8` on 4×4 and 8×8 token grids. For each launch
shape and direction it checked an all-ones baseline, recovered all source byte
addresses with 22 address-bit inputs, asserted a full bijection, and checked
two held-out byte patterns. All four maps completed without errors; their
recorded SHA-256 values match the returned files. The 4×4 maps are physical
identity. The 8×8 forward and inverse maps are nontrivial inverses.

`python tools/validate_volunteer_repack.py <results.zip>` independently checks
DLL/cubin provenance and every returned map against the source-byte address
formula previously derived from SM120 PTX. It reports **zero mismatches** for
all 16,384 + 16,384 + 65,536 + 65,536 entries. This upgrades the **physical
ViT repack map for these two shapes** from static PTX inference to original
SM120 runtime evidence. The exact v0.1 map SHA-256 values are:

| Shape | 2D→1D | 1D→2D |
| --- | --- | --- |
| 4×4 | `999B5382075E99FC59C39652A6D0776F0C73F49866AD762D450569C51A30F5DB` | `999B5382075E99FC59C39652A6D0776F0C73F49866AD762D450569C51A30F5DB` |
| 8×8 | `3254EB6848C973A120C507D2EB7D1EB06D180A491F95B086BCE92234C5435809` | `DC5E95E80202FACA99E7563C2F1C3016F7CFA11B4A6BB0FA3B08676B472B26F6` |

The result does **not** validate the upstream C512 split-view-to-logical-HWC
permutation, the 16-token ViT attention reduction, C256/C32 packed weights or
fused bodies, or the native game's input/output contract. The current connected
bridge also uses 8×4 and 16×4 shapes, which need their own original launches
before those particular extents can be marked runtime-validated.
