# Logical assets or original-kernel evidence: handoff (2026-09-24)

## Fastest route: official full package, no NVIDIA GPU needed

The [upstream DLSS5@AMD README](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
links the regular 0.29 OptiScaler full package at
<https://pan.quark.cn/s/209e04e7acaf>. Its README says package weights
(`*.f16`/`*.f32`) are absent from Git and carried forward in full packages.
This is a lead, not a locally verified archive. The public guest download of
an earlier full package returned Quark code23018 (size limit); a signed-in
download may be needed. Do not substitute the source-code ZIP.

On any machine with the full ZIP or extracted package, copy
`tools/inspect_dlss5_logical_assets.py` and run:

```powershell
python tools/inspect_dlss5_logical_assets.py "C:\path\to\OptiScaler-DLSS5-AMD-0.29.zip" --hash --out "C:\path\to\asset-inventory.json"
```

The script reads ZIP members without extracting or executing them. It checks
the 16 C32 files needed by our recovery, plus the 28 ordinary C256
block15–21/49–55 files, including expected sizes for both groups. Use
`--require c256_followup` if you only need the C256 exit code. Send back
`asset-inventory.json` if
sharing the full ZIP is inconvenient. The report contains filenames, lengths
and hashes, but **cannot itself prove the maps**. No account details or game
files are needed for inventory.

With the ZIP and a matching privately extracted DLL in the same working
checkout, first generate `dlss5-analysis/tensors/` using
`tools/extract_dlssnr.py`, then run:

```powershell
python tools/validate_c256_logical_assets.py --raw-only --dll "C:\path\to\nvngx_dlssnr.dll" --out "C:\path\to\c256-raw-provenance.json"
```

This first check needs no logical assets. It verifies the known DLL SHA256 and
byte-compares all 14 extracted C256 tensors at their recorded DLL offsets.
The expected total is 9,649,248 raw bytes. It does **not** validate logical
coefficient maps. With logical assets, continue with the C32 and C256 checks:

```powershell
python tools/recover_head70_maps.py --assets "C:\path\to\OptiScaler-DLSS5-AMD-0.29.zip" --out "build\head70_recovered_layouts_029"
```

This compares original packed tensor bytes already in this workspace with
logical arrays from the package, infers C32 bit permutations on blocks
1/2/3/67/68, verifies all coefficients, and holds out 69/70. The useful
small return artifacts are `PROVENANCE.json` and the three generated `.npz`
map files in the output directory. They contain mappings and hashes, not
the bulk model assets. The recovery fails without publishing maps when an
input is ambiguous or a held-out comparison differs. C256 assets can then
be used for a separate block15–21/49–55 coefficient and skip-order validation:

```powershell
python tools/validate_c256_logical_assets.py --assets "C:\path\to\OptiScaler-DLSS5-AMD-0.29.zip" --dll "C:\path\to\nvngx_dlssnr.dll" --out "C:\path\to\c256-validation.json"
```

The C256 validator compares predeclared candidate FFN, Q/K/V, projection,
bias, scale and skip coordinates against all 14 supplied logical block pairs.
It fits no C256 coefficients or bit maps. An exact result covers 12,393,584
values; mismatch reports name the component and index without exporting
weights. The ZIP or asset directory and this checkout's raw tensors must be
from the same model build. The tool checks the DLL hash and verifies that each
raw tensor's bytes occur at the recorded DLL offset. Send back
`c256-validation.json`, which contains
file/tensor hashes and counts rather than coefficient arrays. This validates
the supplied package's logical interpretation; its own provenance and original
GPU execution remain separate evidence.

## Blackwell owner: direct original oracle

Before a volunteer installs anything, ask for the exact GPU model (the first
lines of `nvidia-smi`) and the SHA256 of any personally obtained
`nvngx_dlssnr.dll`:

```powershell
Get-FileHash "C:\path\to\nvngx_dlssnr.dll" -Algorithm SHA256
```

An RTX 5080 is listed at compute capability 12.0 on NVIDIA's table below.
The first inventory task above needs only [Python 3.12.10 for Windows
(64-bit)](https://www.python.org/downloads/release/python-31210/) and **no
pip add-ons**; `inspect_dlss5_logical_assets.py` uses the Python standard
library. The standalone [RTX 5080 volunteer kit](volunteer_5080/README.md)
adds only `zstandard==0.25.0` and uses the CUDA Driver API through Python;
it needs neither CUDA Toolkit nor Visual Studio. Its original ViT repack
results are recorded in [ORIGINAL_VIT_REPACK_VALIDATION.md](ORIGINAL_VIT_REPACK_VALIDATION.md).
The harder fused C256/C32 original-kernel probes still need a separate,
validated run recipe.

The signed DLL and extracted SM120 cubins are in the maintainer's private
workspace; they are not in the public repository. The
[NVIDIA compute-capability table](https://developer.nvidia.com/cuda/gpus)
lists RTX 50 cards as 12.0. A volunteer with a compatible NVIDIA GPU,
Windows CUDA toolkit, and a personally obtained matching DLL can use the
CUDA Driver API harness sources from the upstream
[HIP branch](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/tree/HIP/Development)
(`run_original_*.cpp`). First verify their
DLL's SHA256 is
`E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E`;
otherwise treat results as a different model build. The extractor command
for the matching DLL is:

```powershell
python tools/extract_dlssnr.py "C:\path\to\nvngx_dlssnr.dll" --out "C:\path\to\dlss5-analysis" --dump-tensors
```

The most useful return data would be original output bytes and hashes for
block15/49 (C256) and block67/69 or post70 (C32) under the *same supplied
controlled inputs and matching raw weights*. A report must state DLL/cubin/
tensor SHA256, kernel symbol, launch dimensions, input hash/shape/physical
view, output hash/shape, and whether the original launch completed. Without
those fields, a result cannot be compared to our AMD path. The upstream
probes are research harnesses, not a polished one-click runner; C256 fused
tile-sync variants can require special runtime state, so a failed isolated
launch is not evidence that the map is wrong. There is no need to send the
DLL, CUDA binaries, or game files back if outputs and provenance suffice.

The [upstream worklog](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/main/Development/history/porting-worklog.md)
reports exact original-kernel C256 block15 comparisons for real and two
randomized inputs (8,192 values each) after deriving the C256 layout from
measured C64/C128 maps. This is published author evidence, **not a local
replay**. Its preparatory `prepare_native_*_gpu.py` scripts mostly export
already validated capture data; they do not regenerate independent logical
weights from the DLL alone on our AMD host.

## What we can do without either source

The same-image AMD candidate now spans block47/encoder22 through block55 and
the block56 prefix; see [CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md).
Existing raw-index audits compare our C256 candidate
against the public decoder (655,360 exact positions) and another author's
host decoder against the public decoder (671,744 exact C32/C256 positions).
These are valuable packing checks, while public ONNX activations differ from
the native FP8 path and cannot by themselves prove the original channel or
physical-view contract. Keep the result labelled candidate until assets or
original outputs provide an independent holdout.
