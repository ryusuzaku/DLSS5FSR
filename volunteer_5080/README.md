# RTX 5080 volunteer run — DLSS5FSR

Thank you for helping. This small kit collects the original NVIDIA ViT layout
mapping and, if you have a full model package, an inventory of the logical
C32/C256 assets. It does **not** modify a game, install a driver, inject a DLL,
or upload anything automatically. The DLL, cubin, and model assets stay on your
PC. You choose whether to send the small results ZIP at the end.

## What you need

- Windows with a working RTX 5080 NVIDIA driver. Confirm that `nvidia-smi`
  works in PowerShell.
- Your own copy of `nvngx_dlssnr.dll`. We compare against DLL SHA-256
  `E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E`.
  A different hash is still worth reporting, but it is a different build and
  its results cannot be assumed to match our tensors.
- [Python 3.12.10 for Windows, 64-bit](https://www.python.org/downloads/release/python-31210/),
  with the Python launcher (`py`) and pip selected in its installer.
- One Python add-on: `zstandard==0.25.0`. There are no other pip requirements.
  You do **not** need CUDA Toolkit or Visual Studio for this kit.
- Optional: the **full package ZIP** or extracted `native-game-tiled-assets`
  directory. A GitHub source-code ZIP does not contain these assets. The asset
  inventory does not use the GPU.

## Run

Extract this kit ZIP to any folder and open PowerShell **in that folder**.
Run these commands, replacing only the two paths. Omit `--assets` if you do not
have the full package.

```powershell
nvidia-smi
py -3.12 --version
py -3.12 -m pip install --user zstandard==0.25.0
py -3.12 .\run_probe.py --dll "C:\path\to\nvngx_dlssnr.dll" --assets "C:\path\to\full-package.zip"
```

Without a package:

```powershell
py -3.12 .\run_probe.py --dll "C:\path\to\nvngx_dlssnr.dll"
```

If you want to collect only setup and asset information without a GPU launch,
add `--no-gpu`. If you rerun the kit, choose a fresh output folder with
`--out results2` rather than overwriting the first result.

The run may take a few minutes. It reads the DLL locally to find the SM120
ViT repack cubin, launches its original forward and inverse kernels on four
small synthetic layouts (4×4, 8×4, 16×4, 8×8), checks two held-out patterns,
and writes eight output-byte-to-input-byte maps. The 8×4 and 16×4 layouts
directly cover the bridge shapes in our current pipeline. **If you already ran
v0.1**, add `--shapes bridge` to run only these two new shapes (four maps), and
use a fresh output folder such as `--out results2`. It never launches the
C256/C32 fused kernels; those need a separate, validated input/weight harness.

## What to send back

Send **only** `dlss5fsr-5080-results.zip` from your chosen output folder to
the project maintainer.
You can inspect its contents first: `report.json`, up to eight small `.i32` map
files, and `gpu-error.txt` only if a GPU step failed. `report.json` includes
the GPU model/driver, DLL and cubin hashes, map hashes, asset filenames/sizes/
hashes when available, and any errors. It contains no DLL, cubin, weights,
game capture, or full package. An error result is useful too.

Please also fill in [RETURN_FORM.md](RETURN_FORM.md), especially whether the
DLL came from the same package as the optional assets and whether you could
run one follow-up C256/C32 original-kernel probe before your PC is unavailable.
That follow-up is our main unresolved GPU question; this kit first settles the
setup, model provenance, asset availability, and ViT physical layout.

## Scope and verification

The source is in this folder. The DLL parser and asset inventory are copied
from the public project tools so the ZIP is self-contained. The parser, cubin
selection, and metadata-only run were checked against the maintainer's
matching signed DLL. An RTX 5080 volunteer completed the v0.1 CUDA launches
for 4×4 and 8×8, with byte-exact agreement against the PTX-derived physical
map. The new 8×4 and 16×4 shapes still await runtime results. The return
report distinguishes a successful original launch from a failure.
Do not interpret a failed launch as evidence about our candidate C256/C32 map.
