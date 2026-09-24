# DLSS5FSR research workspace

This repository contains source code and offline validation tools for an
experimental DLSS Neural Rendering port to AMD HIP. Its decoder/head path is
still a candidate: exact comparisons against our scalar references do not
establish parity with NVIDIA's original kernels or a working in-game neural
picture.

The public checkout deliberately excludes NVIDIA DLLs, model weights,
extracted cubins, logical asset packages, generated golden headers, game
captures, build output, third-party dependency trees, and local AI/tool
settings. Those files must not be added to public commits. The root
`.gitignore` uses an allowlist for source directories and selected docs.

To help validate the logical weight maps, read
[ASSET_ORACLE_HANDOFF.md](ASSET_ORACLE_HANDOFF.md). A full-package holder can
run `tools/inspect_dlss5_logical_assets.py` without installing or executing
the package. A matching original DLL and full logical assets are needed for
the held-out C32 recovery script; it reads them locally and writes only small
map/provenance files. `ORIGINAL_MAP_VALIDATION.md` records which claims are
local checks and which are attributed to other projects.

The research scripts expect a local `dlss5-analysis/` generated from a
matching, independently obtained `nvngx_dlssnr.dll`. Some also import the
[DLSS5@AMD reference implementation](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
from `ref/dlss5-port/Development`; clone that repository there when needed.
These local data and third-party directories are ignored by Git. The shim
build additionally requires MSVC, the Windows SDK, NVIDIA NGX headers,
Microsoft's D3DX12 header, and an AMD ROCm SDK installation. Set
`ROCM_ROOT` to the ROCm package root when it is not auto-discovered. Generated
fixture headers are intentionally absent, so this checkout is a research
source publication rather than a turnkey game DLL release.

Source code here is experimental. No NVIDIA binaries or weights are bundled.
