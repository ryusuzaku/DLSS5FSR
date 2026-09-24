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

[CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md) records a same-image
block47→56 candidate run. The matching encoder22 skip is now used at block48;
its AMD/scalar checks are exact, while comparisons to the public FP16 ONNX
model are approximate because the candidate rounds activations to FP8.
The same-image candidate has also been propagated through blocks56–69 and
the full-frame head; the document gives the measured RGB comparison and
original-kernel validation limits.

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

The same-image candidate scripts additionally expect the public optimized
FP16 ONNX model and its blue-marble example from
[taowen/dlss5-onnx](https://huggingface.co/taowen/dlss5-onnx) in
`ref/dlss5-onnx/`, plus Python packages `numpy`, `onnx`, `onnxruntime`, and
`Pillow`. The full same-image extraction order is
`extract_peer_preblock0_skip.py`, `extract_peer_coherent_head_inputs.py`,
`extract_peer_decoder66_inputs.py`, `extract_peer_decoder62_inputs.py`,
`extract_peer_decoder56_inputs.py`, then
`extract_peer_decoder48_inputs.py`. Run
`check_upsample48_peer_image.py`, `check_decoder48_55_peer_image.py`, and
`check_upsample56_peer_image.py --candidate-block55` after building the HIP
test executables with `tools/build_split512_block.sh`.

Source code here is experimental. No NVIDIA binaries or weights are bundled.
