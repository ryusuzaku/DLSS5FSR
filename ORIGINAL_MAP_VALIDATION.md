# C256/C32 map validation and original-kernel route (S269, 2026-09-24)

## What is available in the maintainer's private workspace

`dlss5/nvngx_dlssnr.dll` is the signed original DLL (not published here; SHA256
`E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E`).
Its `sm_120` cubins and PTX are in `dlss5-analysis/cubins/`. The host has an
RX 9070 XT and no NVIDIA adapter. CUDA Driver API probes in
`ref/dlss5-port-HIP/Development/run_original_*.cpp` can launch the cubins
on a compatible `sm_120` NVIDIA GPU; they cannot execute on this AMD GPU.
[NVIDIA lists RTX 50 cards, including RTX 5090, as compute capability 12.0](https://developer.nvidia.com/cuda/gpus).

## Offline checks completed here

`tools/audit_c256_peer_basis_candidate.py` checks our extrapolated C256
FFN/attention raw-index maps against the public QMMA decoder after the
audited channel permutation. W1, headwise W2, W3, Q/K/V, attention projection
and bias agree at all **655,360** positions. Manifest:
`build/c256_peer_basis_candidate_audit.json`.

`tools/audit_opendlss_weight_layout.py` independently transcribes
[OpenDLSS-NR's host weight index and attention-bias decoders](https://github.com/maanHimself/OpenDLSS-NR/blob/main/src/nr_model.cpp)
and checks them against the public QMMA decoder: **671,744/671,744** C32/C256
positions agree, including all tested matrices and relative bias. The original
raw tensor hashes for blocks48 and67 are recorded in
`build/opendlss_weight_layout_audit.json`. This corroborates packing across
independently written decoders. It is still a check of *interpretations of the
same serialized bytes*, not a replay of an original kernel.

[The public ONNX reconstruction's author reports direct RTX 5090 `sm_120`
probes](https://huggingface.co/taowen/dlss5-onnx/blob/main/docs/DLSS5_PYTORCH_FIXES.md):
C32 FFN matched all 32 output channels for constant and varying input; C32
down projection matched the full 64×32 matrix under channel impulses; 4096
Swin bias positions were recovered; bias-only attention matched 4094/4096
exactly, with the remainder within 0.0078125. These are **source-attributed
local operator results**, not captures we can independently replay here.
[OpenDLSS-NR likewise reports bit-exact block-boundary comparisons](https://github.com/maanHimself/OpenDLSS-NR),
but says its original captures are not in that repository.

The [DLSS5@AMD author's porting worklog](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/main/Development/history/porting-worklog.md)
also reports exact original C256 block15 output for one real and two
randomized inputs (8,192 values each), after extending layouts measured at
C64/C128. This is useful external original-kernel evidence for that block,
but its raw captures are not available locally, and our AMD candidate has
not been run against the same original inputs here.

## Independent validation still needed

The official [DLSS5@AMD source README](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting)
says the complete model assets are separate from Git and are included in
the full packages. Its [regular 0.29 OptiScaler package](https://pan.quark.cn/s/209e04e7acaf)
is a possible source for `DLSS5-AMD/native-game-tiled-assets`; this package
has **not** been downloaded or checked here. Earlier 0.28 guest download
attempts failed with Quark code23018 (file-size limit). The local 0.28/0.28.1
source ZIPs and other peer ZIPs do not contain these assets.

`tools/inspect_dlss5_logical_assets.py` checks a full ZIP or extracted asset
directory without installing the package. The four local source/peer ZIPs
have 0/16 required C32 files and 0/28 C256 follow-up files. See
`ASSET_ORACLE_HANDOFF.md` for commands and return artifacts.

`tools/validate_c256_logical_assets.py --raw-only` verifies all 14 ordinary
C256 packed tensors byte-for-byte against the signed DLL at the extracted
offsets: **9,649,248/9,649,248** bytes. The full validator is ready to check
the predeclared C256 maps against 28 supplied logical files (12,393,584
values), but no genuine matching logical package has been obtained here.
Synthetic directory/ZIP and one-coefficient-mutation tests check report and
failure behavior only; they are not logical-map evidence.

If the full asset directory or ZIP becomes available, run
`python tools/recover_head70_maps.py --assets <path>` to infer C32 maps from
logical weights, verify every coefficient, and hold out blocks69/70. Then
run `python tools/validate_c256_logical_assets.py --assets <path> --dll <path>
--out <report.json>` for the C256 block15–21/49–55 exact comparison before
promoting their status. A Blackwell machine instead allows direct probes of
the local signed cubins. Neither route can be replaced by a small RGB MAE
against the optimized public FP16 ONNX graph, whose arithmetic differs from
the native FP8 path.

No production DLL or game file was changed by these audits.
