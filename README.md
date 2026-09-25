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
map/provenance files. `tools/validate_c256_logical_assets.py --raw-only`
checks extracted C256 tensors against the matching DLL; with logical assets,
the same tool checks all 14 ordinary C256 blocks without fitting C256 maps.
`ORIGINAL_MAP_VALIDATION.md` records which claims are
local checks and which are attributed to other projects.
The [original RTX 5080 ViT repack result](ORIGINAL_VIT_REPACK_VALIDATION.md)
validates the 4×4 and 8×8 physical byte maps against SM120 runtime output.
`tools/audit_vit16_reduction_sensitivity.py` quantifies how plausible 16-key
half-sum orders affect the saved candidate ViT stages without claiming an
original attention match.
`tools/audit_vit16_bridge_components.py` links a supplied original-map ZIP to
the saved candidate frame while keeping the unmeasured C512 cell view explicit.

An RTX 5080 volunteer can download the standalone
[volunteer kit](volunteer_5080/README.md). It records model/asset provenance and
runs a small original ViT repack probe without CUDA Toolkit; C256/C32 fused
kernel parity still requires a separate follow-up harness.

[CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md) records same-image
block39→head and block47→head candidate runs, plus an AMD block39→head path
started from public ViT38 and split-encoder30 tensors. The matching encoder22 skip is used at block48;
its AMD/scalar checks are exact, while comparisons to the public FP16 ONNX
model are approximate because the candidate rounds activations to FP8.
The same-image candidate has also been propagated through blocks56–69 and
the full-frame head; the document gives the measured RGB comparison and
original-kernel validation limits. The newer full-frame path starts at public
block39 and uses the candidate native C512 window schedule; public encoder
skips and preblock inputs still provide its upstream boundaries.

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
`extract_peer_decoder48_inputs.py`, `extract_peer_split512_inputs.py`, then
`extract_peer_decoder39_inputs.py`, then
`extract_peer_split512_encoder_inputs.py`.
Run `check_split512_peer_image.py --teacher-forced` for the native C512
window schedule and `check_split512_peer_image.py --unshifted-control
--teacher-forced` for the public ONNX window control. The public model's C512
blocks use zero-shift windows, so its FP16 outputs cannot validate the
native shifted schedule. Run
`check_upsample48_peer_image.py`, `check_decoder48_55_peer_image.py`, and
`check_upsample56_peer_image.py --candidate-block55` after building the HIP
test executables with `tools/build_split512_block.sh`.

The additional same-image encoder branch starts from public block22, runs
candidate AMD split-encoder blocks23–30 and the 4×4 C1024 head, and checks
the source-derived logical ViT bridge plus candidate ViT31–38. Run
`check_split512_encoder_peer_image.py` for the native shifted schedule and
`check_split512_encoder_peer_image.py --unshifted-control` to isolate the
public model's zero-shift behavior. `check_vit31_peer_image.py` consumes the
native-schedule head, then `check_vit16_peer_image_chain.py` and
`audit_vit16_peer_image.py`. [CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md)
gives the continuation through block39–69 and the full-frame head. The 16-token
attention reduction is experimental: its original-kernel order and physical
ViT bridge remain unverified.

The newer same-image branch begins at the public block14 downsample, runs
candidate AMD encoder15–22, and feeds both its block22 downsample and skip
through the connected ViT/decoder/head chain. Its source, replay commands,
RGB measurements, and original-map limits are in
[CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md). The C256 maps remain
candidate extensions pending independent logical assets or an original oracle.

An experimental upstream branch now starts at the public block8 downsample,
runs candidate AMD encoder9–14, and carries both its block14 downsample and
decoder skip through block70. The C128 coefficient and downsample maps have
measured upstream support; the attention residual order and full-image output
still need original-kernel validation. This branch improves on the input-image
baseline but compares worse to the public FP16 graph than the branch starting
at public block14. Measurements and replay commands are in
[CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md).

The latest branch starts at the public block4 downsample, runs AMD encoder5–8
with measured C64 maps, and carries its block8 downsample and skip through the
same connected encoder, ViT, decoder, and full-frame head. All declared
HIP/scalar stages and handoffs pass exactly. The public-gain blended frame has
0.005367 MAE against the same-image public FP16 output; see
[CANDIDATE_IMAGE_CHAIN.md](CANDIDATE_IMAGE_CHAIN.md) for replay and limits.

An opt-in **fixed-image game preview** can check whether that candidate frame
reaches the shim's D3D12/HIP model texture. Run
`python tools/export_candidate_preview.py` with the locally generated frame and
connected-GPU manifests, then set `CandidatePreviewPath` to the exported `.bin`
and `DebugView=2` in the game's `dlssnr_shim.ini`. The game should display a
centered square of the candidate image; the log reports an exact GPU pixel
readback and upload time. Set `CandidatePreviewPath=` and `DebugView=0` to
restore the normal game path. This preview replays one blue-marble image on
every frame. It does not apply the candidate network to the live scene, and
its upload timing is not inference timing. The live shim still uses its earlier
64-token encoder path; connecting the full 256×256 encoder, ViT, decoder, and
head to each game frame remains open work. If the preview DLL replaced an
existing game install, `tools/restore_candidate_preview.ps1 -BackupDir
<backup-folder> -GameDir <game-bin-x64-folder>` restores the saved DLLs and
configuration after the game is closed. A Cyberpunk 2077 run at 991×620
confirmed the candidate square, exact HIP readback and model-texture copies
without device errors. `DebugView=2` now outputs opaque alpha; the game's HUD
and some later-rendered elements still appear above the diagnostic view.

For a slow scene-reactive experiment, the [scene preview guide](SCENE_PREVIEW.md)
connects repeated staged captures to the offline candidate runner and lets
the game reload each completed preview. Live game passes have taken 6.7–8.1
minutes per update on the development machine. The paired HIP-input mode
checks and uses the game's same-frame GPU-prepared tensor; on one scene, tiny
CPU/HIP input rounding differences materially changed the gain-1 candidate
image. Both modes keep
the model assumptions above; it is not a real-time full-network game path.

For the next live-input boundary, `CandidateInputCapturePath` in
`dlssnr_shim.ini` writes one completed staged proxy frame to an absolute `.bin`
path. This capture does not alter the model output. Run
`python tools/prepare_candidate_input.py <capture.bin> --output-dir <directory>`
to center-crop and bilinearly resize it to 256×256, decode its sRGB proxy
values to linear `color_linear.f32`, and make a visual PNG plus manifest. The
binary format begins with `D5INP001`, then little-endian width, height,
bytes-per-pixel, row pitch, passthrough flag, proxy mode and float white point,
followed by the exact pitched rows. The converter accepts RGBA8 and FP16;
its crop/color contract is a diagnostic candidate and has **not** been
validated against the original NVIDIA frontend. The PNG is 8-bit and is not
an exact substitute for the linear `.f32` boundary.
Set `CandidateInputCaptureTrigger=1` to wait for a file named
`<capture.bin>.go` before capturing; create that file once the desired scene
is visible. The shim removes the trigger after saving the capture.
`tools/run_candidate_encoder64_from_capture.py <prepared-directory>
--through-block22` can then extract the pinned public FP16 block-4 boundary
from that input and run candidate HIP encoder blocks 5–22, including the
block-8, block-14 and block-22 downsample paths, checking each stage and
device handoff. Use `--through-block14` for a shorter run or omit both flags
to stop at block 8. The later captured-input runners continue that same
frame through encoder30, ViT31–38, decoder39–69 and the 256×256 head. Each
runner verifies its parent device hash and independently extracts the public
FP16 comparison boundaries from the prepared input. For example, with a
spacious output directory and the HIP test executables built:

```powershell
$py = 'build/peer_onnx_venv/Scripts/python.exe'
$input = '<prepared-directory>'
$out = '<output-directory>'
& $py tools/run_candidate_encoder64_from_capture.py $input --through-block22 --output-root "$out/encoder22"
& $py tools/run_candidate_encoder512_from_capture.py $input "$out/encoder22" --output-root "$out/encoder30"
& $py tools/run_candidate_vit39_from_capture.py $input "$out/encoder30" --output-root "$out/vit39"
& $py tools/run_candidate_decoder512_from_capture.py $input "$out/vit39" --output-root "$out/decoder47"
& $py tools/run_candidate_decoder256_from_capture.py $input "$out/encoder22" "$out/decoder47" --output-root "$out/decoder55"
& $py tools/run_candidate_decoder128_from_capture.py $input "$out/encoder22" "$out/decoder55" --output-root "$out/decoder61"
& $py tools/run_candidate_decoder64_from_capture.py $input "$out/encoder22" "$out/decoder61" --output-root "$out/decoder65"
& $py tools/run_candidate_decoder32_from_capture.py $input "$out/decoder65" --output-root "$out/decoder69"
& $py tools/extract_candidate_head_inputs_from_capture.py $input "$out/decoder69" --output-root "$out/head_inputs"
& $py tools/check_head70_peer_frame.py --latent-case capture_candidate --source-dir "$out/head_inputs" --candidate-latent "$out/decoder69/block69/output/output_device.f32" --candidate-report "$out/decoder69/report.json" --output-root "$out/head70"
& $py tools/check_head70_peer_gpu_chain.py --latent-case capture_candidate --frame-dir "$out/head70" --output-dir "$out/head70_connected_gpu"
```

The head writes `public_gain_blended.png`, `public_final.png`, and manifests
under the selected output directory. A captured Cyberpunk proxy completed
this offline chain with exact candidate GPU/scalar stage checks and direct
head-buffer handoffs. Its public-gain blended RGB had 0.004528 MAE against
the same-input public FP16 final image, compared with 0.005364 for the input
image; the enhanced RGB before blending had 0.017400 MAE. This is a
candidate/public diagnostic, not original-kernel parity or an in-game visual
improvement. Blocks 0–4, the block4 decoder skip and preblock0 skip are
supplied by the public graph. The C256/C32 maps, ViT reduction/physical
bridge, exact native input contract, and production full-frame timing remain
unvalidated. The normal game still uses its earlier live 64-token path.

The next diagnostic input stage has a standalone HIP implementation in
`hip/mvp1/candidate_input_256.hip`. It reads the already-staged RGBA8 or FP16
proxy directly from GPU memory and makes the same 256×256 center-square,
bilinear, sRGB-to-linear RGB tensor as `prepare_candidate_input.py`. Build its
test with `tools/build_candidate_input_256.sh`, then compare a capture with:

```powershell
& build/candidate_input_256_test.exe '<capture.bin>' '<prepared-directory>/color_linear.f32' '<output-device.f32>'
```

On the 991×620 FP16 Cyberpunk capture, all 196,608 floats were within
`5.96e-8` of the CPU converter (MAE `2.11e-9`); 1280×720 RGBA8 and FP16
harness captures also passed. Its roughly `0.0052 ms` warm-cache kernel timing
excludes game synchronization, transfers, network inference, and presentation.
The complete offline candidate can be replayed from a prepared input with
`tools/run_candidate_frame_from_capture.py <prepared-directory> --output-root
<spacious-directory>`; `--start-at <stage>` resumes at a named stage using
existing prior-stage outputs.

A full replay using the GPU-prepared game input passed all declared candidate
HIP/scalar and direct head-buffer checks. The public-gain blended image had
MAE `0.004450` against the same-input public FP16 final image (input baseline
`0.005363`). The GPU and CPU preprocessors differ by at most one float32 ULP,
yet public-gain candidate images differ by MAE `0.004568`; native-gain blended
images differ by only `0.000168`. This sensitivity is an unresolved candidate
gain/quantization issue, not a live-game quality result. The crop/color
contract has not been checked against the original NVIDIA frontend.

The shim now also supports `CandidateInputGpuPath` alongside
`CandidateInputCapturePath`. On the same confirmed staged frame, it runs
that kernel in HIP and saves a 256×256 interleaved linear RGB `.f32` tensor
at the absolute GPU path. `CandidateInputCaptureTrigger=1` gates both
files on the capture path's `.go` trigger. This is a one-shot diagnostic and
does not change the rendered output. In the D3D12/HIP harness, the live-path
GPU tensor was byte-for-byte identical to the standalone HIP result for both
RGBA8 and FP16 proxy formats; all 128 checks passed. A default run passed
117/117. A Cyberpunk scene capture at 991×620 FP16 subsequently passed the
same-frame paired check: the live GPU tensor exactly matched standalone HIP
output, with CPU-converter MAE `3.07e-9` and no clipped channels.

For a paired capture, run `tools/validate_candidate_input_pair.py` with the
raw `.bin` and GPU `.f32` paths. It prepares the CPU tensor, checks every
float, and can run `build/candidate_input_256_test.exe` for a byte-exact
standalone HIP comparison. On success it also writes a `gpu_prepared`
directory that can be passed directly to the full offline replay runner.
Generated images and tensors remain outside Git.

The storefront capture's GPU-prepared input completed that eleven-stage
offline replay, including all declared candidate stage and direct head-buffer
checks. Its public-gain blended output had MAE `0.005557` against the
same-input public FP16 final frame, versus `0.007720` for the input baseline.
This remains a 256×256 offline candidate image, not live full-network output
or original NVIDIA-kernel validation.

Source code here is experimental. No NVIDIA binaries or weights are bundled.
