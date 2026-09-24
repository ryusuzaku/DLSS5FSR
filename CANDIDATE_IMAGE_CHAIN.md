# Same-image candidate decoder chain (2026-09-24)

The optimized public FP16 ONNX model was run once on its 256×256 blue-marble
example. `tools/extract_peer_decoder48_inputs.py` extracted block47, encoder22
skip, block48 merge, and blocks48–55. Its block55 bytes match the earlier
independently extracted block55 boundary exactly. The ONNX model SHA256 is
`7aa891c46f90f3d0a4539701ba009131ac333602634a62ad8675da90d0f8a173`.

## Candidate split encoder on the same image

`tools/extract_peer_split512_encoder_inputs.py` extracts public block22,
blocks23–30, and the 4×4 C1024 head from one blue-marble inference. The
block30 skip bytes match the independent block39 extraction. Block22 and the
head are FP32 public graph boundaries; blocks23–30 are half-rounded.

`tools/check_split512_encoder_peer_image.py` converts public block22 to the
candidate C512 basis and FP8, then runs AMD blocks23–30 at 8×8 with the
native 0,3,1,2,0,3,1,2 shift schedule. All 104 spatial/body stages and eight
device input/output handoffs match the scalar candidate exactly. The block30
raw output feeds the 2×2 pool and C1024 head; both AMD stages match exactly.
Block30 skip vs public FP16 is 0.71386 MAE/0.85549 correlation, and the head
is 0.86382/0.88383. The all-zero-shift control reaches block30 at
0.14838/0.99557 and the head at 0.11039/0.99825. This difference reflects
the public ONNX encoder's zero-shift windows; the native-shift run remains the
candidate path. Neither comparison validates an original NVIDIA kernel.

`tools/check_vit31_peer_image.py` takes the *native-schedule AMD head device
bytes*, applies the 4×4 source-derived logical ViT map on AMD, and runs all of
ViT31, including a **candidate** 16-key attention reduction and final
projection. `tools/check_vit16_peer_image_chain.py` continues its device output
through ViT32–38 and applies the inverse logical map on AMD. Every HIP stage
and device handoff matches its scalar candidate exactly. The bridge's 16,384
values are exact against the declared gather. The final ViT38 projection
device SHA256 is
`2126421cb98ea2f64690d91929d201cb00164ab3c03105e9a41e3799737da361`.

The 16-token attention uses the public graph's 16 valid keys and the recovered
score/exponent transform, with a declared half-rounded reduction. The available
native attention reference covers 64/128/256 tokens; **the original 16-token
reduction order has not been checked against an original kernel**. The original
physical C512→ViT and ViT→C512 maps are also unverified. In the public ViT
basis, the ViT38 candidate has 0.43860 MAE and 0.87659 correlation against
the same-image public FP16 graph. `tools/audit_vit16_peer_image.py` checks
every ViT31–38 public stage: MAE falls from 1.02998 at ViT31 to 0.43860 at
ViT38, while correlation ranges from 0.87659 to 0.90114. These are diagnostics
of the candidate against a different FP16 graph, not original-kernel parity.

The candidate ViT38 output and AMD encoder30 skip then feed AMD decoder39–69
and the full 256² head. All body/prefix/handoff checks remain exact against
the scalar candidate, including all 1,024 head windows and the separate direct
GPU-buffer merge/body/RGB check. Block39 vs public FP16 has 0.61445 MAE and
0.89599 correlation; block69 has 1.08336/0.99061. With the public gain, the
blended RGB image has 0.00493 MAE and 0.99918 correlation against the public
FP16 final image, compared with 0.00805 input-color MAE. The render is finite
and visually coherent. It still uses public block22, encoder22/14/8/4 skips,
preblock skip, and color. No original kernel, independent C256/C32 logical-map
holdout, or production game wiring has been validated.

After the public extractions and HIP builds, replay this candidate line with
large generated fixtures outside Git:

```powershell
$offload = Join-Path $HOME 'DLSS5FSR-build-offload'
python tools/check_split512_encoder_peer_image.py --output-root "$offload/peer_split512_encoder_candidate"
python tools/check_vit31_peer_image.py --encoder-root "$offload/peer_split512_encoder_candidate" --output-root "$offload/peer_vit31_candidate16"
python tools/check_vit16_peer_image_chain.py --vit31-root "$offload/peer_vit31_candidate16" --output-root "$offload/peer_vit16_candidate"
python tools/audit_vit16_peer_image.py --vit31-root "$offload/peer_vit31_candidate16" --chain-root "$offload/peer_vit16_candidate"
python tools/check_decoder39_peer_image.py --candidate-vit38 "$offload/peer_vit16_candidate" --candidate-skip30 "$offload/peer_split512_encoder_candidate" --output-root "$offload/decoder39_candidate_vit16"
python tools/check_split512_peer_image.py --amd-block39 --amd-block39-dir "$offload/decoder39_candidate_vit16" --output-root "$offload/peer_split512_candidate_vit16"
python tools/check_upsample48_peer_image.py --chain-report "$offload/peer_split512_candidate_vit16/report.json" --block39-dir "$offload/decoder39_candidate_vit16" --output-root "$offload/upsample48_candidate_vit16"
python tools/check_decoder48_55_peer_image.py --candidate-vit16 --prefix-root "$offload/upsample48_candidate_vit16" --output-root "$offload/peer_decoder48_candidate_vit16"
python tools/check_upsample56_peer_image.py --candidate-vit16 --chain-root "$offload/peer_decoder48_candidate_vit16" --output-root "$offload/upsample56_candidate_vit16"
foreach ($b in 56..61) { python tools/check_block56_candidate.py --case image_candidate_vit16 --block $b --width 32 --height 32 }
python tools/audit_peer_decoder56_tail.py --case image_candidate_vit16
python tools/check_upsample62_peer_image.py --case from_candidate_vit16_fp8
foreach ($b in 62..65) { python tools/check_block62_candidate.py --case from_candidate_vit16_fp8 --block $b --width 64 --height 64 }
python tools/audit_peer_decoder62_tail.py --case from_candidate_vit16_fp8
python tools/check_upsample66_peer_image.py --case from_candidate_vit16_fp8
foreach ($b in 66..69) { python tools/check_block66_peer_candidate.py --case from_candidate_vit16_fp8 --block $b --width 128 --height 128 }
python tools/audit_peer_decoder66_tail.py --case from_candidate_vit16_fp8
python tools/check_head70_peer_frame.py --latent-case from_candidate_vit16_fp8 --output-root "$offload/peer_head_frame_256_candidate_vit16"
python tools/check_head70_peer_gpu_chain.py --latent-case from_candidate_vit16_fp8 --frame-dir "$offload/peer_head_frame_256_candidate_vit16" --output-dir "$offload/peer_head_frame_256_candidate_vit16_gpu"
```

A separate decoder continuation combines the public *logical* ViT38 with the
candidate AMD block30 skip. `check_decoder39_peer_image.py --candidate-skip30`
then feeds AMD blocks39–69 and the full 256² head via case-specific fixtures.
The block39 output vs public FP16 is 0.32561 MAE/0.96044 correlation; block47
is 0.62656/0.90115; block55 is 1.06460/0.91014; block61 is
1.01223/0.96901; block65 is 1.03792/0.98873; and block69 is
0.66831/0.99560. Every AMD/scalar stage and device handoff passes. The
block69 device SHA256 is
`100e53d44b40cbf84f837f407478082f3f80d706c507487a3e6679c0bffe35f9`.

The head's 1,024 windows each pass 12 body checks, both gain settings pass
their outer checks, and the direct GPU-buffer test matches merge, body, and
both RGB outputs exactly. Public-gain blended RGB vs the public FP16 final
has 0.00481 MAE/0.99923 correlation over 196,608 values; the input-color
baseline is 0.00805 MAE. The render is finite and visually coherent. This
still uses public ViT38, encoder22/14/8/4 skips, preblock skip, and color;
the candidate split encoder supplies only the block30 decoder skip. No
original NVIDIA execution or in-game neural rendering is established.

After the prior public boundary extractions and HIP build, replay this branch
with generated fixtures on a spacious drive:

```powershell
$offload = Join-Path $HOME 'DLSS5FSR-build-offload'
python tools/extract_peer_split512_encoder_inputs.py
python tools/check_split512_encoder_peer_image.py --output-root "$offload/peer_split512_encoder_candidate"
python tools/check_split512_encoder_peer_image.py --unshifted-control --output-root "$offload/peer_split512_encoder_unshifted"
python tools/check_vit31_peer_image.py --encoder-root "$offload/peer_split512_encoder_candidate" --output-root "$offload/peer_vit31_prefix"
python tools/check_decoder39_peer_image.py --candidate-skip30 "$offload/peer_split512_encoder_candidate" --output-root "$offload/decoder39_candidate_skip30"
python tools/check_split512_peer_image.py --amd-block39 --amd-block39-dir "$offload/decoder39_candidate_skip30" --output-root "$offload/peer_split512_candidate_skip30"
python tools/check_upsample48_peer_image.py --chain-report "$offload/peer_split512_candidate_skip30/report.json" --block39-dir "$offload/decoder39_candidate_skip30" --output-root "$offload/upsample48_candidate_skip30"
python tools/check_decoder48_55_peer_image.py --encoder-skip30 --prefix-root "$offload/upsample48_candidate_skip30" --output-root "$offload/peer_decoder48_candidate_skip30"
python tools/check_upsample56_peer_image.py --encoder-skip30 --chain-root "$offload/peer_decoder48_candidate_skip30" --output-root "$offload/upsample56_encoder_skip30"
foreach ($b in 56..61) { python tools/check_block56_candidate.py --case image_encoder_skip30 --block $b --width 32 --height 32 }
python tools/audit_peer_decoder56_tail.py --case image_encoder_skip30
python tools/check_upsample62_peer_image.py --case from_encoder_skip30_fp8
foreach ($b in 62..65) { python tools/check_block62_candidate.py --case from_encoder_skip30_fp8 --block $b --width 64 --height 64 }
python tools/audit_peer_decoder62_tail.py --case from_encoder_skip30_fp8
python tools/check_upsample66_peer_image.py --case from_encoder_skip30_fp8
foreach ($b in 66..69) { python tools/check_block66_peer_candidate.py --case from_encoder_skip30_fp8 --block $b --width 128 --height 128 }
python tools/audit_peer_decoder66_tail.py --case from_encoder_skip30_fp8
python tools/check_head70_peer_frame.py --latent-case from_encoder_skip30_fp8 --output-root "$offload/peer_head_frame_256_encoder_skip30"
python tools/check_head70_peer_gpu_chain.py --latent-case from_encoder_skip30_fp8 --frame-dir "$offload/peer_head_frame_256_encoder_skip30" --output-dir "$offload/peer_head_frame_256_encoder_skip30_gpu"
```

## Same-image AMD block39 boundary

`tools/extract_peer_decoder39_inputs.py` extracts public ViT38, split-encoder30
skip, block39 merge, and block39 output from the same optimized FP16 inference.
The block39 bytes match the separate C512 extraction. The original block39
record's 512×1024 projection has an exact 524,288-position raw-index relation
to the public QMMA decoder in the candidate C1024/C512 channel basis.

`tools/check_decoder39_peer_image.py` rounds those public inputs to FP8,
runs the C1024→C512 projection and 8×8 upsample/skip merge on AMD, and passes
three exact device/scalar stages. The output has 0.04418 mean absolute error
and 0.99941 correlation against public FP16 block39. Its device bytes feed
native-schedule C512 blocks40–47, block48, C256 blocks48–55, and the block56
projection/skip merge. All their stages and handoffs pass exactly; block47
vs public FP16 is 0.39551/0.96129 (MAE/correlation), block55 is
0.71607/0.96112, and the block56 FP8 merge is 0.73472/0.97945.

This path begins with public *logical* ViT38 and split-encoder30 tensors.
Its block39 inverse-map test is an identity control; it does not validate the
native physical ViT→C512 bridge or an AMD ViT/encoder run. The earlier 16×4
decoder39 source-composed control still passes after extending the AMD test
to the image's 4×4 ViT geometry.

The separate `from38_fp8` continuation runs C128 blocks56–61, block62 with
the public encoder8 skip, C64 blocks62–65, block66 with the public encoder4
skip, C32 blocks66–69, and the 256² head. All candidate AMD/scalar stages and
device handoffs pass exactly. Block69 vs public FP16 is 0.67469 mean absolute
error and 0.99580 correlation; its device SHA256 is
`a5ab9a67a38f268bcc115d2daabb04c335160aad1369cc92e3768447afc58e25`.
All 1,024 head windows pass 12 exact checks each. The direct GPU-buffer merge
and body (2,097,152 values each) and both RGB outputs (196,608 values each)
also pass exactly. Public-gain blended RGB vs public FP16 final has 0.00429
mean absolute error and 0.99936 correlation; the input-color baseline has
0.00805 mean absolute error. The rendered image is finite and visually
coherent. This is a same-image public-input candidate run, with no original
NVIDIA kernel or production game wiring validated.

## Upstream C512 same-image boundary

`tools/extract_peer_split512_inputs.py` extracts public block39 and blocks40–47
from the same 256×256 image and model. Its block47 bytes match the independent
block48 input extraction. `tools/check_split512_peer_image.py --teacher-forced`
feeds public block39 through candidate AMD blocks40–47 at 8×8 using the native
0,3,1,2,0,3,1,2 window schedule. All 104 GPU/scalar stage checks and all eight
device handoffs pass exactly. After the FP8-rounded input, connected block47
has 0.96215 correlation and 0.39006 mean absolute error against the public
FP16 block47. Feeding its device bytes to
`tools/check_upsample48_peer_image.py --split512` gives exact AMD/scalar
projection and merge checks; the merge is 0.98027 correlated with the public
FP16 merge, with 0.50072 mean absolute error. This path starts at public
block39, so its ViT/decoder39 upstream remains unverified on the image.

The public ONNX `SplitSwinBlock` uses zero-shift attention in every block;
the optimized ONNX graph has no attention Pad node in blocks40, 41, or 45.
The candidate native schedule includes shifted windows. Teacher-forcing each
block from its public predecessor isolates this difference: block41/45 native
schedule mean absolute errors are 0.25986/0.21202, while zero-shift controls
are 0.06147/0.05773. A connected all-zero-shift control reaches block47 at
0.99621 correlation and 0.11888 mean absolute error. That control explains
why the public model is an imperfect oracle for native shifted windows; it is
not an alternate native-kernel validation. Both schedules and their metrics
are recorded separately by the script.

`tools/check_decoder48_55_peer_image.py --split512` continues the native
schedule's block48 merge through candidate C256 blocks48–55 on AMD; every
stage and device handoff again agrees exactly with its scalar candidate.
Block55 vs public FP16 has 0.96097 correlation and 0.71727 mean absolute
error. `tools/check_upsample56_peer_image.py --from39` consumes that device
output and the same-image encoder14 skip. Its C256→C128 projection (32,768
values) and skip merge (131,072 values) pass exactly; the FP8 merge vs public
FP16 has 0.97930 correlation and 0.73654 mean absolute error. This connected
native-schedule path then runs through blocks56–61, the encoder8 skip and
block62 prefix, blocks62–65, the encoder4 skip and block66 prefix, and
blocks66–69. Every device stage and handoff matches the scalar candidate
exactly. Block61 vs public FP16 has 0.97731 correlation/0.86468 mean absolute
error; block65 has 0.99035/0.97426; block69 has 0.99582/0.67474. The
block69 device SHA256 is
`e75111a727230658034cecdee841bf01283e45bb683c8b9b9bdf3dc8b613d076`.

The 256² `from39_fp8` head consumed that block69 device output with the
same-image public preblock skip and color. All 1,024 windows passed the 12
exact AMD/scalar body checks each, and both gain settings passed their outer
checks. The separate direct GPU-buffer test passed exact merge and body
comparisons (2,097,152 values each) and RGB comparisons (196,608 values
at each gain). The public-gain blended image has 0.00423 mean absolute error
and 0.99940 correlation against the public FP16 final RGB over 196,608 values;
the input-color baseline has 0.00805 mean absolute error. The rendered image
is finite and visually coherent. This is still a static candidate comparison:
the public block39, encoder skips, preblock skip, and color are input sources;
no original NVIDIA kernel or production game path was executed.

`tools/check_upsample48_peer_image.py` converts the public C512/C256 channel
bases to candidate native order and rounds the activations to FP8. Its
projection index map agrees with the public QMMA decoder at all 131,072 raw
coefficient positions. On the local AMD GPU, the block48 C512→C256 projection
matches the scalar reference for all 16,384 values, and its skip merge matches
for all 65,536 values. The merged FP8 candidate and public FP16 boundary have
0.99942 correlation and 0.07130 mean absolute error.

`tools/check_decoder48_55_peer_image.py` uses that actual same-image skip and
the candidate C256 maps for blocks48–55 at 16×16. Each GPU gather, FFN,
attention and scatter stage agrees exactly with the scalar reference. Compared
in the public model's channel order, block48 output has 0.99220 correlation
and 0.29824 mean absolute error; block55 has 0.98442 correlation and 0.42580
mean absolute error. The block55 device hash is
`1cd436cdba9878ab4bdd7590db4cac0bcc1e463c769c045334bd6d2277347b80`.

`tools/check_upsample56_peer_image.py --candidate-block55` consumes those
block55 device bytes and the same-image public encoder14 skip. Its block56
projection (32,768 values) and merge (131,072 values) both match their scalar
references exactly. The resulting FP8/public FP16 merge comparison has
0.99331 correlation and 0.40122 mean absolute error.

These are candidate coordinate and arithmetic checks. The public model runs
FP16 while the AMD candidate uses native-style FP8 boundaries. No original
NVIDIA C256/C32 kernel was executed, and the candidate C256 body maps remain
unvalidated by independent logical assets or an original-kernel output.
The public repo excludes the ONNX model, packed tensors, original DLL/cubins,
generated fixtures and game captures. A local workspace with matching source
data, ROCm, and the reference decoder is needed to replay the scripts.

## Extension through the full frame

The block56 device merge above was passed through candidate blocks56–61,
then block62 with the same-image encoder8 skip, blocks62–65, block66 with the
same-image encoder4 skip, and blocks66–69. Every prefix, body stage, and
interblock device handoff matched the scalar candidate exactly. At the final
block69 latent, the candidate/public FP16 comparison has 0.99706 correlation
and 0.47920 mean absolute error over 524,288 values. The block69 device hash
is `428f16285506440571ecfda73fe16181cbf71e0a16d160563209c83a8ae09f1d`.

`tools/check_head70_peer_frame.py --latent-case from48_fp8` then used that AMD
latent with the same-image public preblock skip and color. All 1,024 head
windows passed 12 exact GPU/scalar body checks each; both gain settings passed
the outer checks. A separate `tools/check_head70_peer_gpu_chain.py` run passed
direct GPU-buffer merge and body comparisons (2,097,152 values each) and RGB
comparisons (196,608 values at each gain), all exact. The public-gain blended
candidate image differs from the public FP16 final image by 0.00403 mean
absolute error, with 0.99942 correlation over 196,608 RGB values. The input
image baseline is 0.00805 mean absolute error. The resulting image is finite
and visually coherent, but it is a static public-model comparison and does
not validate the original NVIDIA kernels or production game wiring.

The independent C256 coordinate check is prepared in
`tools/validate_c256_logical_assets.py`. Its verdict is pending a matching
full package with logical files; the synthetic parser/checker smoke run is
not map evidence.

This full-frame run used an optional output directory on a second local drive
because generated fixtures exceed the free space in the workspace. The output
directory is not part of the public checkout.

After the public-model extractions listed in `README.md` and a build of
`tools/build_split512_block.sh`, the candidate can be replayed with:

```powershell
python tools/check_upsample48_peer_image.py
python tools/check_decoder48_55_peer_image.py
python tools/check_upsample56_peer_image.py --candidate-block55
foreach ($b in 56..61) { python tools/check_block56_candidate.py --case image_from_block48 --block $b --width 32 --height 32 }
python tools/audit_peer_decoder56_tail.py --case image_from_block48
python tools/check_upsample62_peer_image.py --case from48_fp8
foreach ($b in 62..65) { python tools/check_block62_candidate.py --case from48_fp8 --block $b --width 64 --height 64 }
python tools/audit_peer_decoder62_tail.py --case from48_fp8
python tools/check_upsample66_peer_image.py --case from48_fp8
foreach ($b in 66..69) { python tools/check_block66_peer_candidate.py --case from48_fp8 --block $b --width 128 --height 128 }
python tools/audit_peer_decoder66_tail.py --case from48_fp8
python tools/check_head70_peer_frame.py --latent-case from48_fp8 --output-root "D:\scratch\peer_head_frame_256_from48_fp8"
python tools/check_head70_peer_gpu_chain.py --latent-case from48_fp8 --frame-dir "D:\scratch\peer_head_frame_256_from48_fp8" --output-dir "D:\scratch\peer_head_frame_256_from48_fp8_connected_gpu"
```

The example output directory should be changed to a drive with ample free
space. All generated fixtures stay outside Git.

The upstream C512 comparison can be replayed separately after the same-image
public extraction and C512 HIP test build:

```powershell
& build/peer_onnx_venv/Scripts/python.exe tools/extract_peer_split512_inputs.py
& build/peer_onnx_venv/Scripts/python.exe tools/check_split512_peer_image.py --teacher-forced
& build/peer_onnx_venv/Scripts/python.exe tools/check_split512_peer_image.py --unshifted-control --teacher-forced
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample48_peer_image.py --split512
& build/peer_onnx_venv/Scripts/python.exe tools/check_decoder48_55_peer_image.py --split512
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample56_peer_image.py --from39
foreach ($b in 56..61) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block56_candidate.py --case image_from39 --block $b --width 32 --height 32 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder56_tail.py --case image_from39
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample62_peer_image.py --case from39_fp8
foreach ($b in 62..65) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block62_candidate.py --case from39_fp8 --block $b --width 64 --height 64 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder62_tail.py --case from39_fp8
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample66_peer_image.py --case from39_fp8
foreach ($b in 66..69) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block66_peer_candidate.py --case from39_fp8 --block $b --width 128 --height 128 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder66_tail.py --case from39_fp8
& build/peer_onnx_venv/Scripts/python.exe tools/check_head70_peer_frame.py --latent-case from39_fp8 --output-root "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from39_fp8"
& build/peer_onnx_venv/Scripts/python.exe tools/check_head70_peer_gpu_chain.py --latent-case from39_fp8 --frame-dir "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from39_fp8" --output-dir "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from39_fp8_connected_gpu"
```

For the newer block39 boundary, after the public extractions above:

```powershell
& build/peer_onnx_venv/Scripts/python.exe tools/extract_peer_decoder39_inputs.py
& build/peer_onnx_venv/Scripts/python.exe tools/check_decoder39_peer_image.py
& build/peer_onnx_venv/Scripts/python.exe tools/check_split512_peer_image.py --amd-block39
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample48_peer_image.py --amd-block39
& build/peer_onnx_venv/Scripts/python.exe tools/check_decoder48_55_peer_image.py --amd-block39
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample56_peer_image.py --from38
foreach ($b in 56..61) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block56_candidate.py --case image_from38 --block $b --width 32 --height 32 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder56_tail.py --case image_from38
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample62_peer_image.py --case from38_fp8
foreach ($b in 62..65) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block62_candidate.py --case from38_fp8 --block $b --width 64 --height 64 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder62_tail.py --case from38_fp8
& build/peer_onnx_venv/Scripts/python.exe tools/check_upsample66_peer_image.py --case from38_fp8
foreach ($b in 66..69) { & build/peer_onnx_venv/Scripts/python.exe tools/check_block66_peer_candidate.py --case from38_fp8 --block $b --width 128 --height 128 }
& build/peer_onnx_venv/Scripts/python.exe tools/audit_peer_decoder66_tail.py --case from38_fp8
& build/peer_onnx_venv/Scripts/python.exe tools/check_head70_peer_frame.py --latent-case from38_fp8 --output-root "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from38_fp8"
& build/peer_onnx_venv/Scripts/python.exe tools/check_head70_peer_gpu_chain.py --latent-case from38_fp8 --frame-dir "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from38_fp8" --output-dir "$HOME\DLSS5FSR-build-offload\peer_head_frame_256_from38_fp8_connected_gpu"
```

The scripts default their large C512 fixtures to a home-directory offload
folder; `--output-root` can select another spacious drive for the C512 runner
or block48 prefix. The C256 runner also accepts `--prefix-root` and
`--output-root`; block56 accepts `--chain-root` and `--output-root` to follow
those custom locations. The small source tensors and reports stay under
`build/`. The downstream `from39_fp8` case writes its large body fixtures to
`$HOME\DLSS5FSR-build-offload` and small audit manifests to `build/`.
