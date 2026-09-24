# Same-image candidate decoder chain (2026-09-24)

The optimized public FP16 ONNX model was run once on its 256×256 blue-marble
example. `tools/extract_peer_decoder48_inputs.py` extracted block47, encoder22
skip, block48 merge, and blocks48–55. Its block55 bytes match the earlier
independently extracted block55 boundary exactly. The ONNX model SHA256 is
`7aa891c46f90f3d0a4539701ba009131ac333602634a62ad8675da90d0f8a173`.

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
```

The scripts default their large C512 fixtures to a home-directory offload
folder; `--output-root` can select another spacious drive for the C512 runner
or block48 prefix. The small source tensors and reports stay under `build/`.
