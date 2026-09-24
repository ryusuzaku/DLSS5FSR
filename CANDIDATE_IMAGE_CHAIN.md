# Same-image candidate decoder chain (2026-09-24)

The optimized public FP16 ONNX model was run once on its 256×256 blue-marble
example. `tools/extract_peer_decoder48_inputs.py` extracted block47, encoder22
skip, block48 merge, and blocks48–55. Its block55 bytes match the earlier
independently extracted block55 boundary exactly. The ONNX model SHA256 is
`7aa891c46f90f3d0a4539701ba009131ac333602634a62ad8675da90d0f8a173`.

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
