# Resident C512 component check

The C512 encoder23–30 and decoder40–47 can now execute as connected GPU
chains in a standalone diagnostic. Encoder block30's raw output also feeds
its pool/head kernels directly on the device. The runner uses the existing
kernels and their arithmetic unchanged.

This is a component replay on prepared, source-validated fixtures. It is
not wired into the scene preview or game DLL, does not execute the whole
model, and adds no original NVIDIA parity evidence.

## What stays on the GPU

The runner uploads the initial activation once and preloads all eight blocks'
weights. Two activation buffers alternate as input/output; one scratch
workspace is reused across blocks. There is no file or host activation
transfer between blocks. Encoder pool/head weights and buffers are also
resident when that step is selected.

An initial diagnostic execution reads back all existing stage comparisons
and checks each block's input against its saved fixture. Those later input
fixtures are comparison targets; they are never uploaded as block inputs.
The measured executions then repeat the same initial input with no device
allocation, host upload, readback, reference-file read or output-file write
inside the loop. Each repetition resets from the resident seed with one
device-to-device copy. Final outputs are downloaded after timing and compared
byte for byte with the saved standalone outputs.

HIP event timings include the device reset copy and connected kernel sequence.
They exclude weight loading, allocation, diagnostic readbacks and final output
transfer. They are not full-frame or game timings. The runner records its
explicit allocation and transfer counters in `resident_metrics.json`.

## Run

Build with the existing HIP/MSVC setup:

```bash
bash tools/build_split512_resident.sh
```

The main `tools/build_split512_block.sh` build also includes this executable.
Use candidate fixtures previously produced by the captured-frame runner:

```powershell
$py = '.\build\peer_onnx_venv\Scripts\python.exe'
& $py tools\check_split512_resident.py 'C:\path\to\candidate\encoder30' `
  --output-root 'C:\path\to\resident-encoder' --repeats 10 --controls
& $py tools\check_split512_resident.py 'C:\path\to\candidate\decoder47' `
  --first-block 40 --output-root 'C:\path\to\resident-decoder' --repeats 10 --controls
```

The Python checker verifies saved fixture hashes, compares weights with the
source-validated base records, checks the saved activation handoffs, and
compares the final connected outputs with the standalone device outputs.
It retains source manifest and initial-input hashes in `report.json`.
`--controls` also verifies that changing the second block's expected input
causes rejection before publishing output, and that an invalid extent fails.
Paths containing spaces are supported.

For the existing larger synthetic encoder fixture:

```powershell
& $py tools\check_split512_resident.py build\split512_spatial --width 32 `
  --head-fixture build\split512_bridge_32x8 `
  --output-root 'C:\path\to\resident-encoder32x8' --controls
```

Generate those fixtures first with the existing split512 chain and bridge
checks if they are absent. Output directories should be separate from the
input fixtures.

## Validation result

| Fixture | Stage/handoff comparisons | Repeats | Device interval per repeat | Setup |
|---|---:|---:|---:|---:|
| Captured encoder23–30, 8×8, plus pool/head | 115 | 10 | 4.37 ms | 802 ms |
| Captured decoder40–47, 8×8 | 112 | 10 | 4.37 ms | 713 ms |
| Synthetic encoder23–30, 32×8, plus pool/head | 115 | 10 | 12.49 ms | 1,106 ms |

All three cases matched their standalone final/raw outputs byte for byte,
including encoder head output. All reported zero allocations and zero host
transfers during the measured repetitions; both failure controls passed in
each case. Setup uploaded about 61–63 MB, mostly weights.

This establishes a connected C512 component under the current candidate
assumptions. The changing-input interface and adjacent gather check below
extend it; the current scene preview still uses its existing file-based
diagnostics.

## Changing inputs and GPU output views

`Chain::run_from_device(DeviceTensor)` accepts a borrowed device pointer with
the fixed chain's element count. It copies the activation into its reusable
workspace and leaves caller-owned input unchanged. `final_view()`,
`raw_view()` and `head_view()` expose borrowed output pointers and element
counts so another GPU operation can consume them directly. No weights are
reloaded and no buffers are reallocated for a new frame.

The interface uses one HIP device and its default stream. The caller must
keep the input allocation alive until the ordered copy completes. Output
views are valid until the next submission or chain destruction; operations
consuming them must be ordered before the next submission. The chain is
not a concurrent, multi-stream interface. A null input or wrong element
count fails before dispatch, clears the output views and prevents saving
stale output. A subsequent valid submission can recover. Diagnostic
`verify=true` still checks the original construction fixtures and is intended
for the fixed-input test; changing inputs have their own reference checks.

The new test reuses one chain for captured input A, spatially shifted input B,
zero input, B again, then A again. It prepares independent scalar/standalone
references before starting the resident process. A reusable incoming GPU
buffer represents the preceding stage. Output views feed reusable GPU
consumer buffers; on the encoder path, the head view also directly feeds
the existing ViT gather kernel. The gather arithmetic was moved unchanged
into a shared header. Its logical map remains the current candidate map,
without new original-kernel validation.

```powershell
& $py tools\check_split512_frames.py 'C:\path\to\candidate\encoder30' `
  --output-root 'C:\path\to\changing-encoder'
& $py tools\check_split512_frames.py 'C:\path\to\candidate\decoder47' `
  --first-block 40 --output-root 'C:\path\to\changing-decoder'
```

Use `--width 32 --head-fixture build\split512_bridge_32x8` with the synthetic
`build\split512_spatial` fixture for the larger encoder case. The build script
above produces both fixed-input and changing-input executables.

Across captured encoder/decoder and synthetic 32×8 encoder cases, all 15
submissions matched their references: 50 byte-exact output arrays, including
the two encoder cases' ViT gather outputs. All three distinct inputs also
produced distinct final outputs. Six invalid-view checks rejected null or
wrong-size inputs, invalidated stale output and recovered on the zero frame.
All caller input buffers remained byte-identical after processing.

The measured regions had zero host transfers and zero explicit device-buffer
allocations. The harness uploaded one new activation before each region and
downloaded results afterward; those transfers are accounted separately in
`frames_metrics.json`. Weights were uploaded only during setup. The 8×8
encoder intervals ranged from 4.67–6.73 ms, decoder 4.57–4.76 ms, and larger
32×8 encoder 12.89–13.04 ms. These are C512 feature-grid extents and component
timings, not full image dimensions or whole-model frame times. Intervals
include the input device copy, C512 work, output-consumer device copies and,
for the encoder, ViT gather.

## Resident ViT31–38 after the gather

The changing-input runner can now pass the GPU gather output directly into
eight resident ViT blocks. `vit_resident::Chain` preloads their weights and
reuses its scratch and activation buffers across all submissions. It uses
the existing ViT kernels unchanged, including the explicitly candidate
16-token attention reduction. Both 16-token and 64-token extents are covered;
32-token attention is rejected because it has no supported contract here.

The borrowed input/output lifetime and default-stream rules above also apply
to ViT. Its output view is cleared on a rejected input, and the next valid
submission recovers. Diagnostic verification compares each block's input
and nine arithmetic stages against saved scalar/standalone references. Later
block inputs are never uploaded from those reference files.

After producing a changing-encoder fixture with `check_split512_frames.py`, run:

```powershell
& $py tools\check_resident_encoder_vit.py 'C:\path\to\changing-encoder' `
  --output-root 'C:\path\to\resident-encoder-vit'
```

For the larger synthetic case, use the changing-encoder32x8 output and add
`--head-fixture build\split512_bridge_32x8`. Keep input and output directories
separate. The checker revalidates C512 source manifests and saved frame
hashes, reconstructs the gather map, and generates independent ViT references
for A, shifted B and zero. It then replays A/B/zero/B/A with one resident chain.
Allow about 1.3 GB per case for decoded diagnostic fixtures; none are bundled
with the public source. The existing resident build script builds this mode.

| Input feature grid | ViT tokens | ViT stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|---:|
| Captured encoder, 8×8×512 | 16 | 80 | 25 | 5.57–5.71 ms |
| Synthetic encoder, 32×8×512 | 64 | 80 | 25 | 20.20–20.85 ms |

These intervals include C51223–30, block30 pool/head, gather, ViT31–38,
input device copies and output-consumer device copies. They exclude setup,
host input upload, diagnostic checks and final readback. Both cases report
zero explicit device-buffer allocations and zero host transfers inside
their measured regions. These remain component timings, not whole-model
or in-game frame times.

All five endpoints (C512 final/raw, head, gather and ViT38) match byte for
byte on every submission. The three distinct inputs produce three distinct
ViT outputs. Eight invalid-view controls across the two cases clear stale
outputs and recover. An altered block32 reference input fails diagnostic
verification in each case, and unsupported attention extents are rejected.
The captured case's 152 ViT fixture arrays also match the earlier complete
captured-frame replay. Legacy C512 encoder and decoder modes retain their
30 exact changing-frame outputs.

The logical bridge and 16-token reduction remain candidate assumptions;
this adds no original NVIDIA-kernel parity evidence. The following extension
connects the inverse bridge and decoder39 to the resident decoder40–47 chain.
Game and scene-preview execution have not been switched to these components.

## Resident inverse bridge and decoder39–47

The combined runner now continues from ViT38 to decoder47 without a host
activation transfer. `decoder39_resident::Entry` retains its inverse map,
projection weights, skip scale and scratch buffers. It reads ViT38 and the
same submission's encoder30 final output directly from their device views,
then performs the inverse gather, projection and upsample/skip merge using
the existing kernels. The merge output feeds the existing resident C512
decoder40–47 chain. The encoder skip stays alive until the merge consumes it.

```powershell
& $py tools\check_resident_decoder47.py 'C:\path\to\resident-encoder-vit' `
  --output-root 'C:\path\to\resident-through-decoder47'
```

Use the larger resident-encoder32x8/ViT result with
`--head-fixture build\split512_bridge_32x8` for the 64-token case. Rebuild with
`tools/build_split512_resident.sh` first. The checker verifies its source
hashes and ViT handoffs, generates independent decoder39 and decoder40–47
scalar/standalone references, then exercises A/B/zero/B/A in one resident
process. The inverse map is the inverse of the existing candidate logical
gather; it is not newly recovered from NVIDIA runtime output.

| Input feature grid | ViT/decoder stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| Captured encoder, 8×8×512 | 197 | 50 | 8.97–9.21 ms |
| Synthetic encoder, 32×8×512 | 197 | 50 | 27.16–30.89 ms |

Each case checks 80 ViT stages/handoffs, decoder39's two input handoffs and
three arithmetic stages, and 112 decoder40–47 stages/handoffs. Ten endpoints
per submission match byte for byte, including the inverse-mapped activation,
projection, merged decoder39 output and decoder47 final/raw outputs. All
three distinct inputs produce distinct decoder47 outputs. The captured
case's seven decoder39 arrays and 208 decoder40–47 arrays also match the
earlier full-frame replay. The older C512 encoder, C512 decoder and
encoder/ViT modes retain their 55 exact changing-frame output arrays.

Twenty invalid-view checks across both cases clear stale output and recover
on valid input, covering null/wrong-size ViT and skip inputs as well as the
encoder, ViT and decoder chains. The harness also verifies that decoder39
leaves both borrowed inputs unchanged. Each case rejects an altered skip
expectation, a non-bijective inverse map, and an altered block41 input
expectation; failed diagnostics publish no frame outputs.

The measured interval now covers encoder23–30, pool/head, forward gather,
ViT31–38, inverse/decoder39 and decoder40–47, including device input and
output-consumer copies. It still excludes setup, host uploads and validation
readbacks. Weights are loaded once and the measured regions have zero
explicit device-buffer allocations and zero host transfers. These are
component timings on prepared inputs, not a complete model or game frame.

The extension below adds decoder48's transition and C256 blocks48–55.
Remaining decoder stages, upstream encoder residency and production game
integration are still ahead. Original logical maps, the 16-token reduction
and native input/output contracts remain unvalidated.

## Resident decoder48–55 with an explicit encoder22 skip

`c256_resident::Prefix` reads decoder47's device output and a separate C256
encoder22 skip view. It keeps block48's projection weights, skip scale and
workspace resident. Its projection/upsample/merge kernels feed
`c256_resident::Chain`, which retains all eight blocks' weights, alternating
activation buffers and scratch sized for the largest padded window grid.
The chain uses the existing gather, FFN, attention and scatter kernels with
their candidate arithmetic unchanged.

The skip is a second caller-owned GPU input. This harness uploads it before
each measured region, alongside the initial C512 input. It does not yet
produce encoder22 inside the resident chain. The captured A control uses
the hash-verified AMD encoder22 skip from the same captured-frame replay.
B pairs the prior synthetic shifted C512 input with a two-column shift of
that skip; zero pairs zero C512 input with a zero skip. B and zero are
explicit synthetic pairs, not outputs from a new full upstream inference.
The larger control uses a reproducible synthetic skip (FP8 Gaussian,
seed 4801, standard deviation 0.03125).

```powershell
& $py tools\check_resident_decoder55.py 'C:\path\to\resident-through-decoder47' `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-through-decoder55'
```

The captured candidate directory must contain the matching encoder22,
encoder30 and decoder47 results. For the larger 32×8 C512 input, use the
corresponding resident-through-decoder47 result, replace
`--captured-candidate ...` with `--synthetic-skip`, and add
`--head-fixture build\split512_bridge_32x8`. Rebuild with the existing
`tools/build_split512_resident.sh` script first.

| C512 input / C256 output grid | ViT-through-decoder stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| 8×8×512 / 16×16×256 | 321 | 65 | 14.41–14.87 ms |
| 32×8×512 / 64×16×256 | 321 | 65 | 45.95–47.27 ms |

The new section adds four prefix checks (two input handoffs, projection and
merge) and 120 C256 checks (input, gather, four FFN, eight attention and
scatter comparisons per block). Across A/B/zero/B/A, all 13 endpoints per
submission match independently generated scalar/standalone references byte
for byte. The captured case's seven prefix arrays and 264 C256 arrays also
match the earlier complete captured-frame replay. All four older modes
retain their 105 exact endpoint comparisons.

Both runs check that input and skip buffers remain unchanged. Their 32
invalid-view controls reject null/wrong-size inputs, clear stale results,
and recover on valid input. Separate diagnostics reject an altered skip22
expectation and an altered block49 input expectation before publishing
frame outputs. The original C256 logical maps are still candidate maps;
these checks add no original NVIDIA-kernel parity evidence.

The timing interval covers the connected encoder23–30, ViT31–38 and
decoder39–55 sections, including both bridge directions, transitions and
device copies used by the harness. It excludes setup, the two host input
uploads and validation readbacks. All weights load once, with zero explicit
buffer allocations or host transfers inside the measured regions. This
still is not whole-model or game performance, and neither the game DLL nor
scene-preview path has been switched to this runner.

Next is decoder56's transition and the C128 decoder56–61 section, using the
matching encoder14 skip. The upstream skip producers also need resident
execution before the full path can operate without external skip inputs.
