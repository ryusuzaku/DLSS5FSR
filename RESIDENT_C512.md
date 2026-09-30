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

The following extension adds decoder56's transition and the C128
decoder56–61 section. The upstream skip producers also need resident
execution before the full path can operate without external skip inputs.

## Resident decoder56–61 with an explicit encoder14 skip

`c128_resident::Prefix` takes decoder55's device output and a separate
encoder14 skip. Its projection, skip scale and workspace stay resident.
The merged output feeds six C128 blocks using the existing kernels and
shift schedule `0, 2, 0, 3, 1, 2`. `c128_resident::Chain` retains the weights
and reuses activation and padded-window buffers across submissions.

The harness uploads the initial C512 input, encoder22 skip and encoder14
skip before each measured region. Captured A uses the verified AMD
encoder14 output from the same captured-frame replay. B rolls that skip
four columns, consistent with the control's spatial scale; zero supplies
a zero skip. These are synthetic pairs, not recomputed upstream encoder
results. The larger case uses a reproducible FP8 Gaussian skip with seed
5601 and standard deviation 0.03125.

```powershell
& $py tools\check_resident_decoder61.py 'C:\path\to\resident-through-decoder55' `
  --decoder47-source 'C:\path\to\resident-through-decoder47' `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-through-decoder61'
```

The decoder47 source must match the report recorded by the decoder55 run;
it supplies the earlier ViT and decoder39 fixtures. For the larger case,
use its matching decoder55/decoder47 directories, `--synthetic-skip` instead
of `--captured-candidate ...`, and
`--head-fixture build\split512_bridge_32x8`. The existing resident build
script includes this mode. `native_command.json` records the prepared
invocation for later replay without regenerating scalar references.

| C512 input / C128 output grid | ViT-through-decoder stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| 8×8×512 / 32×32×128 | 415 | 80 | 19.16–19.42 ms |
| 32×8×512 / 128×32×128 | 415 | 80 | 62.55–62.85 ms |

Block56's transition adds four checks and the six C128 bodies add 90
stage/handoff checks. All 16 endpoints per submission match independent
scalar/standalone references through A/B/zero/B/A. The captured case's
seven prefix arrays and 198 C128 arrays also match the earlier full-frame
replay. The five older modes retain all 170 exact endpoint comparisons.

Both cases verify that the supplied encoder14 skip and decoder55 input
remain unchanged. The 44 invalid-view controls across both runs clear stale
outputs and recover, and separate diagnostics reject an altered encoder14
skip expectation and an altered block57 input before publishing frames.
The C128 attention residual order remains a candidate; no original NVIDIA
kernel was executed for these checks.

The measured region now reaches decoder61 and includes device copies used
by the harness. Setup, three host input uploads and validation readbacks
are excluded. It has zero explicit buffer allocations and zero host
transfers, with weights loaded once. These remain component timings on
prepared inputs, not whole-model or in-game frame times. Game and preview
execution have not been switched to this component.

Next is block62's transition and the C64 decoder62–65 section with the
matching encoder8 skip, followed by C32 and the final head. Upstream
encoder/skip residency and production integration also remain unfinished.

## Resident decoder62–65 with an explicit encoder8 skip

`c64_resident::Prefix` now consumes decoder61's device output and a separate
encoder8 skip. Four C64 blocks follow with shifts `0, 3, 1, 2`; their weights,
ping-pong activations and padded-window workspace stay resident. The existing
kernels are unchanged, including block62's sequential float32 projection
accumulation with half rounding after each 32 products. The independent
reference uses `project_sequential_f32` to preserve that rounding order.

```powershell
& $py tools\check_resident_decoder65.py 'C:\path\to\resident-through-decoder61' `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-through-decoder65'
```

Rebuild with `tools/build_split512_resident.sh` first. The checker requires
the earlier decoder61 report, references and `native_command.json`, plus
the ancestor fixture paths recorded there. It verifies saved endpoint/input
hashes, redecodes the C128 weights and block56 prefix, and uses only the
repository's known executable to replay the prepared chain. It generates
independent scalar/standalone references for the new prefix and C64 blocks.
For the larger decoder61 source, use `--synthetic-skip` instead of
`--captured-candidate ...`.

Captured A uses the hash-verified encoder8 output from the same AMD captured
frame. B rolls that skip eight columns; zero supplies a zero skip. These are
synthetic control pairs, not new upstream encoder executions. The larger
case uses an FP8 Gaussian skip with seed 6201 and standard deviation 0.03125.
The initial C512 input and all three external skips upload before timing.

| C512 input / C64 output grid | ViT-through-decoder stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| 8×8×512 / 64×64×64 | 479 | 95 | 22.09–23.12 ms |
| 32×8×512 / 256×64×64 | 479 | 95 | 77.51–80.12 ms |

The transition adds four checks and the C64 bodies add 60. All 19 endpoints
per submission match through A/B/zero/B/A, with three distinct decoder65
outputs. The seven captured prefix arrays and 132 body arrays also match
the earlier full-frame replay byte for byte. All six older modes retain
their 250 exact endpoint comparisons.

Across both sizes, 56 null/wrong-size input controls clear stale outputs and
recover on valid submissions. Separate diagnostics reject altered skip8
and block63 input expectations before publishing frames. The supplied skip
and decoder61 input remain unchanged. No original NVIDIA kernel was run;
the existing logical-map and attention residual-order assumptions remain.

The measured interval covers encoder23–30, ViT31–38 and decoder39–65,
including bridges, transitions and harness device copies. Weights load once;
there are zero explicit buffer allocations or host transfers inside that
interval. Setup, four input uploads and validation readbacks are excluded.
These are component timings, not full-model or in-game frame times, and the
preview still runs its existing diagnostic path.

Next are block66's transition, C32 decoder66–69 and head70. Upstream encoder
and skip residency, candidate-contract validation and game integration
remain necessary for a complete live path.

## Resident C32 decoder66–69 and head70

`c32_resident::Prefix`, `Chain` and `Head` extend the prepared GPU path from
decoder65 through enhanced RGB. Block66 consumes a separate block4 skip,
then four C32 bodies run with shifts `0, 3, 1, 2`. Their gather/scatter passes
use the existing audited transition basis and public QMMA candidate weights.
The head consumes decoder69's device output, a preblock0 skip and input RGB.
It merges directly into window order, runs the C32 body, converts its raw
half-rounded output back to the native channel basis, and computes both
native-gain (0.03125) and public-gain (1.0) enhanced RGB. The finish pass uses
the raw body output, before FP8 quantization.

All weights and GPU workspace remain allocated between submissions. The
new head body shares the same arithmetic kernels as the standalone checks;
the C32 transition and outer head passes also keep their existing kernels.
The checker uses sequential float32 projection sums with 32-product half
boundaries and independent scalar head traces. Bounded standalone body
chunks are assembled in window order for the resident stage checks.

```powershell
& $py tools\check_resident_head70.py 'C:\path\to\resident-through-decoder65' `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-through-head70'
```

Rebuild using `tools/build_split512_resident.sh`. Keep the decoder65 report,
references, `native_command.json` and its ancestor fixtures available. The
checker verifies saved endpoints and inputs, redecodes C64/prefix62 weights,
and checks the capture provenance of the new skips and RGB. For the larger
decoder65 source, replace `--captured-candidate ...` with `--synthetic-skip`.
The prepared invocation is saved again for subsequent replay.

Captured A uses the same-frame public ONNX block4 skip rounded to FP8, the
public preblock0 skip, and prepared linear RGB. These two skip producers
remain outside the resident chain. B rolls block4 by 16 columns and
preblock0/RGB by 32 columns; zero supplies zeros. They are synthetic pairs,
not recomputed upstream results. The larger case uses seed 6670, Gaussian
skips with standard deviation 0.03125 (FP8 block4, FP16 preblock0), and
uniform RGB. The harness uploads seven input arrays before each measurement:
the initial C512 tensor, five skips and RGB.

The connected candidate now has 563 ViT-through-head stage checks and 28
byte-exact endpoints per submission. Null/wrong-size checks include all
three head inputs and clear every exposed head view before valid-input
recovery. Separate diagnostics corrupt the block4 skip, block67 handoff,
preblock0 skip and RGB expectations. Borrowed decoder outputs and all
supplied inputs must remain unchanged.

| C512 input / enhanced RGB grid | ViT-through-head stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| 8×8×512 / 256×256×3 | 563 | 140 | 34.02–46.61 ms |
| 32×8×512 / 1024×256×3 | 563 | 140 | 112.83–135.20 ms |

Both cases run A/B/zero/B/A with three distinct decoder69 and native-gain
RGB outputs. The captured case matches all 369 saved C32/head arrays from
the earlier full replay, including the head's body chunks and both gains.
All 80 invalid-view controls and eight altered-expectation diagnostics pass
across both sizes. Regression runs retain 345 exact endpoints across the
seven older modes; the new decoder69-only mode adds 110 exact endpoints.
The interval covers encoder23–30, ViT31–38, decoder39–69 and head70, plus
the harness's device copies and both RGB gain calculations. It has zero
explicit GPU buffer allocations and zero host transfers. Setup, seven
input uploads and validation readbacks remain outside the interval; these
measurements are not full-model or in-game frame times.

These are still prepared component checks. Original C32 body maps, the
earlier logical-map assumptions and native input/output contracts remain
unverified. Enhanced RGB here precedes the preview's final blend; neither
the preview nor game DLL uses this resident runner yet. Upstream encoder
and skip residency, full-path performance work and live integration remain.

## Resident encoder15–22 and block22 downsample

`c256_resident::Chain` now has an encoder mode, and `c256_resident::Downsample`
adds the block22 pool and C512 projection. With two extra arguments, the
harness uploads one C256 boundary14 tensor per frame in place of the initial
C512 input and skip22. Encoder15–22 run with shifts `0, 3, 1, 2, 0, 3, 1, 2`.
Block22 writes both its FP8 skip and its unquantized projection. The pool
consumes the raw output, never the quantized skip. It rounds horizontal sums,
combines and rounds, multiplies by 0.25, then applies FP8. The projection
keeps its 32-product half boundaries. The downsampled C512 feeds encoder23
directly. The encoder22 skip stays borrowed on the device until block48
consumes it in the same frame.

```powershell
& $py tools\check_resident_encoder22.py `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-encoder22'
& $py tools\check_resident_encoder22.py --synthetic --output-root 'C:\path\to\resident-encoder22-32x8'
```

Each frame's C512 input and skip22 now come from its own encoder execution,
so every downstream reference is regenerated. The checker first builds
independent encoder15–22 and downsample22 references for A, B and zero. It
then reruns the existing checkers from `check_split512_frames.py` through
`check_resident_head70.py`, declaring those C512 inputs and skip22 arrays.
Captured A starts from the saved candidate encoder14 downsample. Its block22
skip, raw output, downsample matrix and downsampled output match the
canonical candidate byte for byte. Captured A and zero then reproduce the
earlier head70 endpoints byte for byte. B rolls the C256 input two columns,
so it is now an actual encoder execution rather than a rolled C512 input.
The larger case uses a synthetic FP8 Gaussian C256 input (seed 2215,
standard deviation 5.6957, the captured boundary14 value). Skip14/8/4/0 and
RGB remain external synthetic pairs, as before.

The harness adds 125 stage checks on the first frame: 122 for the encoder
chain, including block22's raw projection and raw scatter, and 3 for the
downsample. It adds four exact endpoints per submission: skip22, raw22,
pool22 and the downsampled C512 input. Null and wrong-size controls cover
both new stages. Separate diagnostics corrupt the block16 handoff, the
block22 raw output and the pool expectations. Uploads fall from 101·n to
100·n floats per frame (n = width·8·512): the C256 input, skips14/8/4/0 and RGB.

| C256 input / enhanced RGB grid | Encoder-through-head stage checks | Exact output arrays | Combined device interval |
|---|---:|---:|---:|
| 16×16×256 / 256×256×3 | 688 | 160 | 38.10–41.12 ms |
| 64×16×256 / 1024×256×3 | 688 | 160 | 147.66–155.77 ms |

Both cases pass 44 invalid-view controls and all three altered-expectation
diagnostics. The measured interval covers encoder15–30, ViT31–38,
decoder39–69 and head70, plus the harness's device copies. It has zero
explicit GPU buffer allocations and zero host transfers. The nine older
harness modes retain 595 exact endpoints. These are still prepared component
checks, not full-model or in-game frame times. Original C256 logical maps
remain unverified. Encoder5–14 and their skips, the public block4/preblock0
producers, the preview and the game DLL are not connected yet.

## Resident encoder5–14 and the block8/block14 downsamples

`c128_resident` and `c64_resident` now have the same encoder mode and
`Downsample` class. Encoder9–14 run with shifts `0, 3, 1, 2, 0, 3`, and
encoder5–8 with `0, 3, 1, 2`. Blocks 14 and 8 keep their unquantized
projection for the pool, and each downsample feeds the next C256 or C128
chain directly. Two more argument pairs select these fronts. The 24-argument
mode uploads the C128 boundary8 tensor and produces skip14 on the device.
The 26-argument mode uploads the C64 boundary4 tensor and also produces
skip8. Each skip stays borrowed until its decoder prefix consumes it in the
same frame.

```powershell
& $py tools\check_resident_encoder22.py --first-block 5 `
  --captured-candidate 'C:\path\to\captured-frame\candidate' `
  --output-root 'C:\path\to\resident-encoder8'
& $py tools\check_resident_encoder22.py --first-block 5 --synthetic --output-root 'C:\path\to\resident-encoder8-32x8'
```

`--first-block 9` stops at the C128 front; the default of 15 keeps the
previous behavior. The checker builds independent references for every
selected encoder stage and each of A, B and zero. It then regenerates the
downstream cascade, declaring the C512 inputs and the skip22, skip14 and
skip8 arrays. Finally it runs every front mode in turn (22, 24 and 26
arguments) on the same frames. Captured A starts from the saved candidate
block4 boundary. Its block8/14/22 skip, raw output, downsample matrix and
output match the canonical candidate byte for byte at every stage. Captured
A and zero reproduce the earlier head70 endpoints, including skip14 and
skip8. B rolls the upload eight C64 columns. The larger case uses seed 805
and standard deviation 7.0665, the captured boundary4 value. Skip4, skip0
and RGB remain external.

| Front | Upload per frame | Stage checks | Exact arrays | 8×8 interval | 32×8 interval |
|---|---|---:|---:|---:|---:|
| encoder15 | C256, skips 14/8/4/0, RGB (100·n) | 688 | 160 | 37.58–39.72 ms | 132.49–141.64 ms |
| encoder9 | C128, skips 8/4/0, RGB (98·n) | 783 | 180 | 41.16–44.09 ms | 147.70–157.04 ms |
| encoder5 | C64, skips 4/0, RGB (94·n) | 848 | 200 | 45.77–48.15 ms | 161.03–172.14 ms |

The encoder5 mode adds 62 + 3 and the encoder9 mode 92 + 3 first-frame stage
checks. Each adds four exact endpoints per submission (skip, raw, pool and
downsampled output) and four invalid-view controls. Nine altered-expectation
diagnostics per case corrupt the second block's handoff, the last block's
raw output and the pool expectations of every stage. All were rejected
before any output was written. The nine older harness modes still reproduce
all 595 saved endpoints. The interval now covers encoder5–30, ViT31–38,
decoder39–69 and head70. It has zero explicit GPU buffer allocations and
zero host transfers. These are still component timings, not full-model or
in-game frame times.

The larger synthetic inputs exposed rare half-rounding ties between NumPy
matmul and the kernels' sequential non-FMA float32 sums. There was one value
in a C64 QKV, one in a C512 pre-projection, and a few in a C32 body. The
C64/C128/C256 FFN and attention references, the C512 block reference and
the C32 body trace now accumulate in kernel order. The C32 trace still
checks the upstream matmul oracle, allowing only rare differences of up to
four half steps from such ties. The captured case, regenerated with these
references, still reproduces the earlier A and zero endpoints byte for
byte. Older fixtures were not regenerated; their replays are unchanged.

The public block4/preblock0 producers, the preview and the game DLL remain
unconnected. Original C64/C128/C256 logical maps are still candidate
assumptions.

## Resident C32 front end (stem, block0–4, block4 downsample)

`front_resident::FrontEnd` replaces the last public ONNX producers. From
linear RGB it runs:
- the 15-channel stem (noise, constant, twice the centred colour),
- block0 at full resolution,
- a 2×2 pool,
- encoder C32 blocks 1–4 with shifts `0, 3, 0, 3`,
- the block4 C32→C64 downsample.

It writes the resident chain's C64 input, block4 skip and preblock0 skip in
the native basis. The C32 bodies reuse the head70 body kernels unchanged.
The front end itself runs in the public channel basis and converts only at
its outputs, with the same maps as before.

Weights come from the original tensors:
- block0 (`tensor_000`) is a C32 body with a 16×32 FP16 stem inserted at
  `0x2010` (address bits r0 r3 c3 r1 r2 c0 c1 c2 c4; row 15 is padding),
- blocks 1–3 are plain C32 bodies,
- block4 (`tensor_091`) holds its 64×32 FP8 downsample at `0x50B0` in the
  public QMMA layout.

Both recovered layouts reproduce the public graph's coefficients exactly;
`front_end_reference.py` rechecks this on every run. The public graph's
32×32 texture adapter is the identity. Its noise texture is a public
constant, tiled for wider synthetic extents (an assumption). Shifts were
chosen by comparing all four per block against the public graph.

```powershell
& $py tools\check_resident_front.py 'C:\path\to\prepared' --output-root 'C:\path\to\front-check'
& $py tools\check_resident_frame_server.py 'C:\path\to\resident-encoder8' `
  --output-root 'C:\path\to\server-check' --front-fixture 'C:\path\to\front-check\captured'
```

Both extents pass all 78 front-end stage checks, and five repeated runs are
byte-exact. The front end takes 10.4–11.9 ms at 256×256 and 43.6–47.0 ms at
1024×256. On the captured frame, the candidate outputs compare with the
public ones they replace at correlation 0.9974 (C64 input), 0.9929 (block4
skip) and 0.9996 (preblock0 skip). The difference is by design: the
candidate rounds to FP8 between blocks, while the public graph only clamps
in FP16. The server's front-end mode, given only colour, reproduces exactly
the RGB the chain gives when fed the saved front-end outputs. Its full
colour-to-RGB path takes 56–58 ms at 256×256.

`run_candidate_scene_preview.py --engine resident --resident-front
<server-check report>` uses this path, with no ONNX at run time. On the
saved storefront capture its preview is within 3 levels (PSNR 55.6 dB) of
the ONNX-fed resident preview.

## Performance work

`build/resident_profile.exe <engine config> <capture>` times each stage and
each kernel of the engine's network with GPU events. It uses an optional
launch hook that is null in normal runs. At 1024×768 (the 991×620 game
frame), the network went from 630 ms to 253 ms with every regression still
byte-exact:
- `tiled_gemm.hip` stages 32-wide input and weight slices in LDS. Each
  output keeps its sequential float32 sum and half boundary. The kernel
  covers raw, FP8, gate and double-residual epilogues, column groups and K
  partitions, with taller or wider tiles for narrow outputs. The C32 bodies,
  the C64–C256 blocks, and the ViT/C512 blocks above 64 tokens use it.
  Smaller token counts keep the one-thread-per-output kernels, which are
  faster there.
- Softmax denominators and Q/K norms were recomputed for every element.
  They are now computed once per row or head, in two-phase kernels with
  coalesced element passes.
- The tiled h70 rounding uses a half conversion that equals
  `__float2half_rn` for every one of the 2^32 float inputs (checked
  exhaustively), in the form that stays fast inside tiles.
- Verification-only outputs (C64–C256 linear projection, raw expanded
  arrays) are skipped outside stage checks.

Crop mode takes 33 ms per frame, and the 26-argument chain at 256×256
takes about 32 ms.

### Fast path (default) and `RESIDENT_EXACT=1`

The default path takes the network at 1024×768 from 253 ms to about 37 ms.
Two of the changes stay bit-identical and are used in both modes:
- Rounding helpers use hardware conversions. `split512_half`/`head70_half`
  use `__float2half_rn`, which is value-identical to the old volatile cast
  for all 2^32 floats and about 27 times faster. `split512_fp8` and `h70_f`
  use the gfx12 E4M3 instructions after a ±448 clamp, which match the
  software rounding for every non-NaN float and for every half value.
  `volatile` temporaries were removed. They only prevented contraction,
  which `-ffp-contract=off` already does.
- The head70 finish runs once per pixel for both outputs. Products of half
  values are exact in float, and the truncated terms are summed as
  integers. It falls back to the old kernel when the coefficients are not
  half-valued.

Four changes use f16 WMMA (16×16×16, fp32 accumulate). Every operand is an
FP8 or half value, so the conversion to half is exact and so is each
product. Only the fp32 summation order inside a 32-product slice differs
from the sequential reference. The rounding boundaries, epilogues and fixed
softmax denominator orders are unchanged:
- `wmma_gemm.hip`: every `tiled::gemm` (half weight copies cached per
  weight pointer, float activations converted while staging, epilogue
  straight from the accumulator registers).
- `c32_fused.hip`: `k_c32_wmma` runs a whole C32 body window (FFN,
  attention, projection) in one workgroup. `k_c32_fused` is the exact
  scalar form of the same fusion.
- `split512_attention_wmma.hip`: one workgroup per window and head goes from
  raw qkv to the FP8 context for the C64–C512 blocks, replacing seven
  launches.
- `vit_attention_wmma.hip`: the ViT numerator as a query×key×channel
  product, with one denominator per query.

On the captured data, the fast path reproduces every captured-chain check,
the front end at both extents, the engine crop checks, and the whole
1024×768 engine frame byte for byte. The synthetic chain differs at one
rounding tie (one value in 4.7 million, one half step). `RESIDENT_EXACT=1`
selects the sequential kernels everywhere, and the regressions run in that
mode. `RESIDENT_WMMA=0` disables only the GEMM path.

### Full-frame output options

Optional `key=value` lines after the 24 engine config lines control how
the engine writes the full frame (the input's alpha is always kept):
- `transfer=full` (default) writes the network's RGB. `transfer=luma`
  keeps the game's colour and applies only the network's luminance change
  (Y_out / Y_in on the linearized input). The upscaler input in Cyberpunk
  2077 comes before the game's tone mapper and grade (its values are
  washed out when viewed as sRGB), so the network's hue there does not
  match the final image; other AMD ports use the same luminance-only
  transfer for it.
- `limit=L` clamps the luminance ratio to [1-L, 1+L].
- `smooth=A` and `sigma=S` blend the ratio with the previous frame's,
  gated by exp(-|Y_in - Y_in_prev| / S), against frame-to-frame shimmer.
- In luma mode the network produces a per-pixel ratio map; every frame
  the engine applies the latest map to the newest submitted frame
  (`k_compose`), so the image is never an old frame. Where the current
  input luminance differs from the one the map was computed on, the ratio
  fades by exp(-|dY| / G) (`gate=G`, default 0.03). HDR values at or above
  1.0 pass through, and the ratio fades out as a channel nears 1.0, since
  the network only sees clipped white there. `strength=K` scales the change.
- `gain=0|1` overrides the shim's gain. The native gain (.03125) changes
  the frame by well under one 8-bit level; the public gain (1.0) carries
  the network's full change.

