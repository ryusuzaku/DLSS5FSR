# Slow scene-reactive candidate preview

This opt-in diagnostic lets the game capture a staged proxy frame, runs the
current AMD candidate offline, then displays the result as a centered square
in `DebugView=2`. The game keeps showing the last completed square while the
next capture runs. The latest saved-frame pass took about 2.4 minutes with
shared public reference inference and reused static audits. Earlier passes
while Cyberpunk was running took
6.7 and 8.1 minutes; the shared mode has not yet been timed in-game. This is
**not** real-time inference or a production replacement for
DLSS 5. Original C256/C32 maps, the C512 physical view, ViT reduction, and
the game's exact native input/output contracts still need validation.

The current candidate starts at the pinned public ONNX block-4 boundary and
uses public graph decoder skips. Its AMD stages run encoder5–30, ViT31–38,
decoder39–69 and head70 with internally exact HIP/scalar checks. Each pass
also runs public FP16 comparison boundaries from the same captured input.
This is a model-derived scene preview under those assumptions, not an
original NVIDIA-kernel result.

Build the shim and candidate HIP test executables following the main README.
Use the project's Python venv with NumPy, Pillow and ONNX Runtime and leave
the hash-verified public model in its documented local location. Output
files are private and can take several GB; they are excluded from Git.

With the game closed, `tools/activate_candidate_scene_preview.ps1` can back up
the currently installed three DLL names and INI, install the new built DLL,
and set the diagnostic keys below. Give it the game `bin\x64` directory, the
repository `build` directory, an existing private runtime directory, and a
**new** backup directory. Record that backup path for
`tools/restore_candidate_preview.ps1`.

In the game's `dlssnr_shim.ini`, use absolute paths to one writable directory:

```ini
DebugView=2
HipFeLive=0
CandidatePreviewPath=C:\path\to\scene_preview\candidate_preview.bin
CandidatePreviewReload=1
CandidateInputCapturePath=C:\path\to\scene_preview\game_capture.bin
CandidateInputCaptureTrigger=1
CandidateInputCaptureRepeat=1
CandidateInputGpuPath=C:\path\to\scene_preview\game_input_gpu.f32
```

Start the game, then start the sidecar from the repository in another
terminal. Keep the same capture and preview paths as the INI:

```powershell
$py = 'build\peer_onnx_venv\Scripts\python.exe'
& $py tools\run_candidate_scene_preview.py `
  --capture-path 'C:\path\to\scene_preview\game_capture.bin' `
  --output-root 'C:\path\to\scene_preview' `
  --preview-path 'C:\path\to\scene_preview\candidate_preview.bin' `
  --gpu-input-path 'C:\path\to\scene_preview\game_input_gpu.f32'
```

The sidecar creates `game_capture.bin.go`. The shim captures the next
completed staged frame and consumes the trigger. After all offline stages
pass, the sidecar atomically replaces `candidate_preview.bin`. With
`--gpu-input-path`, it verifies the same-frame HIP input against the raw
capture and uses that GPU tensor for the candidate. The shim
checks for replacements every 60 rendered frames. It then arms the next
capture. `status.json` reports the current stage, source capture hash,
preview hash, completion time and timings. Stop with Ctrl+C; `--max-updates 1`
processes one scene and exits. A stopped game causes the capture wait to time
out after 120 seconds, leaving the last valid preview available. If a hard
console shutdown leaves `game_capture.bin.go` behind, the next sidecar run
adopts that pending trigger and waits for the game to consume it.

The default preview exports the public-graph gain-1 enhanced image.
`--preview-gain native_gain` instead uses the `.03125` head gain reported
for original launches upstream; this gain has not been validated on our
game frame against an original GPU output. On one captured storefront
frame, CPU and HIP input tensors differed by at most `5.96e-8`, yet their
gain-1 candidate enhanced images differed by RGB MAE `0.0216`. The paired
HIP-input mode therefore matters for a prospective GPU runtime. Both gain
variants remain diagnostic choices, not proven original visual parity.

The frame runner and sidecar use one shared public ONNX inference per capture
by default. It supplies the same 85 named comparison boundaries used by the
individual stages (74 unique tensors). Each consumer verifies the prepared
input, source capture, source model, graph and tensor hashes. Public outputs
and all candidate checks are recomputed for every frame. Only the extracted
graph is cached between frames; damaged or stale reference data is rejected.

Pass `--reference-mode independent` to the frame runner or sidecar to run the
separate public branches for comparison. Standalone stage commands still use
their own branches unless given `--public-reference <directory>`. Their graph
extraction caches remain available. Neither mode provides real-time inference.

`tools/audit_shared_public_reference.py` checks a shared reference against a
saved run made with independent branches and exercises invalid-reference
rejection. For example:

```powershell
& $py tools\audit_shared_public_reference.py 'C:\path\to\prepared' `
  --reference-dir 'C:\path\to\shared-run\candidate\public_reference' `
  --baseline-root 'C:\path\to\independent-run\candidate' `
  --output-report 'C:\path\to\reference-audit.json'
```

Both runs must use the same prepared input. On two captured game scenes,
all 170 named boundaries matched byte for byte. The complete storefront
candidate also matched all 2,301 saved intermediate `.f32` files and the
exported preview. This verifies equivalence to the previous candidate, not
original-kernel or native game parity.

On the development machine, separate runs of that saved storefront frame
took 247 seconds with independent branches and 187 seconds with the shared
reference (about 24% less time). These are saved-frame measurements, not
in-game timings. The shared public pass took 11 seconds; candidate execution
and scalar checks still account for most of the remaining time.

Candidate C64/C128/C256 blocks also share their static PTX residual-address
audit within each stage process. The three PTX files are hashed on every
request; changed contents force a new audit. Every block still runs all of
its scalar/GPU comparisons. The standalone
`tools/audit_c256_residual_ptx.py` command always performs a fresh audit;
`tools/check_residual_audit_cache.py` verifies reuse and rejects a source edit
even when its file size and timestamp are preserved.

That change reduced full residual audits from 36 to four per saved-frame run.
The latest pass took 142 seconds and again matched all 2,301 intermediate
`.f32` files and the final preview exactly. Timings of unchanged stages also
varied, so the entire difference from the earlier 187-second run should not
be attributed to audit reuse.

The C32/C64/C128/C256, ViT and head candidate tests can now reuse one HIP
worker per Python stage. Build it with `bash tools/build_candidate_gpu_worker.sh` (also included
in `tools/build_split512_block.sh`), then run the frame runner or sidecar as
usual. Rebuild it after changing any included kernel or test source.

`--gpu-test-mode auto`, the default, uses the built worker and otherwise uses
the standalone test executables. `--gpu-test-mode worker` requires the worker;
`--gpu-test-mode process` selects the old execution path. Standalone stage
commands accept the same choice through `DLSS5_GPU_TEST_MODE`. The selected
mode is recorded in the frame report and completed sidecar status; stage
logs record worker startup and successful job counts.

The dispatcher runs 23 existing test entry points, including gather, FFN,
attention, scatter, downsample, ViT bridges, decoder39, block66 prefix and
head outer passes, with their exact comparisons. It keeps the HIP context alive
between jobs, but still allocates/frees buffers and reads/writes intermediate
files for each test. C512 and other launchers not included in the dispatcher
still use their existing processes. Captured diagnostic output preserves the
standalone PASS counts used by ViT, C32 prefix and head checks. This is an
offline diagnostic optimization.

Worker requests use length-prefixed arguments, including paths containing
spaces. Responses have a 120-second timeout and matching request IDs. Any
comparison, protocol, I/O or timeout failure retires the worker and fails the
request; a failed worker job is never retried through a standalone process.
`tools/check_gpu_test_runner.py` exercises repeat jobs, process equivalence,
missing-worker fallback, comparison failure, fatal I/O, timeout, malformed
requests and clean shutdown using synthetic data.

A complete saved storefront replay with the initial C64/C128/C256 worker matched all 2,301
intermediate `.f32` files, both final head manifests and the exported preview
byte for byte. Four workers handled 150 jobs (78 encoder, 32 C256 decoder,
24 C128 decoder and 16 C64 decoder). Those four stages took 31 seconds versus
74 seconds in the preceding standalone run. The full replay took 124 seconds
versus 142 seconds; unaffected stage timings also varied. Six repeated small
gather/scatter jobs separately took about 0.5 seconds in one worker versus
3.3 seconds in standalone processes. These are saved-frame diagnostic timings,
not live-game performance or evidence of original NVIDIA parity.

Extending the worker to ViT, C32 and the head also preserved all 2,301 tensors,
both final manifests and the preview exactly. Seven workers now handle 192
jobs per frame, including 11 ViT/decoder39 jobs, 13 C32 decoder jobs and 18
head jobs. The full saved-frame replay took 95 seconds versus the previous
124 seconds. C32 decoder time fell from 17.6 to 8.8 seconds and head time from
17.4 to 7.2 seconds; ViT took 22.5 seconds versus 21.3 previously. Other stages
also varied, so the overall difference is not a controlled benchmark of the
worker change alone. Scalar references and fixture processing remain substantial.

Replaying just those 42 newly supported GPU jobs against the same already
prepared fixtures took 16.1 seconds in standalone processes and 2.2 seconds
in workers. All 335 PASS lines matched, and the full tensor comparison still
passed afterward. This isolates test-launch overhead from the Python scalar
and fixture-generation work included in full-frame timings.

To process an already saved capture without a game, pass `--existing-capture
--max-updates 1` with its path. To return to normal rendering, stop the
sidecar, set `DebugView=0`, clear `CandidatePreviewPath` and
`CandidateInputCapturePath`, and restart the game. If you deployed over an
existing game shim, restore its saved DLLs and INI with
`tools/restore_candidate_preview.ps1` after closing the game.

The preview is intentionally a square at 256×256 candidate resolution. The
game's HUD may be drawn over it. Its age is visible in `status.json`; it is
not temporally aligned to current gameplay and should not be used for
quality or performance claims about a real-time port.
