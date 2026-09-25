# Slow scene-reactive candidate preview

This opt-in diagnostic lets the game capture a staged proxy frame, runs the
current AMD candidate offline, then displays the result as a centered square
in `DebugView=2`. The game keeps showing the last completed square while the
next capture runs. Saved-frame passes took about four minutes; two passes
while Cyberpunk was running took 6.7 and 8.1 minutes on the development
machine. This is **not** real-time inference or a production replacement for
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
