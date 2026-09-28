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
assumptions. The next integration step is accepting each new frame's GPU
activation while retaining weights/workspace, then connecting this component
to its neighboring stages. The current scene preview still uses its existing
file-based diagnostics.
