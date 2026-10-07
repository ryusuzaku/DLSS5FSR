#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/split512_resident_test.hip -o build/split512_resident_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/split512_frames_test.hip -o build/split512_frames_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/resident_frame_server.hip -o build/resident_frame_server.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/front_resident_test.hip -o build/front_resident_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include -shared hip/mvp1/resident_engine.hip -o build/resident_engine.dll
# Half-precision activations (S332): the same network without E4M3 activation
# rounding, far steadier frame to frame; the game's engine since S332.
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS -DH70_NO_FP8 \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include -shared hip/mvp1/resident_engine.hip -o build/resident_engine_hp.dll
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/resident_engine_test.hip -o build/resident_engine_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/resident_engine_full_test.hip -o build/resident_engine_full_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/resident_engine_frame.hip -o build/resident_engine_frame.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/resident_engine_seq.hip -o build/resident_engine_seq.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/resident_engine_sync_test.hip -o build/resident_engine_sync_test.exe -ld3d12 -ldxgi
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include hip/mvp1/vit_attention_chunks_test.hip -o build/vit_attention_chunks_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -I ext/rocWMMA/library/include -include functional hip/mvp1/resident_profile.hip -o build/resident_profile.exe
