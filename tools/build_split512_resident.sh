#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/split512_resident_test.hip -o build/split512_resident_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/split512_frames_test.hip -o build/split512_frames_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/resident_frame_server.hip -o build/resident_frame_server.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/front_resident_test.hip -o build/front_resident_test.exe
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 -shared hip/mvp1/resident_engine.hip -o build/resident_engine.dll
MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/resident_engine_test.hip -o build/resident_engine_test.exe
