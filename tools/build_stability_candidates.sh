#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
mkdir -p build
for test in motion_noise stabilize_guided resident_engine_motion_noise; do
  MSYS_NO_PATHCONV=1 "$ROCM_ROOT/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off \
    -D_CRT_SECURE_NO_WARNINGS --offload-arch=gfx1201 "hip/mvp1/${test}_test.hip" -o "build/${test}_test.exe"
done
