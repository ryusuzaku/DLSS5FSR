#!/usr/bin/env bash
# Standalone correctness test; builds our own HIP kernels only.
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
ROCM="${ROCM_ROOT:-}"
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
    --offload-arch=gfx1201 hip/mvp1/head70_test.hip -o build/head70_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
    --offload-arch=gfx1201 hip/mvp1/head70_body_test.hip -o build/head70_body_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
    --offload-arch=gfx1201 hip/mvp1/head70_normalized_test.hip -o build/head70_normalized_test.exe
