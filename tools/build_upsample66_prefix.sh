#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
ROCM="${ROCM_ROOT:-}"
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
    --offload-arch=gfx1201 hip/mvp1/upsample66_prefix_test.hip -o build/upsample66_prefix_test.exe
