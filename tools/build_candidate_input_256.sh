#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
ROCM="${ROCM_ROOT:-}"
if [ ! -x "$ROCM/bin/hipcc.exe" ]; then
    echo "ROCM_ROOT must name the installed ROCm SDK" >&2
    exit 2
fi
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off \
  -D_CRT_SECURE_NO_WARNINGS --offload-arch=gfx1201 \
  hip/mvp1/candidate_input_256_test.hip -o build/candidate_input_256_test.exe
