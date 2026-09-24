#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/msvc.sh
ROCM="${ROCM_ROOT:-}"
mkdir -p build
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/split512_block_test.hip -o build/split512_block_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/split512_window_test.hip -o build/split512_window_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/split512_bridge_test.hip -o build/split512_bridge_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/vit_bridge_ptx_test.hip -o build/vit_bridge_ptx_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/vit_expand_chain_test.hip -o build/vit_expand_chain_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/decoder39_entry_test.hip -o build/decoder39_entry_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/upsample48_prefix_test.hip -o build/upsample48_prefix_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial256_window_test.hip -o build/spatial256_window_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c256_ffn_candidate_test.hip -o build/c256_ffn_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c256_attention_candidate_test.hip -o build/c256_attention_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial256_output_test.hip -o build/spatial256_output_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/upsample56_prefix_test.hip -o build/upsample56_prefix_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c128_ffn_candidate_test.hip -o build/c128_ffn_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c128_attention_candidate_test.hip -o build/c128_attention_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial128_test.hip -o build/spatial128_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial128_output_test.hip -o build/spatial128_output_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/upsample62_prefix_test.hip -o build/upsample62_prefix_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c64_ffn_candidate_test.hip -o build/c64_ffn_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c64_attention_candidate_test.hip -o build/c64_attention_candidate_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial64_test.hip -o build/spatial64_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial64_output_test.hip -o build/spatial64_output_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/upsample66_prefix_test.hip -o build/upsample66_prefix_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial32_peer_test.hip -o build/spatial32_peer_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/c32_peer_body_test.hip -o build/c32_peer_body_test.exe
MSYS_NO_PATHCONV=1 "$ROCM/bin/hipcc.exe" -std=c++17 -O2 -ffp-contract=off -D_CRT_SECURE_NO_WARNINGS \
  --offload-arch=gfx1201 hip/mvp1/spatial32_peer_output_test.hip -o build/spatial32_peer_output_test.exe
