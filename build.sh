#!/usr/bin/env bash
# Build the nvngx.dll shim and the harness.
#
#   ./build.sh            build both
#   ./build.sh clean      remove build/
#
# Requires tools/msvc.sh (MSVC 14.44 + Windows SDK 10.0.26100).
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"

# Relative on purpose: absolute MSYS paths can be mangled by POSIX-path
# conversion on their way to cl.exe.
OUT="build"

source tools/msvc.sh

INC="-Iext/nvngx_sdk -Iext/d3dx12 -Isrc/ngx -I."
LIBS="d3d12.lib dxgi.lib"

# tools/msvc.sh discovers the pip ROCm SDK or honors ROCM_ROOT. Without it,
# the shim builds without the HIP backend.
ROCM="${ROCM_ROOT:-}"

if [ "${1:-}" = "clean" ]; then
    rm -rf "$OUT"
    echo "cleaned"
    exit 0
fi

mkdir -p "$OUT"

# ---------------------------------------------------------------- shaders --
# fxc /Fh emits a C header containing the compiled blob.
echo "== blit.hlsl -> blit_cs.h"
(cd src/ngx/shaders && MSYS_NO_PATHCONV=1 "$FXC" /nologo /T cs_5_0 /E main \
     /Vn g_main /Fh blit_cs.h blit.hlsl)

echo "== dlssnr.hlsl -> dlssnr_cs.h"
(cd src/ngx/shaders && MSYS_NO_PATHCONV=1 "$FXC" /nologo /T cs_5_0 /E main \
     /Vn g_dlssnr /Fh dlssnr_cs.h dlssnr.hlsl)

echo "== counter.hlsl -> counter_cs.h"
(cd src/ngx/shaders && MSYS_NO_PATHCONV=1 "$FXC" /nologo /T cs_5_0 /E main \
     /Vn g_counter /Fh counter_cs.h counter.hlsl)

# ------------------------------------------------------------------- dll ---
echo "== compiling shim"
OBJS=()
for f in log parameter feature gpu ngx_api dllmain; do
    msvc_build /c /Fo:"$OUT/$f.obj" $INC src/ngx/$f.cpp || exit 1
    OBJS+=("$OUT/$f.obj")
done

# The HIP backend needs ROCm's headers for the interop TYPES (it loads the
# functions themselves dynamically; nothing links against amdhip64).
if [ -d "$ROCM/include" ]; then
    echo "== compiling hip_backend (ROCM: $ROCM)"
    msvc_build /c /Fo:"$OUT/hip_backend.obj" $INC -D__HIP_PLATFORM_AMD__ \
        -I"$ROCM/include" src/ngx/hip_backend.cpp || exit 1
    OBJS+=("$OUT/hip_backend.obj")
else
    echo "== ROCm not found at $ROCM -- building WITHOUT the HIP backend"
    echo "   (set ROCM_ROOT to a _rocm_sdk_core directory to enable it)"
fi

echo "== linking nvngx.dll"
MSYS_NO_PATHCONV=1 "$LINK" /nologo /DLL /DEF:src/ngx/exports.def \
    /OUT:"$OUT/nvngx.dll" "${OBJS[@]}" $LIBS || exit 1

# --------------------------------------------------------------- harness ---
echo "== compiling harness"
# /Fo matters here: without it MSVC drops ngx_harness.obj in the CWD, i.e. the
# repo root, while every other object goes to $OUT. Found by a hygiene pass.
msvc_build /Fe:"$OUT/ngx_harness.exe" /Fo:"$OUT/ngx_harness.obj" $INC \
    tests/ngx_harness.cpp $LIBS || exit 1

# ----------------------------------------------------------------- config --
# -n so a hand-edited build/dlssnr_shim.ini (dump settings, log level) is not
# reset on every rebuild.
cp -n src/ngx/dlssnr_shim.ini "$OUT/" 2>/dev/null || true

# The ROCm runtime DLLs must sit next to whatever loads nvngx.dll (the exe's
# directory is searched before PATH). Best effort -- missing ones simply mean
# the shim logs one degradation line and runs the identity model.
if [ -d "$ROCM/bin" ]; then
    echo "== copying ROCm runtime DLLs"
    for dll in amdhip64_7 amd_comgr rocm_kpack hiprtc0715 \
               hiprtc-builtins0715; do
        cp -u "$ROCM/bin/$dll.dll" "$OUT/" 2>/dev/null || \
            echo "   (warning: $dll.dll not copied)"
    done
fi

echo
echo "== done"
ls -la "$OUT/nvngx.dll" "$OUT/ngx_harness.exe"
echo
echo "run:  cd build && ./ngx_harness.exe"
