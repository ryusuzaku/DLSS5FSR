# Source this from bash to get an x64 MSVC command line:
#   source tools/msvc.sh
#
# Done by hand rather than via vcvars64.bat because cmd.exe /c does not survive
# being invoked from this Git Bash (the /c flag is eaten and cmd starts
# interactively). Everything below is what vcvars64.bat would have set.

MSVC_VER=14.44.35207
SDK_VER=10.0.26100.0
MSVC="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/$MSVC_VER"
KITS="C:/Program Files (x86)/Windows Kits/10"

# cl.exe wants INCLUDE/LIB as native Windows paths with ';' separators, so they
# are written out in Windows form directly rather than converted.
# winrt is in the list only because <wrl/client.h> (ComPtr) lives there.
export INCLUDE="C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\$MSVC_VER\\include;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\$SDK_VER\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\$SDK_VER\\um;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\$SDK_VER\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\$SDK_VER\\winrt"
export LIB="C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\$MSVC_VER\\lib\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\$SDK_VER\\ucrt\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\$SDK_VER\\um\\x64"
export PATH="$MSVC/bin/Hostx64/x64:$KITS/bin/$SDK_VER/x64:$PATH"

CL="$MSVC/bin/Hostx64/x64/cl.exe"
LINK="$MSVC/bin/Hostx64/x64/link.exe"
FXC="$KITS/bin/$SDK_VER/x64/fxc.exe"

# The pip-distributed ROCm SDK is optional; callers may override discovery.
if [ -z "${ROCM_ROOT:-}" ]; then
    ROCM_ROOT="$(python -c 'import importlib.util,pathlib; s=importlib.util.find_spec("_rocm_sdk_core"); print(pathlib.Path(s.origin).parent.as_posix() if s and s.origin else "")' 2>/dev/null || true)"
fi
export ROCM_ROOT

msvc_build() {
    MSYS_NO_PATHCONV=1 "$CL" /nologo /EHsc /std:c++17 /W3 /MD /O2 \
        /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX "$@"
}
