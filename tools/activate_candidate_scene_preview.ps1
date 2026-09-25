param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$RuntimeDir,
    [Parameter(Mandatory = $true)][string]$BackupDir
)

$ErrorActionPreference = 'Stop'
if (Get-Process -Name Cyberpunk2077 -ErrorAction SilentlyContinue) {
    throw 'Close Cyberpunk 2077 before replacing the shim.'
}
$game = (Resolve-Path -LiteralPath $GameDir).Path
$build = (Resolve-Path -LiteralPath $BuildDir).Path
$runtime = (Resolve-Path -LiteralPath $RuntimeDir).Path
$dll = Join-Path $build 'nvngx.dll'
if (!(Test-Path -LiteralPath $dll -PathType Leaf)) {
    throw "Built nvngx.dll not found: $dll"
}
$names = @('nvngx.dll', 'nvngx.dll_dlssnr.dll',
           'nvngx_dlssnr.dll', 'dlssnr_shim.ini')
foreach ($name in $names) {
    if (!(Test-Path -LiteralPath (Join-Path $game $name) -PathType Leaf)) {
        throw "Game shim file missing: $name"
    }
}
$backup = [IO.Path]::GetFullPath($BackupDir)
if (Test-Path -LiteralPath $backup) {
    throw "Use a new backup directory: $backup"
}
New-Item -ItemType Directory -Path $backup | Out-Null
foreach ($name in $names) {
    Copy-Item -LiteralPath (Join-Path $game $name) `
              -Destination (Join-Path $backup $name)
    $before = (Get-FileHash -LiteralPath (Join-Path $game $name) -Algorithm SHA256).Hash
    $saved = (Get-FileHash -LiteralPath (Join-Path $backup $name) -Algorithm SHA256).Hash
    if ($before -ne $saved) { throw "Backup verification failed: $name" }
}

$settings = [ordered]@{
    DebugView = '2'
    HipFeLive = '0'
    CandidatePreviewPath = (Join-Path $runtime 'candidate_preview.bin')
    CandidatePreviewReload = '1'
    CandidateInputCapturePath = (Join-Path $runtime 'game_capture.bin')
    CandidateInputCaptureTrigger = '1'
    CandidateInputCaptureRepeat = '1'
    CandidateInputGpuPath = ''
}
$ini = Join-Path $game 'dlssnr_shim.ini'
$lines = [Collections.Generic.List[string]]::new()
foreach ($line in [IO.File]::ReadAllLines($ini)) { $lines.Add($line) }
foreach ($key in $settings.Keys) {
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($key))\s*=") {
            $lines[$i] = "$key=$($settings[$key])"
            $found = $true
        }
    }
    if (!$found) { $lines.Add("$key=$($settings[$key])") }
}

try {
    foreach ($name in $names[0..2]) {
        Copy-Item -LiteralPath $dll -Destination (Join-Path $game $name) -Force
        if ((Get-FileHash -LiteralPath (Join-Path $game $name) -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash) {
            throw "Installed DLL verification failed: $name"
        }
    }
    [IO.File]::WriteAllLines($ini, $lines, [Text.UTF8Encoding]::new($false))
    Write-Output "scene preview activated; backup: $backup"
    Write-Output "capture: $($settings.CandidateInputCapturePath)"
    Write-Output "preview: $($settings.CandidatePreviewPath)"
} catch {
    foreach ($name in $names) {
        Copy-Item -LiteralPath (Join-Path $backup $name) `
                  -Destination (Join-Path $game $name) -Force
    }
    throw
}
