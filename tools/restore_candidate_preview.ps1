param(
    [Parameter(Mandatory = $true)][string]$BackupDir,
    [Parameter(Mandatory = $true)][string]$GameDir
)

$ErrorActionPreference = 'Stop'
if (Get-Process -Name Cyberpunk2077 -ErrorAction SilentlyContinue) {
    throw 'Close Cyberpunk 2077 before restoring the prior shim.'
}
$backup = (Resolve-Path -LiteralPath $BackupDir).Path
$game = (Resolve-Path -LiteralPath $GameDir).Path
$names = @('nvngx.dll', 'nvngx.dll_dlssnr.dll',
           'nvngx_dlssnr.dll', 'dlssnr_shim.ini')
foreach ($name in $names) {
    if (!(Test-Path -LiteralPath (Join-Path $backup $name))) {
        throw "Backup is incomplete: $name"
    }
}
foreach ($name in $names) {
    Copy-Item -LiteralPath (Join-Path $backup $name) `
              -Destination (Join-Path $game $name) -Force
    $expected = (Get-FileHash -LiteralPath (Join-Path $backup $name) -Algorithm SHA256).Hash
    $actual = (Get-FileHash -LiteralPath (Join-Path $game $name) -Algorithm SHA256).Hash
    if ($expected -ne $actual) { throw "Restore verification failed: $name" }
    Write-Output "restored $name ($actual)"
}
