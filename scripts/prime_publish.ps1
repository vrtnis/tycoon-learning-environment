param(
    [switch]$SkipSmokeTest,
    [switch]$NoAutoBump
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$keyVarName = "PRIME" + "_API" + "_KEY"
$keyValue = [Environment]::GetEnvironmentVariable($keyVarName)
if ([string]::IsNullOrWhiteSpace($keyValue)) {
    throw "Set the Prime API key environment variable in your shell before publishing. The script does not read or store keys."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$prime = Join-Path $repoRoot ".venv\Scripts\prime.exe"
if (-not (Test-Path $prime)) {
    $prime = "prime"
}

$oldPythonIoEncoding = $env:PYTHONIOENCODING
$oldPythonUtf8 = $env:PYTHONUTF8
$oldDisableVersionCheck = $env:PRIME_DISABLE_VERSION_CHECK

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PRIME_DISABLE_VERSION_CHECK = "1"

try {
    Push-Location $repoRoot

    if (-not $SkipSmokeTest) {
        & (Join-Path $PSScriptRoot "prime_smoke_test.ps1")
    }

    $args = @(
        "env",
        "--plain",
        "push",
        "--path",
        ".\environments\tycoonle_eval",
        "--visibility",
        "PUBLIC"
    )
    if (-not $NoAutoBump) {
        $args += "--auto-bump"
    }

    & $prime @args
} finally {
    Pop-Location
    $env:PYTHONIOENCODING = $oldPythonIoEncoding
    $env:PYTHONUTF8 = $oldPythonUtf8
    $env:PRIME_DISABLE_VERSION_CHECK = $oldDisableVersionCheck
}
