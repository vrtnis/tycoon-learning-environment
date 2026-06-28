param(
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @(
    (Join-Path $repoRoot "environments\tycoonle_eval"),
    $repoRoot,
    (Join-Path $repoRoot "python")
) -join [IO.Path]::PathSeparator

try {
    Push-Location $repoRoot

    & $python -c "from tycoon_learning_environment import load_environment; env = load_environment(max_turns=5); assert env.max_turns == 5; assert env.action_budget == 4; print('loader ok: max_turns=5 action_budget=4')"
    & $python -m py_compile environments\tycoonle_eval\tycoonle_eval.py environments\tycoonle_eval\tycoon_learning_environment.py

    if (-not $SkipPytest) {
        & $python -m pytest tests\test_jax_contracts.py tests\test_jax_ppo.py -q
    }

    if (Get-Command rg -ErrorAction SilentlyContinue) {
        $keyVarName = "PRIME" + "_API" + "_KEY"
        $knownKeyFragment = "d713" + "41"
        $secretPattern = "pit_[A-Za-z0-9_]+|$knownKeyFragment|$keyVarName"
        & rg -n $secretPattern --hidden --glob "!/.git/**" --glob "!.venv/**" --glob "!.venv-wsl/**" --glob "!node_modules/**" --glob "!dist/**" --glob "!*.pyc" .
        if ($LASTEXITCODE -eq 1) {
            Write-Host "secret scan ok: no Prime key patterns found"
            $global:LASTEXITCODE = 0
        } elseif ($LASTEXITCODE -ne 0) {
            throw "secret scan failed with exit code $LASTEXITCODE"
        } else {
            throw "secret scan found a Prime key pattern"
        }
    } else {
        Write-Warning "rg not found; skipped secret scan"
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}
