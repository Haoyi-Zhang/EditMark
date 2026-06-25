param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Root = Split-Path -Parent $PSScriptRoot
$ExplicitPython = $null
for ($i = 0; $i -lt $Args.Length; $i++) {
    if ($Args[$i] -eq "--python" -and ($i + 1) -lt $Args.Length) {
        $ExplicitPython = $Args[$i + 1]
        break
    }
}
if ($ExplicitPython) {
    $Python = $ExplicitPython
    $PythonSource = "--python"
}
elseif ($env:PYTHON_BIN) {
    $Python = $env:PYTHON_BIN
    $PythonSource = "PYTHON_BIN"
}
else {
    $RepoVenv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $RepoVenv) {
        $Python = $RepoVenv
        $PythonSource = "repo-local .venv"
    }
    else {
        $ActiveCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $ActiveCommand) {
            $ActiveCommand = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if ($env:VIRTUAL_ENV -and $ActiveCommand) {
            $Candidate = & $ActiveCommand.Source -c "import os, sys; from pathlib import Path; current = Path(sys.executable).resolve(); active = str(os.environ.get('VIRTUAL_ENV', '')).strip(); matched = '';`nif active:`n    try:`n        current.relative_to(Path(active).resolve())`n    except Exception:`n        pass`n    else:`n        matched = str(current)`nprint(matched)"
            if ($Candidate) {
                $Python = $Candidate.Trim()
                $PythonSource = "active virtualenv"
            }
        }
        if (-not $Python -and $ActiveCommand) {
            $Candidate = & $ActiveCommand.Source -c "import sys; from pathlib import Path; current = Path(sys.executable).resolve(); print(str(current) if any(token in current.parts for token in ('.venv', 'posteditbench_release', 'posteditbench_release_clean')) else '')"
            if ($Candidate) {
                $Python = $Candidate.Trim()
                $PythonSource = "current interpreter"
            }
        }
        if (-not $Python) {
            Write-Error "[reviewer_workflow.ps1] Missing Python interpreter. Set PYTHON_BIN, activate a dedicated virtualenv, or create $RepoVenv."
            exit 1
        }
    }
}

Write-Host "[reviewer_workflow.ps1] Using $PythonSource: $Python"
& $Python "$Root/scripts/reviewer_workflow.py" @Args
exit $LASTEXITCODE
