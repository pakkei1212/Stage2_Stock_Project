<#
.SYNOPSIS
  Starts the Stage 2 scheduler on Windows (the entry point Task Scheduler calls).

.DESCRIPTION
  Deliberately thin: it only sets the working directory, loads .env into the
  process environment, and runs `python -m pipeline.scheduler`. All trading-
  calendar, catch-up and idempotency logic lives in the Python application.

  Without arguments it starts the long-running scheduler loop (startup check +
  checks per SCHEDULE_MODE: daily / times / interval). Any arguments are passed through:

    scripts\start_scheduler.ps1                   # long-running loop
    scripts\start_scheduler.ps1 --run-once        # one check now
    scripts\start_scheduler.ps1 --run-once --force
    scripts\start_scheduler.ps1 --dry-run
    scripts\start_scheduler.ps1 --resend 2026-09-14

  .env values never override variables already set in the environment, and are
  never echoed.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SchedulerArgs
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$envFile = Join-Path $RepoRoot '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        $text = $line.Trim()
        if ($text -eq '' -or $text.StartsWith('#')) { continue }
        $eq = $text.IndexOf('=')
        if ($eq -lt 1) { continue }
        $key = $text.Substring(0, $eq).Trim()
        $value = $text.Substring($eq + 1).Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                                      ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        } else {
            $hash = $value.IndexOf(' #')
            if ($hash -ge 0) { $value = $value.Substring(0, $hash).TrimEnd() }
        }
        if ($value -ne '' -and -not [Environment]::GetEnvironmentVariable($key, 'Process')) {
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}

$env:PYTHONUNBUFFERED = '1'
# Never let a console code page (cp1252) turn a log line into an encoding error.
$env:PYTHONIOENCODING = 'utf-8'
$python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

# Windows PowerShell 5.1 turns any native-command stderr output into an error
# record; under 'Stop' that would kill the long-running scheduler the first time
# a library prints a warning. Python's own logs (data/logs/) are the record.
$ErrorActionPreference = 'Continue'
& $python -m pipeline.scheduler @SchedulerArgs
exit $LASTEXITCODE
