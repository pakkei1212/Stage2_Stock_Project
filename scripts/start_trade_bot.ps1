<#
.SYNOPSIS
  Starts the Stage I Telegram trade-journal bot on Windows.

.DESCRIPTION
  Same thin wrapper as start_scheduler.ps1: set the working directory, load
  .env into the process environment, run `python -m pipeline.trade_bot`. All
  logic (authorization, Confirm/Cancel, the ledger) lives in Python.

  The bot is a SEPARATE process from the scheduler. It records the fills you
  report and answers questions about them:

    scripts\start_trade_bot.ps1              # long-poll until stopped
    scripts\start_trade_bot.ps1 --once       # drain queued updates and exit
    scripts\start_trade_bot.ps1 --backlog    # also process updates queued before startup

  It needs TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_CHAT_ID (or TELEGRAM_CHAT_ID)
  in .env; mutating commands from any other chat are ignored.

  Advisory only: no brokerage API, no account access, no order placement.
  .env values never override variables already set in the environment, and are
  never echoed.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $BotArgs
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
# The console code page is cp1252; keep log lines from becoming encoding errors.
$env:PYTHONIOENCODING = 'utf-8'
$python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }

# PowerShell 5.1 turns native stderr into a terminating error under 'Stop',
# which would kill the bot the first time a library prints a warning.
$ErrorActionPreference = 'Continue'
& $python -m pipeline.trade_bot @BotArgs
exit $LASTEXITCODE
