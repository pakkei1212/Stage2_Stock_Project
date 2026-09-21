<#
.SYNOPSIS
  Registers (or removes) the Windows Task Scheduler task that keeps the Stage 2
  scheduler running on this laptop.

.DESCRIPTION
  The task starts scripts\start_scheduler.ps1 (hidden) for the current user:
    * At log on            -> startup catch-up check runs immediately
    * Daily at -EnsureAt   -> restarts the loop if it is not running (e.g. after a
                              crash); if it is already running the new instance
                              is ignored (MultipleInstances = IgnoreNew)
  The task holds no market/calendar logic — the Python scheduler decides whether
  a NASDAQ session needs processing, and runs its own checks per SCHEDULE_MODE
  (daily at LOCAL_SCHEDULER_TIME, SCHEDULE_TIMES, or every
  SCHEDULE_INTERVAL_MINUTES). Changing the check frequency never needs extra
  tasks or a re-registration: edit .env and restart the scheduler.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_windows_task.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_windows_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string] $TaskName = 'Stage2 NASDAQ Scheduler',
    [string] $EnsureAt = '09:05',
    [switch] $Unregister
)

$ErrorActionPreference = 'Stop'

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
    return
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Starter = Join-Path $PSScriptRoot 'start_scheduler.ps1'
$User = "$env:USERDOMAIN\$env:USERNAME"

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Starter`"" `
    -WorkingDirectory $RepoRoot

$triggers = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $User),
    (New-ScheduledTaskTrigger -Daily -At $EnsureAt)
)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Principal $principal -Force `
    -Description 'Keeps the NASDAQ Stage 2 scheduler running (python -m pipeline.scheduler).' | Out-Null

Write-Host "Registered scheduled task '$TaskName' for $User (at log on + daily $EnsureAt)."
Write-Host "Start it now with:  Start-ScheduledTask -TaskName '$TaskName'"
