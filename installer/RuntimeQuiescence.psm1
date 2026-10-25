Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-HermesFullPath {
    param([Parameter(Mandatory)][string]$Path)
    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\','/'))
}

function Test-HermesPathWithin {
    param([string]$Path, [string]$Root)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    try { $candidate = ConvertTo-HermesFullPath $Path } catch { return $false }
    return $candidate.Equals($Root, [StringComparison]::OrdinalIgnoreCase) -or $candidate.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Get-HermesProperty {
    param($Object, [string]$Name)
    if ($null -eq $Object -or [string]::IsNullOrWhiteSpace($Name)) { return $null }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -ne $prop) { return ,$prop.Value }
    return $null
}

function Split-HermesCommandLine {
    param([string]$CommandLine)
    if ([string]::IsNullOrWhiteSpace($CommandLine)) { return @() }
    if (-not ('HermesQuiescence.NativeArgv' -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace HermesQuiescence {
 public static class NativeArgv {
  [DllImport("shell32.dll", SetLastError=true)] public static extern IntPtr CommandLineToArgvW([MarshalAs(UnmanagedType.LPWStr)] string commandLine, out int argc);
  [DllImport("kernel32.dll")] public static extern IntPtr LocalFree(IntPtr memory);
 }
}
"@
    }
    $count = 0
    $ptr = [HermesQuiescence.NativeArgv]::CommandLineToArgvW($CommandLine, [ref]$count)
    if ($ptr -eq [IntPtr]::Zero) { throw 'Cannot parse a candidate process command line.' }
    try {
        $result = @()
        for ($index = 0; $index -lt $count; $index++) {
            $result += [Runtime.InteropServices.Marshal]::PtrToStringUni([Runtime.InteropServices.Marshal]::ReadIntPtr($ptr, $index * [IntPtr]::Size))
        }
        return $result
    } finally { [void][HermesQuiescence.NativeArgv]::LocalFree($ptr) }
}

function Get-HermesProcessBirth {
    param($Process)
    $created = Get-HermesProperty $Process 'CreationDate'
    if ($null -eq $created) { return $null }
    try { return ([DateTimeOffset]([datetime]$created)).ToUnixTimeMilliseconds() / 1000.0 } catch { return $null }
}

function Get-HermesProcessBlockers {
    param([string]$SourceRoot, [object[]]$Processes)
    $stateRoot = Split-Path -Parent $SourceRoot
    foreach ($process in $Processes) {
        $processId = Get-HermesProperty $process 'ProcessId'
        $executable = [string](Get-HermesProperty $process 'ExecutablePath')
        $name = [string](Get-HermesProperty $process 'Name')
        $commandLine = [string](Get-HermesProperty $process 'CommandLine')
        if (Test-HermesPathWithin $executable $stateRoot) {
            "PID $processId is executing inside the target Hermes installation."
            continue
        }
        $candidate = $name -match '^(?i:hermes(?:-agent|-desktop)?|python(?:w|\d+(?:\.\d+)?)?|llama-server)(?:\.exe)?$'
        if (-not $candidate) { continue }
        if ([string]::IsNullOrWhiteSpace($executable)) {
            "PID $processId has a Hermes/runtime executable name but its executable identity is unreadable."
            continue
        }
        if ($name -match '^(?i:hermes(?:-agent|-desktop)?)(?:\.exe)?$') {
            "PID $processId is a Hermes Desktop/CLI candidate outside the target state directory; installation ownership cannot be excluded."
            continue
        }
        if ($name -notmatch '^(?i:python(?:w|\d+(?:\.\d+)?)?)(?:\.exe)?$') { continue }
        if ([string]::IsNullOrWhiteSpace($commandLine)) {
            "PID $processId is a Python runtime with an unreadable command line; target ownership cannot be excluded."
            continue
        }
        $arguments = @(Split-HermesCommandLine $commandLine)
        for ($index = 1; $index -lt $arguments.Count; $index++) {
            $argument = $arguments[$index]
            if ($argument -eq '-c') { break } # Code text is not a process identity.
            if ($argument -eq '-m') {
                if ($index + 1 -lt $arguments.Count -and $arguments[$index + 1] -match '^(hermes_cli(?:\.|$)|gateway(?:\.|$)|tui_gateway$)') {
                    "PID $processId invokes a Hermes Python module; its source installation cannot be proved idle."
                }
                break
            }
            if ($argument -eq '--') { $index++; if ($index -ge $arguments.Count) { break }; $argument = $arguments[$index] }
            elseif ($argument -in @('-W','-X','--check-hash-based-pycs')) { $index++; continue }
            elseif ($argument.StartsWith('-')) { continue }
            if ([IO.Path]::IsPathRooted($argument) -and (Test-HermesPathWithin $argument $SourceRoot)) {
                "PID $processId runs a Python entry point inside the target Hermes source."
            } elseif (-not [IO.Path]::IsPathRooted($argument) -and ([IO.Path]::GetFileName($argument) -in @('hermes.py','run_agent.py','cli.py','tui_gateway.py'))) {
                "PID $processId uses a relative Hermes entry point whose source installation cannot be verified."
            }
            break
        }
    }
}

function Read-HermesIdentityJson {
    param([string]$Path, [switch]$AllowEmpty)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Not a plain state file.' }
        # A gateway keeps a writable handle open. Readers must share write/delete access.
        $stream = $null
        try {
            $stream = [IO.FileStream]::new($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
                ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
            $length = $stream.Length
            if ($length -gt 4194304) { throw 'Identity state is too large to inspect safely.' }
            $bytes = New-Object byte[] ([int]$length)
            $offset = 0
            while ($offset -lt $length) {
                $read = $stream.Read($bytes, $offset, [int]($length - $offset))
                if ($read -eq 0) { throw 'Identity state changed while being read; retry.' }
                $offset += $read
            }
            $after = Get-Item -LiteralPath $Path -Force
            if (($after.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                $after.Length -ne $length -or $after.LastWriteTimeUtc.Ticks -ne $item.LastWriteTimeUtc.Ticks) {
                throw 'Identity state changed while being read; retry.'
            }
            $text = ([Text.UTF8Encoding]::new($false, $true)).GetString($bytes).TrimStart([char]0xFEFF)
        } finally { if ($null -ne $stream) { $stream.Dispose() } }
        if ([string]::IsNullOrWhiteSpace($text)) {
            if ($AllowEmpty) { return $null }
            throw 'Empty identity state.'
        }
        $wrapped = ConvertFrom-Json -InputObject ('{"value":' + $text + '}') -ErrorAction Stop
        $value = $wrapped.value
        if ($null -eq $value) { throw 'Null identity state.' }
        # Preserve an empty JSON array instead of losing it through pipeline enumeration.
        return ,$value
    } catch { throw "Cannot verify Hermes identity state '$Path': unreadable, malformed, or redirected state. If Hermes is open, finish work, close Desktop and run hermes gateway stop before retrying. Do not delete its lock files." }
}

function New-HermesIdentityRecord {
    param($Data, [string]$PidField, [string]$SecondsField, [string]$TicksField, [string]$Label, [switch]$Required)
    $id = Get-HermesProperty $Data $PidField
    if ($null -eq $id) {
        if ($Required) { throw "Missing process identity in $Label." }
        return
    }
    $processId = 0
    if (-not [int]::TryParse([string]$id, [ref]$processId) -or $processId -le 0) { throw "Invalid process identity in $Label." }
    $birth = Get-HermesProperty $Data $SecondsField
    $ticks = Get-HermesProperty $Data $TicksField
    $seconds = $null
    if ($null -ne $birth) {
        $number = 0.0
        if (-not [double]::TryParse([string]$birth, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number) -or $number -le 0 -or [double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "Invalid process birth in $Label." }
        $seconds = $number
    } elseif ($null -ne $ticks) {
        $number = 0.0
        if (-not [double]::TryParse([string]$ticks, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number) -or $number -le 0 -or [double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "Invalid process fingerprint in $Label." }
        $seconds = $number / 100.0
    }
    [pscustomobject]@{ ProcessId = $processId; Birth = $seconds; Label = $Label }
}

function Get-HermesStateIdentities {
    param([string]$SourceRoot)
    $stateRoot = Split-Path -Parent $SourceRoot
    $homes = @($stateRoot)
    $profiles = Join-Path $stateRoot 'profiles'
    if (Test-Path -LiteralPath $profiles) {
        $profileRoot = Get-Item -LiteralPath $profiles -Force
        if ($profileRoot.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cannot verify a redirected Hermes profiles directory.' }
        foreach ($profile in @(Get-ChildItem -LiteralPath $profiles -Directory -Force)) {
            if ($profile.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cannot verify a redirected Hermes profile directory.' }
            $homes += $profile.FullName
        }
    }
    $ledgerPath = Join-Path $stateRoot 'spawn-ledger.json'
    $ledger = Read-HermesIdentityJson $ledgerPath
    if ($null -ne $ledger) {
        if ($ledger -isnot [array]) { throw 'Invalid Hermes spawn ledger: expected an array.' }
        foreach ($entry in $ledger) { New-HermesIdentityRecord $entry 'pid' 'create_time' '' 'spawn ledger' -Required }
    }
    foreach ($runtimeHome in $homes) {
        foreach ($file in @('gateway.pid','gateway.lock','gateway_state.json')) {
            $data = Read-HermesIdentityJson (Join-Path $runtimeHome $file) -AllowEmpty
            if ($null -eq $data) { continue }
            if ($data -is [ValueType]) { $data = [pscustomobject]@{ pid = $data } }
            New-HermesIdentityRecord $data 'pid' '' 'start_time' $file -Required
        }
        $sessions = Read-HermesIdentityJson (Join-Path $runtimeHome 'runtime\active_sessions.json')
        if ($null -ne $sessions) {
            if ($sessions -is [array]) { $entries = $sessions } else { $entries = Get-HermesProperty $sessions 'entries' }
            if ($null -eq $entries -or $entries -isnot [array]) { throw 'Invalid Hermes active sessions state: expected an entries array.' }
            foreach ($entry in $entries) { New-HermesIdentityRecord $entry 'pid' 'process_start_time' '' 'active session' -Required }
        }
        $processes = Read-HermesIdentityJson (Join-Path $runtimeHome 'processes.json')
        if ($null -ne $processes) {
            if ($processes -isnot [array]) { throw 'Invalid Hermes background-process checkpoint: expected an array.' }
            foreach ($entry in $processes) {
                $scope = [string](Get-HermesProperty $entry 'pid_scope')
                if ($scope -and $scope -ne 'host') { throw 'Hermes has a non-host process checkpoint whose completion cannot be verified locally.' }
                New-HermesIdentityRecord $entry 'pid' '' 'host_start_time' 'background process checkpoint' -Required
            }
        }
        $server = Read-HermesIdentityJson (Join-Path $runtimeHome 'runtimes\llamacpp\server.json')
        if ($null -ne $server) {
            New-HermesIdentityRecord $server 'pid' 'create_time' 'start_time' 'local model router' -Required
            New-HermesIdentityRecord $server 'owner_pid' 'owner_create_time' 'owner_start_time' 'local model router owner'
        }
    }
    $lockRoot = $env:HERMES_GATEWAY_LOCK_DIR
    if ([string]::IsNullOrWhiteSpace($lockRoot)) {
        if ($env:XDG_STATE_HOME -and [IO.Path]::IsPathRooted($env:XDG_STATE_HOME)) { $lockRoot = Join-Path $env:XDG_STATE_HOME 'hermes\gateway-locks' }
        else { $lockRoot = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\state\hermes\gateway-locks' }
    }
    foreach ($role in @('gateway','serve','desktop-serve')) {
        $record = Read-HermesIdentityJson (Join-Path $lockRoot "host-$role.json")
        if ($null -eq $record) { continue }
        $recordRole = $record.PSObject.Properties['role']
        if ($null -ne $recordRole -and ($recordRole.Value -isnot [string] -or $recordRole.Value -cne $role)) {
            throw "Invalid role in canonical $role host state."
        }
        $homeProperty = $record.PSObject.Properties['home']
        if ($null -eq $homeProperty -or $homeProperty.Value -isnot [string]) {
            throw "Cannot determine ownership of canonical $role host state."
        }
        $runtimeHome = $homeProperty.Value
        if ($runtimeHome.Length -eq 0) {
            # v0.22.3 web_server publishes serve/desktop-serve with home="".
            # Treat this as an unscoped host identity, never as proof of idleness:
            # live or uncertain PID/birth still blocks; dead/reused PID is stale.
            New-HermesIdentityRecord $record 'pid' 'createTime' 'startTime' "canonical $role host" -Required
            continue
        }
        # A malformed/nonabsolute nonempty home must not look like another
        # installation merely because the containment helper returns false.
        if ([string]::IsNullOrWhiteSpace($runtimeHome) -or $runtimeHome -match '[\x00-\x1f<>"|?*]' -or
            $runtimeHome -notmatch '^(?:[A-Za-z]:[\\/]|[\\/]{2}[^\\/]+[\\/][^\\/]+(?:[\\/]|$))') {
            throw "Cannot determine ownership of canonical $role host state."
        }
        try { $null = ConvertTo-HermesFullPath $runtimeHome }
        catch { throw "Cannot determine ownership of canonical $role host state." }
        if (Test-HermesPathWithin $runtimeHome $stateRoot) {
            New-HermesIdentityRecord $record 'pid' 'createTime' 'startTime' "canonical $role host" -Required
        }
    }
}

function Get-HermesIdentityBlockers {
    param([object[]]$Processes, [object[]]$Identities)
    $byId = @{}
    foreach ($process in $Processes) { $byId[[string](Get-HermesProperty $process 'ProcessId')] = $process }
    foreach ($identity in $Identities) {
        $process = $byId[[string]$identity.ProcessId]
        if ($null -eq $process) { continue } # A stale record alone is not active work.
        $birth = Get-HermesProcessBirth $process
        if ($null -eq $identity.Birth -or $null -eq $birth) {
            "PID $($identity.ProcessId) from $($identity.Label) is alive but its birth identity is unavailable."
        } elseif ([Math]::Abs($birth - $identity.Birth) -le 2.0) {
            "PID $($identity.ProcessId) from $($identity.Label) is still active (PID and process birth match)."
        }
        # A verified birth mismatch means the PID was reused, not that Hermes is active.
    }
}

function Assert-HermesRuntimeQuiescent {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$SourceRoot)
    $source = ConvertTo-HermesFullPath $SourceRoot
    $item = Get-Item -LiteralPath $source -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Hermes source must be a plain directory.' }
    # Two fresh inventories narrow startup races; this is a refusal guard, not an OS startup lock.
    for ($pass = 0; $pass -lt 2; $pass++) {
        try { $inventory = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop) }
        catch { throw "Cannot verify Hermes process activity: $($_.Exception.Message)" }
        if ($inventory.Count -eq 0) { throw 'Cannot verify Hermes activity: the process inventory is empty.' }
        $identities = @(Get-HermesStateIdentities -SourceRoot $source)
        $blockers = @(Get-HermesProcessBlockers -SourceRoot $source -Processes $inventory)
        $blockers += @(Get-HermesIdentityBlockers -Processes $inventory -Identities $identities)
        if ($blockers.Count) {
            throw ("Installation refused. Finish running jobs, close Hermes Desktop, and manually stop its gateway before retrying. No process has been stopped or signalled.`n" + (($blockers | Select-Object -Unique) -join "`n"))
        }
    }
}

Export-ModuleMember -Function Assert-HermesRuntimeQuiescent
