#requires -Version 5.1
[CmdletBinding()]
param([string]$SourceRoot,[string]$PythonExecutable,[switch]$CheckOnly)
$ErrorActionPreference='Stop'
if (-not $SourceRoot) {
 $default=Join-Path $env:LOCALAPPDATA 'hermes\hermes-agent'
 $answer=Read-Host "Hermes source folder (Enter for $default)"
 $SourceRoot=if([string]::IsNullOrWhiteSpace($answer)){$default}else{$answer.Trim('"')}
}
$SourceRoot=[IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
if(-not(Test-Path -LiteralPath (Join-Path $SourceRoot 'hermes_cli\main.py') -PathType Leaf)){throw 'Select the hermes-agent folder containing hermes_cli\main.py.'}
$pins=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'checksums.json') -Raw | ConvertFrom-Json
foreach($p in $pins.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $p.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $p.Value){throw "Integrity failure: $($p.Name)"}}
Import-Module (Join-Path $PSScriptRoot 'RuntimeQuiescence.psm1') -Force
if(-not $CheckOnly){Assert-HermesRuntimeQuiescent -SourceRoot $SourceRoot}
if(-not $PythonExecutable){
 $state=Split-Path -Parent $SourceRoot
 $hash=[Security.Cryptography.SHA256]::Create()
 try{$key=([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($SourceRoot)))).Replace('-','').ToLowerInvariant().Substring(0,16)}finally{$hash.Dispose()}
 $install=Join-Path $state "installs\$key"
 $facts=Join-Path $install 'facts.json'
 if(Test-Path -LiteralPath $facts){
  $data=Get-Content -LiteralPath $facts -Raw | ConvertFrom-Json
  $venv=[IO.Path]::GetFullPath([string]$data.packages.venv.environment)
  if(-not $venv.StartsWith($install+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Managed Python environment is outside its installation.'}
  $PythonExecutable=Join-Path $venv 'Scripts\python.exe'
 }elseif(Test-Path -LiteralPath (Join-Path $SourceRoot '.venv\Scripts\python.exe')){$PythonExecutable=Join-Path $SourceRoot '.venv\Scripts\python.exe'}
}
if(-not $PythonExecutable -or -not(Test-Path -LiteralPath $PythonExecutable -PathType Leaf)){throw 'Hermes Python was not found. Supply -PythonExecutable with the Hermes environment python.exe path.'}
$helper=Join-Path $PSScriptRoot 'apply_update.py'
$action=if($CheckOnly){'check'}else{'apply'}
& $PythonExecutable -X utf8 $helper $action $SourceRoot
if($LASTEXITCODE -ne 0){throw 'Source update failed. See the displayed error.'}
if($CheckOnly){return}
Assert-HermesRuntimeQuiescent -SourceRoot $SourceRoot
Push-Location -LiteralPath $SourceRoot
try{
 & $PythonExecutable -m hermes_cli.main desktop --build-only --force-build
 if($LASTEXITCODE -ne 0){throw 'Desktop build failed. Source backups are retained. Fix the build error and rerun this installer; it will resume without reapplying the source patch.'}
}finally{Pop-Location}
& $PythonExecutable -X utf8 $helper complete $SourceRoot
if($LASTEXITCODE -ne 0){throw 'Build completed but receipt update failed.'}
Write-Host 'Update complete. Start Hermes normally. No application was launched by this installer.'
