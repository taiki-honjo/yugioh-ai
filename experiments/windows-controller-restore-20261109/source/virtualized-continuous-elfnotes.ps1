param([ValidateSet('Install','Start','Stop','Status','Run','DisableAutostart')][string]$Action='Status')
$ErrorActionPreference='Stop'
$TaskName='YugiohAI-Continuous-Elfnotes'
$InstallDir=Join-Path $env:LOCALAPPDATA 'YugiohAI'
$InstalledScript=Join-Path $InstallDir 'continuous-elfnotes.ps1'
$LinuxScript='/home/hatk8/projects/yugioh-ai/scripts/continuous_learning.py'
$LinuxConfig='/home/hatk8/projects/yugioh-ai/config/continuous-elfnotes.json'
$Python='/opt/ygo-ai/.venv/bin/python'
$WslArgs=@('-d','Ubuntu-22.04','-u','hatk8','--exec',$Python,'-B','-u',$LinuxScript)
$ModeFile=Join-Path $InstallDir 'autostart-mode.txt'
$StartupLink=Join-Path ([Environment]::GetFolderPath('Startup')) 'YugiohAI-Continuous-Elfnotes.lnk'

function Get-LearnerStatus {
    $raw=& wsl.exe @WslArgs status --config $LinuxConfig
    if($LASTEXITCODE -ne 0){ throw 'Cannot read the learner status.' }
    return (($raw -join "`n") | ConvertFrom-Json)
}

if($Action -eq 'Install'){
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    if($PSCommandPath -ne $InstalledScript){ Copy-Item -LiteralPath $PSCommandPath -Destination $InstalledScript -Force }
    $identity=[System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $arguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$InstalledScript+'" -Action Run'
    # A per-user Startup entry works without administrator privileges and leaves
    # the local worker independent of this chat and of the desktop app.
    $previous=Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if($previous){ Disable-ScheduledTask -TaskName $TaskName | Out-Null }
    $shell=New-Object -ComObject WScript.Shell
    $link=$shell.CreateShortcut($StartupLink)
    $link.TargetPath=(Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe')
    $link.Arguments=$arguments
    $link.WorkingDirectory=$InstallDir
    $link.WindowStyle=7
    $link.Save()
    Set-Content -LiteralPath $ModeFile -Value 'startup' -Encoding ascii
    Write-Output ('Installed. Mode: '+(Get-Content -LiteralPath $ModeFile))
    exit 0
}

if($Action -eq 'Run'){
    $mutex=[System.Threading.Mutex]::new($false,'Local\YugiohAI-Continuous-Elfnotes')
    $owned=$false
    try {
        try { $owned=$mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned=$true }
        if(-not $owned){ exit 0 }
        $process=Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\wsl.exe') -ArgumentList (@($WslArgs)+@('run','--config',$LinuxConfig)) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $InstallDir 'host.log') -RedirectStandardError (Join-Path $InstallDir 'host-error.log') -PassThru -Wait
        $code=$process.ExitCode
    } finally {
        if($owned){ $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
    exit $code
}

if($Action -eq 'Start'){
    if(-not (Test-Path -LiteralPath $ModeFile)){ throw 'Run Install first.' }
    & wsl.exe @WslArgs resume --config $LinuxConfig
    if($LASTEXITCODE -ne 0){ throw 'Cannot enable the learner.' }
    if((Get-Content -LiteralPath $ModeFile).Trim() -eq 'task'){
        Enable-ScheduledTask -TaskName $TaskName | Out-Null
        Start-ScheduledTask -TaskName $TaskName
    } else {
        $arguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$InstalledScript+'" -Action Run'
        Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -ArgumentList $arguments -WindowStyle Hidden
    }
    for($i=0;$i -lt 15;$i++){
        Start-Sleep -Seconds 1
        $s=Get-LearnerStatus
        if($s.worker_running){ Write-Output 'Background learner is running.'; break }
    }
} elseif($Action -eq 'Stop'){
    & wsl.exe @WslArgs stop --now --config $LinuxConfig
    if($LASTEXITCODE -ne 0){ throw 'Cannot request stop.' }
    for($i=0;$i -lt 30;$i++){
        $s=Get-LearnerStatus
        if(-not $s.worker_running){ Write-Output 'Stopped. Checkpoints retained.'; break }
        Start-Sleep -Seconds 1
    }
} elseif($Action -eq 'DisableAutostart'){
    if(Test-Path -LiteralPath $ModeFile){
        if((Get-Content -LiteralPath $ModeFile).Trim() -eq 'task'){
            Disable-ScheduledTask -TaskName $TaskName | Out-Null
        } elseif(Test-Path -LiteralPath $StartupLink){ Remove-Item -LiteralPath $StartupLink }
    }
    Write-Output 'Autostart disabled. Stop the learner separately if it is running.'
}

$s=Get-LearnerStatus
[pscustomobject]@{
    Running=$s.worker_running
    Status=$s.status
    Phase=$s.phase
    CompletedCycles=$s.completed_cycles
    AdditionalDecisions=$s.added_steps
    LastDecision=$s.last_decision
    LastError=$s.last_error
    UpdatedJST=$s.updated_jst
    ActiveLog=$s.active_log
} | Format-List

