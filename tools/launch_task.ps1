<#
.SYNOPSIS
    把长任务挂到 Windows 计划任务上跑，让它不再随 Codex 桌面 App 一起死。

.DESCRIPTION
    踩坑记录（2026-10-22）：baseline15 练到 epoch 32 时整条进程链在 04:12:23 消失，
    04:13:49 ChatGPT/codex 进程全部重启 —— **Codex 桌面 App 一重启就把它 shell 派生的
    整棵子进程树带走了**。之前以为用 WMI Win32_Process.Create 脱离宿主就安全，
    实测 run_b3_after.ps1(PID 36840) 一样没活下来。三个多小时的训练白跑。

    Task Scheduler 不一样：任务由 Task Scheduler 服务启动，进程不属于任何 App 的
    job object，App 重启 / 终端关闭 / 锁屏都动不了它。所以**任何超过 10 分钟的活
    都该用这个脚本挂上去**，别再直接 & 了。

    顺带管两件会静默毁掉长训练的事：
    1. 电源：这台机器是 S0 新型待机，AC 睡眠超时出厂就是 600 秒（10 分钟），
       03:24 真进过一次待机。``-HardenPower`` 会把 AC/DC 睡眠与休眠超时都设成 0。
    2. 显存：8GB 卡，训练前顺手关掉腾讯视频/弹幕/壁纸这类纯娱乐的 GPU 占用者。
       （-FreeVram 默认关，因为会动用户自己的程序。）

.PARAMETER Script
    要跑的 .ps1，相对仓库根或绝对路径都行，例如 tools\run_resume_chain.ps1。

.PARAMETER Arguments
    透传给脚本的参数串，例如 "-Name baseline15 -Epochs 100"。

.PARAMETER TaskName
    计划任务名，默认从脚本名生成（InduVision_<脚本名>）。

.PARAMETER Hours
    执行时限，默认 6 小时，超时由任务计划程序自己终止。

.PARAMETER HardenPower
    把 AC/DC 的睡眠、休眠、磁盘超时都设成"从不"。

.PARAMETER FreeVram
    关掉常见娱乐类 GPU 占用进程，给训练腾显存。

.PARAMETER Unregister
    删掉同名计划任务（跑完清理用）。

.PARAMETER Status
    只看任务状态和最近结果，不启动。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\launch_task.ps1 `
        -Script tools\run_resume_chain.ps1 -HardenPower

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\launch_task.ps1 -Status
#>
[CmdletBinding()]
param(
    [string]$Script,
    [string]$Arguments = "",
    [string]$TaskName = "",
    [int]$Hours = 6,
    [switch]$HardenPower,
    [switch]$FreeVram,
    [switch]$Unregister,
    [switch]$Status,
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$logDir = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$launchLog = Join-Path $logDir "_launch_task.log"

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath $launchLog -Append
}

# 娱乐类 GPU 占用者。只列确定安全的，微信/浏览器这些不动。
$VramHogs = @("QQLive", "MagicDanmaku", "wallpaper64", "wallpaper32",
              "LivelyWallpaper", "TMPThumbHD")

function Show-Status([string]$name) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $t) { Log "    没有名为 $name 的计划任务"; return }
    $i = Get-ScheduledTaskInfo -TaskName $name -ErrorAction SilentlyContinue
    Log ("    {0}  State={1}  LastRun={2}  LastResult={3}  NextRun={4}" -f `
        $name, $t.State, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime)
    # 这两个码都不是失败，别误读：
    #   267009      = 0x41301 任务正在运行
    #   2147946720  = 0x800710E0 触发器到点时任务已在跑，被 MultipleInstances=IgnoreNew 拒掉
    #                 （手动 Start-ScheduledTask 之后一次性触发器又触发，属正常）
    Log "    （LastResult 267009=正在运行；2147946720=触发器撞上已运行实例被忽略，都不是错误）"
}

if ($Status) {
    Log "=== launch_task -Status ==="
    if ($TaskName) { Show-Status $TaskName }
    else { Get-ScheduledTask -TaskName "InduVision*" -ErrorAction SilentlyContinue | ForEach-Object { Show-Status $_.TaskName } }
    exit 0
}

if (-not $Script) { Log "!! 需要 -Script（或用 -Status 查看现有任务）"; exit 2 }

$scriptAbs = if ([System.IO.Path]::IsPathRooted($Script)) { $Script } else { Join-Path $root $Script }
if (-not (Test-Path $scriptAbs)) { Log "!! abort：找不到脚本 $scriptAbs"; exit 2 }
if ([string]::IsNullOrWhiteSpace($TaskName)) {
    $TaskName = "InduVision_" + [System.IO.Path]::GetFileNameWithoutExtension($scriptAbs)
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Log "已删除计划任务 $TaskName"
    exit 0
}

if ($HardenPower) {
    Log "--- 电源加固：睡眠/休眠/磁盘超时全部设为从不（这台机器是 S0 新型待机，默认 AC 10 分钟就睡）"
    powercfg /change standby-timeout-ac 0  | Out-Null
    powercfg /change standby-timeout-dc 0  | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-dc 0 | Out-Null
    powercfg /change disk-timeout-ac 0     | Out-Null
    Log "    standby-timeout-ac = $((powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String '当前交流').ToString().Trim())"
} else {
    $idx = (powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "当前交流电源设置索引")
    if ($idx -and $idx -notmatch "0x00000000") {
        Log "!! 警告：AC 睡眠超时不是 0（$($idx.ToString().Trim())），长训练可能被待机打断。加 -HardenPower。"
    }
}

if ($FreeVram) {
    foreach ($n in $VramHogs) {
        $ps = Get-Process -Name $n -ErrorAction SilentlyContinue
        foreach ($p in $ps) { Log "    kill $n pid=$($p.Id)"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds 4
}
$gpu = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>$null)
Log "--- GPU: $gpu"

$argLine = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptAbs`""
if (-not [string]::IsNullOrWhiteSpace($Arguments)) { $argLine += " $Arguments" }

Log "=== 注册计划任务 $TaskName ==="
Log "    powershell.exe $argLine"
Log "    WorkingDirectory=$root  ExecutionTimeLimit=${Hours}h  RestartCount=3"
if ($Dry) { Log "=== dry-run，未注册 ==="; exit 0 }

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
$action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argLine -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours $Hours) `
              -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 3) -MultipleInstances IgnoreNew
$user = "$env:USERDOMAIN\$env:USERNAME"
try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -User $user -RunLevel Limited -Force | Out-Null
} catch {
    Log "!! 注册失败（$($_.Exception.Message)），退回当前用户默认身份再试一次"
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
}
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 6
Show-Status $TaskName
Log "=== 已启动。监控：Get-Content runs\logs\_resume_chain.log -Tail 20 ==="
Log "    停掉：Stop-ScheduledTask -TaskName $TaskName ; 删除：-Unregister"