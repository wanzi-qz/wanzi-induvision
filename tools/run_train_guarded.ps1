<#
.SYNOPSIS
    带崩溃重试的单次训练守护：跑 tools/train_seg.py，非正常结束就搬残档、等 GPU 恢复、重跑。

.DESCRIPTION
    为什么要单独一个守护脚本：这台机器的显卡驱动在持续负载下会 TDR。
    2026-10-22 一晚就中了两次 —— 02:12:34 直接蓝屏（bugcheck 0x00000116
    VIDEO_TDR_FAILURE，可疑驱动 nvlddmkm.sys，转储 C:\WINDOWS\Minidump\102226-14937-01.dmp），
    02:55:40~02:56:11 是 nvlddmkm 事件 153/14，进程抛 `CUDA error: unknown error`
    退出但系统还活着。所以**任何长训练都得假设会被打断**，靠人工守着不现实。

    判定"跑完了"的唯一依据是 runs\segment\<name>\train_args.json：
    train_seg.py 只在 model.train() 正常返回后才写它。没有这个文件 = 中断 = 必须重跑，
    绝不拿截断的 run 去和跑满的 run 比。

    重跑前把上一次的残档整体挪进 runs\_killed\<name>_a<k>_ep<n>\：
    ultralytics 在 exist_ok=True 且非 resume 时会 unlink 掉旧 results.csv，
    指标不会串，但残档留着能事后看它崩在第几轮、崩之前什么趋势。

.PARAMETER Name
    run 名，同时是 runs\segment\<Name> 目录名和日志名。

.PARAMETER Data
    data.yaml 路径（相对仓库根或绝对路径都行）。

.PARAMETER ExtraArgs
    追加给 train_seg.py 的参数，例如 @("--focal-gamma","2.0","--focal-alpha","0.25")。

.PARAMETER Epochs
    计划 epoch，默认 100，只用于写日志，真实早停由 patience 决定。

.PARAMETER MaxAttempts
    最多尝试次数，默认 3。设 1 等于关掉重试。

.PARAMETER Dry
    只打印命令，不真跑。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_train_guarded.ps1 `
        -Name baseline15 -Data out/yolo_mvtec15/data.yaml

.EXAMPLE
    # 脱离宿主后台跑
    Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
        CommandLine      = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_train_guarded.ps1" -Name baseline15 -Data out/yolo_mvtec15/data.yaml'
        CurrentDirectory = '<repo>'
    }
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$Data,
    [string[]]$ExtraArgs = @(),
    [int]$Epochs = 100,
    [int]$MaxAttempts = 3,
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$py  = "W:\Anaconda\envs\induvision\python.exe"
$log = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath (Join-Path $log "_guard.log") -Append
}

function Gpu-Info {
    $q = & nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($q)) { return $null }
    $p = ($q | Select-Object -First 1) -split ","
    return @{ used = [int]$p[0].Trim(); temp = [int]$p[1].Trim(); util = [int]$p[2].Trim() }
}

function Wait-Gpu([int]$BelowMiB = 3000, [int]$Tries = 90) {
    # 8GB 卡一次只跑一个；等显存降下来再上，顺便把 GPU 健康探针做了
    for ($i = 0; $i -lt $Tries; $i++) {
        $g = Gpu-Info
        if ($null -ne $g -and $g.used -lt $BelowMiB) { return $g }
        Start-Sleep -Seconds 10
    }
    return $null
}

function Quarantine([string]$name, [int]$attempt) {
    $src = Join-Path $root "runs\segment\$name"
    if (-not (Test-Path $src)) { return }
    $csv = Join-Path $src "results.csv"
    $ep = "-"
    if (Test-Path $csv) { $ep = (Get-Content $csv).Count - 1 }
    New-Item -ItemType Directory -Force -Path (Join-Path $root "runs\_killed") | Out-Null
    $leaf = "${name}_a${attempt}_ep${ep}"
    # 同一次跑里可能反复搬同名残档（attempt 号与 ep 都相同时），加时间戳兜底，
    # 否则 Move-Item 会把新残档塞进已存在的目录，事后分不清哪份是哪次崩的。
    if (Test-Path (Join-Path $root "runs\_killed\$leaf")) { $leaf = "${leaf}_$(Get-Date -Format 'HHmmss')" }
    $dst = Join-Path $root "runs\_killed\$leaf"
    # 路径双保险：目标必须在 runs\_killed 下，叶节点名必须是自己拼出来的那个
    if (-not $dst.StartsWith((Join-Path $root "runs\_killed")) -or (Split-Path $dst -Leaf) -ne $leaf) {
        Log "    [warn] 拒绝搬移，目标路径不合预期：$dst"; return
    }
    try { Move-Item -LiteralPath $src -Destination $dst -Force; Log "    [keep] 残档 -> runs\_killed\$leaf" }
    catch { Log "    [warn] 残档搬移失败：$($_.Exception.Message)" }
}

$dataAbs = Join-Path $root $Data
if (-not (Test-Path $dataAbs)) { Log "!! abort：缺数据集 $dataAbs"; exit 2 }
if (-not (Test-Path $py))        { Log "!! abort：找不到 GPU 环境 $py"; exit 2 }

$done = Join-Path $root "runs\segment\$Name\train_args.json"
if (Test-Path $done) { Log "--- skip $Name：已完成（$done 存在）"; exit 0 }

# 本脚本的契约是"从头跑满"，attempt=1 会把残档整体搬走重练。
# 但残档里的 last.pt 其实能接着练 —— 15 类基线从头跑要 3 小时，续训只花剩下的。
# 所以这里只提示不拦，让人自己决定走哪条路。
$lastPt = Join-Path $root "runs\segment\$Name\weights\last.pt"
if (Test-Path $lastPt) {
    $ep0 = "-"; $csv0 = Join-Path $root "runs\segment\$Name\results.csv"
    if (Test-Path $csv0) { $ep0 = (Get-Content $csv0).Count - 1 }
    Log "!! 注意：$Name 有残档（已练到 ep=$ep0），本脚本会把它搬进 runs\_killed 从头重跑。"
    Log "!!       想接着练省时间，请改用 tools\run_resume_chain.ps1（计划任务托管）。"
}

$cmd = @("tools/train_seg.py", "--data", $Data, "--model", "weights/yolov8s-seg.pt",
         "--name", $Name, "--epochs", "$Epochs") + $ExtraArgs
Log "=== guard start (pid=$PID) name=$Name epochs=$Epochs maxAttempts=$MaxAttempts dry=$Dry ==="
Log "    cmd: & $py $($cmd -join ' ')"
if ($Dry) { Log "=== guard dry-run end ==="; exit 0 }

$ok = $false
for ($a = 1; $a -le $MaxAttempts -and -not $ok; $a++) {
    $g = Wait-Gpu
    if ($null -eq $g) { Log "!! abort：等了 15 分钟显存仍不空或 nvidia-smi 不可用"; exit 1 }
    Log ">>> start $Name  attempt=$a/$MaxAttempts  (gpu used=$($g.used)MiB temp=$($g.temp)C util=$($g.util)%)"

    # attempt=1 也要清场：上一次进程被外力杀掉（Codex App 重启 / nvlddmkm TDR）时
    # 残档还赖在 runs\segment\$Name。ultralytics exist_ok=True 只会 unlink 旧
    # results.csv，混淆矩阵和 PR 曲线这些图不清，新旧混在一起事后没法看。
    if ($a -eq 1) { Quarantine $Name 0 }

    $suffix = if ($a -eq 1) { "" } else { ".attempt$a" }
    & $py @cmd *>&1 | Out-File -FilePath (Join-Path $log "$Name$suffix.log") -Encoding utf8
    $code = $LASTEXITCODE

    if (Test-Path $done) { Log "<<< end   $Name exit=$code  已完成（train_args.json 落盘）"; $ok = $true; break }

    Log "<<< end   $Name exit=$code  !! 没有 train_args.json，属于中断"
    Quarantine $Name $a
    if ($a -lt $MaxAttempts) {
        $h = Gpu-Info
        if ($null -eq $h) { Log "    [gpu ] nvidia-smi 不可用，驱动还没恢复，等 120s"; Start-Sleep -Seconds 120 }
        else              { Log "    [gpu ] nvidia-smi 正常（$($h.used)MiB/$($h.temp)C），等 45s 再重试"; Start-Sleep -Seconds 45 }
    }
}

if ($ok) { Log "=== guard finished OK: $Name ==="; exit 0 }
Log "!! give up $Name：连续 $MaxAttempts 次都没跑完，别自动重试了，去看 runs\logs\$Name*.log 和 nvlddmkm 事件"
exit 1
