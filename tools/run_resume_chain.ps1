<#
.SYNOPSIS
    断点续训链：baseline15 从 last.pt 续跑满 -> test 逐类评测 -> 汇总 -> 自动接 B3 过采样消融。

.DESCRIPTION
    背景（2026-10-22 04:12:23）：baseline15 跑到 epoch 32 时整条进程链消失，
    04:13:49 ChatGPT/codex 进程全部重启 —— Codex 桌面 App 一重启就把它 shell 派生的
    整棵子进程树带走了，连用 WMI Win32_Process 脱离出来的 run_b3_after.ps1(PID 36840)
    也没活下来。所以这一版改由 Task Scheduler 托管：任务由服务启动，不属于任何 App 的
    job object，App 重启 / 锁屏 / 会话切换都动不了它。

    为什么不直接用 run_train_guarded.ps1 续训：它的失败兜底是 Quarantine —— 把整个
    run 目录挪进 runs\_killed。这对"从头重跑"是对的，对"续训"是灾难：last.pt 被挪走后
    下一次 attempt 就无档可续。ultralytics 的 resume=True 会恢复优化器状态、接着
    results.csv 往后写、epoch 号连续，所以这里失败就地再 resume，不搬目录。

    口径说明：resume 时 train_seg.py 写的 train_args.json 里 "model" 字段会记成 last.pt
    的路径而不是 weights/yolov8s-seg.pt。summarize_ablation.py 的 PROTOCOL_KEYS 是
    (seed, imgsz, batch, epochs, patience, amp)，不含 model，所以 --strict 不受影响；
    权重初值也确实是同一个 yolov8s-seg.pt，只是中途接着练。

.PARAMETER Name
    要续训的 run 名，默认 baseline15。

.PARAMETER Data
    data.yaml 路径（相对仓库根）。

.PARAMETER MaxAttempts
    续训重试次数，默认 4。每次都是从当前 last.pt 再续，不会重头开始。

.PARAMETER SkipB3
    只跑续训 + 评测 + 汇总，不接 B3。

.PARAMETER Dry
    只打印命令。

.EXAMPLE
    $a = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"<repo>\tools\run_resume_chain.ps1`"" `
        -WorkingDirectory "<repo>"
    Register-ScheduledTask -TaskName InduVisionResumeChain -Action $a ...
#>
[CmdletBinding()]
param(
    [string]$Name = "baseline15",
    [string]$Data = "out/yolo_mvtec15/data.yaml",
    [int]$Epochs = 100,
    [int]$MaxAttempts = 4,
    [switch]$SkipB3,
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pyBase   = "W:\Anaconda\python.exe"                  # 汇总不碰 GPU
$pyGpu    = "W:\Anaconda\envs\induvision\python.exe"  # 训练/评测必须用带 torch 的 env
$log      = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$chainLog = Join-Path $log "_resume_chain.log"
$runDir   = Join-Path $root "runs\segment\$Name"
$done     = Join-Path $runDir "train_args.json"
$lastPt   = Join-Path $runDir "weights\last.pt"

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath $chainLog -Append
}

function EpochDone {
    $csv = Join-Path $runDir "results.csv"
    if (-not (Test-Path $csv)) { return 0 }
    $n = (Get-Content $csv).Count - 1
    if ($n -lt 0) { return 0 }
    return $n
}

function GpuUsed {
    $q = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($q)) { return -1 }
    return [int](($q | Select-Object -First 1).Trim())
}

function Step([string]$title, [scriptblock]$body) {
    Log "=============================================================="
    Log ">>> $title"
    $t0 = Get-Date
    try { & $body } catch { Log "    [err ] $($_.Exception.Message)" }
    $dt = ((Get-Date) - $t0).TotalMinutes
    Log "<<< $title  用时 $([math]::Round($dt,1)) min"
}

Log "=== resume-chain start (pid=$PID, task-hosted) name=$Name dry=$Dry ==="
Log "    起点：ep=$(EpochDone)/$Epochs   train_args.json=$(Test-Path $done)   last.pt=$(Test-Path $lastPt)   gpu=$(GpuUsed)MiB"

# ---- R1 续训 baseline15 ------------------------------------------------------
Step "R1 $Name 断点续训（最多 $MaxAttempts 次，失败就地再续不搬目录）" {
    if (Test-Path $done) { Log "    [skip] 已完成（train_args.json 存在）"; return }
    if (-not (Test-Path $lastPt)) {
        Log "    [abort] 没有 $lastPt，无档可续。要从头跑请改用 run_train_guarded.ps1"
        return
    }
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        if (Test-Path $done) { break }
        Log "    >>> resume attempt=$a/$MaxAttempts  ep=$(EpochDone)  gpu=$(GpuUsed)MiB"
        if ($Dry) {
            Log "    [dry ] & $pyGpu tools/train_seg.py --data $Data --model runs/segment/$Name/weights/last.pt --name $Name --epochs $Epochs --resume"
            return
        }
        $suffix = if ($a -eq 1) { "" } else { ".resume$a" }
        & $pyGpu "tools/train_seg.py" --data $Data `
            --model "runs/segment/$Name/weights/last.pt" `
            --name $Name --epochs $Epochs --resume *>&1 |
            Out-File -FilePath (Join-Path $log "$Name$suffix.log") -Encoding utf8 -Append
        $code = $LASTEXITCODE
        if (Test-Path $done) {
            Log "    <<< attempt=$a exit=$code  已完成，最终 ep=$(EpochDone)"
            break
        }
        Log "    <<< attempt=$a exit=$code  !! 仍无 train_args.json，ep=$(EpochDone)，等 45s 就地再续"
        Start-Sleep -Seconds 45
    }
    if (-not (Test-Path $done)) { Log "    [giveup] $MaxAttempts 次都没跑满，后面的评测/汇总按现状走" }
}

# ---- R2 test split 逐类评测 --------------------------------------------------
Step "R2 $Name 在 test split 上评测（best.pt 是 val 早停选的，最终成绩要在 test 上重算）" {
    $best = Join-Path $runDir "weights\best.pt"
    $md   = Join-Path $log "${Name}_eval_test.md"
    if ($Dry) { Log "    [dry ] & $pyGpu tools/eval_seg.py --weights runs/segment/$Name/weights/best.pt --data $Data --split test --name $Name --markdown"; return }
    if (-not (Test-Path $best)) { Log "    [skip] 没有 $best"; return }
    & $pyGpu "tools/eval_seg.py" --weights "runs/segment/$Name/weights/best.pt" `
        --data $Data --split test --name $Name --markdown *>&1 |
        Out-File -FilePath $md -Encoding utf8
    Log "    exit=$LASTEXITCODE  -> runs\logs\${Name}_eval_test.md"
}

# ---- R3 汇总消融 -------------------------------------------------------------
Step "R3 汇总消融（--strict 校验协议与数据集环境指纹）" {
    $out = Join-Path $log "ablation_summary.md"
    if ($Dry) { Log "    [dry ] & $pyBase tools/summarize_ablation.py --markdown --strict"; return }
    & $pyBase "tools/summarize_ablation.py" --markdown --strict *>&1 | Out-File -FilePath $out -Encoding utf8
    Log "    exit=$LASTEXITCODE  -> runs\logs\ablation_summary.md"
}

# ---- R4 B3 过采样消融 --------------------------------------------------------
Step "R4 B3 过采样消融（out/yolo_bottle_tb_os，数据已落盘并通过 22 项验收）" {
    if ($SkipB3) { Log "    [skip] -SkipB3"; return }
    $q = Join-Path $root "tools\run_ablation_queue.ps1"
    if ($Dry) { Log "    [dry ] powershell -File $q -Only b3_oversample -MaxAttempts 3"; return }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $q -Only b3_oversample -MaxAttempts 3
    Log "    exit=$LASTEXITCODE"
}

# ---- R5 再汇总（把 B3 并进表） -----------------------------------------------
Step "R5 汇总消融（含 B3）" {
    $out = Join-Path $log "ablation_summary.md"
    if ($Dry) { Log "    [dry ] & $pyBase tools/summarize_ablation.py --markdown --strict"; return }
    & $pyBase "tools/summarize_ablation.py" --markdown --strict *>&1 | Out-File -FilePath $out -Encoding utf8
    Log "    exit=$LASTEXITCODE  -> runs\logs\ablation_summary.md"
}

Log "=== resume-chain finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="