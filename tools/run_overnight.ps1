<#
.SYNOPSIS
    夜间串联：补跑崩掉的消融组 -> 15 类全量基线训练 -> test split 逐类评测。

.DESCRIPTION
    三件事必须串行（8GB 卡一次只能跑一个），而且要跨好几个小时没人守，
    所以合成一个脚本一次挂上。每一步都走带重试的守护，单步失败不影响后面继续。

    1. ``run_ablation_queue.ps1``：幂等，只会补跑没留下 train_args.json 的组
       （本轮就是 02:56 被 nvlddmkm TDR 打断的 b2_focal）。
    2. ``run_train_guarded.ps1 -Name baseline15``：docs/baseline.md §2 的正式基线，
       15 类 / 4490 train 图，8GB 卡上 100 epoch 约 3 小时。**这才是能写简历的数**，
       §3 那两组只有 29 张 val 图，绝对值不可用。
    3. ``eval_seg.py --split test``：best.pt 是靠 val 早停选出来的，
       最终成绩要在从没参与决策的 test split 上重算，并逐类列 AP。

.PARAMETER WaitPid
    等这个 PID 退出再开始（用于让当前正在跑的队列自然收尾，避免两个训练抢显存）。

.PARAMETER Epochs
    baseline15 的计划 epoch，默认 100。

.PARAMETER Dry
    只打印每步命令。

.EXAMPLE
    # 脱离宿主后台跑
    Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
        CommandLine      = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_overnight.ps1"'
        CurrentDirectory = '<repo>'
    }
#>
[CmdletBinding()]
param(
    [int]$WaitPid = 0,
    [int]$Epochs = 100,
    [int]$MaxAttempts = 3,
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pyBase = "W:\Anaconda\python.exe"                  # 评测/汇总不碰 GPU，用 base 即可
$pyGpu  = "W:\Anaconda\envs\induvision\python.exe"  # 训练必须用带 torch+ultralytics 的 env
$log    = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$overLog = Join-Path $log "_overnight.log"

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath $overLog -Append
}

function Step([string]$title, [scriptblock]$body) {
    Log "=============================================================="
    Log ">>> $title"
    $t0 = Get-Date
    try { & $body } catch { Log "    [err ] $($_.Exception.Message)" }
    $dt = ((Get-Date) - $t0).TotalMinutes
    Log "<<< $title  用时 $([math]::Round($dt,1)) min"
}

if ($WaitPid -gt 0) {
    Log "=== overnight start (pid=$PID) 等 PID=$WaitPid 退出 ==="
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
    Log "    PID=$WaitPid 已退出，开始接管 GPU"
} else {
    Log "=== overnight start (pid=$PID, dry=$Dry) ==="
}

# --- 1. 补跑消融 -------------------------------------------------------------
Step "step1 补跑消融（幂等，只跑缺 train_args.json 的组）" {
    $q = Join-Path $root "tools\run_ablation_queue.ps1"
    if ($Dry) { Log "    [dry ] powershell -File $q -MaxAttempts $MaxAttempts -Dry"; return }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $q -MaxAttempts $MaxAttempts
    Log "    exit=$LASTEXITCODE"
}

# --- 2. 15 类全量基线 --------------------------------------------------------
Step "step2 baseline15 训练（15 类 / 4490 train 图，$Epochs epoch）" {
    $g = Join-Path $root "tools\run_train_guarded.ps1"
    if ($Dry) { Log "    [dry ] powershell -File $g -Name baseline15 -Data out/yolo_mvtec15/data.yaml -Epochs $Epochs -Dry"; return }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $g `
        -Name "baseline15" -Data "out/yolo_mvtec15/data.yaml" -Epochs $Epochs -MaxAttempts $MaxAttempts
    Log "    exit=$LASTEXITCODE"
}

# --- 3. test split 逐类评测 --------------------------------------------------
Step "step3 baseline15 在 test split 上评测" {
    $best = Join-Path $root "runs\segment\baseline15\weights\best.pt"
    $md = Join-Path $root "runs\logs\baseline15_eval_test.md"
    if ($Dry) { Log "    [dry ] & $pyGpu tools/eval_seg.py --weights runs/segment/baseline15/weights/best.pt --data out/yolo_mvtec15/data.yaml --split test --name baseline15 --markdown"; return }
    if (-not (Test-Path $best)) { Log "    [skip] 没有 $best，step2 没跑成，跳过评测"; return }
    & $pyGpu tools/eval_seg.py --weights "runs/segment/baseline15/weights/best.pt" `
        --data "out/yolo_mvtec15/data.yaml" --split test --name baseline15 --markdown *>&1 |
        Out-File -FilePath $md -Encoding utf8
    Log "    exit=$LASTEXITCODE  markdown -> runs\logs\baseline15_eval_test.md"
}

# --- 4. 汇总消融表 -----------------------------------------------------------
Step "step4 汇总消融（--strict 会校验超参与数据集环境指纹一致）" {
    if ($Dry) { Log "    [dry ] & $pyBase tools/summarize_ablation.py --markdown --strict"; return }
    $out = Join-Path $root "runs\logs\ablation_summary.md"
    & $pyBase tools/summarize_ablation.py --markdown --strict *>&1 | Out-File -FilePath $out -Encoding utf8
    Log "    exit=$LASTEXITCODE  -> runs\logs\ablation_summary.md"
}

Log "=== overnight finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
