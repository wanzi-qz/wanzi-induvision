<#
.SYNOPSIS
    B3（过采样对照组）排队器：等夜间串联整个退出后再接管 GPU，跑完顺手刷新消融汇总。

.DESCRIPTION
    为什么要单独一个排队器，而不是直接把 B3 塞进 run_overnight.ps1：

      * 8GB 卡一次只能跑一个训练。run_overnight.ps1 已经在跑 baseline15（PID 见 -WaitPid），
        后面还接了 step3 评测和 step4 汇总，任何时刻插进去都会抢显存，两边一起 OOM。
      * run_ablation_queue.ps1 里虽然有 Wait-Gpu，但它只等 10 分钟；baseline15 还要跑两小时，
        等不到就会往下走，等于没保护。
      * 本脚本改成**按 PID 等**：等到 overnight 主进程彻底退出（四步全部结束），
        再睡 -SettleSeconds 让显存和文件句柄释放干净，然后才用 -Only b3_oversample
        调队列。队列自身是幂等的，还会在 attempt=1 之前把同名残档挪进 runs\_killed\。

    B3 的意义：给 B1（Copy-Paste 合成）当对照。如果"把原图整张复制几遍"就能拿到接近的收益，
    那合成那套工程复杂度就不值得写进简历。数据集由 tools/oversample_dataset.py 落盘，
    val/test 与 B0/B1/B2 逐字节一致，所以三组之间只有 train 不同，可比。

    跑完自动执行 summarize_ablation.py --strict，把 B3 并进 runs\logs\ablation_summary.md，
    协议一致性（seed/imgsz/batch/epochs/patience/amp）和数据集环境指纹都会被校验。

.PARAMETER WaitPid
    等这个 PID 退出再开始，默认 27100（本轮 run_overnight.ps1）。传 0 表示不等。

.PARAMETER SettleSeconds
    PID 退出后再等多久才动 GPU，默认 60 秒。

.PARAMETER MaxAttempts
    传给队列的单组最大尝试次数，默认 3。

.PARAMETER Dry
    只打印每步命令，不真跑。

.EXAMPLE
    # 脱离宿主后台挂上（宿主 shell 被回收也不受影响）
    Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
        CommandLine      = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_b3_after.ps1" -WaitPid 27100'
        CurrentDirectory = '<repo>'
    }
#>
[CmdletBinding()]
param(
    [int]$WaitPid = 27100,
    [int]$SettleSeconds = 60,
    [int]$MaxAttempts = 3,
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pyBase = "W:\Anaconda\python.exe"                  # 汇总不碰 GPU，用 base
$log    = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$myLog  = Join-Path $log "_b3_queue.log"

$runName = "b3_oversample"
$dataYaml = Join-Path $root "out\yolo_bottle_tb_os\data.yaml"
$report   = Join-Path $root "out\yolo_bottle_tb_os\oversample_report.json"
$argsJson = Join-Path $root "runs\segment\$runName\train_args.json"

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath $myLog -Append
}

Log "=== b3 queue start (pid=$PID, dry=$Dry) 等 PID=$WaitPid 退出 ==="

if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
    Log "    PID=$WaitPid 已退出"
}
if ($SettleSeconds -gt 0) {
    Log "    等 $SettleSeconds 秒让显存/句柄释放"
    Start-Sleep -Seconds $SettleSeconds
}

# --- 落盘产物体检：宁可跳过，也不要拿一个可疑的数据集去和 B1 对比 ----------------
if (-not (Test-Path $dataYaml)) {
    Log "!! abort：缺数据集 $dataYaml（先跑 tools/oversample_dataset.py）"
    exit 2
}
if (Test-Path $report) {
    try {
        $rep = Get-Content $report -Raw -Encoding UTF8 | ConvertFrom-Json
        Log ("    体检：copies={0} val/test 一致={1} sha 不匹配={2} env_mismatch={3}" -f `
            $rep.copies.images, $rep.verify.val_test_identical, `
            ($rep.verify.sha256_mismatch | Measure-Object).Count, `
            ($rep.env_mismatch | Measure-Object).Count)
        if ($rep.verify.val_test_identical -ne $true) {
            Log "!! abort：oversample_report.json 说 val/test 与源不一致，这组不能用"
            exit 2
        }
        if (($rep.env_mismatch | Measure-Object).Count -gt 0) {
            Log "!! abort：数据集是在环境指纹不一致的情况下落盘的，这组不能用"
            exit 2
        }
    } catch {
        Log "    [warn ] oversample_report.json 解析失败，继续但请人工复核：$($_.Exception.Message)"
    }
} else {
    Log "    [warn ] 没有 $report，无法体检，继续但请人工复核"
}

if (Test-Path $argsJson) {
    Log "=== $runName 已有 train_args.json，说明已经跑完，跳过训练 ==="
} else {
    $q = Join-Path $root "tools\run_ablation_queue.ps1"
    if ($Dry) {
        Log "    [dry ] powershell -File $q -Only $runName -MaxAttempts $MaxAttempts"
    } else {
        Log ">>> 交给队列跑 $runName（幂等 + 崩溃重试 + 残档隔离）"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $q -Only $runName -MaxAttempts $MaxAttempts
        Log "<<< 队列退出 exit=$LASTEXITCODE"
    }
}

# --- 刷新消融汇总（含协议一致性 + 环境指纹校验） --------------------------------
if ($Dry) {
    Log "    [dry ] & $pyBase tools/summarize_ablation.py --markdown --strict"
} else {
    $out = Join-Path $log "ablation_summary.md"
    Log ">>> 刷新消融汇总 -> runs\logs\ablation_summary.md"
    & $pyBase tools/summarize_ablation.py --markdown --strict *>&1 | Out-File -FilePath $out -Encoding utf8
    Log "<<< exit=$LASTEXITCODE（--strict 非零说明协议或指纹有问题，必须查）"
}

Log "=== b3 queue finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="