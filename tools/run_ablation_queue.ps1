<#
.SYNOPSIS
    消融队列：按顺序跑完 B1 / B2 / C1 / B4，**已完成的组自动跳过**。

.DESCRIPTION
    一组 100 epoch 在 8GB 卡上约 15 分钟，四组一小时，没人会守着，只能放后台。
    后台跑会撞上两件事：

      1. 宿主 shell 被回收时，整个进程树会被一起端掉；
      2. Windows Update 半夜自动重启 —— 2026-10-22 02:12:34 就真发生过一次，
         B1 停在 86/100（best@83），因为 patience=20 它本该跑满 100，这一组只能作废重跑。

    所以本脚本做成幂等的：每组正常跑完时 tools/train_seg.py 会在
    runs\segment\<name>\train_args.json 落一份配置快照，重跑本脚本会认这个文件、
    跳过已完成的组，从断点接着往下走，不需要人工回忆"刚才跑到哪了"。

    被中断的组不会留下 train_args.json，因此会被自动重跑 —— 这正是想要的行为：
    消融宁可重跑，也不要拿一个截断的 run 去和跑满的 run 比。

.PARAMETER Dry
    只打印将要执行的命令，不真的启动训练。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_ablation_queue.ps1

.EXAMPLE
    # 脱离宿主后台跑（新进程的父级是 WmiPrvSE，不随 shell 回收而死）
    Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
        CommandLine    = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_ablation_queue.ps1"'
        CurrentDirectory = '<repo>'
    }
#>
[CmdletBinding()]
param(
    [switch]$Dry
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"

# 训练必须用带 torch+ultralytics 的 GPU 环境；数据转换必须固定用同一个 Python 环境
# （cv2 4.10 与 5.0 的 findContours/approxPolyDP 结果差约 1 像素，会悄悄破坏单变量前提）
$py  = "W:\Anaconda\envs\induvision\python.exe"
$log = Join-Path $root "runs\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null

function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath (Join-Path $log "_queue.log") -Append
}

# 编号口径见 docs/baseline.md §3。B3（过采样）还缺落盘 CLI，不排进本轮。
$jobs = @(
    @{ n = "b1_cp";       d = "out/yolo_bottle_tb_cp/data.yaml";     f = @() }
    @{ n = "b2_focal";    d = "out/yolo_bottle_tb/data.yaml";        f = @("--focal-gamma", "2.0", "--focal-alpha", "0.25") }
    @{ n = "c1_cp_focal"; d = "out/yolo_bottle_tb_cp/data.yaml";     f = @("--focal-gamma", "2.0", "--focal-alpha", "0.25") }
    @{ n = "b4_nogood";   d = "out/yolo_bottle_tb_nogood/data.yaml"; f = @() }
)

function Wait-Gpu {
    # 8GB 卡一次只跑一个；等显存降到 3GB 以下再上
    for ($i = 0; $i -lt 60; $i++) {
        $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1)
        if ($null -ne $used -and [int]$used -lt 3000) { return [int]$used }
        Start-Sleep -Seconds 10
    }
    return -1
}

Log "=== queue start (pid=$PID, dry=$Dry) ==="
foreach ($j in $jobs) {
    $done = Join-Path $root "runs\segment\$($j.n)\train_args.json"
    if (Test-Path $done) { Log "--- skip $($j.n)：已完成（$done 存在）"; continue }

    $data = Join-Path $root $j.d
    if (-not (Test-Path $data)) { Log "!! skip $($j.n)：缺数据集 $data"; continue }

    $used = Wait-Gpu
    if ($used -lt 0) { Log "!! abort：等了 10 分钟显存仍不空，先看看是不是有别的训练在跑"; exit 1 }
    Log ">>> start $($j.n)  (gpu used=${used}MiB)"

    $cmd = @("tools/train_seg.py", "--data", $j.d, "--model", "weights/yolov8s-seg.pt",
             "--name", $j.n, "--epochs", "100") + $j.f
    if ($Dry) { Log "    [dry] & $py $($cmd -join ' ')"; continue }

    & $py @cmd *>&1 | Out-File -FilePath (Join-Path $log "$($j.n).log") -Encoding utf8
    $code = $LASTEXITCODE
    if (Test-Path $done) { Log "<<< end   $($j.n) exit=$code  已完成" }
    else { Log "<<< end   $($j.n) exit=$code  !! 没有 train_args.json，属于中断，下次重跑本脚本会自动重来" }
}
Log "=== queue finished ==="
