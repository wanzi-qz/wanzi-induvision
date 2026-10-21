<#
.SYNOPSIS
    消融队列：按顺序跑完 B1 / B2 / C1 / B4，**已完成的组自动跳过**，**崩了的组自动重试**。

.DESCRIPTION
    一组 100 epoch 在 8GB 卡上约 15 分钟，四组一小时，没人会守着，只能放后台。
    后台跑会撞上三件事：

      1. 宿主 shell 被回收时，整个进程树会被一起端掉；
      2. 系统蓝屏重启 —— 2026-10-22 02:12:34 就真发生过一次，B1 停在 86/100（best@83），
         因为 patience=20 它本该跑满 100，这一组只能作废重跑；
      3. GPU 掉线但不重启系统 —— 同日 02:55:40~02:56:11 连报 nvlddmkm 事件 153/14，
         B2 在 epoch 29 抛 `CUDA error: unknown error` 退出，进程没了、系统还活着。

    第 2、3 条查下来是同一个根因：**bugcheck 0x00000116（VIDEO_TDR_FAILURE），
    可疑驱动 nvlddmkm.sys**，转储在 C:\WINDOWS\Minidump\102226-14937-01.dmp。
    也就是说这不是"Windows Update 半夜重启"，是显卡驱动在持续负载下超时复位。
    长训练在这台机器上必须假设**随时会被打断**。

    所以本脚本做成两层防护：

      * 幂等：每组正常跑完时 tools/train_seg.py 会在 runs\segment\<name>\train_args.json
        落一份配置快照，重跑本脚本认这个文件、跳过已完成的组，从断点接着往下走。
      * 重试：单组非零退出且没有 train_args.json 时，等 GPU 恢复后原地重跑，
        最多 -MaxAttempts 次；每次失败的日志单独存 <name>.attempt<k>.log，
        残缺的 results.csv 挪进 runs\_killed\ 留档，不覆盖。

    被中断的组不留 train_args.json，因此一定会重跑 —— 这正是想要的行为：
    **消融宁可重跑，也不要拿一个截断的 run 去和跑满的 run 比**。

.PARAMETER Dry
    只打印将要执行的命令，不真的启动训练。

.PARAMETER MaxAttempts
    单组最多尝试次数，默认 3。设 1 等价于关掉重试。

.PARAMETER Only
    只跑指定组名（可多个），用于补跑崩掉的那一组。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_ablation_queue.ps1

.EXAMPLE
    # 脱离宿主后台跑（新进程的父级是 WmiPrvSE，不随 shell 回收而死）
    Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
        CommandLine      = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_ablation_queue.ps1"'
        CurrentDirectory = '<repo>'
    }
#>
[CmdletBinding()]
param(
    [switch]$Dry,
    [int]$MaxAttempts = 3,
    [string[]]$Only = @()
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

function Gpu-Alive {
    # TDR 之后 nvidia-smi 可能直接失败或查询超时，这里当健康探针用
    try {
        $t = (& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($t)) { return $false }
        return $true
    } catch { return $false }
}

function Quarantine($name, $attempt) {
    # 把崩掉那次的残档挪走留证，避免和重跑的结果混在一个目录里
    $src = Join-Path $root "runs\segment\$name"
    if (-not (Test-Path $src)) { return }
    $csv = Join-Path $src "results.csv"
    $ep = "-"
    if (Test-Path $csv) { $ep = (Get-Content $csv).Count - 1 }
    New-Item -ItemType Directory -Force -Path (Join-Path $root "runs\_killed") | Out-Null
    $leaf = "${name}_a${attempt}_ep${ep}"
    # 同名残档已存在就加时间戳，别让 Move-Item -Force 把新残档塞进旧目录里
    if (Test-Path (Join-Path $root "runs\_killed\$leaf")) { $leaf = "${leaf}_$(Get-Date -Format 'HHmmss')" }
    $dst = Join-Path $root "runs\_killed\$leaf"
    # 路径双保险：目标必须在 runs\_killed 下，叶节点名必须是自己刚拼出来的那个
    if (-not $dst.StartsWith((Join-Path $root "runs\_killed")) -or (Split-Path $dst -Leaf) -ne $leaf) {
        Log "    [warn] 拒绝搬移，目标路径不合预期：$dst"; return
    }
    try { Move-Item -LiteralPath $src -Destination $dst -Force; Log "    [keep] 残档 -> runs\_killed\$leaf" }
    catch { Log "    [warn] 残档搬移失败：$($_.Exception.Message)" }
}

Log "=== queue start (pid=$PID, dry=$Dry, maxAttempts=$MaxAttempts, only=$($Only -join ',')) ==="
foreach ($j in $jobs) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $j.n)) { continue }

    $done = Join-Path $root "runs\segment\$($j.n)\train_args.json"
    if (Test-Path $done) { Log "--- skip $($j.n)：已完成（$done 存在）"; continue }

    $data = Join-Path $root $j.d
    if (-not (Test-Path $data)) { Log "!! skip $($j.n)：缺数据集 $data"; continue }

    $cmd = @("tools/train_seg.py", "--data", $j.d, "--model", "weights/yolov8s-seg.pt",
             "--name", $j.n, "--epochs", "100") + $j.f
    if ($Dry) { Log ">>> [dry] $($j.n): & $py $($cmd -join ' ')"; continue }

    $ok = $false
    for ($a = 1; $a -le $MaxAttempts -and -not $ok; $a++) {
        $used = Wait-Gpu
        if ($used -lt 0) { Log "!! abort：等了 10 分钟显存仍不空，先看看是不是有别的训练在跑"; exit 1 }
        Log ">>> start $($j.n)  attempt=$a/$MaxAttempts  (gpu used=${used}MiB)"

        # attempt=1 之前也要清场：上一次调用遗留的残档会被 ultralytics 原地覆盖，
        # 10-22 的 b2_focal 第一轮 ep29 残档就是这么丢的（见 docs/baseline.md「队列与中断恢复」）
        if ($a -eq 1) { Quarantine $j.n 0 }

        $suffix = if ($a -eq 1) { "" } else { ".attempt$a" }
        & $py @cmd *>&1 | Out-File -FilePath (Join-Path $log "$($j.n)$suffix.log") -Encoding utf8
        $code = $LASTEXITCODE

        if (Test-Path $done) { Log "<<< end   $($j.n) exit=$code  已完成"; $ok = $true; break }

        Log "<<< end   $($j.n) exit=$code  !! 没有 train_args.json，属于中断"
        Quarantine $j.n $a
        if ($a -lt $MaxAttempts) {
            if (-not (Gpu-Alive)) {
                Log "    [gpu ] nvidia-smi 不可用，驱动还没恢复，等 90s"
                Start-Sleep -Seconds 90
            } else {
                Log "    [gpu ] nvidia-smi 正常，等 30s 再重试"
                Start-Sleep -Seconds 30
            }
        }
    }
    if (-not $ok) { Log "!! give up $($j.n)：连续 $MaxAttempts 次都没跑完，别再自动重试了，去看 runs\logs\$($j.n)*.log" }
}
Log "=== queue finished ==="
