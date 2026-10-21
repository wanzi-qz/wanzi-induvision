# 后台拉取 MVTec AD 全量（15 类），日志落 runs/logs/download_mvtec.log
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
# python 输出是 UTF-8 字节，先把控制台解码端也切成 UTF-8，
# 否则 Out-File 落盘的是 GBK 误解码后的乱码（本轮 [verify] 那行就是这么花的）
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$log = Join-Path $PSScriptRoot "..\runs\logs\download_mvtec.log"
"=== download start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') pid=$PID ===" | Out-File -LiteralPath $log -Encoding utf8
& W:\Anaconda\python.exe tools/download_mvtec.py --source hf --jobs 4 2>&1 |
    Out-File -LiteralPath $log -Encoding utf8 -Append
"=== download exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -LiteralPath $log -Encoding utf8 -Append
