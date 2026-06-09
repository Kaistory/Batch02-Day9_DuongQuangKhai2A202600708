# stop_all.ps1 — Dừng toàn bộ service A2A (và demo_server nếu đang chạy)
#
# Cách chạy:  .\stop_all.ps1

$ports = 10000, 10100, 10101, 10102, 10103, 8800
$pids = Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

if (-not $pids) {
    Write-Host "Khong co service nao dang chay tren cac cong $($ports -join ', ')." -ForegroundColor Yellow
    return
}

foreach ($p in $pids) {
    try {
        $name = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName
        Stop-Process -Id $p -Force -ErrorAction Stop
        Write-Host "Da dung PID $p ($name)" -ForegroundColor Green
    } catch {}
}
Write-Host "Hoan tat." -ForegroundColor Green
