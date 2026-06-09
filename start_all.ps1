# start_all.ps1 — Khởi động 5 service A2A trên Windows (PowerShell + uv)
#
# Cách chạy:
#   .\start_all.ps1
#
# Mỗi service mở trong 1 cửa sổ PowerShell riêng để bạn xem log.
# Thứ tự: Registry trước → leaf agents (Tax, Compliance) → Law → Customer.

$root = $PSScriptRoot

function Start-Agent($title, $module) {
    Start-Process pwsh -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle='$title'; Set-Location '$root'; `$env:PYTHONIOENCODING='utf-8'; uv run python -m $module"
    )
}

Write-Host "Khoi dong Registry (10000)..." -ForegroundColor Cyan
Start-Agent "Registry :10000" "registry"
Start-Sleep -Seconds 4

Write-Host "Khoi dong Tax (10102) + Compliance (10103)..." -ForegroundColor Cyan
Start-Agent "Tax :10102" "tax_agent"
Start-Agent "Compliance :10103" "compliance_agent"
Start-Sleep -Seconds 5

Write-Host "Khoi dong Law (10101)..." -ForegroundColor Cyan
Start-Agent "Law :10101" "law_agent"
Start-Sleep -Seconds 5

Write-Host "Khoi dong Customer (10100)..." -ForegroundColor Cyan
Start-Agent "Customer :10100" "customer_agent"

Write-Host ""
Write-Host "Xong! 5 cua so da mo. Doi ~10s cho cac service khoi dong xong." -ForegroundColor Green
Write-Host "Test bang:  uv run python test_client.py" -ForegroundColor Yellow
Write-Host "Dung tat ca: chay .\stop_all.ps1" -ForegroundColor Yellow
