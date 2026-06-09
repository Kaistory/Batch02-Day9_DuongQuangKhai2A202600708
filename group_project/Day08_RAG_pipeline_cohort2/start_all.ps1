# Start all Legal Multi-Agent System services in separate windows
Write-Host "Starting Registry service on port 10000..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe -m registry`""

Start-Sleep -Seconds 2

Write-Host "Starting Tax Agent on port 10102..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe -m criminal_agent`""

Write-Host "Starting Compliance Agent on port 10103..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe -m rehab_agent`""

Start-Sleep -Seconds 3

Write-Host "Starting Law Agent on port 10101..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe -m law_agent`""

Start-Sleep -Seconds 3

Write-Host "Starting Customer Agent on port 10100..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe -m customer_agent`""

Start-Sleep -Seconds 2

Write-Host "Starting Observatory UI on port 8000..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\.venv\Scripts\python.exe app.py`""

Write-Host ""
Write-Host "All services started in separate windows."
Write-Host "Open http://127.0.0.1:8000 to use the live graph and trace log."
