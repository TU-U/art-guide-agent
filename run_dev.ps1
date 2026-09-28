$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

. .\venv\Scripts\Activate.ps1

Write-Host "启动开发模式后端（自动重载）..." -ForegroundColor Cyan
Write-Host "项目目录: $projectRoot"
Write-Host "访问地址: http://localhost:8000"
Write-Host "修改 Python 或 prompts 文件后会自动重启"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
