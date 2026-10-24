@echo off
setlocal
cd /d "%~dp0"
set "WGPU_BACKEND_TYPE=D3D12"
py -u sistema_solar/python/main_solar.py
if errorlevel 1 pause
endlocal
