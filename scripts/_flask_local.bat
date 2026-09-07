@echo off
cd /d "%~dp0\.."
set LOCAL_DEV=1

call "%~dp0_resolve_python.bat"
if errorlevel 1 (
  pause
  exit /b 1
)

echo Using Python: %PYTHON%
"%PYTHON%" app.py
