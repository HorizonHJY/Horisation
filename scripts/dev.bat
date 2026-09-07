@echo off
cd /d "%~dp0\.."

call "%~dp0_resolve_python.bat"
if errorlevel 1 (
  pause
  exit /b 1
)

echo ==============================
echo  Horisation - Dev Mode
echo  Python   : %PYTHON%
echo  Frontend : http://localhost:5173
echo  API      : http://localhost:5000
echo ==============================

echo [1/3] Installing Python dependencies...
if defined PIP (
  "%PIP%" install -r requirements.txt --quiet
) else (
  "%PYTHON%" -m pip install -r requirements.txt --quiet
)
echo       Done.

echo [2/3] Installing frontend dependencies...
cd frontend
call npm install --silent
echo       Done.

echo [3/3] Starting servers...
start "Flask API" cmd /k "%~dp0_flask_local.bat"

echo       Open http://localhost:5173
call npm run dev
