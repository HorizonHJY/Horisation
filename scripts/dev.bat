@echo off
cd /d "%~dp0\.."

echo ==============================
echo  Horisation - Dev Mode
echo  Frontend : http://localhost:5173
echo  API      : http://localhost:5000
echo ==============================

:: Start Flask in a new window
start "Flask API" cmd /k "python app.py"

:: Install deps and start Vite in current window
cd frontend
call npm install --silent
call npm run dev
