@echo off
cd /d "%~dp0\.."

echo ==============================
echo  Horisation - Build and Run
echo ==============================

echo [1/2] Building React app...
cd frontend
call npm install --silent
call npm run build
cd ..

echo [2/2] Starting Flask...
echo       Open http://localhost:5000
echo ==============================
python app.py
