@echo off
rem Resolve a usable Python into %PYTHON% and %PIP%.
rem
rem The dev scripts used to hardcode D:\Anaconda\envs\Horisation, which no
rem longer exists — so both failed with a bare "cannot find the path". Set
rem HORISATION_PYTHON to pin a specific interpreter; otherwise the first
rem candidate that exists on disk wins.

set "PYTHON="

if defined HORISATION_PYTHON if exist "%HORISATION_PYTHON%" set "PYTHON=%HORISATION_PYTHON%"

if not defined PYTHON if exist "D:\Anaconda\envs\Horisation\python.exe" set "PYTHON=D:\Anaconda\envs\Horisation\python.exe"
if not defined PYTHON if exist "D:\Anaconda\python.exe"                 set "PYTHON=D:\Anaconda\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

rem Fall back to PATH, skipping the Windows Store stub that only opens the Store.
if not defined PYTHON (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON (
      echo %%P | find /i "WindowsApps" >nul || set "PYTHON=%%P"
    )
  )
)

if not defined PYTHON (
  echo.
  echo   ERROR: No Python interpreter found.
  echo   Set HORISATION_PYTHON to your interpreter, for example:
  echo       set HORISATION_PYTHON=D:\Anaconda\envs\myenv\python.exe
  echo.
  exit /b 1
)

for %%I in ("%PYTHON%") do set "PYDIR=%%~dpI"
set "PIP=%PYDIR%Scripts\pip.exe"
if not exist "%PIP%" set "PIP="

exit /b 0
