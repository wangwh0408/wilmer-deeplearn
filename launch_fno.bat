@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%
set PYTHONPATH=%SCRIPT_DIR%\workspace

echo ============================================================
echo FNO MODEL TRAINING/EVALUATION LAUNCHER
echo ============================================================
echo.
echo Python: %PYTHON_DIR%\python.exe
echo Workspace: %SCRIPT_DIR%\workspace
echo.
echo ============================================================
echo.

cd /d "%SCRIPT_DIR%\workspace"
"%PYTHON_DIR%\python.exe" "%SCRIPT_DIR%\workspace\launch_fno.py"

pause
