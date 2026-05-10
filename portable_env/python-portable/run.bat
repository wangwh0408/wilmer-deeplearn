@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python
set WORKSPACE=%SCRIPT_DIR%\workspace

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%
set PYTHONPATH=%WORKSPACE%

echo ============================================================
echo FNO Model Training/Evaluation - Portable Python Environment
echo ============================================================
echo Python: %PYTHON_DIR%
echo Workspace: %WORKSPACE%
echo ============================================================

if "%~1"=="" (
    echo Usage: run.bat ^<script^> [args...]
    echo Examples:
    echo   run.bat train_fno2_cfdbench.py
    echo   run.bat evaluate_navier_stokes_model.py --model_path model.pth
    echo.
    echo Or enter Python interactive mode:
    "%PYTHON_DIR%\python.exe"
    pause
    exit /b 0
)

"%PYTHON_DIR%\python.exe" %*
