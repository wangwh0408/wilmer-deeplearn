@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python
set WORKSPACE=%SCRIPT_DIR%\workspace

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%
set PYTHONPATH=%WORKSPACE%

title FNO Python Environment
cd /d "%WORKSPACE%"

echo ============================================================
echo FNO Model Training/Evaluation - Portable Python Environment
echo ============================================================
echo Python: %PYTHON_DIR%
echo Workspace: %WORKSPACE%
echo ============================================================
echo.
echo Available commands:
echo   python train_fno2_cfdbench.py      - Train model
echo   python evaluate_navier_stokes_model.py  - Evaluate model
echo   pip list                             - List installed packages
echo.

cmd /k
