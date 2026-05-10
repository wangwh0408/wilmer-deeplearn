@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%
set PYTHONPATH=%SCRIPT_DIR%\workspace

echo ============================================================
echo FNO MODEL EVALUATION
echo ============================================================
echo.
echo Python: %PYTHON_DIR%\python.exe
echo Workspace: %SCRIPT_DIR%\workspace
echo.
echo ============================================================
echo.

cd /d "%SCRIPT_DIR%\workspace"

echo You can customize evaluation with command line arguments:
echo   --model_path              Path to trained model (default: fno2_cfdbench_quick_model.pth)
echo   --data_root               Dataset root directory (default: C:\traework\data)
echo   --max_cases_per_category  Max cases per category (default: 10)
echo   --output_report           Output report path (default: evaluation_report.json)
echo   --help                    Show all options
echo.
echo Example:
echo   python evaluate_navier_stokes_model.py --model_path my_model.pth --data_root D:\data
echo.
echo ============================================================
echo Starting evaluation with default settings...
echo ============================================================
echo.

"%PYTHON_DIR%\python.exe" "evaluate_navier_stokes_model.py" %*

echo.
echo ============================================================
echo Evaluation complete!
echo ============================================================
pause
