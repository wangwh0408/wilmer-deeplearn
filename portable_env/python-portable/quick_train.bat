@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%
set PYTHONPATH=%SCRIPT_DIR%\workspace

echo ============================================================
echo FNO MODEL QUICK TRAINING (Standalone Version)
echo ============================================================
echo.
echo Python: %PYTHON_DIR%\python.exe
echo Workspace: %SCRIPT_DIR%\workspace
echo.
echo This script uses the standalone training script that does NOT
echo depend on any external directories like 'deeponet-cfd'.
echo.
echo ============================================================
echo.

cd /d "%SCRIPT_DIR%\workspace"

echo You can customize training with command line arguments:
echo   --data_root           Dataset root directory (default: C:\traework\data)
echo   --model_output        Output model path (default: fno2_trained_model.pth)
echo   --epochs              Number of epochs (default: 30)
echo   --batch_size          Batch size (default: 8)
echo   --learning_rate       Learning rate (default: 0.001)
echo   --max_cases_per_category  Max cases per category (default: all)
echo   --help                Show all options
echo.
echo Example:
echo   quick_train.bat --data_root D:\data --model_output D:\models\my_model.pth --epochs 50
echo.
echo ============================================================
echo Starting training with default settings...
echo Data: C:\traework\data
echo Output: fno2_trained_model.pth
echo Epochs: 30
echo ============================================================
echo.

"%PYTHON_DIR%\python.exe" "train_fno2_standalone.py" %*

echo.
echo ============================================================
echo Training complete!
echo ============================================================
pause
