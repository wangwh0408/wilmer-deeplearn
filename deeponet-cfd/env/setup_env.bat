@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   FNO Training Environment Setup
echo   for CFDBench Dataset
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set ENV_DIR=%SCRIPT_DIR%\..\venv-fno
set FRAMEWORK=both

echo Available frameworks:
echo   1. PyTorch only
echo   2. PaddlePaddle only
echo   3. Both (recommended)
echo.

set /p FRAMEWORK_CHOICE="Please select framework [1/2/3] (default: 3): "

if "%FRAMEWORK_CHOICE%"=="" set FRAMEWORK_CHOICE=3

if "%FRAMEWORK_CHOICE%"=="1" (
    set FRAMEWORK=pytorch
    set REQS_FILE=%SCRIPT_DIR%\requirements-fno.txt
    echo Selected: PyTorch only
) else if "%FRAMEWORK_CHOICE%"=="2" (
    set FRAMEWORK=paddle
    set REQS_FILE=%SCRIPT_DIR%\requirements-fno-paddle.txt
    echo Selected: PaddlePaddle only
) else (
    set FRAMEWORK=both
    set REQS_FILE=%SCRIPT_DIR%\requirements-fno.txt
    echo Selected: Both PyTorch and PaddlePaddle
)
echo.

echo [Step 1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

echo [Step 2/5] Creating virtual environment...
if exist "%ENV_DIR%" (
    echo Virtual environment already exists at: %ENV_DIR%
    set /p OVERWRITE="Do you want to recreate it? (y/n): "
    if /i "!OVERWRITE!"=="y" (
        echo Removing existing environment...
        rmdir /s /q "%ENV_DIR%"
    ) else (
        echo Skipping environment creation.
        goto :activate_env
    )
)

echo Creating virtual environment at: %ENV_DIR%
python -m venv "%ENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo Environment created successfully.
echo.

:activate_env
echo [Step 3/5] Activating virtual environment...
call "%ENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo Environment activated.
echo.

echo [Step 4/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [Step 5/5] Installing required packages...
echo Installing from: %REQS_FILE%
echo.

if not exist "%REQS_FILE%" (
    echo [ERROR] Requirements file not found: %REQS_FILE%
    pause
    exit /b 1
)

pip install -r "%REQS_FILE%"
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some packages may have failed to install.
    echo Please check the error messages above.
    echo.
    echo If CUDA-related errors occur, you may need to:
    echo   1. Install CUDA Toolkit 11.8 or 12.1
    echo   2. Install PyTorch/PaddlePaddle with CUDA support manually
    echo.
    echo PyTorch CUDA installation:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo   or
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo PaddlePaddle GPU installation:
    echo   python -m pip install paddlepaddle-gpu==2.6.2.post118 -i https://mirror.baidu.com/pypi/simple
    echo   or
    echo   python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://mirror.baidu.com/pypi/simple
    echo.
) else (
    echo.
    echo All packages installed successfully!
)

echo.
echo ========================================
echo   Verifying installation...
echo ========================================
echo.

python "%SCRIPT_DIR%\verify_environment_paddle.py"

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Installed frameworks:

if "%FRAMEWORK%"=="pytorch" (
    echo   - PyTorch: Yes
    echo   - PaddlePaddle: No
) else if "%FRAMEWORK%"=="paddle" (
    echo   - PyTorch: No
    echo   - PaddlePaddle: Yes
) else (
    echo   - PyTorch: Yes
    echo   - PaddlePaddle: Yes
)

echo.
echo Virtual environment:
echo   %ENV_DIR%
echo.
echo To activate the environment:
echo   call "%ENV_DIR%\Scripts\activate.bat"
echo.
echo To verify the installation:
echo   python "%SCRIPT_DIR%\verify_environment.py"
echo   python "%SCRIPT_DIR%\verify_environment_paddle.py"
echo.
echo To run PyTorch training:
echo   python train_fno_cfdbench.py --help
echo.
echo To run PaddlePaddle training:
echo   python train_fno_cfdbench_paddle.py --help
echo.
pause
