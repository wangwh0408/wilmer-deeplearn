<#
.SYNOPSIS
    Setup portable Python environment for FNO model training and evaluation
.DESCRIPTION
    This script creates a fully portable Python environment that can be copied
    to other Windows machines without requiring Python installation.
#>

param(
    [string]$PythonPath = "C:\Program Files\Python310",
    [string]$TargetDir = "$PSScriptRoot\python-portable",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "FNO Model Training/Evaluation - Portable Python Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to test Python path
function Test-PythonInstallation {
    param([string]$Path)
    return (Test-Path "$Path\python.exe")
}

# Step 1: Check if target directory exists
Write-Host "[Step 1/5] Checking environment..." -ForegroundColor Yellow

if (Test-Path $TargetDir) {
    if ($Force) {
        Write-Host "  Removing existing directory: $TargetDir" -ForegroundColor Yellow
        Remove-Item -Path $TargetDir -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "  Warning: Target directory already exists: $TargetDir" -ForegroundColor Red
        $confirm = Read-Host "  Remove and recreate? (y/n)"
        if ($confirm -eq 'y' -or $confirm -eq 'Y') {
            Write-Host "  Removing existing directory..." -ForegroundColor Yellow
            Remove-Item -Path $TargetDir -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "  Operation cancelled." -ForegroundColor Red
            exit 0
        }
    }
}

# Step 2: Find Python installation
Write-Host "[Step 2/5] Finding Python installation..." -ForegroundColor Yellow

$pythonFound = $false
$pythonLocations = @(
    $PythonPath,
    "C:\Python310",
    "C:\Program Files\Python310",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python310",
    "C:\Users\$env:USERNAME\AppData\Roaming\Python\Python310"
)

foreach ($loc in $pythonLocations) {
    if (Test-PythonInstallation $loc) {
        $PythonPath = $loc
        $pythonFound = $true
        break
    }
}

if (-not $pythonFound) {
    Write-Host "  Error: Python 3.10 installation not found in common locations." -ForegroundColor Red
    Write-Host ""
    $PythonPath = Read-Host "  Please enter your Python installation path (e.g., C:\Python310)"
    if (-not (Test-PythonInstallation $PythonPath)) {
        Write-Host "  Error: Invalid Python path. python.exe not found in: $PythonPath" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  Found Python at: $PythonPath" -ForegroundColor Green

# Step 3: Create directory structure
Write-Host "[Step 3/5] Creating directory structure..." -ForegroundColor Yellow

$dirs = @(
    "$TargetDir\python",
    "$TargetDir\workspace",
    "$TargetDir\workspace\data"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
}

# Step 4: Copy Python files
Write-Host "[Step 4/5] Copying Python files (this may take a few minutes)..." -ForegroundColor Yellow
Write-Host "  Source: $PythonPath" -ForegroundColor Gray
Write-Host "  Target: $TargetDir\python" -ForegroundColor Gray

try {
    Copy-Item -Path "$PythonPath\*" -Destination "$TargetDir\python\" -Recurse -Force -ErrorAction Stop
    Write-Host "  Python files copied successfully!" -ForegroundColor Green
} catch {
    Write-Host "  Warning: Error copying files: $_" -ForegroundColor Yellow
    Write-Host "  Please manually copy the Python directory:" -ForegroundColor Yellow
    Write-Host "    Copy '$PythonPath' to '$TargetDir\python'" -ForegroundColor Yellow
}

# Step 5: Copy workspace files
Write-Host "[Step 5/5] Copying workspace files..." -ForegroundColor Yellow

$workspaceFiles = @(
    "train_fno2_cfdbench.py",
    "run_training_quick.py",
    "evaluate_navier_stokes_model.py",
    "fno2_model.py",
    "fno2_cfdbench_quick_model.pth"
)

$scriptParent = Split-Path $PSScriptRoot -Parent

foreach ($file in $workspaceFiles) {
    $sourcePath = Join-Path $scriptParent $file
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination "$TargetDir\workspace\" -Force
        Write-Host "  Copied: $file" -ForegroundColor Green
    } else {
        Write-Host "  Not found: $file" -ForegroundColor Yellow
    }
}

# Copy requirements
$reqSource = Join-Path $PSScriptRoot "requirements-fno-train.txt"
if (Test-Path $reqSource) {
    Copy-Item -Path $reqSource -Destination "$TargetDir\workspace\" -Force
    Write-Host "  Copied: requirements-fno-train.txt" -ForegroundColor Green
}

# Create launch scripts
Write-Host ""
Write-Host "Creating launch scripts..." -ForegroundColor Yellow

# run.bat
$runBatContent = @"
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
"@
$runBatContent | Out-File -FilePath "$TargetDir\run.bat" -Encoding ASCII
Write-Host "  Created: run.bat" -ForegroundColor Green

# start_cmd.bat
$startCmdContent = @"
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
"@
$startCmdContent | Out-File -FilePath "$TargetDir\start_cmd.bat" -Encoding ASCII
Write-Host "  Created: start_cmd.bat" -ForegroundColor Green

# quick_train.bat
$quickTrainContent = @"
@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

echo ============================================================
echo Starting FNO Model Training...
echo ============================================================

"%PYTHON_DIR%\python.exe" "%SCRIPT_DIR%\workspace\run_training_quick.py"

echo.
echo Training complete!
pause
"@
$quickTrainContent | Out-File -FilePath "$TargetDir\quick_train.bat" -Encoding ASCII
Write-Host "  Created: quick_train.bat" -ForegroundColor Green

# evaluate_model.bat
$evaluateContent = @"
@echo off
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PYTHON_DIR=%SCRIPT_DIR%\python

set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%

echo ============================================================
echo Starting FNO Model Evaluation...
echo ============================================================

"%PYTHON_DIR%\python.exe" "%SCRIPT_DIR%\workspace\evaluate_navier_stokes_model.py" --model_path "%SCRIPT_DIR%\workspace\fno2_cfdbench_quick_model.pth"

echo.
echo Evaluation complete!
pause
"@
$evaluateContent | Out-File -FilePath "$TargetDir\evaluate_model.bat" -Encoding ASCII
Write-Host "  Created: evaluate_model.bat" -ForegroundColor Green

# Verify script
$verifyScript = @'
"""
Verify portable Python environment
Run this script to check if the environment is properly configured
"""
import sys
import os

print("=" * 60)
print("FNO Model Training/Evaluation - Environment Verification")
print("=" * 60)
print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

print("\n" + "=" * 60)
print("Checking core dependencies...")
print("=" * 60)

dependencies = [
    ('torch', 'PyTorch - Deep Learning Framework'),
    ('torchvision', 'TorchVision - Computer Vision Tools'),
    ('torchaudio', 'TorchAudio - Audio Processing Tools'),
    ('numpy', 'NumPy - Numerical Computing'),
    ('scipy', 'SciPy - Scientific Computing'),
    ('matplotlib', 'Matplotlib - Visualization'),
    ('h5py', 'H5Py - HDF5 Data Processing'),
    ('yaml', 'PyYAML - YAML Configuration'),
    ('tqdm', 'TQDM - Progress Bar'),
]

all_ok = True
for module_name, description in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"  [OK] {module_name:15s} v{version:15s} - {description}")
    except ImportError as e:
        print(f"  [ERROR] {module_name:15s} NOT INSTALLED - {description}")
        print(f"          Error: {e}")
        all_ok = False

print("\n" + "=" * 60)
print("Checking PyTorch details...")
print("=" * 60)

try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Using CPU mode")
    print(f"  [OK] PyTorch is working correctly")
except Exception as e:
    print(f"  [ERROR] PyTorch check failed: {e}")
    all_ok = False

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if all_ok:
    print("\n  [SUCCESS] All core dependencies are installed!")
    print("\n  You can now:")
    print("    1. Train model: python train_fno2_cfdbench.py")
    print("    2. Evaluate model: python evaluate_navier_stokes_model.py")
    print("    3. Quick train: python run_training_quick.py")
else:
    print("\n  [FAILED] Some dependencies are missing.")
    print("  Install missing dependencies with:")
    print("    pip install -r requirements-fno-train.txt")

print("\nVerification complete!")
'@
$verifyScript | Out-File -FilePath "$TargetDir\workspace\verify_environment.py" -Encoding UTF8
Write-Host "  Created: verify_environment.py" -ForegroundColor Green

# Create README
$readmeContent = @"
# FNO Model Training/Evaluation - Portable Python Environment

## Overview
This directory contains a fully portable Python environment for FNO (Fourier Neural Operator)
model training and evaluation. It can be copied to other Windows machines without requiring
Python installation.

## Directory Structure
```
python-portable/
├── python/              # Python interpreter and all dependencies
├── workspace/           # Working directory with scripts and data
│   ├── train_fno2_cfdbench.py      # Training script
│   ├── run_training_quick.py        # Quick training script
│   ├── evaluate_navier_stokes_model.py  # Evaluation script
│   ├── fno2_model.py                 # FNO model definition
│   ├── fno2_cfdbench_quick_model.pth # Pre-trained model
│   ├── verify_environment.py         # Environment verification
│   └── requirements-fno-train.txt    # Dependencies list
├── run.bat             # Run Python script
├── start_cmd.bat       # Open command prompt
├── quick_train.bat     # One-click training
└── evaluate_model.bat  # One-click evaluation
```

## Quick Start

### Method 1: Using Batch Scripts (Recommended)

1. **Verify Environment**
   - Double-click `start_cmd.bat` to open command prompt
   - Run: `python verify_environment.py`

2. **Quick Training**
   - Double-click `quick_train.bat`
   - Or in command prompt: `python run_training_quick.py`

3. **Evaluate Model**
   - Double-click `evaluate_model.bat`
   - Or in command prompt: `python evaluate_navier_stokes_model.py --model_path fno2_cfdbench_quick_model.pth`

### Method 2: Using Command Line

1. Double-click `start_cmd.bat` to open configured command prompt
2. Execute Python commands

## Porting to Another Machine

### Steps

1. **Copy the entire directory**
   Copy the `python-portable` directory to any location on the target machine.

2. **Prepare Dataset**
   Ensure the target machine has the same dataset path:
   ```
   C:\traework\data\
   ├── cavity\
   │   ├── bc\
   │   ├── geo\
   │   └── prop\
   ```

3. **Run**
   - Double-click `quick_train.bat` or `evaluate_model.bat`

### Requirements

- ✅ Supported: Windows 10/11 64-bit
- ❌ Not supported: Windows 32-bit, Linux, macOS
- No Python installation required on target machine
- Dataset path must match: `C:\traework\data\`

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.2 | PyTorch Deep Learning Framework |
| torchvision | 0.17.2 | Computer Vision Tools |
| torchaudio | 2.2.2 | Audio Processing Tools |
| numpy | 1.24.4 | Numerical Computing |
| scipy | 1.11.4 | Scientific Computing |
| matplotlib | 3.7.5 | Data Visualization |
| h5py | 3.10.0 | HDF5 Data Processing |
| pyyaml | 6.0.1 | YAML Configuration |
| tqdm | 4.66.2 | Progress Bar |

## Script Reference

### Training Scripts

**train_fno2_cfdbench.py**
- Complete training script
- Supports custom parameters: epochs, batch_size, learning_rate, etc.

**run_training_quick.py**
- Quick training script
- Uses preset parameters for fast training
- 10 samples per category, 30 epochs

### Evaluation Script

**evaluate_navier_stokes_model.py**
- Evaluate trained model
- Output metrics: MSE, RMSE, MAE, R2, MAPE
- Supports command line arguments

## Command Line Arguments

### Training Parameters
```bash
python train_fno2_cfdbench.py `
    --data_root "C:\traework\data" `
    --epochs 30 `
    --batch_size 8 `
    --learning_rate 0.001 `
    --modes 12 `
    --width 32
```

### Evaluation Parameters
```bash
python evaluate_navier_stokes_model.py `
    --model_path "fno2_cfdbench_quick_model.pth" `
    --data_root "C:\traework\data" `
    --max_cases_per_category 10
```

## Troubleshooting

### Issue 1: Dataset Not Found
**Error:**
```
RuntimeError: No data loaded from C:\traework\data
```

**Solution:**
1. Check if dataset path exists: `C:\traework\data\`
2. Ensure correct directory structure:
   ```
   C:\traework\data\cavity\bc\case0000\u.npy
   C:\traework\data\cavity\bc\case0000\v.npy
   ```

### Issue 2: Missing Dependencies
**Solution:**
Run in command prompt:
```bash
pip install -r requirements-fno-train.txt
```

### Issue 3: PyTorch CUDA Error
**Solution:**
1. Check if GPU drivers are properly installed
2. Or use CPU mode (default)
3. Code will auto-detect CUDA availability

## Performance Estimates

| Configuration | Estimated Time |
|----------------|----------------|
| CPU (i7) | ~1-2 minutes per epoch |
| GPU (RTX 3090) | ~10-30 seconds per epoch |

### Memory Requirements
- Python runtime: ~500MB
- PyTorch + dependencies: ~2GB
- During training: ~4-8GB (depends on batch size)

## Notes

- This environment is designed for Windows 64-bit systems only
- Not cross-platform compatible
- First run may require Windows firewall configuration
- The entire environment is approximately 3-4GB (mostly PyTorch)
"@
$readmeContent | Out-File -FilePath "$TargetDir\README.md" -Encoding UTF8
Write-Host "  Created: README.md" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Portable Python Environment Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Location: $TargetDir" -ForegroundColor Green
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor Yellow
Write-Host "  .\start_cmd.bat       - Open command prompt" -ForegroundColor Gray
Write-Host "  .\quick_train.bat     - Start training" -ForegroundColor Gray
Write-Host "  .\evaluate_model.bat  - Start evaluation" -ForegroundColor Gray
Write-Host ""
Write-Host "To verify the environment:" -ForegroundColor Yellow
Write-Host "  1. Double-click start_cmd.bat" -ForegroundColor Gray
Write-Host "  2. Run: python verify_environment.py" -ForegroundColor Gray
Write-Host ""
Write-Host "To port to another machine:" -ForegroundColor Yellow
Write-Host "  1. Copy the entire '$TargetDir' directory" -ForegroundColor Gray
Write-Host "  2. Ensure dataset exists at C:\traework\data\" -ForegroundColor Gray
Write-Host "  3. Double-click quick_train.bat or evaluate_model.bat" -ForegroundColor Gray
Write-Host ""
