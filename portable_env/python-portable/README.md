# FNO Model Training/Evaluation - Portable Python Environment

## Overview
This directory contains a fully portable Python environment for FNO (Fourier Neural Operator)
model training and evaluation. It can be copied to other Windows machines without requiring
Python installation.

## Directory Structure
`
python-portable/
鈹溾攢鈹€ python/              # Python interpreter and all dependencies
鈹溾攢鈹€ workspace/           # Working directory with scripts and data
鈹?  鈹溾攢鈹€ train_fno2_cfdbench.py      # Training script
鈹?  鈹溾攢鈹€ run_training_quick.py        # Quick training script
鈹?  鈹溾攢鈹€ evaluate_navier_stokes_model.py  # Evaluation script
鈹?  鈹溾攢鈹€ fno2_model.py                 # FNO model definition
鈹?  鈹溾攢鈹€ fno2_cfdbench_quick_model.pth # Pre-trained model
鈹?  鈹溾攢鈹€ verify_environment.py         # Environment verification
鈹?  鈹斺攢鈹€ requirements-fno-train.txt    # Dependencies list
鈹溾攢鈹€ run.bat             # Run Python script
鈹溾攢鈹€ start_cmd.bat       # Open command prompt
鈹溾攢鈹€ quick_train.bat     # One-click training
鈹斺攢鈹€ evaluate_model.bat  # One-click evaluation
`

## Quick Start

### Method 1: Using Batch Scripts (Recommended)

1. **Verify Environment**
   - Double-click start_cmd.bat to open command prompt
   - Run: python verify_environment.py

2. **Quick Training**
   - Double-click quick_train.bat
   - Or in command prompt: python run_training_quick.py

3. **Evaluate Model**
   - Double-click evaluate_model.bat
   - Or in command prompt: python evaluate_navier_stokes_model.py --model_path fno2_cfdbench_quick_model.pth

### Method 2: Using Command Line

1. Double-click start_cmd.bat to open configured command prompt
2. Execute Python commands

## Porting to Another Machine

### Steps

1. **Copy the entire directory**
   Copy the python-portable directory to any location on the target machine.

2. **Prepare Dataset**
   Ensure the target machine has the same dataset path:
   `
   C:\traework\data\
   鈹溾攢鈹€ cavity\
   鈹?  鈹溾攢鈹€ bc\
   鈹?  鈹溾攢鈹€ geo\
   鈹?  鈹斺攢鈹€ prop\
   `

3. **Run**
   - Double-click quick_train.bat or evaluate_model.bat

### Requirements

- 鉁?Supported: Windows 10/11 64-bit
- 鉂?Not supported: Windows 32-bit, Linux, macOS
- No Python installation required on target machine
- Dataset path must match: C:\traework\data\

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
`ash
python train_fno2_cfdbench.py 
    --data_root "C:\traework\data" 
    --epochs 30 
    --batch_size 8 
    --learning_rate 0.001 
    --modes 12 
    --width 32
`

### Evaluation Parameters
`ash
python evaluate_navier_stokes_model.py 
    --model_path "fno2_cfdbench_quick_model.pth" 
    --data_root "C:\traework\data" 
    --max_cases_per_category 10
`

## Troubleshooting

### Issue 1: Dataset Not Found
**Error:**
`
RuntimeError: No data loaded from C:\traework\data
`

**Solution:**
1. Check if dataset path exists: C:\traework\data\
2. Ensure correct directory structure:
   `
   C:\traework\data\cavity\bc\case0000\u.npy
   C:\traework\data\cavity\bc\case0000\v.npy
   `

### Issue 2: Missing Dependencies
**Solution:**
Run in command prompt:
`ash
pip install -r requirements-fno-train.txt
`

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
