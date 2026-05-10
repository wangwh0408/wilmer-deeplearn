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
