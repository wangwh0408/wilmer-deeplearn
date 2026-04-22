import os
import sys
import platform
from datetime import datetime


def check_python_version():
    print("=" * 80)
    print("PYTHON ENVIRONMENT CHECK")
    print("=" * 80)
    
    print(f"\nPython Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    version_info = sys.version_info
    major, minor = version_info.major, version_info.minor
    
    if major >= 3 and minor >= 10:
        print(f"  Status: OK (Python {major}.{minor} >= 3.10)")
        return True
    else:
        print(f"  WARNING: Python {major}.{minor} may not be fully compatible.")
        print(f"           Recommended: Python 3.10+")
        return False


def check_imports():
    print("\n" + "=" * 80)
    print("PACKAGE IMPORT CHECK")
    print("=" * 80)
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("h5py", "h5py"),
        ("torch", "torch"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
    ]
    
    all_ok = True
    for pkg_name, display_name in packages:
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {display_name:15s}: OK (version {version})")
        except ImportError as e:
            print(f"  {display_name:15s}: FAILED - {e}")
            all_ok = False
    
    return all_ok


def check_pytorch():
    print("\n" + "=" * 80)
    print("PYTORCH CHECK")
    print("=" * 80)
    
    try:
        import torch
        
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                props = torch.cuda.get_device_properties(i)
                print(f"\n  CUDA Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            print("\nTesting CUDA operations...")
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = x @ y
                print(f"  CUDA Matrix Multiplication: OK")
                print(f"  Result shape: {z.shape}")
            except Exception as e:
                print(f"  CUDA Matrix Multiplication: FAILED - {e}")
            
            return True
        else:
            print("\nWARNING: CUDA is not available.")
            print("Training will use CPU, which may be slow for large datasets.")
            print("\nTo enable CUDA:")
            print("  1. Install NVIDIA CUDA Toolkit 11.8 or 12.1")
            print("  2. Install PyTorch with CUDA support:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("     or")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
            print("\nTesting CPU operations...")
            try:
                x = torch.randn(1000, 1000)
                y = torch.randn(1000, 1000)
                z = x @ y
                print(f"  CPU Matrix Multiplication: OK")
                print(f"  Result shape: {z.shape}")
            except Exception as e:
                print(f"  CPU Matrix Multiplication: FAILED - {e}")
            
            return True
            
    except ImportError as e:
        print(f"\nPyTorch import FAILED: {e}")
        print("\nPlease install PyTorch:")
        print("  pip install torch torchvision torchaudio")
        return False


def check_fno_model():
    print("\n" + "=" * 80)
    print("FNO MODEL CHECK")
    print("=" * 80)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import AutoRegressiveFNO, count_parameters
        
        print("\nCreating FNO model (small configuration)...")
        model = AutoRegressiveFNO(
            modes1=8,
            modes2=8,
            width=16,
            in_channels=2,
            out_channels=2,
            n_layers=2,
            hidden_dim=64
        )
        
        print(f"Model Parameters: {count_parameters(model):,}")
        
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"\nTesting forward pass on {device}...")
        batch_size = 2
        grid_size = 64
        
        x = torch.randn(batch_size, grid_size, grid_size, 2).to(device)
        
        with torch.no_grad():
            output = model(x)
            print(f"  Input shape:  {x.shape}")
            print(f"  Output shape: {output.shape}")
            
            print(f"\nTesting multi-step autoregressive prediction...")
            multi_step = model.generate_many(x, n_steps=5)
            print(f"  Input shape:    {x.shape}")
            print(f"  Output shape:   {multi_step.shape}")
            print(f"  (batch, steps, H, W, channels)")
        
        print(f"\nFNO Model: OK")
        return True
        
    except ImportError as e:
        print(f"\nFNO Model import FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFNO Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cfdbench_loader():
    print("\n" + "=" * 80)
    print("CFDBENCH LOADER CHECK")
    print("=" * 80)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data import CFDBenchDataset, create_cfdbench_dataloaders
        
        print(f"\nCFDBench Data Loader: OK")
        print(f"Available classes:")
        print(f"  - CFDBenchDataset: Load CFDBench dataset")
        print(f"  - create_cfdbench_dataloaders: Create DataLoaders for training/testing")
        
        print(f"\nLoader methods:")
        print(f"  - load(): Load data from directory")
        print(f"  - get_normalization_params(): Get data normalization parameters")
        print(f"  - __len__(): Get number of samples")
        print(f"  - __getitem__(): Get sample by index")
        
        return True
        
    except ImportError as e:
        print(f"\nCFDBench Loader import FAILED: {e}")
        return False


def main():
    print("=" * 80)
    print("FNO TRAINING ENVIRONMENT VERIFICATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Package Imports", check_imports()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("FNO Model", check_fno_model()))
    results.append(("CFDBench Loader", check_cfdbench_loader()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "OK" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"  {name:20s}: {status}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL CHECKS PASSED!")
        print("\nYour environment is ready for FNO training on CFDBench dataset.")
        print("\nNext steps:")
        print("  1. Place CFDBench data in 'data/' directory")
        print("  2. Run training:")
        print("     python train_fno_cfdbench.py --help")
        print("     python train_fno_cfdbench.py --data_root data --problems cavity --epochs 100")
    else:
        print("SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
