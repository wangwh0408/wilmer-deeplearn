import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_imports():
    print("=" * 80)
    print("VERIFYING DEEPONET PROJECT IMPORTS")
    print("=" * 80)
    
    all_ok = True
    
    print("\n1. Verifying config module...")
    try:
        from config import setup_logger, TrainingLogger
        print("   OK: config module imported successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n2. Verifying models module...")
    try:
        from models import (
            BranchNet, TrunkNet, DeepONet, DeepONetConfig,
            BranchNetPaddle, TrunkNetPaddle, DeepONetPaddle, DeepONetConfigPaddle
        )
        print("   OK: models module imported successfully")
        
        import numpy as np
        print("\n   Testing DeepONet (PyTorch) initialization...")
        try:
            import torch
            model = DeepONet(
                branch_input_dim=100,
                trunk_input_dim=2,
                branch_hidden_layers=[64, 64],
                trunk_hidden_layers=[64, 64],
                output_dim=100
            )
            print(f"      OK: DeepONet created - {model.count_parameters()} parameters")
            print(f"      Model info: {model.get_model_info()}")
        except ImportError:
            print("      PyTorch not installed, skipping PyTorch model test")
        except Exception as e:
            print(f"      ERROR: {e}")
            all_ok = False
        
        print("\n   Testing DeepONetPaddle initialization...")
        try:
            import paddle
            model_paddle = DeepONetPaddle(
                branch_input_dim=100,
                trunk_input_dim=2,
                branch_hidden_layers=[64, 64],
                trunk_hidden_layers=[64, 64],
                output_dim=100
            )
            print(f"      OK: DeepONetPaddle created - {model_paddle.count_parameters()} parameters")
            print(f"      Model info: {model_paddle.get_model_info()}")
        except ImportError:
            print("      PaddlePaddle not installed, skipping Paddle model test")
        except Exception as e:
            print(f"      ERROR: {e}")
            all_ok = False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n3. Verifying data module...")
    try:
        from data import (
            CFDDataGenerator,
            CFDDatasetPyTorch,
            CFDDatasetPaddle,
            create_datasets,
            HAS_TORCH,
            HAS_PADDLE
        )
        print("   OK: data module imported successfully")
        print(f"   HAS_TORCH: {HAS_TORCH}")
        print(f"   HAS_PADDLE: {HAS_PADDLE}")
        
        print("\n   Testing CFDDataGenerator...")
        try:
            generator = CFDDataGenerator(
                n_samples=10,
                grid_size=32
            )
            branch, trunk, output = generator.generate_dataset("poisson")
            print(f"      OK: Generated dataset")
            print(f"         branch shape: {branch.shape}")
            print(f"         trunk shape: {trunk.shape}")
            print(f"         output shape: {output.shape}")
        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n4. Verifying train module...")
    try:
        from train import (
            train_deeponet_pytorch, quick_train_pytorch,
            train_deeponet_paddle, quick_train_paddle
        )
        print("   OK: train module imported successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n5. Verifying main script...")
    try:
        import train_deeponet
        print("   OK: train_deeponet.py imported successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n" + "=" * 80)
    if all_ok:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
    print("=" * 80)
    
    return all_ok


def show_usage_examples():
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("""
1. Command Line Usage:

   # PyTorch, default parameters
   python train_deeponet.py --framework pytorch
   
   # PaddlePaddle, custom epochs and batch size
   python train_deeponet.py --framework paddle --epochs 200 --batch_size 128
   
   # Quick training (small dataset)
   python train_deeponet.py --framework pytorch --quick_train --epochs 5
   
   # Save model and log
   python train_deeponet.py --framework pytorch --model_save_path models/deeponet.pth --log_file logs/train.log
   
   # Custom network architecture
   python train_deeponet.py --framework pytorch --branch_hidden_layers 512 512 --trunk_hidden_layers 512 512
   
   # Show help
   python train_deeponet.py --help

2. Python API Usage:

   from train.trainer_pytorch import train_deeponet_pytorch, quick_train_pytorch
   
   # Full training
   result = train_deeponet_pytorch(
       epochs=100,
       batch_size=64,
       learning_rate=1e-3,
       n_train_samples=1000,
       n_test_samples=200,
       grid_size=64,
       model_save_path='deeponet_model.pth',
       log_file='training.log'
   )
   
   # Quick training
   result = quick_train_pytorch(
       epochs=10,
       batch_size=32
   )
   
   # Access results
   print(f"Final Train Loss: {result['final_train_loss']}")
   print(f"Final Test Loss: {result['final_test_loss']}")
   print(f"Model Path: {result['model_path']}")
   model = result['model']
""")
    print("=" * 80)


if __name__ == "__main__":
    success = verify_imports()
    
    if success:
        show_usage_examples()
    
    sys.exit(0 if success else 1)
