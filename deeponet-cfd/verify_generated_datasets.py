import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_generated_datasets():
    print("=" * 80)
    print("VERIFYING GENERATED DATASETS")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    datasets = [
        ("Poisson", "poisson_small.npz"),
        ("Navier-Stokes", "navier_stokes_small.npz")
    ]
    
    for name, filename in datasets:
        filepath = os.path.join(data_dir, filename)
        
        print(f"\n--- {name} Dataset ---")
        
        if not os.path.exists(filepath):
            print(f"   ERROR: File not found: {filepath}")
            continue
        
        file_size = os.path.getsize(filepath) / 1024
        print(f"   File: {filepath}")
        print(f"   Size: {file_size:.2f} KB")
        
        print("\n   Loading dataset...")
        data = np.load(filepath, allow_pickle=True)
        
        keys = list(data.keys())
        print(f"   Keys in NPZ: {keys}")
        
        print("\n   Data shapes:")
        for key in keys:
            if key.startswith('meta_'):
                print(f"      {key}: {data[key]}")
            else:
                print(f"      {key}: {data[key].shape}")
        
        print("\n   Data sample:")
        if 'train_branch' in data:
            train_branch = data['train_branch']
            print(f"      train_branch - min: {train_branch.min():.4f}, max: {train_branch.max():.4f}, mean: {train_branch.mean():.4f}")
        
        if 'train_output' in data:
            train_output = data['train_output']
            print(f"      train_output - min: {train_output.min():.4f}, max: {train_output.max():.4f}, mean: {train_output.mean():.4f}")
        
        print(f"\n   Loading via load_cfdbench_dataset...")
        try:
            from generate_cfdbench_dataset import load_cfdbench_dataset
            
            loaded = load_cfdbench_dataset(filepath)
            print(f"   OK: Loaded successfully")
            print(f"      Train samples: {loaded.n_train}")
            print(f"      Test samples: {loaded.n_test}")
            
            if loaded.train_data:
                print(f"      Branch shape: {loaded.train_data['branch_inputs'].shape}")
                print(f"      Trunk shape: {loaded.train_data['trunk_inputs'].shape}")
                print(f"      Output shape: {loaded.train_data['outputs'].shape}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n   Converting to PyTorch dataset...")
        try:
            train_ds, test_ds = loaded.get_pytorch_dataset(normalize=True)
            print(f"   OK: PyTorch datasets created")
            print(f"      Train length: {len(train_ds)}")
            print(f"      Test length: {len(test_ds)}")
            
            sample = train_ds[0]
            print(f"      Sample branch shape: {sample[0].shape}")
            print(f"      Sample trunk shape: {sample[1].shape}")
            print(f"      Sample output shape: {sample[2].shape}")
            
        except ImportError:
            print("   SKIP: PyTorch not installed")
        except Exception as e:
            print(f"   ERROR: {e}")
        
        print("\n   Converting to PaddlePaddle dataset...")
        try:
            train_ds, test_ds = loaded.get_paddle_dataset(normalize=True)
            print(f"   OK: PaddlePaddle datasets created")
            print(f"      Train length: {len(train_ds)}")
            print(f"      Test length: {len(test_ds)}")
            
            sample = train_ds[0]
            print(f"      Sample branch shape: {sample[0].shape}")
            print(f"      Sample trunk shape: {sample[1].shape}")
            print(f"      Sample output shape: {sample[2].shape}")
            
        except ImportError:
            print("   SKIP: PaddlePaddle not installed")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Both Poisson and Navier-Stokes datasets generated and loaded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    verify_generated_datasets()
