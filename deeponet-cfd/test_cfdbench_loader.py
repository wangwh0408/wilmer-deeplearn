import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.cfdbench_loader import CFDBenchDataset, create_cfdbench_dataloaders


def test_data_loading():
    print("=" * 80)
    print("TESTING CFDBENCH DATA LOADING")
    print("=" * 80)
    
    data_root = r"c:\traework\a\wilmer-deeplearn\data"
    
    print(f"\nData root: {data_root}")
    print(f"Directory exists: {os.path.exists(data_root)}")
    
    if os.path.exists(data_root):
        print(f"\nContents of data directory:")
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                print(f"  [DIR] {item}")
            else:
                print(f"  [FILE] {item}")
    
    print("\n" + "=" * 80)
    print("TEST 1: Load cavity/prop category")
    print("=" * 80)
    
    dataset1 = CFDBenchDataset(
        data_root=data_root,
        problems=['cavity'],
        categories=['prop'],
        input_steps=1,
        output_steps=1,
        max_cases_per_category=10
    )
    dataset1.load()
    
    print(f"\nLoaded {len(dataset1.all_cases)} cases, {len(dataset1.samples)} samples")
    
    if len(dataset1) > 0:
        sample = dataset1[0]
        print(f"\nSample info:")
        print(f"  Case name: {sample['case_name']}")
        print(f"  Start time: {sample['start_t']}")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Output shape: {sample['output'].shape}")
        print(f"  Input dtype: {sample['input'].dtype}")
        
        input_data = sample['input'].numpy()
        output_data = sample['output'].numpy()
        print(f"\nData stats:")
        print(f"  Input - min: {input_data.min():.4f}, max: {input_data.max():.4f}, mean: {input_data.mean():.4f}")
        print(f"  Output - min: {output_data.min():.4f}, max: {output_data.max():.4f}, mean: {output_data.mean():.4f}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Load cavity/bc category")
    print("=" * 80)
    
    dataset2 = CFDBenchDataset(
        data_root=data_root,
        problems=['cavity'],
        categories=['bc'],
        input_steps=1,
        output_steps=1,
        max_cases_per_category=10
    )
    dataset2.load()
    
    print(f"Loaded {len(dataset2.all_cases)} cases, {len(dataset2.samples)} samples")
    
    if len(dataset2) > 0:
        sample = dataset2[0]
        print(f"\nSample info:")
        print(f"  Case name: {sample['case_name']}")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Output shape: {sample['output'].shape}")
    
    print("\n" + "=" * 80)
    print("TEST 3: Load multiple problems")
    print("=" * 80)
    
    dataset3 = CFDBenchDataset(
        data_root=data_root,
        problems=['cavity', 'cylinder', 'dam', 'tube'],
        categories=['prop'],
        input_steps=1,
        output_steps=1,
        max_cases_per_category=5
    )
    dataset3.load()
    
    print(f"Loaded {len(dataset3.all_cases)} cases, {len(dataset3.samples)} samples")
    print(f"\nNormalization params:")
    print(f"  u: mean={dataset3.mean_u:.6f}, std={dataset3.std_u:.6f}")
    print(f"  v: mean={dataset3.mean_v:.6f}, std={dataset3.std_v:.6f}")
    
    print("\n" + "=" * 80)
    print("TEST 4: Check a single case in detail")
    print("=" * 80)
    
    if len(dataset1.all_cases) > 0:
        case = dataset1.all_cases[0]
        print(f"\nCase: {case.name}")
        print(f"  Time steps: {case.time_steps}")
        print(f"  Grid size: {case.grid_size}")
        
        if case.u_data is not None:
            print(f"\n  u_data shape: {case.u_data.shape}")
            print(f"  u_data - min: {case.u_data.min():.4f}, max: {case.u_data.max():.4f}")
        
        if case.v_data is not None:
            print(f"\n  v_data shape: {case.v_data.shape}")
            print(f"  v_data - min: {case.v_data.min():.4f}, max: {case.v_data.max():.4f}")
    
    print("\n" + "=" * 80)
    print("TEST 5: Create DataLoaders")
    print("=" * 80)
    
    train_loader, test_loader, norm_params = create_cfdbench_dataloaders(
        data_root=data_root,
        problems=['cavity'],
        categories=['prop'],
        batch_size=4,
        train_ratio=0.8,
        max_cases_per_category=10
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"\nNormalization params: {norm_params}")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nTrain batch {batch_idx}:")
        print(f"  Input shape: {batch['input'].shape}")
        print(f"  Output shape: {batch['output'].shape}")
        print(f"  Case names: {batch['case_name'][:3]}...")
        break
    
    print("\n" + "=" * 80)
    print("ALL DATA LOADING TESTS PASSED!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    test_data_loading()
