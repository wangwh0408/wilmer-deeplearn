"""
Quick test script for CFDBench FNO2 training.

This script:
1. Tests if CFDBench data can be loaded
2. Tests if the model can be created and trained on a small batch
3. Saves a small test model

Run this first to verify everything works before full training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from typing import Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deeponet-cfd'))

try:
    from data.cfdbench_loader import CFDBenchDataset
    CFD_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import cfdbench_loader: {e}")
    print("Please ensure deeponet-cfd/data/cfdbench_loader.py exists")
    CFD_LOADER_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))
try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import fno2_model: {e}")
    FNO2D_AVAILABLE = False


class GridCoordinates:
    """Add grid coordinates to input."""
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        self.grid_size = grid_size
        self.coords = self._generate_coords()
    
    def _generate_coords(self) -> torch.Tensor:
        h, w = self.grid_size
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1)
        return coords
    
    def add_coords(self, velocity_field: torch.Tensor) -> torch.Tensor:
        if velocity_field.dim() == 3:
            velocity_field = velocity_field.unsqueeze(0)
        batch_size = velocity_field.shape[0]
        coords = self.coords.to(velocity_field.device)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([velocity_field, coords], dim=-1)


def test_data_loading(
    data_root: str = r"C:\traework\data",
    max_cases: int = 5
) -> bool:
    """Test if CFDBench data can be loaded."""
    
    print("\n" + "=" * 50)
    print("TEST 1: Data Loading")
    print("=" * 50)
    
    if not CFD_LOADER_AVAILABLE:
        print("❌ FAILED: CFDBench loader not available")
        return False
    
    if not os.path.exists(os.path.join(data_root, 'cavity')):
        print(f"❌ FAILED: Data directory not found: {os.path.join(data_root, 'cavity')}")
        print(f"   Expected structure: {data_root}/cavity/bc/case0000/")
        return False
    
    try:
        dataset = CFDBenchDataset(
            data_root=data_root,
            problems=['cavity'],
            categories=['bc'],
            input_steps=1,
            output_steps=1,
            normalize=True,
            max_cases_per_category=max_cases
        )
        dataset.load()
        
        if len(dataset) == 0:
            print(" [FAIL] FAILED: No samples loaded")
            return False
        
        print(f" [OK] Loaded {len(dataset.all_cases)} cases, {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"   Sample shape:")
        print(f"   - Input: {sample['input'].shape}")
        print(f"   - Output: {sample['output'].shape}")
        print(f"   - Case: {sample['case_name']}")
        
        if sample['input'].shape[-1] == 2:
            print(" [OK] Input has correct shape: [H, W, 2] (u, v)")
        else:
            print(f" [WARN] Input has unexpected shape: {sample['input'].shape}")
        
        return True
        
    except Exception as e:
        print(f" [FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation() -> bool:
    """Test if FNO2 model can be created with correct channels."""
    
    print("\n" + "=" * 50)
    print("TEST 2: Model Creation")
    print("=" * 50)
    
    if not FNO2D_AVAILABLE:
        print(" [FAIL] FAILED: FNO2d model not available")
        return False
    
    try:
        model = FNO2d(
            modes1=12,
            modes2=12,
            width=32,
            in_channels=4,
            out_channels=2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f" [OK] Model created successfully")
        print(f"   - Parameters: {total_params:,}")
        print(f"   - Input channels: 4 (u, v, x, y)")
        print(f"   - Output channels: 2 (u, v)")
        
        return True
        
    except Exception as e:
        print(f" [FAIL] FAILED: {e}")
        return False


def test_forward_pass() -> bool:
    """Test if model can do a forward pass."""
    
    print("\n" + "=" * 50)
    print("TEST 3: Forward Pass")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        grid_coords = GridCoordinates((64, 64))
        model = FNO2d(
            modes1=12,
            modes2=12,
            width=32,
            in_channels=4,
            out_channels=2
        ).to(device)
        
        batch_size = 2
        velocity_input = torch.randn(batch_size, 64, 64, 2).to(device)
        
        model_input = velocity_input  # 模型会自动添加坐标
        print(f"Input shape: {model_input.shape} (expected: [2, 64, 64, 2])")
        
        output = model(model_input)
        print(f"Output shape: {output.shape} (expected: [2, 64, 64, 2])")
        
        if output.shape == (batch_size, 64, 64, 2):
            print(" [OK] Forward pass successful!")
            return True
        else:
            print(f" [FAIL] Output shape mismatch: expected (2, 64, 64, 2), got {output.shape}")
            return False
            
    except Exception as e:
        print(f" [FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step() -> bool:
    """Test one training step."""
    
    print("\n" + "=" * 50)
    print("TEST 4: Training Step")
    print("=" * 50)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        grid_coords = GridCoordinates((64, 64))
        model = FNO2d(
            modes1=12,
            modes2=12,
            width=32,
            in_channels=4,
            out_channels=2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        batch_size = 4
        velocity_input = torch.randn(batch_size, 64, 64, 2).to(device)
        velocity_target = torch.randn(batch_size, 64, 64, 2).to(device)
        
        model.train()
        optimizer.zero_grad()
        
        output = model(velocity_input)  # 模型自动添加坐标
        loss = criterion(output, velocity_target)
        loss.backward()
        optimizer.step()
        
        print(f" [OK] Training step successful!")
        print(f"   Loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f" [FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(
    data_root: str = r"C:\traework\data",
    quick: bool = True
) -> Dict[str, bool]:
    """Run all tests."""
    
    print("\n" + "=" * 60)
    print("FNO2 CFDBENCH TRAINING - QUICK TEST")
    print("=" * 60)
    
    results = {}
    
    results['data_loading'] = test_data_loading(
        data_root=data_root,
        max_cases=3 if quick else 10
    )
    
    results['model_creation'] = test_model_creation()
    results['forward_pass'] = test_forward_pass()
    results['training_step'] = test_training_step()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = " [OK] PASS" if passed else " [FAIL] FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print(" [OK] ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Run full training with: python train_fno2_cfdbench.py")
        print("  2. Or use: quick_train_cfdbench(epochs=50, max_cases_per_category=20)")
    else:
        print(" [FAIL] SOME TESTS FAILED")
        print("\nPlease fix the issues before running full training.")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test for CFDBench FNO2 training')
    parser.add_argument('--data_root', type=str, default=r"C:\traework\data",
                        help='Root directory of CFDBench dataset')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Run full test (not quick)')
    
    args = parser.parse_args()
    
    run_all_tests(data_root=args.data_root, quick=not args.full)
