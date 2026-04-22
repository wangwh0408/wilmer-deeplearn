import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_imports():
    print("=" * 80)
    print("VERIFYING FNO FOR CFDBENCH - IMPORTS")
    print("=" * 80)
    
    try:
        from data import (
            CFDBCase,
            CFDBCategory,
            CFDBenchDataset,
            create_cfdbench_dataloaders
        )
        print("  data module: OK")
    except Exception as e:
        print(f"  data module: FAILED - {e}")
        return False
    
    try:
        from models import (
            SpectralConv2d,
            FNOBlock,
            FNO2d,
            AutoRegressiveFNO,
            count_parameters
        )
        print("  models module: OK")
    except Exception as e:
        print(f"  models module: FAILED - {e}")
        return False
    
    try:
        from train_fno_cfdbench import (
            TrainingConfig,
            TrainingResult,
            CFDBenchTrainer
        )
        print("  train_fno_cfdbench module: OK")
    except Exception as e:
        print(f"  train_fno_cfdbench module: FAILED - {e}")
        return False
    
    try:
        from test_fno_cfdbench import (
            EvaluationMetrics,
            CFDBenchEvaluator
        )
        print("  test_fno_cfdbench module: OK")
    except Exception as e:
        print(f"  test_fno_cfdbench module: FAILED - {e}")
        return False
    
    return True


def verify_model():
    print("\n" + "=" * 80)
    print("VERIFYING FNO MODEL")
    print("=" * 80)
    
    from models import AutoRegressiveFNO, count_parameters
    
    model = AutoRegressiveFNO(
        modes1=12,
        modes2=12,
        width=32,
        in_channels=2,
        out_channels=2,
        n_layers=4,
        hidden_dim=128,
        use_coordinates=True,
        use_mask=False
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
    
    batch_size = 4
    grid_size = 64
    x = torch.randn(batch_size, grid_size, grid_size, 2)
    
    with torch.no_grad():
        output = model(x)
        print(f"\nSingle-step prediction:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        multi_step = model.generate_many(x, n_steps=5)
        print(f"\nMulti-step autoregressive prediction:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {multi_step.shape}")
    
    return True


def verify_metrics():
    print("\n" + "=" * 80)
    print("VERIFYING EVALUATION METRICS")
    print("=" * 80)
    
    from test_fno_cfdbench import EvaluationMetrics
    
    np.random.seed(42)
    target = np.random.randn(100, 64, 64, 2)
    pred = target + 0.1 * np.random.randn(*target.shape)
    
    metrics = EvaluationMetrics.compute_all(pred, target)
    print("Combined metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.6f}")
    
    u_pred = pred[..., 0]
    u_target = target[..., 0]
    v_pred = pred[..., 1]
    v_target = target[..., 1]
    
    print("\nU-component metrics:")
    for k, v in EvaluationMetrics.compute_all(u_pred, u_target).items():
        print(f"  {k:15s}: {v:.6f}")
    
    print("\nV-component metrics:")
    for k, v in EvaluationMetrics.compute_all(v_pred, v_target).items():
        print(f"  {k:15s}: {v:.6f}")
    
    return True


def verify_synthetic_data():
    print("\n" + "=" * 80)
    print("VERIFYING WITH SYNTHETIC DATA")
    print("=" * 80)
    
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticCFDDataset(Dataset):
        def __init__(self, n_samples=100, grid_size=64):
            self.n_samples = n_samples
            self.grid_size = grid_size
            
            self.inputs = torch.randn(n_samples, grid_size, grid_size, 2)
            self.outputs = torch.randn(n_samples, grid_size, grid_size, 2)
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            return {
                'input': self.inputs[idx],
                'output': self.outputs[idx],
                'case_name': f'case_{idx:04d}',
                'start_t': 0
            }
    
    print("Creating synthetic dataset...")
    dataset = SyntheticCFDDataset(n_samples=100, grid_size=64)
    print(f"  Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Input shape:  {sample['input'].shape}")
    print(f"  Output shape: {sample['output'].shape}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    from models import AutoRegressiveFNO
    from train_fno_cfdbench import TrainingConfig
    
    model = AutoRegressiveFNO(
        modes1=6,
        modes2=6,
        width=16,
        in_channels=2,
        out_channels=2,
        n_layers=2,
        hidden_dim=64
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    print("\nTraining on synthetic data (2 epochs)...")
    for epoch in range(2):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch['input']
            targets = batch['output']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                targets = batch['output']
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.6f}, test_loss={test_loss/len(test_loader):.6f}")
    
    return True


def main():
    print("=" * 80)
    print("FNO FOR CFDBENCH - VERIFICATION SCRIPT")
    print("=" * 80)
    print()
    
    all_passed = True
    
    all_passed = verify_imports() and all_passed
    all_passed = verify_model() and all_passed
    all_passed = verify_metrics() and all_passed
    all_passed = verify_synthetic_data() and all_passed
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VERIFICATIONS PASSED!")
        print("\nThe FNO training code for CFDBench is ready.")
        print("\nTo use with real CFDBench data:")
        print("1. Download CFDBench dataset from HuggingFace")
        print("2. Place data in 'data/' directory with structure:")
        print("   data/")
        print("   ├── cavity/")
        print("   │   ├── bc/")
        print("   │   │   ├── case0000/")
        print("   │   │   │   ├── u.npy")
        print("   │   │   │   └── v.npy")
        print("   │   │   ├── case0001/")
        print("   │   │   ...")
        print("   │   ├── geo/")
        print("   │   └── prop/")
        print("   ├── tube/")
        print("   ├── dam/")
        print("   └── cylinder/")
        print("\n3. Run training:")
        print("   python train_fno_cfdbench.py --data_root data --problems cavity --epochs 100")
        print("\n4. Run evaluation:")
        print("   python test_fno_cfdbench.py --model_path path/to/model.pth --data_root data")
    else:
        print("SOME VERIFICATIONS FAILED!")
        print("Please check the errors above.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
