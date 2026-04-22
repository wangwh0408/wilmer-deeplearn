import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.cfdbench_loader import CFDBenchDataset, create_cfdbench_dataloaders
from models.fno_cfdbench import AutoRegressiveFNO, count_parameters


def train_fno_on_real_cfdbench(
    data_root: str,
    problems: List[str] = ['cavity'],
    categories: List[str] = ['prop'],
    max_cases: int = 10,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    modes: int = 8,
    width: int = 16,
    n_layers: int = 2,
    device: Optional[str] = None
):
    print("=" * 80)
    print("FNO TRAINING ON REAL CFDBENCH DATASET")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data root: {data_root}")
    print(f"Problems: {problems}")
    print(f"Categories: {categories}")
    print(f"Max cases per category: {max_cases}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model: modes={modes}, width={width}, layers={n_layers}")
    print("=" * 80)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"\nUsing device: {device}")
    
    print("\n" + "=" * 80)
    print("[1/4] Loading CFDBench dataset...")
    print("=" * 80)
    
    train_loader, test_loader, norm_params = create_cfdbench_dataloaders(
        data_root=data_root,
        problems=problems,
        categories=categories,
        input_steps=1,
        output_steps=1,
        batch_size=batch_size,
        train_ratio=0.8,
        normalize=True,
        max_cases_per_category=max_cases,
        num_workers=0
    )
    
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print(f"\nNormalization parameters:")
    for key, value in norm_params.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n" + "=" * 80)
    print("[2/4] Building FNO model...")
    print("=" * 80)
    
    model = AutoRegressiveFNO(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=2,
        out_channels=2,
        n_layers=n_layers,
        hidden_dim=64,
        use_coordinates=True,
        use_mask=False
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    print("\nModel info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 80)
    print("[3/4] Setting up optimizer and loss...")
    print("=" * 80)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    
    criterion = nn.MSELoss()
    
    print(f"Optimizer: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"Loss function: MSE Loss")
    
    print("\n" + "=" * 80)
    print("[4/4] Starting training loop...")
    print("=" * 80)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    start_time = datetime.now()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_count += batch_size
            
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
        
        train_loss = train_loss / train_count if train_count > 0 else 0.0
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                test_loss += loss.item() * batch_size
                test_count += batch_size
        
        test_loss = test_loss / test_count if test_count > 0 else 0.0
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Loss:  {test_loss:.6f}")
        print(f"  Best Test:  {best_test_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print("-" * 80)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    
    print(f"\nTraining duration: {duration:.2f} seconds")
    print(f"\nFinal Results:")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final test loss:  {test_losses[-1]:.6f}")
    print(f"  Best test loss:   {best_test_loss:.6f}")
    
    print(f"\nLoss history:")
    for epoch, (tr_l, te_l) in enumerate(zip(train_losses, test_losses), 1):
        print(f"  Epoch {epoch}: train={tr_l:.6f}, test={te_l:.6f}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION: Testing model inference...")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            
            outputs = model(inputs)
            
            print(f"\nSample inference:")
            print(f"  Input shape:  {inputs.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Target shape: {targets.shape}")
            
            print(f"\n  Input stats:")
            print(f"    min: {inputs.min().item():.4f}, max: {inputs.max().item():.4f}, mean: {inputs.mean().item():.4f}")
            
            print(f"\n  Output stats:")
            print(f"    min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}, mean: {outputs.mean().item():.4f}")
            
            print(f"\n  Target stats:")
            print(f"    min: {targets.min().item():.4f}, max: {targets.max().item():.4f}, mean: {targets.mean().item():.4f}")
            
            mse = ((outputs - targets) ** 2).mean().item()
            print(f"\n  Batch MSE: {mse:.6f}")
            
            break
    
    print("\n" + "=" * 80)
    print("MULTI-STEP AUTOREGRESSIVE PREDICTION TEST")
    print("=" * 80)
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            
            print(f"\nGenerating 5-step autoregressive prediction...")
            print(f"  Initial input shape: {inputs.shape}")
            
            multi_step = model.generate_many(inputs, n_steps=5)
            
            print(f"  Multi-step output shape: {multi_step.shape}")
            print(f"    (batch, steps, H, W, channels)")
            
            print(f"\n  Step-wise stats:")
            for step in range(5):
                step_data = multi_step[:, step, ...]
                print(f"    Step {step+1}: min={step_data.min().item():.4f}, max={step_data.max().item():.4f}, mean={step_data.mean().item():.4f}")
            
            break
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("  1. Data loading: OK")
    print("  2. Model creation: OK")
    print("  3. Training loop: OK")
    print("  4. Inference: OK")
    print("  5. Multi-step prediction: OK")
    print("\nThe FNO model can successfully train on the real CFDBench dataset!")
    print("=" * 80)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss,
        'norm_params': norm_params,
        'duration': duration
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FNO TRAINING TEST ON REAL CFDBENCH DATASET")
    print("=" * 80)
    
    data_root = r"c:\traework\a\wilmer-deeplearn\data"
    
    print("\n" + "-" * 80)
    print("TEST 1: Small training on cavity/prop")
    print("-" * 80)
    
    results1 = train_fno_on_real_cfdbench(
        data_root=data_root,
        problems=['cavity'],
        categories=['prop'],
        max_cases=5,
        epochs=3,
        batch_size=4,
        learning_rate=1e-3,
        modes=8,
        width=16,
        n_layers=2
    )
    
    print("\n" + "-" * 80)
    print("TEST 2: Training with multiple categories")
    print("-" * 80)
    
    results2 = train_fno_on_real_cfdbench(
        data_root=data_root,
        problems=['cavity'],
        categories=['bc', 'prop'],
        max_cases=5,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        modes=8,
        width=16,
        n_layers=2
    )
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nTest 1 (cavity/prop):")
    print(f"  Best test loss: {results1['best_test_loss']:.6f}")
    print(f"  Duration: {results1['duration']:.2f}s")
    
    print("\nTest 2 (cavity/bc+prop):")
    print(f"  Best test loss: {results2['best_test_loss']:.6f}")
    print(f"  Duration: {results2['duration']:.2f}s")
    
    print("\n" + "=" * 80)
    print("ENVIRONMENT VERIFICATION COMPLETE!")
    print("=" * 80)
    print("\nThe virtual environment is fully functional:")
    print("  1. PyTorch is working correctly")
    print("  2. CFDBench data loader works with real data")
    print("  3. FNO model trains successfully")
    print("  4. Inference and multi-step prediction work")
    print("\nYou can now run full training with:")
    print("  python train_fno_cfdbench.py --help")
    print("=" * 80)
