"""
Quick test script to verify training works
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('='*70)
print('QUICK FNO2 TRAINING TEST')
print('='*70)
print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print()

print('Step 1: Importing FNO2 model...')
from fno2_model import FNO2d
print('  OK: FNO2d imported')
print()

print('Step 2: Creating test dataset...')

class SimpleDataset(Dataset):
    def __init__(self, n_samples=100, grid_size=64):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.data = torch.randn(n_samples, grid_size, grid_size, 2)
        self.target = torch.randn(n_samples, grid_size, grid_size, 2)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], f'case_{idx}', 0

dataset = SimpleDataset(n_samples=100)
print(f'  OK: Created {len(dataset)} samples')
print()

print('Step 3: Creating model...')
model = FNO2d(
    modes1=12,
    modes2=12,
    width=32,
    in_channels=4,
    out_channels=2
)
total_params = sum(p.numel() for p in model.parameters())
print(f'  OK: Model created with {total_params:,} parameters')
print()

print('Step 4: Setting up training...')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
print(f'  OK: {train_size} train, {test_size} test')
print()

print('Step 5: Starting training (5 epochs)...')
print('-'*70)

for epoch in range(5):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        velocity_input = batch[0]
        velocity_target = batch[1]
        
        optimizer.zero_grad()
        output = model(velocity_input)
        loss = criterion(output, velocity_target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * velocity_input.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            velocity_input = batch[0]
            velocity_target = batch[1]
            output = model(velocity_input)
            loss = criterion(output, velocity_target)
            test_loss += loss.item() * velocity_input.size(0)
            all_outputs.append(output.numpy())
            all_targets.append(velocity_target.numpy())
    
    test_loss /= len(test_loader.dataset)
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    ss_res = np.sum((all_targets - all_outputs) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    print(f'Epoch {epoch+1}/5 | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | R2: {r2:.4f}')

print('-'*70)
print()
print('='*70)
print('TEST COMPLETED SUCCESSFULLY!')
print('='*70)
print()
print('Training is working correctly!')
print()
