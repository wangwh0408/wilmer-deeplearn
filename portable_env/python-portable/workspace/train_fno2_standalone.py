"""
Standalone FNO2 Training Script for Navier-Stokes Equations on CFDBench

This script is completely standalone and does NOT depend on any external directories
like 'deeponet-cfd'. It uses the same data loading logic as the evaluation script.

Task: Time series prediction
  - Input:  velocity field at time t (u, v) - 2 channels
  - Model auto-adds: x, y coordinates - 2 channels
  - Total input: 4 channels
  - Output: velocity field at time t+1 (u, v) - 2 channels

Usage:
  python train_fno2_standalone.py
  python train_fno2_standalone.py --data_root D:\data --model_output D:\models\my_model.pth
  python train_fno2_standalone.py --epochs 50 --batch_size 16
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import Dict, List, Tuple, Optional, Any
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import FNO2d model: {e}")
    print("Please ensure fno2_model.py exists in the same directory.")
    FNO2D_AVAILABLE = False


class CFDBenchNavierStokesDataset(Dataset):
    """
    CFDBench Navier-Stokes Dataset
    
    Loads u and v velocity components:
    - Input:  velocity at time t [batch, H, W, 2] (u, v)
    - Output: velocity at time t+1 [batch, H, W, 2] (u, v)
    
    The FNO2 model automatically adds coordinate channels (x, y).
    """
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        max_cases_per_category: int = None,
        normalize: bool = True,
        resolution: int = 64
    ):
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.max_cases_per_category = max_cases_per_category
        self.normalize = normalize
        self.resolution = resolution
        
        self.samples: List[Dict] = []
        
        self.u_mean = 0.0
        self.u_std = 1.0
        self.u_min = 0.0
        self.u_max = 1.0
        
        self.v_mean = 0.0
        self.v_std = 1.0
        self.v_min = 0.0
        self.v_max = 1.0
        
        self._load_data()
    
    def _load_data(self):
        """Load CFDBench data"""
        print(f"\nLoading CFDBench data from: {self.data_root}")
        print(f"Problems: {self.problems}")
        print(f"Categories: {self.categories}")
        
        all_u_data = []
        all_v_data = []
        
        for problem in self.problems:
            problem_path = os.path.join(self.data_root, problem)
            if not os.path.exists(problem_path):
                print(f"  Warning: Problem directory not found: {problem_path}")
                continue
            
            for category in self.categories:
                category_path = os.path.join(problem_path, category)
                if not os.path.exists(category_path):
                    print(f"  Warning: Category directory not found: {category_path}")
                    continue
                
                case_dirs = sorted(glob.glob(os.path.join(category_path, 'case*')))
                
                if self.max_cases_per_category:
                    case_dirs = case_dirs[:self.max_cases_per_category]
                
                for case_dir in case_dirs:
                    case_name = os.path.basename(case_dir)
                    
                    u_path = os.path.join(case_dir, 'u.npy')
                    v_path = os.path.join(case_dir, 'v.npy')
                    
                    if not os.path.exists(u_path) or not os.path.exists(v_path):
                        continue
                    
                    try:
                        u_data = np.load(u_path).astype(np.float32)
                        v_data = np.load(v_path).astype(np.float32)
                        
                        if u_data.ndim == 2:
                            u_data = u_data[np.newaxis, ...]
                        if v_data.ndim == 2:
                            v_data = v_data[np.newaxis, ...]
                        
                        if u_data.ndim == 4:
                            u_data = u_data[..., 0]
                        if v_data.ndim == 4:
                            v_data = v_data[..., 0]
                        
                        all_u_data.append(u_data)
                        all_v_data.append(v_data)
                        
                        time_steps = u_data.shape[0]
                        for t in range(time_steps - 1):
                            self.samples.append({
                                'input_u': u_data[t].copy(),
                                'input_v': v_data[t].copy(),
                                'output_u': u_data[t + 1].copy(),
                                'output_v': v_data[t + 1].copy(),
                                'case_name': f"{problem}/{category}/{case_name}",
                                'time_step': t
                            })
                        
                    except Exception as e:
                        print(f"  Warning: Failed to load {case_dir}: {e}")
        
        print(f"\n  Loaded {len(self.samples)} time-step pairs")
        
        if self.normalize and len(all_u_data) > 0 and len(all_v_data) > 0:
            all_u_concat = np.concatenate(all_u_data, axis=0)
            all_v_concat = np.concatenate(all_v_data, axis=0)
            
            self.u_mean = float(np.mean(all_u_concat))
            self.u_std = float(np.std(all_u_concat) + 1e-8)
            self.u_min = float(np.min(all_u_concat))
            self.u_max = float(np.max(all_u_concat))
            
            self.v_mean = float(np.mean(all_v_concat))
            self.v_std = float(np.std(all_v_concat) + 1e-8)
            self.v_min = float(np.min(all_v_concat))
            self.v_max = float(np.max(all_v_concat))
            
            print(f"\n  Normalization stats:")
            print(f"    U: mean={self.u_mean:.6f}, std={self.u_std:.6f}")
            print(f"    V: mean={self.v_mean:.6f}, std={self.v_std:.6f}")
            
            for sample in self.samples:
                sample['input_u'] = (sample['input_u'] - self.u_mean) / self.u_std
                sample['input_v'] = (sample['input_v'] - self.v_mean) / self.v_std
                sample['output_u'] = (sample['output_u'] - self.u_mean) / self.u_std
                sample['output_v'] = (sample['output_v'] - self.v_mean) / self.v_std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        sample = self.samples[idx]
        
        input_u = torch.tensor(sample['input_u'], dtype=torch.float32).unsqueeze(-1)
        input_v = torch.tensor(sample['input_v'], dtype=torch.float32).unsqueeze(-1)
        input_tensor = torch.cat([input_u, input_v], dim=-1)
        
        output_u = torch.tensor(sample['output_u'], dtype=torch.float32).unsqueeze(-1)
        output_v = torch.tensor(sample['output_v'], dtype=torch.float32).unsqueeze(-1)
        output_tensor = torch.cat([output_u, output_v], dim=-1)
        
        return input_tensor, output_tensor, sample['case_name'], sample['time_step']
    
    def get_normalization_params(self) -> Dict[str, float]:
        """Get normalization parameters"""
        return {
            'u_mean': self.u_mean,
            'u_std': self.u_std,
            'u_min': self.u_min,
            'u_max': self.u_max,
            'v_mean': self.v_mean,
            'v_std': self.v_std,
            'v_min': self.v_min,
            'v_max': self.v_max,
        }


def train_fno2_standalone(
    data_root: str = r"C:\traework\data",
    problems: List[str] = None,
    categories: List[str] = None,
    modes: int = 12,
    width: int = 32,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    normalize: bool = True,
    max_cases_per_category: int = None,
    scheduler_step_size: int = 15,
    scheduler_gamma: float = 0.5,
    model_save_path: Optional[str] = 'fno2_trained_model.pth',
    loss_plot_path: Optional[str] = 'training_loss.png',
    device: Optional[str] = None,
    verbose: bool = True,
    in_channels: int = 4,
    out_channels: int = 2,
) -> Dict[str, Any]:
    """
    Train FNO2 model on CFDBench dataset for Navier-Stokes equations.
    
    This is a standalone function that does NOT depend on external directories.
    """
    
    if not FNO2D_AVAILABLE:
        raise RuntimeError("FNO2d model not available. Please check fno2_model.py exists.")
    
    if problems is None:
        problems = ['cavity']
    if categories is None:
        categories = ['bc', 'geo', 'prop']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    config = {
        'data_root': data_root,
        'problems': problems,
        'categories': categories,
        'modes': modes,
        'width': width,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_ratio': train_ratio,
        'normalize': normalize,
        'max_cases_per_category': max_cases_per_category,
        'in_channels': in_channels,
        'out_channels': out_channels,
    }
    
    if verbose:
        print('=' * 70)
        print('FNO2 TRAINING FOR NAVIER-STOKES (CFDBENCH)')
        print('=' * 70)
        print(f'Device: {device}')
        print(f'Modes: {modes}, Width: {width}')
        print(f'Input channels: {in_channels} (2 data + 2 coords auto-added)')
        print(f'Output channels: {out_channels} (u, v)')
        print(f'Epochs: {epochs}, Batch Size: {batch_size}')
        print(f'Learning Rate: {learning_rate}')
        print(f'Data Root: {data_root}')
        print(f'Problems: {problems}')
        print(f'Categories: {categories}')
        if max_cases_per_category:
            print(f'Max cases per category: {max_cases_per_category}')
        print('=' * 70)
    
    if verbose:
        print('\nLoading CFDBench dataset...')
    
    dataset = CFDBenchNavierStokesDataset(
        data_root=data_root,
        problems=problems,
        categories=categories,
        max_cases_per_category=max_cases_per_category,
        normalize=normalize
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No data loaded from {data_root}. Please check the data path.")
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    if verbose:
        print(f'   OK: {n_total} samples loaded')
        print(f'   Split: {n_train} train, {n_test} test')
    
    train_size = int(n_total * train_ratio)
    test_size = n_total - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print('\nCreating FNO2 model...')
    
    model = FNO2d(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'   OK: Model created: {total_params:,} parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )
    criterion = nn.MSELoss()
    
    if verbose:
        print('\nStarting training...')
        print('-' * 70)
    
    train_losses = []
    test_losses = []
    test_metrics_history = []
    
    best_test_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            velocity_input = batch[0].to(device)
            velocity_target = batch[1].to(device)
            
            optimizer.zero_grad()
            output = model(velocity_input)
            loss = criterion(output, velocity_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * velocity_input.size(0)
        
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                velocity_input = batch[0].to(device)
                velocity_target = batch[1].to(device)
                
                output = model(velocity_input)
                
                loss = criterion(output, velocity_target)
                test_loss += loss.item() * velocity_input.size(0)
                
                all_outputs.append(output.cpu().numpy())
                all_targets.append(velocity_target.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mse = np.mean((all_outputs - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_outputs - all_targets))
        
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        test_metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        test_metrics_history.append(test_metrics)
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            is_best = ' (BEST)' if test_loss == best_test_loss else ''
            print(
                f'Epoch {epoch+1:3d}/{epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Test Loss: {test_loss:.6f}{is_best} | '
                f'R2: {r2:.4f} | '
                f'LR: {current_lr:.2e}'
            )
    
    if verbose:
        print('-' * 70)
        print('Training completed!')
        print(f'Best test loss: {best_test_loss:.6f}')
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_path = None
    if model_save_path is not None:
        if verbose:
            print(f'\nSaving model to: {model_save_path}')
        
        save_dict = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_metrics_history': test_metrics_history,
            'best_test_loss': best_test_loss,
            'modes': modes,
            'width': width,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'normalize': normalize,
            'normalization_params': dataset.get_normalization_params(),
            'config': config,
        }
        
        torch.save(save_dict, model_save_path)
        model_path = os.path.abspath(model_save_path)
        
        if verbose:
            print(f'   OK: Model saved to: {model_path}')
    
    final_train_loss = train_losses[-1] if train_losses else None
    final_test_loss = test_losses[-1] if test_losses else None
    final_test_metrics = test_metrics_history[-1] if test_metrics_history else None
    
    result = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_metrics_history': test_metrics_history,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'final_test_metrics': final_test_metrics,
        'best_test_loss': best_test_loss,
        'config': config,
        'model_path': model_path,
        'device': str(device),
        'normalization_params': dataset.get_normalization_params(),
    }
    
    if verbose:
        print('\n' + '=' * 70)
        print('TRAINING SUMMARY')
        print('=' * 70)
        print(f'Final Train Loss: {final_train_loss:.6f}')
        print(f'Final Test Loss:  {final_test_loss:.6f}')
        print(f'Best Test Loss:   {best_test_loss:.6f}')
        if final_test_metrics:
            print(f'Final Test R2:    {final_test_metrics["r2"]:.4f}')
            print(f'Final Test RMSE:  {final_test_metrics["rmse"]:.6f}')
        if model_path:
            print(f'Model saved at:   {model_path}')
        print('=' * 70)
    
    return result


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Standalone FNO2 Training for Navier-Stokes on CFDBench',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default=r'C:\traework\data',
        help='Root directory of CFDBench dataset'
    )
    
    parser.add_argument(
        '--model_output',
        type=str,
        default='fno2_trained_model.pth',
        help='Output path for trained model file'
    )
    
    parser.add_argument(
        '--problems',
        type=str,
        default='cavity',
        help='Comma-separated list of problems (e.g., cavity,tube,dam,cylinder)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        default='bc,geo,prop',
        help='Comma-separated list of categories (e.g., bc,geo,prop)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--modes',
        type=int,
        default=12,
        help='Number of Fourier modes (FNO hyperparameter)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=32,
        help='Channel width (FNO hyperparameter)'
    )
    
    parser.add_argument(
        '--max_cases_per_category',
        type=int,
        default=None,
        help='Maximum number of cases per category (use small for quick tests)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training data (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--loss_plot',
        type=str,
        default='training_loss.png',
        help='Output path for loss plot'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    print('=' * 70)
    print('STANDALONE FNO2 TRAINING FOR NAVIER-STOKES (CFDBENCH)')
    print('=' * 70)
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    problems = [p.strip() for p in args.problems.split(',')]
    categories = [c.strip() for c in args.categories.split(',')]
    
    result = train_fno2_standalone(
        data_root=args.data_root,
        problems=problems,
        categories=categories,
        modes=args.modes,
        width=args.width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_cases_per_category=args.max_cases_per_category,
        train_ratio=args.train_ratio,
        model_save_path=args.model_output,
        loss_plot_path=args.loss_plot,
        device=args.device,
        verbose=not args.quiet
    )
    
    print()
    print('=' * 70)
    print('TRAINING COMPLETED')
    print('=' * 70)
    print(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    if result.get('model_path'):
        print(f'\nModel saved to: {result["model_path"]}')
    
    print()
    print('NEXT STEPS:')
    print(f'  To evaluate the trained model:')
    print(f'    python evaluate_navier_stokes_model.py --model_path {args.model_output} --data_root {args.data_root}')
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
