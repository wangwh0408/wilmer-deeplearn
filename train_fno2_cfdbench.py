"""
Train FNO2 model on CFDBench dataset for Navier-Stokes equations.

This script trains a 2D Fourier Neural Operator (FNO2) to predict
velocity field evolution in time for fluid dynamics problems.

Task: Time series prediction
  - Input:  velocity field at time t (u, v) + grid coordinates
  - Output: velocity field at time t+1 (u, v)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from fno2_model import FNO2d
import os
import sys
from typing import Optional, Dict, Tuple, Any, List
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deeponet-cfd'))

try:
    from data.cfdbench_loader import CFDBenchDataset, create_cfdbench_dataloaders
    CFD_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import cfdbench_loader: {e}")
    print("Please ensure deeponet-cfd/data/cfdbench_loader.py exists")
    CFD_LOADER_AVAILABLE = False


class GridCoordinates:
    """
    Add grid coordinates to the input.
    
    The FNO2 model expects input with coordinates for spatial attention.
    This class generates x and y coordinate grids and appends them to the input.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        self.grid_size = grid_size
        self.coords = self._generate_coords()
    
    def _generate_coords(self) -> torch.Tensor:
        """Generate normalized coordinate grid [0, 1]."""
        h, w = self.grid_size
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1)
        return coords
    
    def add_coords(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """
        Add coordinate channels to velocity field.
        
        Args:
            velocity_field: Shape [batch, H, W, 2] or [H, W, 2]
        
        Returns:
            Shape [batch, H, W, 4] or [H, W, 4] (u, v, x_coord, y_coord)
        """
        if velocity_field.dim() == 3:
            velocity_field = velocity_field.unsqueeze(0)
        
        batch_size = velocity_field.shape[0]
        coords = self.coords.to(velocity_field.device)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return torch.cat([velocity_field, coords], dim=-1)


def train_fno2_cfdbench(
    data_root: str = r"C:\traework\data",
    problems: List[str] = None,
    categories: List[str] = None,
    modes: int = 12,
    width: int = 32,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    grid_size: Tuple[int, int] = (64, 64),
    train_ratio: float = 0.8,
    normalize: bool = True,
    max_cases_per_category: int = None,
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.5,
    model_save_path: Optional[str] = 'fno2_cfdbench_model.pth',
    loss_plot_path: Optional[str] = 'training_loss_cfdbench.png',
    sample_plot_path: Optional[str] = 'prediction_samples.png',
    device: Optional[torch.device] = None,
    verbose: bool = True,
    in_channels: int = 4,
    out_channels: int = 2,
    input_steps: int = 1,
    output_steps: int = 1,
) -> Dict[str, Any]:
    """
    Train a 2D Fourier Neural Operator (FNO2) model on CFDBench dataset
    for Navier-Stokes equations.
    
    Parameters
    ----------
    data_root : str
        Root directory of CFDBench dataset
    problems : List[str]
        List of problems to use: ['cavity', 'tube', 'dam', 'cylinder']
    categories : List[str]
        List of categories to use: ['bc', 'geo', 'prop']
    modes : int
        Number of Fourier modes to keep in each spatial dimension
    width : int
        Number of channels (feature dimension) in the FNO layers
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training and validation
    learning_rate : float
        Initial learning rate for Adam optimizer
    weight_decay : float
        Weight decay (L2 regularization) coefficient
    grid_size : Tuple[int, int]
        Spatial resolution of the input/output grids
    train_ratio : float
        Ratio of training data (0.0 to 1.0)
    normalize : bool
        Whether to normalize the data
    max_cases_per_category : int
        Maximum number of cases per category (None = all)
    scheduler_step_size : int
        Step size for learning rate scheduler
    scheduler_gamma : float
        Multiplicative factor for learning rate decay
    model_save_path : str
        Path to save the trained model checkpoint
    loss_plot_path : str
        Path to save the training loss curve plot
    sample_plot_path : str
        Path to save prediction sample plots
    device : torch.device
        Device to use for training (cuda/cpu). If None, auto-detects.
    verbose : bool
        Whether to print progress information during training
    in_channels : int
        Number of input channels (u, v, x_coord, y_coord = 4)
    out_channels : int
        Number of output channels (u, v = 2)
    input_steps : int
        Number of input time steps (currently only 1 supported)
    output_steps : int
        Number of output time steps (currently only 1 supported)
    
    Returns
    -------
    Dict[str, Any]
        Training results including model, losses, metrics, etc.
    """
    
    if problems is None:
        problems = ['cavity']
    if categories is None:
        categories = ['bc', 'geo', 'prop']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    grid_coords = None  # 模型会自动添加坐标，不需要手动添加
    
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
        'grid_size': grid_size,
        'train_ratio': train_ratio,
        'normalize': normalize,
        'max_cases_per_category': max_cases_per_category,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'input_steps': input_steps,
        'output_steps': output_steps,
    }
    
    if verbose:
        print('=' * 70)
        print('FNO2 TRAINING FOR NAVIER-STOKES (CFDBENCH)')
        print('=' * 70)
        print(f'Device: {device}')
        print(f'Modes: {modes}, Width: {width}')
        print(f'Input channels: {in_channels} (u, v, x, y)')
        print(f'Output channels: {out_channels} (u, v)')
        print(f'Epochs: {epochs}, Batch Size: {batch_size}')
        print(f'Learning Rate: {learning_rate}, Weight Decay: {weight_decay}')
        print(f'Grid Size: {grid_size[0]}x{grid_size[1]}')
        print(f'Problems: {problems}')
        print(f'Categories: {categories}')
        if max_cases_per_category:
            print(f'Max cases per category: {max_cases_per_category}')
        print('=' * 70)
    
    if verbose:
        print('\nLoading CFDBench dataset...')
    
    if not CFD_LOADER_AVAILABLE:
        raise RuntimeError(
            "CFDBench loader not available. "
            "Please ensure deeponet-cfd/data/cfdbench_loader.py exists."
        )
    
    dataset = CFDBenchDataset(
        data_root=data_root,
        problems=problems,
        categories=categories,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=normalize,
        grid_size=grid_size,
        max_cases_per_category=max_cases_per_category
    )
    dataset.load()
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"No data loaded from {data_root}. "
            f"Please check the data path and structure."
        )
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    if verbose:
        print(f'   OK: {n_total} samples loaded')
        print(f'   Split: {n_train} train, {n_test} test')
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
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
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'   OK: Model created: {total_params:,} parameters ({trainable_params:,} trainable)')
    
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
    train_metrics_history = []
    test_metrics_history = []
    
    best_test_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            velocity_input = batch['input'].to(device)
            velocity_target = batch['output'].to(device)
            
            optimizer.zero_grad()
            output = model(velocity_input)  # 模型自动添加坐标
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
                velocity_input = batch['input'].to(device)
                velocity_target = batch['output'].to(device)
                
                output = model(velocity_input)  # 模型自动添加坐标
                
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
            'grid_size': grid_size,
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
    
    if loss_plot_path is not None and len(train_losses) > 0:
        if verbose:
            print(f'\nGenerating loss curve plot...')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MSE Loss', fontsize=12)
        ax1.set_title('Training & Test Loss (Log Scale)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2 = axes[1]
        r2_values = [m['r2'] for m in test_metrics_history]
        ax2.plot(r2_values, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('R2 Score', fontsize=12)
        ax2.set_title('Test R2 Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f'   OK: Loss curve saved to: {os.path.abspath(loss_plot_path)}')
    
    if sample_plot_path is not None and len(test_dataset) > 0:
        if verbose:
            print(f'\nGenerating prediction sample plot...')
        
        model.eval()
        with torch.no_grad():
            sample_idx = 0
            sample_data = test_dataset[sample_idx]
            velocity_input = sample_data['input'].unsqueeze(0).to(device)
            velocity_target = sample_data['output'].unsqueeze(0).to(device)
            
            model_input = grid_coords.add_coords(velocity_input)
            prediction = model(model_input)
            
            input_np = velocity_input[0].cpu().numpy()
            target_np = velocity_target[0].cpu().numpy()
            pred_np = prediction[0].cpu().numpy()
            
            norm_params = dataset.get_normalization_params()
            if normalize:
                def denormalize_u(v):
                    return v * norm_params['std_u'] + norm_params['mean_u']
                def denormalize_v(v):
                    return v * norm_params['std_v'] + norm_params['mean_v']
                
                input_u = denormalize_u(input_np[..., 0])
                input_v = denormalize_v(input_np[..., 1])
                target_u = denormalize_u(target_np[..., 0])
                target_v = denormalize_v(target_np[..., 1])
                pred_u = denormalize_u(pred_np[..., 0])
                pred_v = denormalize_v(pred_np[..., 1])
            else:
                input_u = input_np[..., 0]
                input_v = input_np[..., 1]
                target_u = target_np[..., 0]
                target_v = target_np[..., 1]
                pred_u = pred_np[..., 0]
                pred_v = pred_np[..., 1]
            
            error_u = np.abs(target_u - pred_u)
            error_v = np.abs(target_v - pred_v)
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            
            vmin_u = min(input_u.min(), target_u.min(), pred_u.min())
            vmax_u = max(input_u.max(), target_u.max(), pred_u.max())
            vmin_v = min(input_v.min(), target_v.min(), pred_v.min())
            vmax_v = max(input_v.max(), target_v.max(), pred_v.max())
            
            im1 = axes[0, 0].imshow(input_u, cmap='jet', vmin=vmin_u, vmax=vmax_u)
            axes[0, 0].set_title('Input u (t)', fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(target_u, cmap='jet', vmin=vmin_u, vmax=vmax_u)
            axes[0, 1].set_title('Target u (t+1)', fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(pred_u, cmap='jet', vmin=vmin_u, vmax=vmax_u)
            axes[0, 2].set_title('Predicted u (t+1)', fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=axes[0, 2])
            
            im4 = axes[0, 3].imshow(error_u, cmap='jet')
            axes[0, 3].set_title(f'Error u\nMSE={np.mean(error_u**2):.6f}', fontsize=12, fontweight='bold')
            plt.colorbar(im4, ax=axes[0, 3])
            
            im5 = axes[1, 0].imshow(input_v, cmap='jet', vmin=vmin_v, vmax=vmax_v)
            axes[1, 0].set_title('Input v (t)', fontsize=12, fontweight='bold')
            plt.colorbar(im5, ax=axes[1, 0])
            
            im6 = axes[1, 1].imshow(target_v, cmap='jet', vmin=vmin_v, vmax=vmax_v)
            axes[1, 1].set_title('Target v (t+1)', fontsize=12, fontweight='bold')
            plt.colorbar(im6, ax=axes[1, 1])
            
            im7 = axes[1, 2].imshow(pred_v, cmap='jet', vmin=vmin_v, vmax=vmax_v)
            axes[1, 2].set_title('Predicted v (t+1)', fontsize=12, fontweight='bold')
            plt.colorbar(im7, ax=axes[1, 2])
            
            im8 = axes[1, 3].imshow(error_v, cmap='jet')
            axes[1, 3].set_title(f'Error v\nMSE={np.mean(error_v**2):.6f}', fontsize=12, fontweight='bold')
            plt.colorbar(im8, ax=axes[1, 3])
            
            axes[2, 0].axis('off')
            axes[2, 0].text(0.5, 0.5, f'Case: {sample_data["case_name"]}\n'
                                       f'Input: t={sample_data["start_t"]}\n'
                                       f'Target: t={sample_data["start_t"]+1}\n'
                                       f'R2 u: {1 - np.sum((target_u-pred_u)**2)/np.sum((target_u-np.mean(target_u))**2):.4f}\n'
                                       f'R2 v: {1 - np.sum((target_v-pred_v)**2)/np.sum((target_v-np.mean(target_v))**2):.4f}',
                           transform=axes[2, 0].transAxes,
                           fontsize=11, verticalalignment='center',
                           horizontalalignment='center')
            
            axes[2, 1].axis('off')
            axes[2, 2].axis('off')
            axes[2, 3].axis('off')
            
            plt.suptitle('FNO2 Prediction Sample on CFDBench', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(sample_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f'   OK: Sample plot saved to: {os.path.abspath(sample_plot_path)}')
    
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
        'device': device,
        'normalization_params': dataset.get_normalization_params(),
        'grid_coords': grid_coords,
    }
    
    if verbose:
        print('\n' + '=' * 70)
        print('TRAINING SUMMARY (NAVIER-STOKES)')
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


def quick_train_cfdbench(
    epochs: int = 20,
    max_cases_per_category: int = 5,
    batch_size: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick training with reduced settings for testing purposes.
    """
    return train_fno2_cfdbench(
        epochs=epochs,
        max_cases_per_category=max_cases_per_category,
        batch_size=batch_size,
        **kwargs
    )


if __name__ == '__main__':
    print("=" * 70)
    print("FNO2 TRAINING FOR NAVIER-STOKES EQUATIONS (CFDBENCH)")
    print("=" * 70)
    
    result = train_fno2_cfdbench(
        data_root=r"C:\traework\data",
        problems=['cavity'],
        categories=['bc', 'geo', 'prop'],
        modes=12,
        width=32,
        epochs=100,
        batch_size=8,
        learning_rate=0.001,
        grid_size=(64, 64),
        train_ratio=0.8,
        normalize=True,
        max_cases_per_category=None,
        model_save_path='fno2_cfdbench_model.pth',
        loss_plot_path='training_loss_cfdbench.png',
        sample_plot_path='prediction_samples.png',
        verbose=True
    )
