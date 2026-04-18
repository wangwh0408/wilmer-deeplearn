import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from fno2_model import FNO2d
import os
from typing import Optional, Dict, Tuple, Any


class PoissonDataset(Dataset):
    def __init__(self, n_samples=1000, resolution=64):
        self.n_samples = n_samples
        self.resolution = resolution
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.n_samples):
            x, y, f, u = self._generate_poisson_problem()
            data.append((f, u))
        return data

    def _generate_poisson_problem(self):
        res = self.resolution
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)

        num_sources = np.random.randint(2, 6)
        f = np.zeros((res, res))
        for _ in range(num_sources):
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(1.0, 5.0)
            f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

        u = self._solve_poisson_fft(f)
        return X, Y, f, u

    def _solve_poisson_fft(self, f):
        res = self.resolution
        kx = np.fft.fftfreq(res) * res
        ky = np.fft.fftfreq(res) * res
        KX, KY = np.meshgrid(kx, ky)

        f_hat = np.fft.fft2(f)
        u_hat = np.zeros_like(f_hat, dtype=np.complex128)

        mask = (KX**2 + KY**2) > 0
        u_hat[mask] = -f_hat[mask] / (4 * np.pi**2 * (KX[mask]**2 + KY[mask]**2))

        u = np.fft.ifft2(u_hat).real
        u = u - np.mean(u)
        u = (u - np.min(u)) / (np.max(u) - np.min(u))
        return u

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        f, u = self.data[idx]
        f_tensor = torch.tensor(f, dtype=torch.float32).unsqueeze(-1)
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(-1)
        return f_tensor, u_tensor


def train_fno2(
    modes: int = 12,
    width: int = 32,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    resolution: int = 64,
    n_train_samples: int = 800,
    n_test_samples: int = 200,
    scheduler_step_size: int = 20,
    scheduler_gamma: float = 0.5,
    model_save_path: Optional[str] = 'fno2_model.pth',
    loss_plot_path: Optional[str] = 'training_loss.png',
    device: Optional[torch.device] = None,
    verbose: bool = True,
    in_channels: int = 3,
    out_channels: int = 1
) -> Dict[str, Any]:
    """
    Train a 2D Fourier Neural Operator (FNO2) model for solving Poisson's equation.
    
    This function is designed to be called externally with customizable parameters.
    
    Parameters
    ----------
    modes : int, optional
        Number of Fourier modes to keep in each spatial dimension (default: 12)
    width : int, optional
        Number of channels (feature dimension) in the FNO layers (default: 32)
    epochs : int, optional
        Number of training epochs (default: 50)
    batch_size : int, optional
        Batch size for training and validation (default: 16)
    learning_rate : float, optional
        Initial learning rate for Adam optimizer (default: 0.001)
    weight_decay : float, optional
        Weight decay (L2 regularization) coefficient (default: 1e-4)
    resolution : int, optional
        Spatial resolution of the input/output grids (default: 64)
    n_train_samples : int, optional
        Number of training samples to generate (default: 800)
    n_test_samples : int, optional
        Number of test samples to generate (default: 200)
    scheduler_step_size : int, optional
        Step size for learning rate scheduler (default: 20)
    scheduler_gamma : float, optional
        Multiplicative factor for learning rate decay (default: 0.5)
    model_save_path : str, optional
        Path to save the trained model checkpoint. If None, model is not saved.
        (default: 'fno2_model.pth')
    loss_plot_path : str, optional
        Path to save the training loss curve plot. If None, plot is not saved.
        (default: 'training_loss.png')
    device : torch.device, optional
        Device to use for training (cuda/cpu). If None, auto-detects.
        (default: None)
    verbose : bool, optional
        Whether to print progress information during training (default: True)
    in_channels : int, optional
        Number of input channels (includes grid coordinates) (default: 3)
    out_channels : int, optional
        Number of output channels (default: 1)
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'model': Trained FNO2d model
        - 'train_losses': List of training losses per epoch
        - 'test_losses': List of test losses per epoch
        - 'final_train_loss': Final training loss
        - 'final_test_loss': Final test loss
        - 'best_train_loss': Best (minimum) training loss
        - 'best_test_loss': Best (minimum) test loss
        - 'config': Dictionary of all training parameters used
        - 'model_path': Path where model was saved (if saved)
        - 'device': Device used for training
    
    Examples
    --------
    >>> # Basic training with default parameters
    >>> result = train_fno2()
    >>> model = result['model']
    >>> print(f"Final test loss: {result['final_test_loss']:.6f}")
    
    >>> # Custom training parameters
    >>> result = train_fno2(
    ...     modes=16,
    ...     width=64,
    ...     epochs=100,
    ...     batch_size=32,
    ...     learning_rate=1e-3,
    ...     n_train_samples=1000,
    ...     n_test_samples=300,
    ...     model_save_path='my_custom_model.pth'
    ... )
    
    >>> # Quick training for testing
    >>> result = train_fno2(
    ...     epochs=5,
    ...     n_train_samples=100,
    ...     n_test_samples=50,
    ...     resolution=32,
    ...     model_save_path=None,
    ...     loss_plot_path=None
    ... )
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'modes': modes,
        'width': width,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'resolution': resolution,
        'n_train_samples': n_train_samples,
        'n_test_samples': n_test_samples,
        'scheduler_step_size': scheduler_step_size,
        'scheduler_gamma': scheduler_gamma,
        'in_channels': in_channels,
        'out_channels': out_channels
    }
    
    if verbose:
        print('=' * 70)
        print('FNO2 TRAINING CONFIGURATION')
        print('=' * 70)
        print(f'Device: {device}')
        print(f'Modes: {modes}, Width: {width}')
        print(f'Epochs: {epochs}, Batch Size: {batch_size}')
        print(f'Learning Rate: {learning_rate}, Weight Decay: {weight_decay}')
        print(f'Resolution: {resolution}x{resolution}')
        print(f'Training Samples: {n_train_samples}, Test Samples: {n_test_samples}')
        print('=' * 70)
    
    if verbose:
        print('\nGenerating dataset...')
    
    train_dataset = PoissonDataset(n_samples=n_train_samples, resolution=resolution)
    test_dataset = PoissonDataset(n_samples=n_test_samples, resolution=resolution)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print(f'   OK: Dataset generated: {len(train_dataset)} training, {len(test_dataset)} test samples')
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
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
        
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch+1:3d}/{epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Test Loss: {test_loss:.6f} | '
                f'LR: {current_lr:.2e}'
            )
    
    if verbose:
        print('-' * 70)
        print('Training completed!')
    
    model_path = None
    if model_save_path is not None:
        if verbose:
            print(f'\nSaving model to: {model_save_path}')
        
        save_dict = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses,
            'test_loss': test_losses,
            'modes': modes,
            'width': width,
            'resolution': resolution,
            'config': config
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
        ax2.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
        ax2.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MSE Loss', fontsize=12)
        ax2.set_title('Training & Test Loss (Linear Scale)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f'   OK: Loss curve saved to: {os.path.abspath(loss_plot_path)}')
    
    final_train_loss = train_losses[-1] if train_losses else None
    final_test_loss = test_losses[-1] if test_losses else None
    best_train_loss = min(train_losses) if train_losses else None
    best_test_loss = min(test_losses) if test_losses else None
    
    result = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'best_train_loss': best_train_loss,
        'best_test_loss': best_test_loss,
        'config': config,
        'model_path': model_path,
        'device': device
    }
    
    if verbose:
        print('\n' + '=' * 70)
        print('TRAINING SUMMARY')
        print('=' * 70)
        print(f'Final Train Loss: {final_train_loss:.6f}')
        print(f'Final Test Loss:  {final_test_loss:.6f}')
        print(f'Best Train Loss:  {best_train_loss:.6f}')
        print(f'Best Test Loss:   {best_test_loss:.6f}')
        if model_path:
            print(f'Model saved at:   {model_path}')
        print('=' * 70)
    
    return result


def train_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train FNO2 model using a configuration dictionary.
    
    This is a convenience function that wraps train_fno2() and accepts
    a dictionary of parameters instead of keyword arguments.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary containing training parameters. Keys should match
        the parameter names of train_fno2().
    
    Returns
    -------
    Dict[str, Any]
        Same return value as train_fno2()
    
    Examples
    --------
    >>> config = {
    ...     'modes': 12,
    ...     'width': 32,
    ...     'epochs': 50,
    ...     'batch_size': 16
    ... }
    >>> result = train_with_config(config)
    """
    return train_fno2(**config)


def quick_train(
    epochs: int = 10,
    n_train_samples: int = 200,
    n_test_samples: int = 100,
    resolution: int = 32,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick training with reduced settings for testing purposes.
    
    This function provides sensible defaults for quick testing while
    still allowing full customization through kwargs.
    
    Parameters
    ----------
    epochs : int, optional
        Number of epochs (default: 10)
    n_train_samples : int, optional
        Number of training samples (default: 200)
    n_test_samples : int, optional
        Number of test samples (default: 100)
    resolution : int, optional
        Spatial resolution (default: 32)
    **kwargs
        Additional arguments passed to train_fno2()
    
    Returns
    -------
    Dict[str, Any]
        Same return value as train_fno2()
    
    Examples
    --------
    >>> # Very quick test
    >>> result = quick_train(epochs=3)
    
    >>> # Quick test with custom model size
    >>> result = quick_train(
    ...     epochs=20,
    ...     modes=16,
    ...     width=64,
    ...     model_save_path='quick_model.pth'
    ... )
    """
    return train_fno2(
        epochs=epochs,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        resolution=resolution,
        **kwargs
    )


if __name__ == '__main__':
    result = train_fno2(
        modes=12,
        width=32,
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        resolution=64,
        n_train_samples=800,
        n_test_samples=200,
        model_save_path='fno2_model.pth',
        loss_plot_path='training_loss.png',
        verbose=True
    )
