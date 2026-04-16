import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fno2_model import FNO2d
from train_fno2 import PoissonDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime


def load_model(model_path='fno2_model.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'\n{"="*60}\n'
            f'ERROR: Trained model file not found!\n'
            f'{"="*60}\n'
            f'Expected file: {os.path.abspath(model_path)}\n\n'
            f'Possible solutions:\n'
            f'  1. Run: python train_fno2.py (to train a new model)\n'
            f'  2. Or specify a different model path\n\n'
            f'Current directory: {os.getcwd()}\n'
            f'{"="*60}'
        )

    print(f'Loading model from: {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    
    modes = checkpoint.get('modes', 12)
    width = checkpoint.get('width', 32)
    epochs = checkpoint.get('epoch', 0)
    resolution = checkpoint.get('resolution', 64)

    model = FNO2d(modes1=modes, modes2=modes, width=width, in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'✓ Model loaded successfully!')
    print(f'  - Modes: {modes}')
    print(f'  - Width: {width}')
    print(f'  - Trained Epochs: {epochs}')
    print(f'  - Resolution: {resolution}x{resolution}')
    
    if 'train_loss' in checkpoint and 'test_loss' in checkpoint:
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        print(f'  - Final Train Loss: {train_loss[-1]:.6f}')
        print(f'  - Final Test Loss: {test_loss[-1]:.6f}')

    return model, checkpoint, resolution


def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    max_ae = np.max(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    rel_error = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MaxAE': max_ae,
        'R2': r2,
        'RelError (%)': rel_error
    }


def generate_test_case(case_type='standard', resolution=64):
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    f = np.zeros((resolution, resolution))
    
    if case_type == 'standard':
        sources = [
            (0.3, 0.3, 0.08, 3.0),
            (0.7, 0.5, 0.1, 2.5),
            (0.4, 0.7, 0.07, 2.0)
        ]
    elif case_type == 'single_source':
        sources = [(0.5, 0.5, 0.1, 5.0)]
    elif case_type == 'multiple_sources':
        sources = [
            (0.2, 0.2, 0.06, 2.0),
            (0.8, 0.2, 0.06, 2.0),
            (0.2, 0.8, 0.06, 2.0),
            (0.8, 0.8, 0.06, 2.0),
            (0.5, 0.5, 0.1, 3.0)
        ]
    elif case_type == 'boundary_near':
        sources = [
            (0.1, 0.1, 0.05, 4.0),
            (0.1, 0.9, 0.05, 4.0),
            (0.9, 0.1, 0.05, 4.0),
            (0.9, 0.9, 0.05, 4.0)
        ]
    elif case_type == 'smooth':
        sources = [
            (0.5, 0.3, 0.2, 1.5),
            (0.3, 0.7, 0.15, 1.0),
            (0.7, 0.7, 0.15, 1.0)
        ]
    else:
        sources = [(0.5, 0.5, 0.1, 3.0)]
    
    for x0, y0, sigma, amplitude in sources:
        f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    def solve_poisson_fft(f_data):
        res = f_data.shape[0]
        kx = np.fft.fftfreq(res) * res
        ky = np.fft.fftfreq(res) * res
        KX, KY = np.meshgrid(kx, ky)
        
        f_hat = np.fft.fft2(f_data)
        u_hat = np.zeros_like(f_hat, dtype=np.complex128)
        
        mask = (KX**2 + KY**2) > 0
        u_hat[mask] = -f_hat[mask] / (4 * np.pi**2 * (KX[mask]**2 + KY[mask]**2))
        
        u = np.fft.ifft2(u_hat).real
        u = u - np.mean(u)
        u = (u - np.min(u)) / (np.max(u) - np.min(u) + 1e-8)
        return u
    
    u_true = solve_poisson_fft(f)
    return X, Y, f, u_true, case_type


def predict(model, f, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    f_tensor = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        output = model(f_tensor)
    
    u_pred = output.squeeze().cpu().numpy()
    return u_pred


def visualize_training_history(checkpoint, save_path='training_history.png'):
    if 'train_loss' not in checkpoint or 'test_loss' not in checkpoint:
        print('No training history found in checkpoint.')
        return None
    
    train_losses = checkpoint['train_loss']
    test_losses = checkpoint['test_loss']
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training & Test Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    ax2.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss (Linear)', fontsize=12)
    ax2.set_title('Loss Curve (Linear Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ Training history saved to: {save_path}')
    
    return {
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'min_train_loss': min(train_losses),
        'min_test_loss': min(test_losses),
        'epochs_trained': len(train_losses)
    }


def visualize_test_case(X, Y, f, u_true, u_pred, case_name, save_path=None):
    metrics = calculate_metrics(u_true, u_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    im1 = axes[0, 0].pcolormesh(X, Y, f, cmap='viridis', shading='auto')
    axes[0, 0].set_title(f'Source Term (f) - {case_name}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].pcolormesh(X, Y, u_true, cmap='viridis', shading='auto')
    axes[0, 1].set_title('True Solution (u_true)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].pcolormesh(X, Y, u_pred, cmap='viridis', shading='auto')
    axes[0, 2].set_title('FNO2 Prediction (u_pred)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    error = np.abs(u_true - u_pred)
    im4 = axes[1, 0].pcolormesh(X, Y, error, cmap='hot', shading='auto')
    axes[1, 0].set_title(f'Absolute Error\n(Max: {metrics["MaxAE"]:.4f})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    rel_error = np.abs(u_true - u_pred) / (np.abs(u_true) + 1e-8) * 100
    im5 = axes[1, 1].pcolormesh(X, Y, rel_error, cmap='hot', shading='auto', vmin=0, vmax=20)
    axes[1, 1].set_title(f'Relative Error (%)\n(Mean: {metrics["RelError (%)"]:.2f}%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])
    
    axes[1, 2].axis('off')
    metrics_text = 'Performance Metrics:\n\n'
    for key, value in metrics.items():
        if key == 'R2':
            metrics_text += f'{key}: {value:.4f}\n'
        elif key == 'RelError (%)':
            metrics_text += f'{key}: {value:.2f}%\n'
        else:
            metrics_text += f'{key}: {value:.6f}\n'
    
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, 
                     verticalalignment='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].set_title('Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Test Case: {case_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✓ Visualization saved to: {save_path}')
    
    plt.close()
    return metrics


def evaluate_on_dataset(model, n_samples=200, resolution=64, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'\n{"="*60}')
    print(f'Dataset Evaluation ({n_samples} samples)')
    print(f'{"="*60}')
    
    test_dataset = PoissonDataset(n_samples=n_samples, resolution=resolution)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    avg_mse = total_loss / len(test_loader.dataset)
    metrics = calculate_metrics(all_targets.flatten(), all_preds.flatten())
    
    print(f'\nDataset Evaluation Results:')
    print(f'  - Total Samples: {n_samples}')
    print(f'  - Resolution: {resolution}x{resolution}')
    print(f'  - Average MSE: {metrics["MSE"]:.6f}')
    print(f'  - RMSE: {metrics["RMSE"]:.6f}')
    print(f'  - MAE: {metrics["MAE"]:.6f}')
    print(f'  - Max AE: {metrics["MaxAE"]:.6f}')
    print(f'  - R² Score: {metrics["R2"]:.4f}')
    print(f'  - Mean Rel Error: {metrics["RelError (%)"]:.2f}%')
    
    return metrics, all_preds, all_targets


def generate_report(model_info, training_stats, test_cases_results, dataset_metrics, save_path='test_report.txt'):
    report = []
    report.append('='*70)
    report.append('FNO2 MODEL TESTING REPORT')
    report.append('='*70)
    report.append(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.append('')
    
    report.append('-'*70)
    report.append('1. MODEL INFORMATION')
    report.append('-'*70)
    for key, value in model_info.items():
        report.append(f'  {key}: {value}')
    report.append('')
    
    if training_stats:
        report.append('-'*70)
        report.append('2. TRAINING STATISTICS')
        report.append('-'*70)
        for key, value in training_stats.items():
            if 'loss' in key.lower():
                report.append(f'  {key}: {value:.6f}')
            else:
                report.append(f'  {key}: {value}')
        report.append('')
    
    report.append('-'*70)
    report.append('3. TEST CASES RESULTS')
    report.append('-'*70)
    
    metric_names = ['MSE', 'RMSE', 'MAE', 'MaxAE', 'R2', 'RelError (%)']
    
    header = f'{"Test Case":<25}'
    for name in metric_names:
        header += f'{name:>15}'
    report.append(header)
    report.append('-'*120)
    
    for case_name, metrics in test_cases_results.items():
        line = f'{case_name:<25}'
        for name in metric_names:
            val = metrics.get(name, 0)
            if name == 'R2':
                line += f'{val:>15.4f}'
            elif name == 'RelError (%)':
                line += f'{val:>13.2f}%'
            else:
                line += f'{val:>15.6f}'
        report.append(line)
    report.append('')
    
    if dataset_metrics:
        report.append('-'*70)
        report.append('4. DATASET EVALUATION')
        report.append('-'*70)
        for key, value in dataset_metrics.items():
            if key == 'R2':
                report.append(f'  {key}: {value:.4f}')
            elif key == 'RelError (%)':
                report.append(f'  {key}: {value:.2f}%')
            else:
                report.append(f'  {key}: {value:.6f}')
        report.append('')
    
    report.append('-'*70)
    report.append('5. SUMMARY & CONCLUSIONS')
    report.append('-'*70)
    
    avg_r2 = np.mean([m['R2'] for m in test_cases_results.values()])
    avg_mse = np.mean([m['MSE'] for m in test_cases_results.values()])
    
    report.append(f'  Average MSE across test cases: {avg_mse:.6f}')
    report.append(f'  Average R² across test cases: {avg_r2:.4f}')
    
    if avg_r2 > 0.95:
        report.append('  ✓ Excellent model performance (R² > 0.95)')
    elif avg_r2 > 0.9:
        report.append('  ✓ Good model performance (R² > 0.90)')
    elif avg_r2 > 0.8:
        report.append('  ⚠ Moderate performance, consider more training')
    else:
        report.append('  ✗ Poor performance, need more training or tuning')
    
    report.append('')
    report.append('='*70)
    report.append('END OF REPORT')
    report.append('='*70)
    
    report_text = '\n'.join(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f'\n✓ Test report saved to: {save_path}')
    return report_text


def test_fno2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'PyTorch version: {torch.__version__}')
    
    print('\n' + '='*70)
    print('FNO2 MODEL VALIDATION TEST')
    print('='*70)
    
    model_path = 'fno2_model.pth'
    
    try:
        model, checkpoint, resolution = load_model(model_path, device)
    except FileNotFoundError as e:
        print(e)
        print('\nTo train a new model, run:')
        print('  python train_fno2.py')
        return
    except Exception as e:
        print(f'\nUnexpected error loading model: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return
    
    model_info = {
        'Model Path': os.path.abspath(model_path),
        'Modes': checkpoint.get('modes', 12),
        'Width': checkpoint.get('width', 32),
        'Epochs Trained': checkpoint.get('epoch', 0),
        'Resolution': f"{checkpoint.get('resolution', 64)}x{checkpoint.get('resolution', 64)}",
        'Device': str(device)
    }
    
    print('\n' + '='*70)
    print('1. ANALYZING TRAINING HISTORY')
    print('='*70)
    
    training_stats = visualize_training_history(checkpoint, 'training_history.png')
    
    print('\n' + '='*70)
    print('2. RUNNING MULTIPLE TEST CASES')
    print('='*70)
    
    test_cases = [
        ('standard', 'Standard (3 sources)'),
        ('single_source', 'Single Source'),
        ('multiple_sources', 'Multiple Sources (5)'),
        ('boundary_near', 'Boundary Near'),
        ('smooth', 'Smooth Variation')
    ]
    
    test_cases_results = {}
    
    for case_type, case_name in test_cases:
        print(f'\n--- Testing: {case_name} ---')
        
        X, Y, f, u_true, _ = generate_test_case(case_type=case_type, resolution=resolution)
        print(f'  Generated test case with {resolution}x{resolution} resolution')
        
        u_pred = predict(model, f, device)
        print(f'  Prediction completed')
        
        save_path = f'test_{case_type}.png'
        metrics = visualize_test_case(X, Y, f, u_true, u_pred, case_name, save_path)
        
        test_cases_results[case_name] = metrics
        
        print(f'  Results: MSE={metrics["MSE"]:.6f}, R²={metrics["R2"]:.4f}')
    
    print('\n' + '='*70)
    print('3. DATASET EVALUATION')
    print('='*70)
    
    dataset_metrics, _, _ = evaluate_on_dataset(model, n_samples=200, resolution=resolution, device=device)
    
    print('\n' + '='*70)
    print('4. GENERATING TEST REPORT')
    print('='*70)
    
    report = generate_report(
        model_info=model_info,
        training_stats=training_stats,
        test_cases_results=test_cases_results,
        dataset_metrics=dataset_metrics,
        save_path='test_report.txt'
    )
    
    print('\n' + '='*70)
    print('TESTING COMPLETED!')
    print('='*70)
    
    print('\nGenerated files:')
    print('  - training_history.png  - Training loss curves')
    print('  - test_*.png            - Individual test case visualizations')
    print('  - test_report.txt       - Detailed test report')
    
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    
    avg_mse = np.mean([m['MSE'] for m in test_cases_results.values()])
    avg_r2 = np.mean([m['R2'] for m in test_cases_results.values()])
    
    print(f'Average MSE (test cases): {avg_mse:.6f}')
    print(f'Average R² (test cases):  {avg_r2:.4f}')
    print(f'Dataset R² Score:          {dataset_metrics["R2"]:.4f}')
    
    if avg_r2 > 0.95:
        print('\n✓ Excellent! Model is well-trained and ready for use.')
    elif avg_r2 > 0.9:
        print('\n✓ Good! Model performs well, but could improve with more training.')
    elif avg_r2 > 0.8:
        print('\n⚠ Moderate performance. Consider running more training epochs.')
    else:
        print('\n✗ Poor performance. Need more training or hyperparameter tuning.')


if __name__ == '__main__':
    test_fno2()
