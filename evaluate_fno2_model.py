"""
FNO2模型评测脚本
使用真实CFD数据集或生成的Poisson问题对训练好的FNO2模型进行评测

评测指标:
- MSE (Mean Squared Error): 均方误差
- RMSE (Root Mean Squared Error): 均方根误差
- MAE (Mean Absolute Error): 平均绝对误差
- R2 (Coefficient of Determination): 决定系数
- MAPE (Mean Absolute Percentage Error): 平均百分比误差
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"Warning: fno2_model not found: {e}")
    FNO2D_AVAILABLE = False


class MetricsCalculator:
    """
    评测指标计算器
    
    支持的指标:
    - MSE: 均方误差 = mean((pred - true)^2)
    - RMSE: 均方根误差 = sqrt(MSE)
    - MAE: 平均绝对误差 = mean(|pred - true|)
    - R2: 决定系数 = 1 - SSE/SST
    - MAPE: 平均百分比误差 = mean(|(pred - true)/true|) * 100
    """
    
    @staticmethod
    def mse(pred: np.ndarray, true: np.ndarray) -> float:
        """均方误差"""
        return float(np.mean((pred - true) ** 2))
    
    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        """均方根误差"""
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    
    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        """平均绝对误差"""
        return float(np.mean(np.abs(pred - true)))
    
    @staticmethod
    def r2(pred: np.ndarray, true: np.ndarray) -> float:
        """决定系数"""
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
        """平均百分比误差"""
        mask = np.abs(true) > eps
        if np.sum(mask) == 0:
            return 0.0
        return float(np.mean(np.abs((true[mask] - pred[mask]) / (true[mask] + eps))) * 100)
    
    @staticmethod
    def compute_all(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'mse': MetricsCalculator.mse(pred, true),
            'rmse': MetricsCalculator.rmse(pred, true),
            'mae': MetricsCalculator.mae(pred, true),
            'r2': MetricsCalculator.r2(pred, true),
            'mape': MetricsCalculator.mape(pred, true)
        }


class PoissonTestDataset(Dataset):
    """
    Poisson问题测试数据集
    使用FFT求解生成测试数据
    """
    
    def __init__(self, n_samples: int = 200, resolution: int = 64, seed: int = 42):
        self.n_samples = n_samples
        self.resolution = resolution
        self.seed = seed
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        np.random.seed(self.seed)
        data = []
        for _ in range(self.n_samples):
            f, u = self._generate_poisson_problem()
            data.append((f, u))
        return data
    
    def _generate_poisson_problem(self) -> Tuple[np.ndarray, np.ndarray]:
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
        return f, u
    
    def _solve_poisson_fft(self, f: np.ndarray) -> np.ndarray:
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
        u = (u - np.min(u)) / (np.max(u) - np.min(u) + 1e-8)
        return u
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f, u = self.data[idx]
        f_tensor = torch.tensor(f, dtype=torch.float32).unsqueeze(-1)
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(-1)
        return f_tensor, u_tensor


class ModelEvaluator:
    """
    FNO2模型评测器
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model_path = model_path
        self.verbose = verbose
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.model_info = None
        self.metrics_history = []
    
    def load_model(self) -> Tuple[bool, str]:
        """加载训练好的模型"""
        if not FNO2D_AVAILABLE:
            return False, "FNO2d model class not available"
        
        if not os.path.exists(self.model_path):
            return False, f"Model file not found: {self.model_path}"
        
        try:
            if self.verbose:
                print(f"Loading model from: {self.model_path}")
                print(f"Device: {self.device}")
            
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            modes = checkpoint.get('modes', 12)
            width = checkpoint.get('width', 32)
            in_channels = checkpoint.get('config', {}).get('in_channels', 3)
            out_channels = checkpoint.get('config', {}).get('out_channels', 1)
            resolution = checkpoint.get('resolution', 64)
            
            self.model = FNO2d(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_info = {
                'path': self.model_path,
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'resolution': resolution,
                'device': str(self.device),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
            
            if self.verbose:
                print(f"Model loaded successfully!")
                print(f"  Parameters: {self.model_info['total_parameters']:,}")
                print(f"  Resolution: {resolution}x{resolution}")
                print(f"  Modes: {modes}, Width: {width}")
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        在测试集上评测模型
        
        Returns:
            包含所有评测指标的字典
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating on {dataset_name} Dataset")
            print(f"{'='*60}")
        
        all_preds = []
        all_trues = []
        all_losses = []
        
        criterion = nn.MSELoss()
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            output = self.model(data)
            
            loss = criterion(output, target)
            all_losses.append(loss.item())
            
            all_preds.append(output.cpu().numpy())
            all_trues.append(target.cpu().numpy())
            
            if self.verbose and (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}: Loss = {loss.item():.6f}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        avg_loss = np.mean(all_losses)
        
        metrics = MetricsCalculator.compute_all(all_preds.flatten(), all_trues.flatten())
        
        metrics['avg_mse_loss'] = float(avg_loss)
        metrics['n_samples'] = len(all_preds)
        
        sample_metrics = []
        for i in range(len(all_preds)):
            sample_metric = MetricsCalculator.compute_all(all_preds[i].flatten(), all_trues[i].flatten())
            sample_metrics.append(sample_metric)
        
        metrics['per_sample'] = {
            'mse_mean': float(np.mean([m['mse'] for m in sample_metrics])),
            'mse_std': float(np.std([m['mse'] for m in sample_metrics])),
            'r2_mean': float(np.mean([m['r2'] for m in sample_metrics])),
            'r2_std': float(np.std([m['r2'] for m in sample_metrics])),
            'mae_mean': float(np.mean([m['mae'] for m in sample_metrics])),
        }
        
        best_idx = np.argmin([m['mse'] for m in sample_metrics])
        worst_idx = np.argmax([m['mse'] for m in sample_metrics])
        
        metrics['best_sample'] = {
            'index': int(best_idx),
            **sample_metrics[best_idx]
        }
        metrics['worst_sample'] = {
            'index': int(worst_idx),
            **sample_metrics[worst_idx]
        }
        
        self.metrics_history.append({
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        if self.verbose:
            self._print_metrics(metrics, dataset_name)
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any], dataset_name: str):
        """打印评测指标"""
        print(f"\n{dataset_name} Dataset Evaluation Results:")
        print(f"{'-'*60}")
        print(f"  Number of samples: {metrics['n_samples']}")
        print(f"  Average MSE Loss:  {metrics['avg_mse_loss']:.6f}")
        print(f"{'-'*60}")
        print(f"  Metrics (overall):")
        print(f"    MSE  (Mean Squared Error):      {metrics['mse']:.8f}")
        print(f"    RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}")
        print(f"    MAE  (Mean Absolute Error):     {metrics['mae']:.6f}")
        print(f"    R²   (Coefficient of Determination): {metrics['r2']:.6f}")
        print(f"    MAPE (Mean Absolute Percentage Error): {metrics['mape']:.4f}%")
        print(f"{'-'*60}")
        print(f"  Metrics (per sample):")
        print(f"    MSE:  mean={metrics['per_sample']['mse_mean']:.8f}, std={metrics['per_sample']['mse_std']:.8f}")
        print(f"    R²:   mean={metrics['per_sample']['r2_mean']:.6f}, std={metrics['per_sample']['r2_std']:.6f}")
        print(f"    MAE:  mean={metrics['per_sample']['mae_mean']:.6f}")
        print(f"{'-'*60}")
        print(f"  Best sample (idx={metrics['best_sample']['index']}):")
        print(f"    MSE={metrics['best_sample']['mse']:.8f}, R²={metrics['best_sample']['r2']:.6f}")
        print(f"  Worst sample (idx={metrics['worst_sample']['index']}):")
        print(f"    MSE={metrics['worst_sample']['mse']:.8f}, R²={metrics['worst_sample']['r2']:.6f}")
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """生成评测报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model_info,
            'metrics': metrics,
            'metrics_explanation': {
                'mse': 'Mean Squared Error - 均方误差，值越小越好',
                'rmse': 'Root Mean Squared Error - 均方根误差，值越小越好',
                'mae': 'Mean Absolute Error - 平均绝对误差，值越小越好',
                'r2': 'Coefficient of Determination - 决定系数，1表示完美预测',
                'mape': 'Mean Absolute Percentage Error - 平均百分比误差，值越小越好'
            }
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"\nReport saved to: {output_path}")
        
        return json.dumps(report, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FNO2 Model Evaluation')
    parser.add_argument('--model_path', type=str, default='fno2_model.pth',
                        help='Path to trained model file (default: fno2_model.pth)')
    parser.add_argument('--n_test_samples', type=int, default=200,
                        help='Number of test samples to generate (default: 200)')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Spatial resolution (default: 64)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for test data generation (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--output_report', type=str, default='evaluation_report.json',
                        help='Path to save evaluation report (default: evaluation_report.json)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('FNO2 MODEL EVALUATION')
    print('=' * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Test Samples: {args.n_test_samples}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print('=' * 60)
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        verbose=args.verbose
    )
    
    success, message = evaluator.load_model()
    if not success:
        print(f"Error: {message}")
        print("\nPlease ensure:")
        print(f"1. The model file exists at: {args.model_path}")
        print(f"2. fno2_model.py is in the parent directory")
        print(f"3. PyTorch is installed correctly")
        sys.exit(1)
    
    if args.verbose:
        print(f"\nGenerating test dataset with {args.n_test_samples} samples...")
    
    test_dataset = PoissonTestDataset(
        n_samples=args.n_test_samples,
        resolution=args.resolution,
        seed=args.seed
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    if args.verbose:
        print(f"Test dataset ready: {len(test_dataset)} samples")
    
    metrics = evaluator.evaluate(test_loader, dataset_name="Poisson Test")
    
    if args.output_report:
        evaluator.generate_report(metrics, args.output_report)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  R² Score:  {metrics['r2']:.6f}")
    print(f"  MSE:       {metrics['mse']:.8f}")
    print(f"  RMSE:      {metrics['rmse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  MAPE:      {metrics['mape']:.4f}%")
    
    if metrics['r2'] > 0.95:
        print(f"\n✅ Excellent performance! R² > 0.95")
    elif metrics['r2'] > 0.9:
        print(f"\n✅ Good performance! R² > 0.9")
    elif metrics['r2'] > 0.8:
        print(f"\n⚠️ Moderate performance. Consider more training.")
    else:
        print(f"\n❌ Poor performance. Model needs improvement.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
