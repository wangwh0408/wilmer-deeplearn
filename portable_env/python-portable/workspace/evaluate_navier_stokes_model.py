"""
使用真实CFDBench数据对Navier-Stokes FNO2模型进行评测

本模型是为Navier-Stokes方程训练的:
  - 输入: u, v 两个速度分量 (2通道)
  - 模型自动添加: x, y 坐标 (2通道)
  - 总输入: 4通道
  - 输出: u, v 两个速度分量 (2通道)

评测方案:
  - 任务: 时间序列预测 (t时刻 -> t+1时刻)
  - 同时评测u分量和v分量
  - 使用所有评测指标: MSE, RMSE, MAE, R2, MAPE
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
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"Warning: fno2_model not found: {e}")
    FNO2D_AVAILABLE = False


class MetricsCalculator:
    """
    评测指标计算器
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


class CFDBenchNavierStokesDataset(Dataset):
    """
    CFDBench Navier-Stokes数据集
    
    同时加载u和v两个速度分量:
    - 输入: t时刻的速度分量 [batch, H, W, 2] (u, v)
    - 输出: t+1时刻的速度分量 [batch, H, W, 2] (u, v)
    
    模型会自动添加坐标通道，所以实际进入模型的是4通道。
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
        """
        Args:
            data_root: 数据根目录
            problems: 问题类型列表 ['cavity', 'tube', 'dam', 'cylinder']
            categories: 类别列表 ['bc', 'geo', 'prop']
            max_cases_per_category: 每类最大样本数
            normalize: 是否归一化
            resolution: 目标分辨率
        """
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
        """加载数据"""
        print(f"\nLoading CFDBench Navier-Stokes data from: {self.data_root}")
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
            print(f"    U: mean={self.u_mean:.6f}, std={self.u_std:.6f}, min={self.u_min:.6f}, max={self.u_max:.6f}")
            print(f"    V: mean={self.v_mean:.6f}, std={self.v_std:.6f}, min={self.v_min:.6f}, max={self.v_max:.6f}")
            
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
    
    def denormalize_u(self, data: np.ndarray) -> np.ndarray:
        """反归一化u分量"""
        if self.normalize:
            return data * self.u_std + self.u_mean
        return data
    
    def denormalize_v(self, data: np.ndarray) -> np.ndarray:
        """反归一化v分量"""
        if self.normalize:
            return data * self.v_std + self.v_mean
        return data


class NavierStokesModelEvaluator:
    """
    Navier-Stokes FNO2模型评测器
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
    
    def load_model(self) -> Tuple[bool, str]:
        """加载训练好的模型"""
        if not FNO2D_AVAILABLE:
            return False, "FNO2d model class not available"
        
        if not os.path.exists(self.model_path):
            return False, f"Model file not found: {self.model_path}"
        
        try:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Loading model from: {self.model_path}")
                print(f"Device: {self.device}")
            
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )
            
            modes = checkpoint.get('modes', 12)
            width = checkpoint.get('width', 32)
            in_channels = checkpoint.get('config', {}).get('in_channels', 4)
            out_channels = checkpoint.get('config', {}).get('out_channels', 2)
            resolution = checkpoint.get('resolution', 64)
            
            if self.verbose:
                print(f"\nModel configuration:")
                print(f"  Modes: {modes}")
                print(f"  Width: {width}")
                print(f"  Input channels: {in_channels} (2 data + 2 coords auto-added)")
                print(f"  Output channels: {out_channels} (u, v)")
            
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
            }
            
            if self.verbose:
                print(f"\nModel loaded successfully!")
                print(f"  Total parameters: {self.model_info['total_parameters']:,}")
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed to load model: {str(e)}"
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: CFDBenchNavierStokesDataset
    ) -> Dict[str, Any]:
        """
        在数据集上评测模型
        
        Args:
            dataset: 数据集
        
        Returns:
            包含所有评测指标的字典
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATING NAVIER-STOKES MODEL")
            print(f"{'='*60}")
            print(f"Number of samples: {len(dataset)}")
        
        all_preds_u = []
        all_preds_v = []
        all_trues_u = []
        all_trues_v = []
        
        all_preds_u_denorm = []
        all_preds_v_denorm = []
        all_trues_u_denorm = []
        all_trues_v_denorm = []
        
        sample_metrics = []
        
        criterion = nn.MSELoss()
        total_loss = 0.0
        
        for idx in range(len(dataset)):
            input_tensor, output_tensor, case_name, time_step = dataset[idx]
            
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            output_tensor = output_tensor.unsqueeze(0).to(self.device)
            
            pred = self.model(input_tensor)
            
            loss = criterion(pred, output_tensor)
            total_loss += loss.item()
            
            pred_np = pred.cpu().numpy()
            true_np = output_tensor.cpu().numpy()
            
            pred_u = pred_np[..., 0]
            pred_v = pred_np[..., 1]
            true_u = true_np[..., 0]
            true_v = true_np[..., 1]
            
            all_preds_u.append(pred_u)
            all_preds_v.append(pred_v)
            all_trues_u.append(true_u)
            all_trues_v.append(true_v)
            
            pred_u_denorm = dataset.denormalize_u(pred_u)
            pred_v_denorm = dataset.denormalize_v(pred_v)
            true_u_denorm = dataset.denormalize_u(true_u)
            true_v_denorm = dataset.denormalize_v(true_v)
            
            all_preds_u_denorm.append(pred_u_denorm)
            all_preds_v_denorm.append(pred_v_denorm)
            all_trues_u_denorm.append(true_u_denorm)
            all_trues_v_denorm.append(true_v_denorm)
            
            u_metrics = MetricsCalculator.compute_all(pred_u_denorm.flatten(), true_u_denorm.flatten())
            v_metrics = MetricsCalculator.compute_all(pred_v_denorm.flatten(), true_v_denorm.flatten())
            
            sample_metrics.append({
                'case_name': case_name,
                'time_step': time_step,
                'u_metrics': u_metrics,
                'v_metrics': v_metrics
            })
            
            if self.verbose and (idx + 1) % max(1, len(dataset) // 10) == 0:
                print(f"  Sample {idx + 1}/{len(dataset)}: Loss = {loss.item():.6f}")
        
        all_preds_u = np.concatenate(all_preds_u, axis=0)
        all_preds_v = np.concatenate(all_preds_v, axis=0)
        all_trues_u = np.concatenate(all_trues_u, axis=0)
        all_trues_v = np.concatenate(all_trues_v, axis=0)
        
        all_preds_u_denorm = np.concatenate(all_preds_u_denorm, axis=0)
        all_preds_v_denorm = np.concatenate(all_preds_v_denorm, axis=0)
        all_trues_u_denorm = np.concatenate(all_trues_u_denorm, axis=0)
        all_trues_v_denorm = np.concatenate(all_trues_v_denorm, axis=0)
        
        avg_loss = total_loss / len(dataset)
        
        u_metrics_normalized = MetricsCalculator.compute_all(all_preds_u.flatten(), all_trues_u.flatten())
        v_metrics_normalized = MetricsCalculator.compute_all(all_preds_v.flatten(), all_trues_v.flatten())
        
        u_metrics_denormalized = MetricsCalculator.compute_all(all_preds_u_denorm.flatten(), all_trues_u_denorm.flatten())
        v_metrics_denormalized = MetricsCalculator.compute_all(all_preds_v_denorm.flatten(), all_trues_v_denorm.flatten())
        
        combined_pred = np.stack([all_preds_u_denorm, all_preds_v_denorm], axis=-1)
        combined_true = np.stack([all_trues_u_denorm, all_trues_v_denorm], axis=-1)
        combined_metrics = MetricsCalculator.compute_all(combined_pred.flatten(), combined_true.flatten())
        
        result = {
            'n_samples': len(dataset),
            'avg_mse_loss': float(avg_loss),
            'u_component': {
                'metrics_normalized': u_metrics_normalized,
                'metrics_denormalized': u_metrics_denormalized
            },
            'v_component': {
                'metrics_normalized': v_metrics_normalized,
                'metrics_denormalized': v_metrics_denormalized
            },
            'combined': {
                'metrics_denormalized': combined_metrics
            },
            'sample_metrics': sample_metrics,
        }
        
        if sample_metrics:
            u_sorted = sorted(sample_metrics, key=lambda x: x['u_metrics']['mse'])
            v_sorted = sorted(sample_metrics, key=lambda x: x['v_metrics']['mse'])
            
            result['best_sample_u'] = {
                'index': sample_metrics.index(u_sorted[0]),
                **u_sorted[0]
            }
            result['worst_sample_u'] = {
                'index': sample_metrics.index(u_sorted[-1]),
                **u_sorted[-1]
            }
            
            result['best_sample_v'] = {
                'index': sample_metrics.index(v_sorted[0]),
                **v_sorted[0]
            }
            result['worst_sample_v'] = {
                'index': sample_metrics.index(v_sorted[-1]),
                **v_sorted[-1]
            }
            
            result['per_sample_stats'] = {
                'u_mse_mean': float(np.mean([m['u_metrics']['mse'] for m in sample_metrics])),
                'u_mse_std': float(np.std([m['u_metrics']['mse'] for m in sample_metrics])),
                'u_r2_mean': float(np.mean([m['u_metrics']['r2'] for m in sample_metrics])),
                'u_r2_std': float(np.std([m['u_metrics']['r2'] for m in sample_metrics])),
                'v_mse_mean': float(np.mean([m['v_metrics']['mse'] for m in sample_metrics])),
                'v_mse_std': float(np.std([m['v_metrics']['mse'] for m in sample_metrics])),
                'v_r2_mean': float(np.mean([m['v_metrics']['r2'] for m in sample_metrics])),
                'v_r2_std': float(np.std([m['v_metrics']['r2'] for m in sample_metrics])),
            }
        
        if self.verbose:
            self._print_metrics(result)
        
        return result
    
    def _print_metrics(self, result: Dict[str, Any]):
        """打印评测指标"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Number of samples: {result['n_samples']}")
        print(f"Average MSE Loss (normalized): {result['avg_mse_loss']:.6f}")
        
        print(f"\n{'-'*60}")
        print(f"  U COMPONENT (denormalized - actual velocity):")
        u = result['u_component']['metrics_denormalized']
        print(f"    MSE:  {u['mse']:.8f}")
        print(f"    RMSE: {u['rmse']:.6f}")
        print(f"    MAE:  {u['mae']:.6f}")
        print(f"    R2:   {u['r2']:.6f}")
        print(f"    MAPE: {u['mape']:.4f}%")
        
        print(f"\n{'-'*60}")
        print(f"  V COMPONENT (denormalized - actual velocity):")
        v = result['v_component']['metrics_denormalized']
        print(f"    MSE:  {v['mse']:.8f}")
        print(f"    RMSE: {v['rmse']:.6f}")
        print(f"    MAE:  {v['mae']:.6f}")
        print(f"    R2:   {v['r2']:.6f}")
        print(f"    MAPE: {v['mape']:.4f}%")
        
        print(f"\n{'-'*60}")
        print(f"  COMBINED (U + V):")
        c = result['combined']['metrics_denormalized']
        print(f"    MSE:  {c['mse']:.8f}")
        print(f"    RMSE: {c['rmse']:.6f}")
        print(f"    MAE:  {c['mae']:.6f}")
        print(f"    R2:   {c['r2']:.6f}")
        print(f"    MAPE: {c['mape']:.4f}%")
        
        if 'per_sample_stats' in result:
            print(f"\n{'-'*60}")
            print(f"  Per-sample statistics:")
            s = result['per_sample_stats']
            print(f"    U: MSE mean={s['u_mse_mean']:.8f}, std={s['u_mse_std']:.8f}")
            print(f"    U: R2  mean={s['u_r2_mean']:.6f}, std={s['u_r2_std']:.6f}")
            print(f"    V: MSE mean={s['v_mse_mean']:.8f}, std={s['v_mse_std']:.8f}")
            print(f"    V: R2  mean={s['v_r2_mean']:.6f}, std={s['v_r2_std']:.6f}")
        
        if 'best_sample_u' in result:
            print(f"\n{'-'*60}")
            print(f"  Best U sample: {result['best_sample_u']['case_name']} (t={result['best_sample_u']['time_step']})")
            print(f"    MSE={result['best_sample_u']['u_metrics']['mse']:.8f}, R2={result['best_sample_u']['u_metrics']['r2']:.6f}")
            print(f"  Worst U sample: {result['worst_sample_u']['case_name']} (t={result['worst_sample_u']['time_step']})")
            print(f"    MSE={result['worst_sample_u']['u_metrics']['mse']:.8f}, R2={result['worst_sample_u']['u_metrics']['r2']:.6f}")
            
            print(f"\n  Best V sample: {result['best_sample_v']['case_name']} (t={result['best_sample_v']['time_step']})")
            print(f"    MSE={result['best_sample_v']['v_metrics']['mse']:.8f}, R2={result['best_sample_v']['v_metrics']['r2']:.6f}")
            print(f"  Worst V sample: {result['worst_sample_v']['case_name']} (t={result['worst_sample_v']['time_step']})")
            print(f"    MSE={result['worst_sample_v']['v_metrics']['mse']:.8f}, R2={result['worst_sample_v']['v_metrics']['r2']:.6f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Navier-Stokes FNO2 Evaluation on CFDBench Data')
    parser.add_argument('--model_path', type=str, default='fno2_cfdbench_quick_model.pth',
                        help='Path to trained model file (default: fno2_cfdbench_quick_model.pth)')
    parser.add_argument('--data_root', type=str, default=r'C:\traework\data',
                        help='Root directory of CFDBench data')
    parser.add_argument('--problems', type=str, default='cavity',
                        help='Comma-separated list of problems (default: cavity)')
    parser.add_argument('--categories', type=str, default='bc,geo,prop',
                        help='Comma-separated list of categories (default: bc,geo,prop)')
    parser.add_argument('--max_cases_per_category', type=int, default=10,
                        help='Maximum number of cases per category (default: 10)')
    parser.add_argument('--no_normalize', action='store_true', default=False,
                        help='Disable data normalization')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--output_report', type=str, default='navier_stokes_evaluation_report.json',
                        help='Path to save evaluation report')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('NAVIER-STOKES FNO2 MODEL EVALUATION ON CFD BENCH DATA')
    print('=' * 60)
    print(f"\nModel type: Navier-Stokes (2-channel input/output)")
    print(f"  - Input:  velocity field (u, v) at time t")
    print(f"  - Output: velocity field (u, v) at time t+1")
    print(f"  - Model auto-adds: x, y coordinates")
    print('=' * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Data Root: {args.data_root}")
    print(f"  Problems: {args.problems}")
    print(f"  Categories: {args.categories}")
    print(f"  Max Cases Per Category: {args.max_cases_per_category}")
    print(f"  Normalize: {not args.no_normalize}")
    print('=' * 60)
    
    problems = [p.strip() for p in args.problems.split(',')]
    categories = [c.strip() for c in args.categories.split(',')]
    
    evaluator = NavierStokesModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        verbose=args.verbose
    )
    
    success, message = evaluator.load_model()
    if not success:
        print(f"\nError: {message}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Loading dataset...")
    print(f"{'='*60}")
    
    dataset = CFDBenchNavierStokesDataset(
        data_root=args.data_root,
        problems=problems,
        categories=categories,
        max_cases_per_category=args.max_cases_per_category,
        normalize=not args.no_normalize
    )
    
    if len(dataset) == 0:
        print("\nError: No data loaded")
        sys.exit(1)
    
    result = evaluator.evaluate_dataset(dataset)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': evaluator.model_info,
        'evaluation_type': 'Navier-Stokes Time Series Prediction',
        'task_description': 'Predict t+1 velocity (u, v) from t velocity (u, v)',
        'configuration': {
            'model_path': args.model_path,
            'data_root': args.data_root,
            'problems': problems,
            'categories': categories,
            'max_cases_per_category': args.max_cases_per_category,
            'normalize': not args.no_normalize,
        },
        'data_stats': {
            'n_samples': len(dataset),
            'u_mean': dataset.u_mean,
            'u_std': dataset.u_std,
            'u_min': dataset.u_min,
            'u_max': dataset.u_max,
            'v_mean': dataset.v_mean,
            'v_std': dataset.v_std,
            'v_min': dataset.v_min,
            'v_max': dataset.v_max,
        },
        'results': result,
        'metrics_explanation': {
            'mse': 'Mean Squared Error - lower is better',
            'rmse': 'Root Mean Squared Error - lower is better',
            'mae': 'Mean Absolute Error - lower is better',
            'r2': 'Coefficient of Determination - 1.0 is perfect',
            'mape': 'Mean Absolute Percentage Error (%) - lower is better',
            'normalized': 'Metrics computed on normalized data (0-1 range)',
            'denormalized': 'Metrics computed on actual velocity values',
        }
    }
    
    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report) if os.path.dirname(args.output_report) else '.', exist_ok=True)
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Report saved to: {args.output_report}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
