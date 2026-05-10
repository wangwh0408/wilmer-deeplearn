"""
使用真实CFDBench数据对FNO2模型进行评测

注意: 
- 本模型是为Poisson方程训练的(标量输入→标量输出)
- CFDBench数据是Navier-Stokes速度场(时间序列,向量场)
- 这是一个分布外(OOD)/零样本测试

评测方案:
- 任务: 时间序列预测 (t时刻 → t+1时刻)
- 对u分量和v分量分别进行评测
- 使用所有评测指标: MSE, RMSE, MAE, R², MAPE
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


class CFDBenchTimeSeriesDataset(Dataset):
    """
    CFDBench时间序列数据集
    
    将多时间步数据转换为:
    - 输入: t时刻的速度分量
    - 输出: t+1时刻的同一速度分量
    
    这样可以利用10个时间步生成9个预测任务
    """
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        component: str = 'u',
        max_cases: int = None,
        normalize: bool = True,
        resolution: int = 64
    ):
        """
        Args:
            data_root: 数据根目录
            problems: 问题类型列表 ['cavity', 'tube', 'dam', 'cylinder']
            categories: 类别列表 ['bc', 'geo', 'prop']
            component: 使用的速度分量 'u' 或 'v'
            max_cases: 最大样本数
            normalize: 是否归一化
            resolution: 目标分辨率
        """
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.component = component
        self.max_cases = max_cases
        self.normalize = normalize
        self.resolution = resolution
        
        self.samples: List[Dict] = []
        self.mean = 0.0
        self.std = 1.0
        self.min_val = 0.0
        self.max_val = 1.0
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        print(f"\nLoading CFDBench data from: {self.data_root}")
        print(f"Problems: {self.problems}")
        print(f"Categories: {self.categories}")
        print(f"Component: {self.component}")
        
        all_data = []
        
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
                
                if self.max_cases:
                    case_dirs = case_dirs[:self.max_cases]
                
                for case_dir in case_dirs:
                    case_name = os.path.basename(case_dir)
                    
                    data_path = os.path.join(case_dir, f'{self.component}.npy')
                    if not os.path.exists(data_path):
                        continue
                    
                    try:
                        data = np.load(data_path).astype(np.float32)
                        
                        if data.ndim == 2:
                            data = data[np.newaxis, ...]
                        
                        if data.ndim == 4:
                            data = data[..., 0]
                        
                        all_data.append(data)
                        
                        time_steps = data.shape[0]
                        for t in range(time_steps - 1):
                            self.samples.append({
                                'input_data': data[t].copy(),
                                'output_data': data[t + 1].copy(),
                                'case_name': f"{problem}/{category}/{case_name}",
                                'time_step': t
                            })
                        
                    except Exception as e:
                        print(f"  Warning: Failed to load {case_dir}: {e}")
        
        print(f"\n  Loaded {len(self.samples)} time-step pairs from {len(all_data)} cases")
        
        if self.normalize and len(all_data) > 0:
            all_data_concat = np.concatenate(all_data, axis=0)
            self.mean = float(np.mean(all_data_concat))
            self.std = float(np.std(all_data_concat) + 1e-8)
            self.min_val = float(np.min(all_data_concat))
            self.max_val = float(np.max(all_data_concat))
            
            print(f"  Normalization stats for {self.component}:")
            print(f"    mean={self.mean:.6f}, std={self.std:.6f}")
            print(f"    min={self.min_val:.6f}, max={self.max_val:.6f}")
            
            for sample in self.samples:
                sample['input_data'] = (sample['input_data'] - self.mean) / self.std
                sample['output_data'] = (sample['output_data'] - self.mean) / self.std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        sample = self.samples[idx]
        
        input_tensor = torch.tensor(sample['input_data'], dtype=torch.float32).unsqueeze(-1)
        output_tensor = torch.tensor(sample['output_data'], dtype=torch.float32).unsqueeze(-1)
        
        return input_tensor, output_tensor, sample['case_name'], sample['time_step']
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """反归一化"""
        if self.normalize:
            return data * self.std + self.mean
        return data


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
            }
            
            if self.verbose:
                print(f"Model loaded successfully!")
                print(f"  Parameters: {self.model_info['total_parameters']:,}")
                print(f"  Modes: {modes}, Width: {width}")
                print(f"  Input channels: {in_channels}, Output channels: {out_channels}")
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: CFDBenchTimeSeriesDataset,
        component_name: str = 'u'
    ) -> Dict[str, Any]:
        """
        在数据集上评测模型
        
        Args:
            dataset: 数据集
            component_name: 速度分量名称 ('u' 或 'v')
        
        Returns:
            包含所有评测指标的字典
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating on {component_name} component")
            print(f"{'='*60}")
            print(f"Number of samples: {len(dataset)}")
        
        all_preds = []
        all_trues = []
        all_preds_denorm = []
        all_trues_denorm = []
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
            
            all_preds.append(pred_np)
            all_trues.append(true_np)
            
            pred_denorm = dataset.denormalize(pred_np)
            true_denorm = dataset.denormalize(true_np)
            all_preds_denorm.append(pred_denorm)
            all_trues_denorm.append(true_denorm)
            
            metrics = MetricsCalculator.compute_all(pred_denorm.flatten(), true_denorm.flatten())
            metrics['case_name'] = case_name
            metrics['time_step'] = time_step
            sample_metrics.append(metrics)
            
            if self.verbose and (idx + 1) % max(1, len(dataset) // 10) == 0:
                print(f"  Sample {idx + 1}/{len(dataset)}: Loss = {loss.item():.6f}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        all_preds_denorm = np.concatenate(all_preds_denorm, axis=0)
        all_trues_denorm = np.concatenate(all_trues_denorm, axis=0)
        
        avg_loss = total_loss / len(dataset)
        
        metrics_normalized = MetricsCalculator.compute_all(all_preds.flatten(), all_trues.flatten())
        metrics_denormalized = MetricsCalculator.compute_all(all_preds_denorm.flatten(), all_trues_denorm.flatten())
        
        result = {
            'component': component_name,
            'n_samples': len(dataset),
            'avg_mse_loss': float(avg_loss),
            'metrics_normalized': metrics_normalized,
            'metrics_denormalized': metrics_denormalized,
            'sample_metrics': sample_metrics,
        }
        
        if sample_metrics:
            sorted_by_mse = sorted(sample_metrics, key=lambda x: x['mse'])
            result['best_sample'] = {
                'index': sample_metrics.index(sorted_by_mse[0]),
                **sorted_by_mse[0]
            }
            result['worst_sample'] = {
                'index': sample_metrics.index(sorted_by_mse[-1]),
                **sorted_by_mse[-1]
            }
            
            result['per_sample_stats'] = {
                'mse_mean': float(np.mean([m['mse'] for m in sample_metrics])),
                'mse_std': float(np.std([m['mse'] for m in sample_metrics])),
                'r2_mean': float(np.mean([m['r2'] for m in sample_metrics])),
                'r2_std': float(np.std([m['r2'] for m in sample_metrics])),
                'mae_mean': float(np.mean([m['mae'] for m in sample_metrics])),
            }
        
        if self.verbose:
            self._print_metrics(result)
        
        return result
    
    def _print_metrics(self, result: Dict[str, Any]):
        """打印评测指标"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {result['component']} component")
        print(f"{'='*60}")
        print(f"Number of samples: {result['n_samples']}")
        print(f"Average MSE Loss (normalized): {result['avg_mse_loss']:.6f}")
        print(f"{'-'*60}")
        print(f"  Metrics (normalized):")
        m = result['metrics_normalized']
        print(f"    MSE:  {m['mse']:.8f}")
        print(f"    RMSE: {m['rmse']:.6f}")
        print(f"    MAE:  {m['mae']:.6f}")
        print(f"    R²:   {m['r2']:.6f}")
        print(f"    MAPE: {m['mape']:.4f}%")
        print(f"{'-'*60}")
        print(f"  Metrics (denormalized - actual velocity values):")
        m = result['metrics_denormalized']
        print(f"    MSE:  {m['mse']:.8f}")
        print(f"    RMSE: {m['rmse']:.6f}")
        print(f"    MAE:  {m['mae']:.6f}")
        print(f"    R²:   {m['r2']:.6f}")
        print(f"    MAPE: {m['mape']:.4f}%")
        
        if 'per_sample_stats' in result:
            print(f"{'-'*60}")
            print(f"  Per-sample stats (denormalized):")
            s = result['per_sample_stats']
            print(f"    MSE:  mean={s['mse_mean']:.8f}, std={s['mse_std']:.8f}")
            print(f"    R²:   mean={s['r2_mean']:.6f}, std={s['r2_std']:.6f}")
        
        if 'best_sample' in result:
            print(f"{'-'*60}")
            print(f"  Best sample: {result['best_sample']['case_name']} (t={result['best_sample']['time_step']})")
            print(f"    MSE={result['best_sample']['mse']:.8f}, R²={result['best_sample']['r2']:.6f}")
            print(f"  Worst sample: {result['worst_sample']['case_name']} (t={result['worst_sample']['time_step']})")
            print(f"    MSE={result['worst_sample']['mse']:.8f}, R²={result['worst_sample']['r2']:.6f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FNO2 Evaluation on CFDBench Data')
    parser.add_argument('--model_path', type=str, default='fno2_model.pth',
                        help='Path to trained model file (default: fno2_model.pth)')
    parser.add_argument('--data_root', type=str, default=r'C:\traework\data',
                        help='Root directory of CFDBench data')
    parser.add_argument('--problems', type=str, default='cavity',
                        help='Comma-separated list of problems (default: cavity)')
    parser.add_argument('--categories', type=str, default='bc,geo,prop',
                        help='Comma-separated list of categories (default: bc,geo,prop)')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Maximum number of cases per category (default: all)')
    parser.add_argument('--no_normalize', action='store_true', default=False,
                        help='Disable data normalization')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--output_report', type=str, default='cfdbench_evaluation_report.json',
                        help='Path to save evaluation report')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('FNO2 MODEL EVALUATION ON CFD BENCH DATA')
    print('=' * 60)
    print(f"\nIMPORTANT NOTICE:")
    print(f"  - This model was trained for Poisson equation (scalar input -> scalar output)")
    print(f"  - CFDBench contains Navier-Stokes velocity field data (time series)")
    print(f"  - This is a distribution-out-of-distribution (OOD) / zero-shot test")
    print(f"  - Task: Predict t+1 velocity from t velocity (time series prediction)")
    print('=' * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Data Root: {args.data_root}")
    print(f"  Problems: {args.problems}")
    print(f"  Categories: {args.categories}")
    print(f"  Max Cases: {args.max_cases}")
    print(f"  Normalize: {not args.no_normalize}")
    print('=' * 60)
    
    problems = [p.strip() for p in args.problems.split(',')]
    categories = [c.strip() for c in args.categories.split(',')]
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        verbose=args.verbose
    )
    
    success, message = evaluator.load_model()
    if not success:
        print(f"\nError: {message}")
        sys.exit(1)
    
    all_results = {}
    
    for component in ['u', 'v']:
        print(f"\n{'#'*60}")
        print(f"# Processing component: {component}")
        print(f"{'#'*60}")
        
        dataset = CFDBenchTimeSeriesDataset(
            data_root=args.data_root,
            problems=problems,
            categories=categories,
            component=component,
            max_cases=args.max_cases,
            normalize=not args.no_normalize
        )
        
        if len(dataset) == 0:
            print(f"Warning: No data loaded for component {component}")
            continue
        
        result = evaluator.evaluate_dataset(dataset, component_name=component)
        all_results[component] = {
            'metrics': result,
            'data_stats': {
                'n_samples': len(dataset),
                'normalized': not args.no_normalize,
                'mean': dataset.mean,
                'std': dataset.std,
                'min': dataset.min_val,
                'max': dataset.max_val,
            }
        }
    
    if len(all_results) == 0:
        print("\nError: No data was loaded for any component")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for comp, result in all_results.items():
        m = result['metrics']['metrics_denormalized']
        print(f"\n  {comp.upper()} Component:")
        print(f"    Samples: {result['metrics']['n_samples']}")
        print(f"    MSE:  {m['mse']:.8f}")
        print(f"    RMSE: {m['rmse']:.6f}")
        print(f"    R²:   {m['r2']:.6f}")
        print(f"    MAPE: {m['mape']:.4f}%")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': evaluator.model_info,
        'evaluation_type': 'CFDBench Time Series Prediction (OOD/Zero-shot)',
        'task_description': 'Predict t+1 velocity from t velocity',
        'configuration': {
            'model_path': args.model_path,
            'data_root': args.data_root,
            'problems': problems,
            'categories': categories,
            'max_cases': args.max_cases,
            'normalize': not args.no_normalize,
        },
        'results': all_results,
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
