import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cfdbench_loader import CFDBenchDataset, create_cfdbench_dataloaders
from models.fno_cfdbench import AutoRegressiveFNO
from train_fno_cfdbench import CFDBenchTrainer


class EvaluationMetrics:
    
    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))
    
    @staticmethod
    def rmse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.sqrt(EvaluationMetrics.mse(pred, target)))
    
    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - target)))
    
    @staticmethod
    def max_ae(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.max(np.abs(pred - target)))
    
    @staticmethod
    def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
        return float(np.mean(np.abs(pred - target) / (np.abs(target) + eps)))
    
    @staticmethod
    def compute_all(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        return {
            'mse': EvaluationMetrics.mse(pred, target),
            'rmse': EvaluationMetrics.rmse(pred, target),
            'mae': EvaluationMetrics.mae(pred, target),
            'max_ae': EvaluationMetrics.max_ae(pred, target),
            'r2_score': EvaluationMetrics.r2_score(pred, target),
            'relative_error': EvaluationMetrics.relative_error(pred, target)
        }


class CFDBenchEvaluator:
    
    def __init__(
        self,
        model: AutoRegressiveFNO,
        test_loader: torch.utils.data.DataLoader,
        normalization_params: Dict[str, float],
        device: Optional[str] = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.normalization_params = normalization_params
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.criterion = nn.MSELoss()
    
    def denormalize(self, data: np.ndarray, field: str = 'u') -> np.ndarray:
        mean_key = f'mean_{field}'
        std_key = f'std_{field}'
        
        if mean_key in self.normalization_params and std_key in self.normalization_params:
            mean = self.normalization_params[mean_key]
            std = self.normalization_params[std_key]
            return data * std + mean
        
        return data
    
    def evaluate(self) -> Dict[str, any]:
        print("=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)
        
        all_preds = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                all_losses.append(loss.item())
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.test_loader)}: loss={loss.item():.6f}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        results = {
            'test_loss': np.mean(all_losses),
            'n_samples': len(all_preds)
        }
        
        u_pred = all_preds[..., 0]
        u_target = all_targets[..., 0]
        v_pred = all_preds[..., 1]
        v_target = all_targets[..., 1]
        
        results['u_metrics'] = EvaluationMetrics.compute_all(u_pred, u_target)
        results['v_metrics'] = EvaluationMetrics.compute_all(v_pred, v_target)
        
        combined_pred = np.stack([u_pred, v_pred], axis=-1)
        combined_target = np.stack([u_target, v_target], axis=-1)
        results['combined_metrics'] = EvaluationMetrics.compute_all(combined_pred, combined_target)
        
        if self.normalization_params:
            u_pred_denorm = self.denormalize(u_pred, 'u')
            u_target_denorm = self.denormalize(u_target, 'u')
            v_pred_denorm = self.denormalize(v_pred, 'v')
            v_target_denorm = self.denormalize(v_target, 'v')
            
            results['u_metrics_denorm'] = EvaluationMetrics.compute_all(u_pred_denorm, u_target_denorm)
            results['v_metrics_denorm'] = EvaluationMetrics.compute_all(v_pred_denorm, v_target_denorm)
            
            combined_pred_denorm = np.stack([u_pred_denorm, v_pred_denorm], axis=-1)
            combined_target_denorm = np.stack([u_target_denorm, v_target_denorm], axis=-1)
            results['combined_metrics_denorm'] = EvaluationMetrics.compute_all(combined_pred_denorm, combined_target_denorm)
        
        return results
    
    def evaluate_multi_step(
        self,
        n_steps: int = 10
    ) -> Dict[str, any]:
        print(f"\n{'=' * 80}")
        print(f"MULTI-STEP AUTOREGRESSIVE EVALUATION ({n_steps} steps)")
        print("=" * 80)
        
        all_preds_multi = []
        all_targets_multi = []
        step_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['input'].to(self.device)
                
                multi_step_pred = self.model.generate_many(inputs, n_steps=n_steps)
                all_preds_multi.append(multi_step_pred.cpu().numpy())
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}/{len(self.test_loader)}")
        
        all_preds_multi = np.concatenate(all_preds_multi, axis=0)
        
        step_metrics = {}
        for step in range(n_steps):
            step_pred = all_preds_multi[:, step, ...]
            step_metrics[f'step_{step+1}'] = {
                'shape': list(step_pred.shape)
            }
        
        results = {
            'multi_step_n': n_steps,
            'predictions_shape': list(all_preds_multi.shape),
            'step_metrics': step_metrics
        }
        
        return results
    
    def save_results(self, results: Dict, filepath: str):
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        results_serializable[key][k] = list(v) if isinstance(v, np.ndarray) else v
                    else:
                        results_serializable[key][k] = v
            elif isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {os.path.abspath(filepath)}")


def print_evaluation_report(results: Dict):
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\nTest Loss (MSE): {results['test_loss']:.6f}")
    print(f"Number of samples: {results['n_samples']}")
    
    print("\n--- Normalized Metrics ---")
    
    print("\nU-component (x-velocity):")
    for metric_name, value in results['u_metrics'].items():
        print(f"  {metric_name:15s}: {value:.6f}")
    
    print("\nV-component (y-velocity):")
    for metric_name, value in results['v_metrics'].items():
        print(f"  {metric_name:15s}: {value:.6f}")
    
    print("\nCombined (u + v):")
    for metric_name, value in results['combined_metrics'].items():
        print(f"  {metric_name:15s}: {value:.6f}")
    
    if 'u_metrics_denorm' in results:
        print("\n--- Denormalized Metrics (Physical Units) ---")
        
        print("\nU-component (x-velocity):")
        for metric_name, value in results['u_metrics_denorm'].items():
            print(f"  {metric_name:15s}: {value:.6f}")
        
        print("\nV-component (y-velocity):")
        for metric_name, value in results['v_metrics_denorm'].items():
            print(f"  {metric_name:15s}: {value:.6f}")
        
        print("\nCombined (u + v):")
        for metric_name, value in results['combined_metrics_denorm'].items():
            print(f"  {metric_name:15s}: {value:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained FNO model on CFDBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing CFDBench data (default: data)"
    )
    
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["cavity"],
        choices=["cavity", "tube", "dam", "cylinder"],
        help="CFD problems to use (default: cavity)"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["bc"],
        choices=["bc", "geo", "prop"],
        help="Categories to use (default: bc)"
    )
    
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Maximum number of cases per category (default: all)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)"
    )
    
    parser.add_argument(
        "--input_steps",
        type=int,
        default=1,
        help="Number of input time steps (default: 1)"
    )
    
    parser.add_argument(
        "--output_steps",
        type=int,
        default=1,
        help="Number of output time steps (default: 1)"
    )
    
    parser.add_argument(
        "--multi_step",
        type=int,
        default=None,
        help="Number of steps for multi-step autoregressive evaluation (default: None)"
    )
    
    parser.add_argument(
        "--results_save_path",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON format, default: None)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("FNO EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_root}")
    print(f"Problems: {args.problems}")
    print(f"Categories: {args.categories}")
    print("=" * 80)
    
    print("\n[1/4] Loading trained model...")
    model, info = CFDBenchTrainer.load_model(args.model_path, device=args.device)
    
    normalization_params = info.get('normalization_params', {})
    print(f"Normalization params loaded: {list(normalization_params.keys())}")
    
    print("\n[2/4] Loading test data...")
    train_loader, test_loader, _ = create_cfdbench_dataloaders(
        data_root=args.data_root,
        problems=args.problems,
        categories=args.categories,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        normalize=True,
        max_cases_per_category=args.max_cases,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print("\n[3/4] Initializing evaluator...")
    evaluator = CFDBenchEvaluator(
        model=model,
        test_loader=test_loader,
        normalization_params=normalization_params,
        device=args.device
    )
    
    print("\n[4/4] Running evaluation...")
    results = evaluator.evaluate()
    
    print_evaluation_report(results)
    
    if args.multi_step:
        multi_step_results = evaluator.evaluate_multi_step(n_steps=args.multi_step)
        results['multi_step'] = multi_step_results
        
        print(f"\nMulti-step predictions shape: {multi_step_results['predictions_shape']}")
    
    if args.results_save_path:
        evaluator.save_results(results, args.results_save_path)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
