"""
Enhanced Evaluation Script for Navier-Stokes FNO2 Model

This script provides DETAILED LOGGING and FLEXIBLE METRIC CALCULATION
throughout the entire evaluation process.

Features:
  - Detailed data loading logs
  - Per-sample evaluation progress
  - Detailed metric calculations (configurable via --metrics)
  - Real-time progress display
  - Memory usage tracking
  - Timing information
  - Python standard logging module support

Available Metrics:
  - mse: Mean Squared Error
  - rmse: Root Mean Squared Error
  - mae: Mean Absolute Error
  - r2: Coefficient of Determination (R²)
  - mape: Mean Absolute Percentage Error (%)
  - nrmse: Normalized Root Mean Squared Error (by std)
  - nmae: Normalized Mean Absolute Error (by std)
  - pcc: Pearson Correlation Coefficient
  - max_error: Maximum Absolute Error
  - min_error: Minimum Absolute Error
  - median_error: Median Absolute Error

Usage:
  python evaluate_navier_stokes_enhanced.py --model_path fno2_full_model.pth --data_root C:\traework\data
  python evaluate_navier_stokes_enhanced.py --metrics mse,rmse,r2 (only specific metrics)
  python evaluate_navier_stokes_enhanced.py --metrics all (all metrics)
  python evaluate_navier_stokes_enhanced.py --log_level DEBUG (verbose logging)
  python evaluate_navier_stokes_enhanced.py --quiet (disable console output)
"""

import os
import sys
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import FNO2d model: {e}")
    FNO2D_AVAILABLE = False


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """Setup Python standard logging module for evaluation."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    log_level_int = level_map.get(log_level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    class EvaluationFormatter(logging.Formatter):
        """Custom log formatter with elapsed time support"""
        
        def __init__(self, fmt: str = None, datefmt: str = None):
            super().__init__(fmt=fmt, datefmt=datefmt)
            self.start_time = time.time()
        
        def format(self, record):
            elapsed = time.time() - self.start_time
            record.elapsed = f"{elapsed:.2f}s"
            return super().format(record)
    
    log_format = '%(asctime)s [%(levelname)s] [%(elapsed)s] %(message)s'
    formatter = EvaluationFormatter(log_format)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if log_to_file and log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


class EnhancedLogger:
    """Enhanced logger using Python standard logging module for evaluation."""
    
    _start_time = None
    _phase_start_time = None
    _logger = None
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        self._start_time = time.time()
        self._phase_start_time = None
        self._log_file = log_file
        
        self._logger = setup_logging(
            log_file=log_file,
            log_level=log_level,
            log_to_console=log_to_console,
            log_to_file=log_to_file
        )
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with specified level"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
            'PHASE': logging.INFO,
            'DATA': logging.INFO,
            'DETAIL': logging.DEBUG,
            'PROGRESS': logging.INFO,
            'METRIC': logging.INFO,
        }
        
        log_level = level_map.get(level.upper(), logging.INFO)
        self._logger.log(log_level, message)
    
    def debug(self, message: str):
        self._logger.debug(message)
    
    def info(self, message: str):
        self._logger.info(message)
    
    def warning(self, message: str):
        self._logger.warning(message)
    
    def error(self, message: str):
        self._logger.error(message)
    
    def critical(self, message: str):
        self._logger.critical(message)
    
    def start_phase(self, phase_name: str):
        self._phase_start_time = time.time()
        self.info(f"=== STARTING: {phase_name} ===")
    
    def end_phase(self, phase_name: str, details: str = ""):
        if self._phase_start_time:
            elapsed = time.time() - self._phase_start_time
            self.info(f"=== COMPLETED: {phase_name} (took {elapsed:.2f}s) ===")
            if details:
                self.info(f"    Details: {details}")
            self._phase_start_time = None
    
    def get_memory_info(self) -> str:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved"
        else:
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                return f"CPU Memory: {mem_info.rss / 1024**2:.2f}MB RSS"
            except:
                return "CPU Memory: (unavailable)"


class MetricsCalculator:
    """Enhanced metrics calculator with detailed logging and configurable metrics"""
    
    AVAILABLE_METRICS = {
        'mse': 'Mean Squared Error',
        'rmse': 'Root Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'r2': 'Coefficient of Determination (R2)',
        'mape': 'Mean Absolute Percentage Error (percent)',
        'nrmse': 'Normalized Root Mean Squared Error (by std)',
        'nmae': 'Normalized Mean Absolute Error (by std)',
        'pcc': 'Pearson Correlation Coefficient',
        'max_error': 'Maximum Absolute Error',
        'min_error': 'Minimum Absolute Error',
        'median_error': 'Median Absolute Error',
    }
    
    DEFAULT_METRICS = ['mse', 'rmse', 'mae', 'r2', 'mape']
    
    @staticmethod
    def mse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean((pred - true) ** 2))
    
    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    
    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - true)))
    
    @staticmethod
    def r2(pred: np.ndarray, true: np.ndarray) -> float:
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
        mask = np.abs(true) > eps
        if np.sum(mask) == 0:
            return 0.0
        return float(np.mean(np.abs((true[mask] - pred[mask]) / (true[mask] + eps))) * 100)
    
    @staticmethod
    def nrmse(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
        """Normalized Root Mean Squared Error (normalized by standard deviation of true values)"""
        rmse_val = np.sqrt(np.mean((pred - true) ** 2))
        std_true = np.std(true)
        if std_true < eps:
            return rmse_val
        return float(rmse_val / std_true)
    
    @staticmethod
    def nmae(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
        """Normalized Mean Absolute Error (normalized by standard deviation of true values)"""
        mae_val = np.mean(np.abs(pred - true))
        std_true = np.std(true)
        if std_true < eps:
            return mae_val
        return float(mae_val / std_true)
    
    @staticmethod
    def pcc(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
        """Pearson Correlation Coefficient"""
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        
        pred_std = np.std(pred_flat)
        true_std = np.std(true_flat)
        
        if pred_std < eps or true_std < eps:
            return 0.0
        
        covariance = np.mean((pred_flat - np.mean(pred_flat)) * (true_flat - np.mean(true_flat)))
        return float(covariance / (pred_std * true_std))
    
    @staticmethod
    def max_error(pred: np.ndarray, true: np.ndarray) -> float:
        """Maximum Absolute Error"""
        return float(np.max(np.abs(pred - true)))
    
    @staticmethod
    def min_error(pred: np.ndarray, true: np.ndarray) -> float:
        """Minimum Absolute Error"""
        return float(np.min(np.abs(pred - true)))
    
    @staticmethod
    def median_error(pred: np.ndarray, true: np.ndarray) -> float:
        """Median Absolute Error"""
        return float(np.median(np.abs(pred - true)))
    
    @classmethod
    def compute_selected(
        cls, 
        pred: np.ndarray, 
        true: np.ndarray, 
        metrics: List[str] = None,
        name: str = "", 
        logger: Optional[EnhancedLogger] = None
    ) -> Dict[str, float]:
        """Compute selected metrics with optional logging
        
        Args:
            pred: Predicted values
            true: True values
            metrics: List of metrics to compute. If None, use DEFAULT_METRICS.
                    Use ['all'] to compute all available metrics.
            name: Name for the metrics (for logging)
            logger: Optional logger for output
        
        Returns:
            Dictionary of computed metrics
        """
        if metrics is None:
            metrics = cls.DEFAULT_METRICS
        
        if 'all' in [m.lower() for m in metrics]:
            metrics = list(cls.AVAILABLE_METRICS.keys())
        
        results = {}
        
        for metric in metrics:
            metric_lower = metric.lower()
            
            if metric_lower not in cls.AVAILABLE_METRICS:
                if logger:
                    logger.warning(f"Unknown metric: {metric}, skipping")
                continue
            
            method_name = metric_lower
            if hasattr(cls, method_name):
                metric_func = getattr(cls, method_name)
                results[metric_lower] = metric_func(pred, true)
        
        if logger and name:
            logger.info(f"  {name}:")
            for metric_name, metric_value in results.items():
                metric_desc = cls.AVAILABLE_METRICS.get(metric_name, metric_name)
                
                if metric_name == 'mape':
                    logger.info(f"    {metric_name.upper()}:  {metric_value:.4f}%  # {metric_desc}")
                elif metric_name in ['mse', 'rmse', 'mae']:
                    logger.info(f"    {metric_name.upper()}:  {metric_value:.8f}  # {metric_desc}")
                elif metric_name == 'r2':
                    logger.info(f"    {metric_name.upper()}:   {metric_value:.6f}  # {metric_desc}")
                else:
                    logger.info(f"    {metric_name}: {metric_value:.6f}  # {metric_desc}")
        
        return results
    
    @classmethod
    def get_available_metrics(cls) -> Dict[str, str]:
        """Get dictionary of available metrics and their descriptions"""
        return cls.AVAILABLE_METRICS.copy()
    
    @classmethod
    def get_default_metrics(cls) -> List[str]:
        """Get list of default metrics"""
        return cls.DEFAULT_METRICS.copy()


class CFDBenchNavierStokesDataset(Dataset):
    """
    CFDBench Navier-Stokes Dataset with detailed logging
    """
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        max_cases_per_category: int = None,
        normalize: bool = True,
        logger: Optional[EnhancedLogger] = None
    ):
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.max_cases_per_category = max_cases_per_category
        self.normalize = normalize
        self.logger = logger
        
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
    
    def _log(self, msg: str, level: str = "DATA"):
        if self.logger:
            self.logger.log(msg, level)
        else:
            print(f"[{level}] {msg}")
    
    def _load_data(self):
        """Load CFDBench data with detailed logging"""
        self._log("=" * 60)
        self._log("DATA LOADING PROCESS")
        self._log("=" * 60)
        self._log(f"Data root: {self.data_root}")
        self._log(f"Problems: {self.problems}")
        self._log(f"Categories: {self.categories}")
        if self.max_cases_per_category:
            self._log(f"Max cases per category: {self.max_cases_per_category}")
        self._log(f"Normalize: {self.normalize}")
        self._log("")
        
        all_u_data = []
        all_v_data = []
        total_cases_loaded = 0
        total_cases_skipped = 0
        
        for problem in self.problems:
            problem_path = os.path.join(self.data_root, problem)
            if not os.path.exists(problem_path):
                self._log(f"[SKIP] Problem directory not found: {problem_path}")
                continue
            
            self._log(f"Processing problem: {problem}")
            
            for category in self.categories:
                category_path = os.path.join(problem_path, category)
                if not os.path.exists(category_path):
                    self._log(f"  [SKIP] Category directory not found: {category_path}")
                    continue
                
                case_dirs = sorted(glob.glob(os.path.join(category_path, 'case*')))
                total_cases = len(case_dirs)
                
                if self.max_cases_per_category:
                    case_dirs = case_dirs[:self.max_cases_per_category]
                
                self._log(f"  Category: {category} - Found {total_cases} cases, using {len(case_dirs)}")
                
                for case_idx, case_dir in enumerate(case_dirs):
                    case_name = os.path.basename(case_dir)
                    
                    u_path = os.path.join(case_dir, 'u.npy')
                    v_path = os.path.join(case_dir, 'v.npy')
                    
                    if not os.path.exists(u_path) or not os.path.exists(v_path):
                        self._log(f"    [SKIP] {case_name}: u.npy or v.npy not found")
                        total_cases_skipped += 1
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
                        
                        time_steps = u_data.shape[0]
                        grid_size = u_data.shape[1:]
                        
                        all_u_data.append(u_data)
                        all_v_data.append(v_data)
                        
                        for t in range(time_steps - 1):
                            self.samples.append({
                                'input_u': u_data[t].copy(),
                                'input_v': v_data[t].copy(),
                                'output_u': u_data[t + 1].copy(),
                                'output_v': v_data[t + 1].copy(),
                                'case_name': f"{problem}/{category}/{case_name}",
                                'time_step': t
                            })
                        
                        self._log(f"    [LOAD] {case_name}: {time_steps} time steps, {time_steps-1} samples, shape={grid_size}")
                        total_cases_loaded += 1
                        
                    except Exception as e:
                        self._log(f"    [ERROR] {case_name}: {e}")
                        total_cases_skipped += 1
        
        self._log("")
        self._log("-" * 60)
        self._log("DATA LOADING SUMMARY")
        self._log("-" * 60)
        self._log(f"  Total cases loaded: {total_cases_loaded}")
        self._log(f"  Total cases skipped: {total_cases_skipped}")
        self._log(f"  Total time-step pairs: {len(self.samples)}")
        
        if self.normalize and len(all_u_data) > 0 and len(all_v_data) > 0:
            self._log("")
            self._log("Computing normalization statistics...")
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
            
            self._log(f"  U Component:")
            self._log(f"    mean = {self.u_mean:.6f}")
            self._log(f"    std  = {self.u_std:.6f}")
            self._log(f"    min  = {self.u_min:.6f}")
            self._log(f"    max  = {self.u_max:.6f}")
            self._log(f"  V Component:")
            self._log(f"    mean = {self.v_mean:.6f}")
            self._log(f"    std  = {self.v_std:.6f}")
            self._log(f"    min  = {self.v_min:.6f}")
            self._log(f"    max  = {self.v_max:.6f}")
            
            self._log("")
            self._log("Normalizing data...")
            for i, sample in enumerate(self.samples):
                sample['input_u'] = (sample['input_u'] - self.u_mean) / self.u_std
                sample['input_v'] = (sample['input_v'] - self.v_mean) / self.v_std
                sample['output_u'] = (sample['output_u'] - self.u_mean) / self.u_std
                sample['output_v'] = (sample['output_v'] - self.v_mean) / self.v_std
                
                if (i + 1) % 50 == 0:
                    self._log(f"  Normalized {i+1}/{len(self.samples)} samples")
            
            self._log(f"  Normalization complete: {len(self.samples)} samples")
        
        self._log("")
        self._log("=" * 60)
        self._log("DATA LOADING COMPLETE")
        self._log("=" * 60)
    
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
        return {
            'u_mean': self.u_mean, 'u_std': self.u_std,
            'u_min': self.u_min, 'u_max': self.u_max,
            'v_mean': self.v_mean, 'v_std': self.v_std,
            'v_min': self.v_min, 'v_max': self.v_max,
        }


def evaluate_model_enhanced(
    model_path: str,
    data_root: str,
    problems: List[str] = None,
    categories: List[str] = None,
    max_cases_per_category: int = None,
    normalize: bool = True,
    batch_size: int = 8,
    device: Optional[str] = None,
    log_file: Optional[str] = 'evaluation_enhanced.log',
    output_report: str = 'evaluation_report_enhanced.json',
    print_every: int = 10,
    metrics: List[str] = None,
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Enhanced model evaluation with detailed logging and configurable metrics
    
    Args:
        model_path: Path to the trained model file
        data_root: Root directory of CFDBench dataset
        problems: List of problems to evaluate (e.g., ['cavity'])
        categories: List of categories to evaluate (e.g., ['bc', 'geo', 'prop'])
        max_cases_per_category: Maximum cases per category (None for all)
        normalize: Whether to normalize the data
        batch_size: Batch size for evaluation
        device: Device to use (cuda/cpu, None for auto)
        log_file: Log file path
        output_report: Output report file path (JSON)
        print_every: Print progress every N samples
        metrics: List of metrics to compute. Use ['all'] for all available metrics.
                 Available: mse, rmse, mae, r2, mape, nrmse, nmae, pcc, 
                           max_error, min_error, median_error
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
    """
    
    if not FNO2D_AVAILABLE:
        raise RuntimeError("FNO2d model not available.")
    
    if problems is None:
        problems = ['cavity']
    if categories is None:
        categories = ['bc', 'geo', 'prop']
    
    if metrics is None:
        metrics = MetricsCalculator.get_default_metrics()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger = EnhancedLogger(
        log_file=log_file if log_to_file else None,
        log_level=log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED FNO2 MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_root}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Metrics: {metrics}")
    logger.info("")
    
    config = {
        'model_path': model_path, 'data_root': data_root,
        'problems': problems, 'categories': categories,
        'max_cases_per_category': max_cases_per_category,
        'normalize': normalize, 'batch_size': batch_size,
        'device': str(device),
        'metrics': metrics,
        'log_level': log_level,
    }
    
    logger.info("EVALUATION CONFIGURATION:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    logger.start_phase("Model Loading")
    
    logger.log(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    modes = checkpoint.get('modes', 12)
    width = checkpoint.get('width', 32)
    in_channels = checkpoint.get('in_channels', 4)
    out_channels = checkpoint.get('out_channels', 2)
    
    logger.log(f"Model configuration:")
    logger.log(f"  Modes: {modes}")
    logger.log(f"  Width: {width}")
    logger.log(f"  Input channels: {in_channels}")
    logger.log(f"  Output channels: {out_channels}")
    
    if 'config' in checkpoint:
        logger.log(f"  Training config: {checkpoint['config']}")
    
    model = FNO2d(
        modes1=modes, modes2=modes, width=width,
        in_channels=in_channels, out_channels=out_channels
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model loaded successfully!")
    logger.log(f"  Total parameters: {total_params:,}")
    
    logger.end_phase("Model Loading", f"{total_params:,} parameters")
    logger.log(f"Memory: {logger.get_memory_info()}")
    
    logger.start_phase("Data Loading")
    
    dataset = CFDBenchNavierStokesDataset(
        data_root=data_root, problems=problems, categories=categories,
        max_cases_per_category=max_cases_per_category, normalize=normalize,
        logger=logger
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No data loaded from {data_root}")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = len(data_loader)
    
    logger.end_phase("Data Loading", f"{len(dataset)} samples, {num_batches} batches")
    logger.log(f"Memory: {logger.get_memory_info()}")
    
    logger.start_phase("Model Evaluation")
    logger.log("=" * 80)
    logger.log("EVALUATION PROCESS")
    logger.log("=" * 80)
    logger.log(f"Number of samples: {len(dataset)}")
    logger.log(f"Number of batches: {num_batches}")
    logger.log(f"Print progress every {print_every} samples")
    logger.log("")
    
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_case_names = []
    all_time_steps = []
    
    per_sample_metrics_u = []
    per_sample_metrics_v = []
    
    primary_metric = metrics[0] if metrics else 'mse'
    
    def compute_metrics_for_sample(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """Dynamically compute all user-specified metrics for a sample"""
        result = {}
        for metric_name in metrics:
            metric_name_lower = metric_name.lower()
            if metric_name_lower == 'mse':
                result['mse'] = MetricsCalculator.mse(pred, true)
            elif metric_name_lower == 'rmse':
                result['rmse'] = MetricsCalculator.rmse(pred, true)
            elif metric_name_lower == 'mae':
                result['mae'] = MetricsCalculator.mae(pred, true)
            elif metric_name_lower == 'r2':
                result['r2'] = MetricsCalculator.r2(pred, true)
            elif metric_name_lower == 'mape':
                result['mape'] = MetricsCalculator.mape(pred, true)
            elif metric_name_lower == 'nrmse':
                result['nrmse'] = MetricsCalculator.nrmse(pred, true)
            elif metric_name_lower == 'nmae':
                result['nmae'] = MetricsCalculator.nmae(pred, true)
            elif metric_name_lower == 'pcc':
                result['pcc'] = MetricsCalculator.pcc(pred, true)
            elif metric_name_lower == 'max_error':
                result['max_error'] = MetricsCalculator.max_error(pred, true)
            elif metric_name_lower == 'min_error':
                result['min_error'] = MetricsCalculator.min_error(pred, true)
            elif metric_name_lower == 'median_error':
                result['median_error'] = MetricsCalculator.median_error(pred, true)
        return result
    
    def init_best_worst() -> Dict[str, Any]:
        """Initialize best/worst sample tracker with all metrics"""
        result = {'case': None, 'time_step': None}
        for metric_name in metrics:
            metric_name_lower = metric_name.lower()
            if metric_name_lower in ['r2', 'pcc']:
                result[metric_name_lower + '_best'] = -float('inf')
                result[metric_name_lower + '_worst'] = float('inf')
            else:
                result[metric_name_lower + '_best'] = float('inf')
                result[metric_name_lower + '_worst'] = -float('inf')
        return result
    
    best_u = init_best_worst()
    worst_u = init_best_worst()
    best_v = init_best_worst()
    worst_v = init_best_worst()
    
    sample_counter = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            velocity_input = batch[0].to(device)
            velocity_target = batch[1].to(device)
            case_names = batch[2]
            time_steps = batch[3]
            
            output = model(velocity_input)
            loss = criterion(output, velocity_target)
            
            batch_size_current = velocity_input.size(0)
            total_loss += loss.item() * batch_size_current
            
            output_np = output.cpu().numpy()
            target_np = velocity_target.cpu().numpy()
            
            all_outputs.append(output_np)
            all_targets.append(target_np)
            
            for i in range(batch_size_current):
                sample_counter += 1
                
                pred_u = output_np[i, ..., 0]
                pred_v = output_np[i, ..., 1]
                true_u = target_np[i, ..., 0]
                true_v = target_np[i, ..., 1]
                
                metrics_u = compute_metrics_for_sample(pred_u, true_u)
                metrics_v = compute_metrics_for_sample(pred_v, true_v)
                
                per_sample_metrics_u.append({
                    **metrics_u,
                    'case': case_names[i],
                    'time_step': time_steps[i].item()
                })
                per_sample_metrics_v.append({
                    **metrics_v,
                    'case': case_names[i],
                    'time_step': time_steps[i].item()
                })
                
                primary_metric_lower = primary_metric.lower()
                current_primary_u = metrics_u.get(primary_metric_lower, float('inf'))
                current_primary_v = metrics_v.get(primary_metric_lower, float('inf'))
                
                if primary_metric_lower in ['r2', 'pcc']:
                    if current_primary_u > best_u.get(primary_metric_lower + '_best', -float('inf')):
                        best_u = {**{k + '_best': metrics_u[k] for k in metrics_u},
                                  'case': case_names[i], 'time_step': time_steps[i].item()}
                    if current_primary_u < worst_u.get(primary_metric_lower + '_worst', float('inf')):
                        worst_u = {**{k + '_worst': metrics_u[k] for k in metrics_u},
                                   'case': case_names[i], 'time_step': time_steps[i].item()}
                    
                    if current_primary_v > best_v.get(primary_metric_lower + '_best', -float('inf')):
                        best_v = {**{k + '_best': metrics_v[k] for k in metrics_v},
                                  'case': case_names[i], 'time_step': time_steps[i].item()}
                    if current_primary_v < worst_v.get(primary_metric_lower + '_worst', float('inf')):
                        worst_v = {**{k + '_worst': metrics_v[k] for k in metrics_v},
                                   'case': case_names[i], 'time_step': time_steps[i].item()}
                else:
                    if current_primary_u < best_u.get(primary_metric_lower + '_best', float('inf')):
                        best_u = {**{k + '_best': metrics_u[k] for k in metrics_u},
                                  'case': case_names[i], 'time_step': time_steps[i].item()}
                    if current_primary_u > worst_u.get(primary_metric_lower + '_worst', -float('inf')):
                        worst_u = {**{k + '_worst': metrics_u[k] for k in metrics_u},
                                   'case': case_names[i], 'time_step': time_steps[i].item()}
                    
                    if current_primary_v < best_v.get(primary_metric_lower + '_best', float('inf')):
                        best_v = {**{k + '_best': metrics_v[k] for k in metrics_v},
                                  'case': case_names[i], 'time_step': time_steps[i].item()}
                    if current_primary_v > worst_v.get(primary_metric_lower + '_worst', -float('inf')):
                        worst_v = {**{k + '_worst': metrics_v[k] for k in metrics_v},
                                   'case': case_names[i], 'time_step': time_steps[i].item()}
                
                if sample_counter % print_every == 0:
                    avg_loss = total_loss / sample_counter
                    logger.log(
                        f"  [Sample {sample_counter}/{len(dataset)}] "
                        f"Loss = {loss.item():.6f} | "
                        f"Avg Loss = {avg_loss:.6f} | "
                        f"Case: {case_names[i]} (t={time_steps[i].item()})"
                    )
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_loss = total_loss / len(dataset)
    
    logger.log("")
    logger.log("-" * 80)
    logger.log("EVALUATION COMPLETE - COMPUTING METRICS")
    logger.log("-" * 80)
    
    logger.log(f"\nAverage MSE Loss (normalized): {avg_loss:.6f}")
    
    output_u = all_outputs[..., 0]
    output_v = all_outputs[..., 1]
    target_u = all_targets[..., 0]
    target_v = all_targets[..., 1]
    
    if normalize:
        norm_params = dataset.get_normalization_params()
        u_mean, u_std = norm_params['u_mean'], norm_params['u_std']
        v_mean, v_std = norm_params['v_mean'], norm_params['v_std']
        
        logger.log("")
        logger.log("=" * 80)
        logger.log("DENORMALIZED METRICS (Actual Velocity Values)")
        logger.log("=" * 80)
        
        output_u_denorm = output_u * u_std + u_mean
        output_v_denorm = output_v * v_std + v_mean
        target_u_denorm = target_u * u_std + u_mean
        target_v_denorm = target_v * v_std + v_mean
        
        logger.log("")
        logger.log("-" * 60)
        metrics_u = MetricsCalculator.compute_selected(
            output_u_denorm, target_u_denorm, metrics, "U COMPONENT", logger
        )
        
        logger.log("")
        logger.log("-" * 60)
        metrics_v = MetricsCalculator.compute_selected(
            output_v_denorm, target_v_denorm, metrics, "V COMPONENT", logger
        )
        
        logger.log("")
        logger.log("-" * 60)
        combined_output = np.concatenate([output_u_denorm, output_v_denorm])
        combined_target = np.concatenate([target_u_denorm, target_v_denorm])
        metrics_combined = MetricsCalculator.compute_selected(
            combined_output, combined_target, metrics, "COMBINED (U + V)", logger
        )
        
        logger.log("")
        logger.log("-" * 60)
        logger.log("PER-SAMPLE STATISTICS")
        logger.log("-" * 60)
        
        def format_per_sample_stat(metric_name: str, values_u: List[float], values_v: List[float]) -> str:
            """Format per-sample statistics for logging"""
            mean_u = np.mean(values_u)
            std_u = np.std(values_u)
            mean_v = np.mean(values_v)
            std_v = np.std(values_v)
            
            metric_name_upper = metric_name.upper()
            if metric_name in ['r2', 'pcc']:
                return (
                    f"  U: {metric_name_upper}  mean={mean_u:.6f}, std={std_u:.6f}\n"
                    f"  V: {metric_name_upper}  mean={mean_v:.6f}, std={std_v:.6f}"
                )
            elif metric_name == 'mape':
                return (
                    f"  U: {metric_name_upper} mean={mean_u:.4f}%, std={std_u:.4f}%\n"
                    f"  V: {metric_name_upper} mean={mean_v:.4f}%, std={std_v:.4f}%"
                )
            else:
                return (
                    f"  U: {metric_name_upper} mean={mean_u:.8f}, std={std_u:.8f}\n"
                    f"  V: {metric_name_upper} mean={mean_v:.8f}, std={std_v:.8f}"
                )
        
        per_sample_stats_u = {}
        per_sample_stats_v = {}
        
        for metric_name in metrics:
            metric_name_lower = metric_name.lower()
            if metric_name_lower in per_sample_metrics_u[0] if per_sample_metrics_u else []:
                values_u = [m[metric_name_lower] for m in per_sample_metrics_u]
                values_v = [m[metric_name_lower] for m in per_sample_metrics_v]
                
                per_sample_stats_u[metric_name_lower + '_mean'] = float(np.mean(values_u))
                per_sample_stats_u[metric_name_lower + '_std'] = float(np.std(values_u))
                per_sample_stats_v[metric_name_lower + '_mean'] = float(np.mean(values_v))
                per_sample_stats_v[metric_name_lower + '_std'] = float(np.std(values_v))
                
                logger.log(format_per_sample_stat(metric_name_lower, values_u, values_v))
        
        logger.log("")
        logger.log("-" * 60)
        logger.log("BEST / WORST SAMPLES")
        logger.log("-" * 60)
        
        def format_best_worst_sample(sample: Dict, is_best: bool, component: str) -> str:
            """Format best/worst sample for logging"""
            if sample.get('case') is None:
                return f"  No {'best' if is_best else 'worst'} {component} sample"
            
            lines = [f"\n  {'Best' if is_best else 'Worst'} {component} sample: {sample['case']} (t={sample['time_step']})"]
            metric_strs = []
            for k, v in sample.items():
                if k.endswith('_best') or k.endswith('_worst'):
                    metric_base = k[:-5] if k.endswith('_best') else k[:-6]
                    metric_strs.append(f"{metric_base.upper()}={v:.8f}")
            lines.append(f"    {', '.join(metric_strs)}")
            return "\n".join(lines)
        
        logger.log(format_best_worst_sample(best_u, True, "U"))
        logger.log(format_best_worst_sample(worst_u, False, "U"))
        logger.log(format_best_worst_sample(best_v, True, "V"))
        logger.log(format_best_worst_sample(worst_v, False, "V"))
    
    logger.end_phase("Model Evaluation")
    
    logger.start_phase("Report Saving")
    
    def get_best_worst_dict(sample: Dict) -> Dict:
        """Convert best/worst sample to dict for report"""
        if sample.get('case') is None:
            return {}
        result = {'case': sample['case'], 'time_step': sample['time_step']}
        for k, v in sample.items():
            if k.endswith('_best') or k.endswith('_worst'):
                result[k] = v
        return result
    
    report = {
        'model_path': model_path,
        'data_root': data_root,
        'config': config,
        'normalization_params': dataset.get_normalization_params() if normalize else None,
        'num_samples': len(dataset),
        'average_loss': avg_loss,
        'metrics': {
            'u': metrics_u,
            'v': metrics_v,
            'combined': metrics_combined
        },
        'per_sample_stats': {
            'u': per_sample_stats_u,
            'v': per_sample_stats_v
        },
        'best_worst_samples': {
            'best_u': get_best_worst_dict(best_u),
            'worst_u': get_best_worst_dict(worst_u),
            'best_v': get_best_worst_dict(best_v),
            'worst_v': get_best_worst_dict(worst_v)
        },
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.log(f"Report saved to: {os.path.abspath(output_report)}")
    logger.end_phase("Report Saving")
    
    logger.log("\n" + "=" * 80)
    logger.log("EVALUATION COMPLETE - FINAL SUMMARY")
    logger.log("=" * 80)
    logger.log(f"Model: {model_path}")
    logger.log(f"Data: {data_root}")
    logger.log(f"Samples: {len(dataset)}")
    logger.log(f"Average Loss: {avg_loss:.6f}")
    logger.log("")
    logger.log("Final Metrics (Denormalized):")
    
    def format_metric(name: str, value: float) -> str:
        """Format metric value based on metric type"""
        if name == 'mape':
            return f"  {name.upper()}:  {value:.4f}%"
        elif name == 'r2':
            return f"  {name.upper()}:   {value:.6f}"
        elif name in ['mse', 'rmse', 'mae']:
            return f"  {name.upper()}:  {value:.8f}"
        else:
            return f"  {name}: {value:.6f}"
    
    common_metrics = ['r2', 'rmse', 'mse', 'mae', 'mape']
    available_common = [m for m in common_metrics if m in metrics_u]
    additional_metrics = [m for m in metrics_u.keys() if m not in common_metrics]
    
    if available_common:
        logger.log(f"  U Component:")
        for m in available_common:
            logger.log(format_metric(m, metrics_u[m]))
        
        logger.log(f"  V Component:")
        for m in available_common:
            logger.log(format_metric(m, metrics_v[m]))
        
        logger.log(f"  Combined (U + V):")
        for m in available_common:
            logger.log(format_metric(m, metrics_combined[m]))
    
    if additional_metrics:
        logger.log("")
        logger.log("  Additional Metrics:")
        for m in sorted(additional_metrics):
            metric_desc = MetricsCalculator.AVAILABLE_METRICS.get(m, m)
            logger.log(f"    {m} ({metric_desc}):")
            logger.log(f"      U: {metrics_u[m]:.6f}")
            logger.log(f"      V: {metrics_v[m]:.6f}")
            logger.log(f"      Combined: {metrics_combined[m]:.6f}")
    
    logger.log("")
    if log_file:
        logger.log(f"Log saved at: {os.path.abspath(log_file)}")
    logger.log(f"Report saved at: {os.path.abspath(output_report)}")
    logger.log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 80)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced FNO2 Model Evaluation with Detailed Logging and Configurable Metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='fno2_full_model.pth',
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default=r'C:\traework\data',
        help='Root directory of CFDBench dataset'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        default='evaluation_enhanced.log',
        help='Log file path'
    )
    
    parser.add_argument(
        '--output_report',
        type=str,
        default='evaluation_report_enhanced.json',
        help='Output report file path'
    )
    
    parser.add_argument(
        '--problems',
        type=str,
        default='cavity',
        help='Problems (comma-separated)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        default='bc,geo,prop',
        help='Categories (comma-separated)'
    )
    
    parser.add_argument(
        '--max_cases_per_category',
        type=int,
        default=None,
        help='Maximum cases per category (use None for all)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    
    parser.add_argument(
        '--print_every',
        type=int,
        default=10,
        help='Print progress every N samples'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    
    available_metrics = MetricsCalculator.get_available_metrics()
    default_metrics = MetricsCalculator.get_default_metrics()
    
    metrics_help = (
        "Metrics to compute (comma-separated). Use 'all' for all metrics.\n"
        f"Default: {default_metrics}\n"
        "Available metrics: mse, rmse, mae, r2, mape, nrmse, nmae, pcc, max_error, min_error, median_error"
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        default=','.join(default_metrics),
        help=metrics_help
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Disable console output (only log to file)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--no_file_log',
        action='store_true',
        default=False,
        help='Disable file logging (only log to console)'
    )
    
    parser.add_argument(
        '--list_metrics',
        action='store_true',
        default=False,
        help='List all available metrics and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_metrics:
        print("\n" + "=" * 60)
        print("AVAILABLE METRICS")
        print("=" * 60)
        print(f"\nDefault metrics: {default_metrics}")
        print("\nAll available metrics:")
        for metric_name, metric_desc in available_metrics.items():
            default_mark = " [DEFAULT]" if metric_name in default_metrics else ""
            print(f"  - {metric_name}: {metric_desc}{default_mark}")
        print("\n" + "=" * 60)
        return 0
    
    if args.verbose:
        args.log_level = 'DEBUG'
    
    problems = [p.strip() for p in args.problems.split(',')]
    categories = [c.strip() for c in args.categories.split(',')]
    
    if args.metrics:
        metrics = [m.strip().lower() for m in args.metrics.split(',')]
    else:
        metrics = default_metrics
    
    log_to_console = not args.quiet
    log_to_file = not args.no_file_log
    
    report = evaluate_model_enhanced(
        model_path=args.model_path,
        data_root=args.data_root,
        problems=problems,
        categories=categories,
        max_cases_per_category=args.max_cases_per_category,
        batch_size=args.batch_size,
        print_every=args.print_every,
        log_file=args.log_file,
        output_report=args.output_report,
        device=args.device,
        metrics=metrics,
        log_level=args.log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
