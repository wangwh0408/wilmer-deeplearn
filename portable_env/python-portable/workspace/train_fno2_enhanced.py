"""
Enhanced FNO2 Training Script for Navier-Stokes Equations on CFDBench

This script provides EXTREMELY DETAILED LOGGING using Python's standard logging module.

KEY FEATURES:
  - classNum parameter: Control number of categories to use (1=bc, 2=bc+geo, 3=all)
  - Gradient statistics: Track gradient mean/max/min to detect vanishing/exploding gradients
  - Parameter statistics: Track parameter updates
  - Learning rate change alerts
  - Prediction samples: Show actual vs predicted values
  - Detailed loss tracking: Per-batch, per-epoch, running averages
  - Anomaly detection: Detect NaN, Inf values
  - Batch information: Input/output shapes, data statistics
  - Progress bar simulation
  - Epoch summary with trend analysis

NEW LOGGING FEATURES (using Python standard logging module):
  - Multiple log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Configurable output: Console and/or file logging
  - --quiet: Disable console output (only log to file)
  - --verbose: Enable DEBUG level logging
  - --no_file_log: Disable file logging (only log to console)
  - --log_level: Set specific log level

Usage:
  python train_fno2_enhanced.py --data_root C:\traework\data --epochs 30 --classNum 3 --max_cases_per_category 10
  python train_fno2_enhanced.py --classNum 1 (use only bc category)
  python train_fno2_enhanced.py --classNum 2 (use bc + geo categories)
  python train_fno2_enhanced.py --classNum 3 (use all categories: bc, geo, prop)
  
  # Logging options:
  python train_fno2_enhanced.py --verbose (DEBUG level)
  python train_fno2_enhanced.py --quiet (only log to file)
  python train_fno2_enhanced.py --log_level WARNING (only warnings and errors)
  python train_fno2_enhanced.py --no_file_log (only log to console)
"""

import os
import sys
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime

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
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup Python standard logging module.
    
    Args:
        log_file: Path to log file (if None, file logging is disabled)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file (requires log_file)
        log_format: Custom log format (if None, use default format)
    
    Returns:
        Configured logger instance
    """
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
    
    if log_format is None:
        log_format = '%(asctime)s [%(levelname)s] [%(elapsed)s] %(message)s'
    
    formatter = CustomFormatter(log_format)
    
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


class CustomFormatter(logging.Formatter):
    """Custom log formatter with elapsed time support"""
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.start_time = time.time()
    
    def format(self, record):
        elapsed = time.time() - self.start_time
        record.elapsed = f"{elapsed:.2f}s"
        return super().format(record)


class EnhancedLogger:
    """
    Enhanced logger using Python standard logging module.
    
    This class wraps the standard logging module to provide:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Console and file output (configurable)
    - Elapsed time tracking
    - Phase timing
    - Memory usage tracking
    """
    
    _start_time = None
    _phase_start_time = None
    _logger = None
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        logger_name: str = "fno2_training"
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
            'TENSOR': logging.DEBUG,
            'METRIC': logging.INFO,
            'ALERT': logging.WARNING,
        }
        
        log_level = level_map.get(level.upper(), logging.INFO)
        self._logger.log(log_level, message)
    
    def debug(self, message: str):
        """Log debug message"""
        self._logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self._logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self._logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self._logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self._logger.critical(message)
    
    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        self._phase_start_time = time.time()
        self.info(f"=== STARTING: {phase_name} ===")
    
    def end_phase(self, phase_name: str, details: str = ""):
        """End timing a phase"""
        if self._phase_start_time:
            elapsed = time.time() - self._phase_start_time
            self.info(f"=== COMPLETED: {phase_name} (took {elapsed:.2f}s) ===")
            if details:
                self.info(f"    Details: {details}")
            self._phase_start_time = None
    
    def get_memory_info(self) -> str:
        """Get memory usage information"""
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
    
    def log_progress_bar(self, current: int, total: int, prefix: str = "Progress", length: int = 50):
        """Log a progress bar"""
        percent = (current / total) * 100
        filled = int(length * current // total)
        bar = '█' * filled + '-' * (length - filled)
        self.info(f"{prefix}: |{bar}| {percent:.1f}% ({current}/{total})")
    
    def log_dict(self, data: Dict, title: str = "", indent: int = 2):
        """Log a dictionary with proper formatting"""
        if title:
            self.info(f"  {title}:")
        for key, value in data.items():
            if isinstance(value, float):
                self.info(f"{' ' * indent}{key}: {value:.6f}")
            else:
                self.info(f"{' ' * indent}{key}: {value}")
    
    def log_tensor_stats(self, tensor: torch.Tensor, name: str = "tensor"):
        """Log detailed tensor statistics"""
        if tensor is None:
            return
        
        with torch.no_grad():
            stats = {
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'mean': float(torch.mean(tensor)),
                'std': float(torch.std(tensor)) if tensor.numel() > 1 else 0.0,
                'min': float(torch.min(tensor)),
                'max': float(torch.max(tensor)),
                'median': float(torch.median(tensor)),
            }
        
        self.debug(f"  Tensor Stats - {name}:")
        self.log_dict(stats, indent=4)
        
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            if has_nan and has_inf:
                anomaly_msg = f"{name} contains NaN and Inf values"
            elif has_nan:
                anomaly_msg = f"{name} contains NaN values"
            else:
                anomaly_msg = f"{name} contains Inf values"
            self.warning(f"    [WARNING] ANOMALY: {anomaly_msg}")


class TrainingStatsTracker:
    """Track detailed training statistics"""
    
    def __init__(self):
        self.gradient_stats = []
        self.param_stats = []
        self.loss_history = []
        self.running_loss = 0.0
        self.running_count = 0
    
    @staticmethod
    def _safe_to_float(value, default: float = 0.0) -> float:
        """Safely convert tensor value to float, handling Inf/NaN"""
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                return float_val
            return default
        except (RuntimeError, OverflowError, ValueError):
            return default
    
    def update_gradient_stats(self, model: nn.Module):
        """Track gradient statistics with safety checks for Inf/NaN and complex numbers"""
        grad_norms = []
        grad_means = []
        grad_maxs = []
        grad_mins = []
        
        valid_count = 0
        invalid_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                
                if has_nan or has_inf:
                    invalid_count += 1
                    continue
                
                is_complex = grad.is_complex()
                
                norm_val = self._safe_to_float(torch.norm(grad))
                
                if is_complex:
                    grad_abs = torch.abs(grad)
                    mean_val = self._safe_to_float(torch.mean(grad_abs))
                    max_val = self._safe_to_float(torch.max(grad_abs))
                    min_val = self._safe_to_float(torch.min(grad_abs))
                else:
                    mean_val = self._safe_to_float(torch.mean(grad))
                    max_val = self._safe_to_float(torch.max(grad))
                    min_val = self._safe_to_float(torch.min(grad))
                
                if np.isfinite(norm_val) and np.isfinite(mean_val):
                    grad_norms.append(norm_val)
                    grad_means.append(mean_val)
                    grad_maxs.append(max_val)
                    grad_mins.append(min_val)
                    valid_count += 1
                else:
                    invalid_count += 1
        
        if grad_norms:
            stats = {
                'grad_norm_mean': float(np.mean(grad_norms)),
                'grad_norm_max': float(np.max(grad_norms)),
                'grad_norm_min': float(np.min(grad_norms)),
                'grad_mean_mean': float(np.mean(grad_means)),
                'grad_max_max': float(np.max(grad_maxs)),
                'grad_min_min': float(np.min(grad_mins)),
                'valid_gradients': valid_count,
                'invalid_gradients': invalid_count,
            }
            self.gradient_stats.append(stats)
            return stats
        return None
    
    def update_param_stats(self, model: nn.Module):
        """Track parameter statistics with safety checks for complex numbers"""
        param_norms = []
        param_means = []
        
        for name, param in model.named_parameters():
            if param.data is not None:
                is_complex = param.data.is_complex()
                
                norm_val = self._safe_to_float(torch.norm(param.data))
                
                if is_complex:
                    mean_val = self._safe_to_float(torch.mean(torch.abs(param.data)))
                else:
                    mean_val = self._safe_to_float(torch.mean(param.data))
                
                if np.isfinite(norm_val) and np.isfinite(mean_val):
                    param_norms.append(norm_val)
                    param_means.append(mean_val)
        
        if param_norms:
            stats = {
                'param_norm_mean': float(np.mean(param_norms)),
                'param_norm_max': float(np.max(param_norms)),
                'param_means_mean': float(np.mean(param_means)),
            }
            self.param_stats.append(stats)
            return stats
        return None
    
    def update_loss(self, loss: float, batch_size: int = 1):
        """Update loss tracking"""
        self.loss_history.append(loss)
        self.running_loss += loss * batch_size
        self.running_count += batch_size
    
    def get_running_avg(self) -> float:
        """Get running average loss"""
        if self.running_count > 0:
            return self.running_loss / self.running_count
        return 0.0
    
    def reset_running(self):
        """Reset running statistics"""
        self.running_loss = 0.0
        self.running_count = 0


class AnomalyDetector:
    """Detect anomalies in training"""
    
    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> Tuple[bool, str]:
        """Check for NaN or Inf values"""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan and has_inf:
            return True, f"{name} contains NaN and Inf values"
        elif has_nan:
            return True, f"{name} contains NaN values"
        elif has_inf:
            return True, f"{name} contains Inf values"
        return False, ""
    
    @staticmethod
    def check_loss_anomaly(loss: float, prev_loss: float, threshold: float = 10.0) -> Tuple[bool, str]:
        """Check if loss has abnormal spike"""
        if prev_loss > 0 and loss > prev_loss * threshold:
            return True, f"Loss spiked by {loss/prev_loss:.2f}x"
        return False, ""
    
    @staticmethod
    def check_gradient_anomaly(grad_stats: Dict, threshold: float = 1e6) -> Tuple[bool, str]:
        """Check for gradient explosion/vanishing"""
        if grad_stats is None:
            return False, ""
        
        if grad_stats['grad_norm_max'] > threshold:
            return True, f"Gradient explosion detected: max norm = {grad_stats['grad_norm_max']:.2e}"
        
        if grad_stats['grad_norm_mean'] < 1e-10:
            return True, f"Gradient vanishing detected: mean norm = {grad_stats['grad_norm_mean']:.2e}"
        
        return False, ""


class CFDBenchNavierStokesDataset(Dataset):
    """CFDBench Navier-Stokes Dataset with detailed logging"""
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        classNum: int = 3,
        max_cases_per_category: int = None,
        normalize: bool = True,
        logger: Optional[EnhancedLogger] = None
    ):
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.classNum = classNum
        
        if categories is None:
            if classNum == 1:
                self.categories = ['bc']
            elif classNum == 2:
                self.categories = ['bc', 'geo']
            else:
                self.categories = ['bc', 'geo', 'prop']
        else:
            self.categories = categories
        
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
        self._log(f"classNum: {self.classNum}")
        self._log(f"Categories to use: {self.categories}")
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


def train_fno2_enhanced(
    data_root: str = r"C:\traework\data",
    problems: List[str] = None,
    categories: List[str] = None,
    classNum: int = 3,
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
    model_save_path: Optional[str] = 'fno2_enhanced_model.pth',
    log_file: Optional[str] = 'training_enhanced.log',
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    device: Optional[str] = None,
    print_every: int = 1,
    show_gradient_stats: bool = True,
    show_param_stats: bool = True,
    show_tensor_stats: bool = True,
    show_prediction_samples: bool = True,
) -> Dict[str, Any]:
    """Enhanced training with EXTREMELY DETAILED logging using Python standard logging module"""
    
    if not FNO2D_AVAILABLE:
        raise RuntimeError("FNO2d model not available.")
    
    if problems is None:
        problems = ['cavity']
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger = EnhancedLogger(
        log_file=log_file,
        log_level=log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    stats_tracker = TrainingStatsTracker()
    prev_epoch_loss = None
    
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED FNO2 TRAINING FOR NAVIER-STOKES (CFDBENCH)")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log to console: {log_to_console}")
    logger.info(f"Log to file: {log_to_file}")
    logger.info("")
    
    config = {
        'data_root': data_root, 'problems': problems, 'categories': categories,
        'classNum': classNum, 'modes': modes, 'width': width,
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
        'weight_decay': weight_decay, 'train_ratio': train_ratio,
        'normalize': normalize, 'max_cases_per_category': max_cases_per_category,
        'in_channels': 4, 'out_channels': 2,
        'show_gradient_stats': show_gradient_stats,
        'show_param_stats': show_param_stats,
        'show_tensor_stats': show_tensor_stats,
        'show_prediction_samples': show_prediction_samples,
    }
    
    logger.log("TRAINING CONFIGURATION:")
    for key, value in config.items():
        logger.log(f"  {key}: {value}")
    logger.log("")
    
    logger.start_phase("Data Loading")
    dataset = CFDBenchNavierStokesDataset(
        data_root=data_root, problems=problems, categories=categories,
        classNum=classNum, max_cases_per_category=max_cases_per_category,
        normalize=normalize, logger=logger
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No data loaded from {data_root}")
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    train_size = int(n_total * train_ratio)
    test_size = n_total - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.end_phase("Data Loading", f"{n_train} train, {n_test} test, {len(train_loader)} batches per epoch")
    logger.log(f"Memory: {logger.get_memory_info()}")
    
    logger.start_phase("Model Creation")
    model = FNO2d(
        modes1=modes, modes2=modes, width=width,
        in_channels=4, out_channels=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f"Model Architecture:")
    logger.log(f"  FNO2d with modes={modes}, width={width}")
    logger.log(f"  Input channels: 4 (2 data + 2 coords auto-added)")
    logger.log(f"  Output channels: 2 (u, v velocity)")
    logger.log(f"  Total parameters: {total_params:,}")
    logger.log(f"  Trainable parameters: {trainable_params:,}")
    
    logger.log(f"\n  Layer-wise parameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.log(f"    {name}: {param.numel():,} params")
    
    logger.end_phase("Model Creation", f"{total_params:,} parameters")
    logger.log(f"Memory: {logger.get_memory_info()}")
    
    logger.start_phase("Optimizer Setup")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.MSELoss()
    
    logger.log(f"Optimizer: Adam")
    logger.log(f"  Learning rate: {learning_rate}")
    logger.log(f"  Weight decay: {weight_decay}")
    logger.log(f"Scheduler: StepLR (step={scheduler_step_size}, gamma={scheduler_gamma})")
    logger.log(f"Loss function: MSELoss")
    logger.end_phase("Optimizer Setup")
    
    logger.start_phase("Training")
    logger.log("=" * 80)
    logger.log("TRAINING PROCESS - DETAILED LOGGING")
    logger.log("=" * 80)
    logger.log("")
    logger.log("Logging features enabled:")
    logger.log(f"  - Gradient statistics: {'[YES]' if show_gradient_stats else '[NO]'}")
    logger.log(f"  - Parameter statistics: {'[YES]' if show_param_stats else '[NO]'}")
    logger.log(f"  - Tensor statistics: {'[YES]' if show_tensor_stats else '[NO]'}")
    logger.log(f"  - Prediction samples: {'[YES]' if show_prediction_samples else '[NO]'}")
    logger.log("")
    
    train_losses = []
    test_losses = []
    test_metrics_history = []
    best_test_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    
    training_start_time = time.time()
    initial_lr = learning_rate
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log("\n" + "=" * 80)
        logger.log(f"[EPOCH {epoch+1}/{epochs}] START")
        logger.log("=" * 80)
        logger.log(f"  Current learning rate: {current_lr:.2e}")
        
        if epoch > 0 and current_lr != initial_lr:
            logger.log(f"  [ALERT] LEARNING RATE CHANGED: {initial_lr:.2e} -> {current_lr:.2e}", "ALERT")
            initial_lr = current_lr
        
        logger.log(f"  Number of batches: {num_batches}")
        logger.log(f"  Memory: {logger.get_memory_info()}")
        logger.log("")
        
        stats_tracker.reset_running()
        
        for batch_idx, batch in enumerate(train_loader):
            velocity_input = batch[0].to(device)
            velocity_target = batch[1].to(device)
            case_names = batch[2]
            time_steps = batch[3]
            
            batch_size_current = velocity_input.size(0)
            
            if show_tensor_stats and (batch_idx + 1) % print_every == 0:
                logger.log("-" * 60)
                logger.log(f"[Batch {batch_idx+1}/{num_batches}] INPUT DATA STATS", "DETAIL")
                logger.log(f"  Case: {case_names[0]} (t={time_steps[0]})")
                logger.log_tensor_stats(velocity_input[0], "Input (first sample)")
                logger.log_tensor_stats(velocity_target[0], "Target (first sample)")
            
            optimizer.zero_grad()
            output = model(velocity_input)
            loss = criterion(output, velocity_target)
            
            has_anomaly, anomaly_msg = AnomalyDetector.check_nan_inf(loss, "Loss")
            if has_anomaly:
                logger.log(f"  [WARNING] ANOMALY DETECTED: {anomaly_msg}", "WARNING")
                logger.log(f"     Skipping this batch and continuing...", "WARNING")
                continue
            
            loss.backward()
            
            if show_gradient_stats and (batch_idx + 1) % print_every == 0:
                grad_stats = stats_tracker.update_gradient_stats(model)
                if grad_stats:
                    logger.log(f"\n  [Batch {batch_idx+1}/{num_batches}] GRADIENT STATISTICS", "DETAIL")
                    logger.log_dict(grad_stats, "Gradient Stats", indent=4)
                    
                    has_grad_anomaly, grad_anomaly_msg = AnomalyDetector.check_gradient_anomaly(grad_stats)
                    if has_grad_anomaly:
                        logger.log(f"  [WARNING] GRADIENT ANOMALY: {grad_anomaly_msg}", "WARNING")
            
            optimizer.step()
            
            if show_param_stats and (batch_idx + 1) % print_every == 0:
                param_stats = stats_tracker.update_param_stats(model)
                if param_stats:
                    logger.log(f"\n  [Batch {batch_idx+1}/{num_batches}] PARAMETER STATISTICS", "DETAIL")
                    logger.log_dict(param_stats, "Param Stats", indent=4)
            
            batch_loss = loss.item()
            train_loss += batch_loss * batch_size_current
            stats_tracker.update_loss(batch_loss, batch_size_current)
            
            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
                running_avg = stats_tracker.get_running_avg()
                
                logger.log(f"\n  [Batch {batch_idx+1}/{num_batches}] TRAINING PROGRESS", "PROGRESS")
                logger.log(f"    Batch Loss:      {batch_loss:.8f}")
                logger.log(f"    Running Avg:     {running_avg:.8f}")
                
                if prev_epoch_loss:
                    loss_ratio = batch_loss / prev_epoch_loss
                    logger.log(f"    vs Last Epoch:   {loss_ratio:.2f}x")
                    
                    if loss_ratio > 10.0:
                        logger.log(f"    [WARNING] Loss spike detected!", "WARNING")
                
                logger.log(f"    Case:            {case_names[0]} (t={time_steps[0]})")
                
                if show_prediction_samples and (batch_idx + 1) % (print_every * 5) == 0:
                    logger.log(f"\n  [Batch {batch_idx+1}/{num_batches}] PREDICTION SAMPLE", "DETAIL")
                    
                    with torch.no_grad():
                        sample_idx = 0
                        pred_sample = output[sample_idx].cpu().numpy()
                        true_sample = velocity_target[sample_idx].cpu().numpy()
                        
                        logger.log(f"    Sample: {case_names[sample_idx]} (t={time_steps[sample_idx]})")
                        logger.log(f"    Predicted U stats:  mean={pred_sample[..., 0].mean():.6f}, std={pred_sample[..., 0].std():.6f}")
                        logger.log(f"    Target U stats:     mean={true_sample[..., 0].mean():.6f}, std={true_sample[..., 0].std():.6f}")
                        logger.log(f"    Predicted V stats:  mean={pred_sample[..., 1].mean():.6f}, std={pred_sample[..., 1].std():.6f}")
                        logger.log(f"    Target V stats:     mean={true_sample[..., 1].mean():.6f}, std={true_sample[..., 1].std():.6f}")
                        
                        mse_u = np.mean((pred_sample[..., 0] - true_sample[..., 0]) ** 2)
                        mse_v = np.mean((pred_sample[..., 1] - true_sample[..., 1]) ** 2)
                        logger.log(f"    Sample MSE (U): {mse_u:.8f}")
                        logger.log(f"    Sample MSE (V): {mse_v:.8f}")
        
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.log("\n" + "-" * 60)
        logger.log(f"[EPOCH {epoch+1}/{epochs}] TRAIN COMPLETE")
        logger.log("-" * 60)
        logger.log(f"  Average Train Loss: {train_loss:.8f}")
        logger.log(f"  Epoch time: {epoch_time:.2f}s")
        
        if prev_epoch_loss:
            train_loss_change = (train_loss - prev_epoch_loss) / prev_epoch_loss * 100
            if train_loss_change < 0:
                logger.log(f"  Loss change: {train_loss_change:.2f}% (↓ improved)")
            else:
                logger.log(f"  Loss change: +{train_loss_change:.2f}% (↑ worsened)")
        
        prev_epoch_loss = train_loss
        
        logger.log(f"\n[EPOCH {epoch+1}/{epochs}] EVALUATION ON TEST SET")
        model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                velocity_input = batch[0].to(device)
                velocity_target = batch[1].to(device)
                
                output = model(velocity_input)
                loss = criterion(output, velocity_target)
                test_loss += loss.item() * velocity_input.size(0)
                
                all_outputs.append(output.cpu().numpy())
                all_targets.append(velocity_target.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mse = np.mean((all_outputs - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_outputs - all_targets))
        
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        output_u = all_outputs[..., 0]
        output_v = all_outputs[..., 1]
        target_u = all_targets[..., 0]
        target_v = all_targets[..., 1]
        
        r2_u = 1 - (np.sum((target_u - output_u) ** 2) / (np.sum((target_u - np.mean(target_u)) ** 2) + 1e-8))
        r2_v = 1 - (np.sum((target_v - output_v) ** 2) / (np.sum((target_v - np.mean(target_v)) ** 2) + 1e-8))
        
        test_metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'r2_u': r2_u, 'r2_v': r2_v}
        test_metrics_history.append(test_metrics)
        
        best_marker = " *** BEST ***" if is_best else ""
        logger.log(f"\n  Test Loss: {test_loss:.8f}{best_marker}")
        logger.log(f"  Test Metrics:")
        logger.log(f"    MSE:  {mse:.8f}")
        logger.log(f"    RMSE: {rmse:.6f}")
        logger.log(f"    MAE:  {mae:.6f}")
        logger.log(f"    R2 (Overall): {r2:.6f}")
        logger.log(f"    R2 (U component): {r2_u:.6f}")
        logger.log(f"    R2 (V component): {r2_v:.6f}")
        
        logger.log(f"\n[EPOCH {epoch+1}/{epochs}] END - Time: {epoch_time:.2f}s")
        logger.log(f"Memory: {logger.get_memory_info()}")
    
    total_training_time = time.time() - training_start_time
    logger.end_phase("Training", f"Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.start_phase("Model Saving")
    model_path = None
    if model_save_path is not None:
        logger.log(f"Saving model to: {model_save_path}")
        
        save_dict = {
            'epoch': epochs, 'best_epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses, 'test_losses': test_losses,
            'test_metrics_history': test_metrics_history,
            'best_test_loss': best_test_loss,
            'modes': modes, 'width': width,
            'in_channels': 4, 'out_channels': 2,
            'normalize': normalize,
            'normalization_params': dataset.get_normalization_params(),
            'config': config,
            'total_training_time': total_training_time,
            'gradient_stats': stats_tracker.gradient_stats if stats_tracker.gradient_stats else None,
            'param_stats': stats_tracker.param_stats if stats_tracker.param_stats else None,
        }
        
        torch.save(save_dict, model_save_path)
        model_path = os.path.abspath(model_save_path)
        
        logger.log(f"Model saved successfully: {model_path}")
        logger.log(f"File size: {os.path.getsize(model_save_path) / 1024/1024:.2f} MB")
    
    logger.end_phase("Model Saving")
    
    logger.log("\n" + "=" * 80)
    logger.log("TRAINING COMPLETE - FINAL SUMMARY")
    logger.log("=" * 80)
    logger.log(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    logger.log(f"Average time per epoch: {total_training_time/epochs:.2f} seconds")
    logger.log("")
    logger.log(f"Final Train Loss: {train_losses[-1]:.8f}")
    logger.log(f"Final Test Loss:  {test_losses[-1]:.8f}")
    logger.log(f"Best Test Loss:   {best_test_loss:.8f} (Epoch {best_epoch})")
    logger.log("")
    logger.log(f"Final Test Metrics:")
    logger.log(f"  MSE:  {test_metrics_history[-1]['mse']:.8f}")
    logger.log(f"  RMSE: {test_metrics_history[-1]['rmse']:.6f}")
    logger.log(f"  MAE:  {test_metrics_history[-1]['mae']:.6f}")
    logger.log(f"  R2:   {test_metrics_history[-1]['r2']:.6f}")
    logger.log("")
    
    if stats_tracker.gradient_stats:
        logger.log(f"Gradient Statistics Summary:")
        grad_means = [s['grad_norm_mean'] for s in stats_tracker.gradient_stats]
        logger.log(f"  Average gradient norm: {np.mean(grad_means):.6f}")
        logger.log(f"  Max gradient norm:     {np.max(grad_means):.6f}")
        logger.log(f"  Min gradient norm:     {np.min(grad_means):.6f}")
    
    if model_path:
        logger.log(f"Model saved at: {model_path}")
    logger.log(f"Log saved at: {os.path.abspath(log_file) if log_file else 'None'}")
    logger.log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 80)
    
    return {
        'model': model, 'train_losses': train_losses, 'test_losses': test_losses,
        'test_metrics_history': test_metrics_history,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'final_test_metrics': test_metrics_history[-1] if test_metrics_history else None,
        'best_test_loss': best_test_loss, 'best_epoch': best_epoch,
        'config': config, 'model_path': model_path,
        'device': str(device), 'total_training_time': total_training_time,
        'normalization_params': dataset.get_normalization_params(),
        'gradient_stats': stats_tracker.gradient_stats,
        'param_stats': stats_tracker.param_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced FNO2 Training with Python Standard Logging Module',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_root', type=str, default=r'C:\traework\data', help='Dataset root')
    parser.add_argument('--model_output', type=str, default='fno2_full_model.pth', help='Model output path')
    parser.add_argument('--log_file', type=str, default='training_full.log', help='Log file path')
    parser.add_argument('--problems', type=str, default='cavity', help='Problems (comma-separated)')
    parser.add_argument('--categories', type=str, default='bc,geo,prop', help='Categories (comma-separated, overrides classNum)')
    
    parser.add_argument('--classNum', type=int, default=3, 
                        help='Number of categories to use: 1=bc, 2=bc+geo, 3=bc+geo+prop')
    
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=32, help='Channel width')
    parser.add_argument('--max_cases_per_category', type=int, default=10, help='Max cases per category')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train data ratio')
    parser.add_argument('--print_every', type=int, default=1, help='Print every N batches')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    parser.add_argument('--no_gradient_stats', action='store_true', default=False, help='Disable gradient statistics')
    parser.add_argument('--no_param_stats', action='store_true', default=False, help='Disable parameter statistics')
    parser.add_argument('--no_tensor_stats', action='store_true', default=False, help='Disable tensor statistics')
    parser.add_argument('--no_prediction_samples', action='store_true', default=False, help='Disable prediction samples')
    
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level: DEBUG=most verbose, INFO=standard, WARNING=only warnings+')
    parser.add_argument('--quiet', action='store_true', default=False, 
                        help='Disable console output (only log to file)')
    parser.add_argument('--verbose', action='store_true', default=False, 
                        help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--no_file_log', action='store_true', default=False, 
                        help='Disable file logging (only log to console)')
    
    args = parser.parse_args()
    
    if args.verbose:
        args.log_level = 'DEBUG'
    
    log_to_console = not args.quiet
    log_to_file = not args.no_file_log
    
    problems = [p.strip() for p in args.problems.split(',')]
    
    categories = None
    if args.categories != 'bc,geo,prop':
        categories = [c.strip() for c in args.categories.split(',')]
    
    temp_logger = EnhancedLogger(
        log_file=args.log_file if log_to_file else None,
        log_level=args.log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    
    temp_logger.info("\n" + "=" * 80)
    temp_logger.info("FNO2 ENHANCED TRAINING CONFIGURATION (Python Standard Logging)")
    temp_logger.info("=" * 80)
    temp_logger.info(f"classNum: {args.classNum}")
    
    if args.classNum == 1:
        temp_logger.info("  -> Using only: bc category")
    elif args.classNum == 2:
        temp_logger.info("  -> Using: bc + geo categories")
    else:
        temp_logger.info("  -> Using: bc + geo + prop categories (all)")
    
    if categories:
        temp_logger.info(f"  -> Overridden with custom categories: {categories}")
    
    temp_logger.info(f"Max cases per category: {args.max_cases_per_category}")
    temp_logger.info(f"Epochs: {args.epochs}")
    temp_logger.info(f"Batch size: {args.batch_size}")
    temp_logger.info(f"Learning rate: {args.learning_rate}")
    temp_logger.info(f"Log level: {args.log_level}")
    temp_logger.info(f"Log to console: {log_to_console}")
    temp_logger.info(f"Log to file: {log_to_file}")
    temp_logger.info("=" * 80)
    temp_logger.info("")
    
    result = train_fno2_enhanced(
        data_root=args.data_root,
        problems=problems,
        categories=categories,
        classNum=args.classNum,
        modes=args.modes,
        width=args.width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_cases_per_category=args.max_cases_per_category,
        train_ratio=args.train_ratio,
        model_save_path=args.model_output,
        log_file=args.log_file,
        log_level=args.log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        device=args.device,
        print_every=args.print_every,
        show_gradient_stats=not args.no_gradient_stats,
        show_param_stats=not args.no_param_stats,
        show_tensor_stats=not args.no_tensor_stats,
        show_prediction_samples=not args.no_prediction_samples,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
