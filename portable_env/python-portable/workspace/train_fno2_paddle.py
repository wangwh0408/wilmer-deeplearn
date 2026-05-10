"""
Enhanced FNO2 Training Script for Navier-Stokes Equations on CFDBench (PaddlePaddle Version)

This is the PaddlePaddle version of train_fno2_enhanced.py.
All features are identical to the PyTorch version.

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
  python train_fno2_paddle.py --data_root C:\traework\data --epochs 30 --classNum 3 --max_cases_per_category 10
  python train_fno2_paddle.py --classNum 1 (use only bc category)
  python train_fno2_paddle.py --classNum 2 (use bc + geo categories)
  python train_fno2_paddle.py --classNum 3 (use all categories: bc, geo, prop)
  
  # Logging options:
  python train_fno2_paddle.py --verbose (DEBUG level)
  python train_fno2_paddle.py --quiet (only log to file)
  python train_fno2_paddle.py --log_level WARNING (only warnings and errors)
  python train_fno2_paddle.py --no_file_log (only log to console)
"""

import os
import sys
import glob
import time
import logging
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fno2_model_paddle import FNO2dPaddle, count_parameters_paddle
    FNO2D_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import FNO2dPaddle model: {e}")
    FNO2D_AVAILABLE = False


class CustomFormatter(logging.Formatter):
    """Custom log formatter with elapsed time"""
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.start_time = time.time()
    
    def format(self, record):
        elapsed = time.time() - self.start_time
        record.elapsed = f"{elapsed:.2f}s"
        return super().format(record)


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


class EnhancedLogger:
    """Enhanced logger using Python standard logging module"""
    
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
            'TENSOR': logging.DEBUG,
            'GRADIENT': logging.DEBUG,
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
        try:
            if paddle.device.is_compiled_with_cuda():
                allocated = paddle.device.cuda.memory_allocated() / 1024**2
                reserved = paddle.device.cuda.memory_reserved() / 1024**2
                return f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved"
        except:
            pass
        
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return f"CPU Memory: {mem_info.rss / 1024**2:.2f}MB RSS"
        except:
            return "CPU Memory: (unavailable)"


class TrainingStatsTracker:
    """Track detailed training statistics for PaddlePaddle"""
    
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
    
    def update_gradient_stats(self, model: nn.Layer):
        """Track gradient statistics with safety checks for Inf/NaN and complex numbers"""
        grad_norms = []
        grad_means = []
        grad_maxs = []
        grad_mins = []
        
        valid_count = 0
        invalid_count = 0
        
        for name, param in model.named_parameters():
            if not param.stop_gradient and param.grad is not None:
                grad = param.grad
                
                has_nan = paddle.isnan(grad).any().item()
                has_inf = paddle.isinf(grad).any().item()
                
                if has_nan or has_inf:
                    invalid_count += 1
                    continue
                
                is_complex = grad.dtype in [paddle.complex64, paddle.complex128]
                
                norm_val = self._safe_to_float(paddle.norm(grad))
                
                if is_complex:
                    grad_abs = paddle.abs(grad)
                    mean_val = self._safe_to_float(paddle.mean(grad_abs))
                    max_val = self._safe_to_float(paddle.max(grad_abs))
                    min_val = self._safe_to_float(paddle.min(grad_abs))
                else:
                    mean_val = self._safe_to_float(paddle.mean(grad))
                    max_val = self._safe_to_float(paddle.max(grad))
                    min_val = self._safe_to_float(paddle.min(grad))
                
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
    
    def update_param_stats(self, model: nn.Layer):
        """Track parameter statistics with safety checks for complex numbers"""
        param_norms = []
        param_means = []
        
        for name, param in model.named_parameters():
            is_complex = param.dtype in [paddle.complex64, paddle.complex128]
            
            norm_val = self._safe_to_float(paddle.norm(param))
            
            if is_complex:
                mean_val = self._safe_to_float(paddle.mean(paddle.abs(param)))
            else:
                mean_val = self._safe_to_float(paddle.mean(param))
            
            if np.isfinite(norm_val) and np.isfinite(mean_val):
                param_norms.append(norm_val)
                param_means.append(mean_val)
        
        if param_norms:
            stats = {
                'param_norm_mean': float(np.mean(param_norms)),
                'param_norm_max': float(np.max(param_norms)),
                'param_norm_min': float(np.min(param_norms)),
                'param_mean_mean': float(np.mean(param_means)),
            }
            self.param_stats.append(stats)
            return stats
        return None
    
    def update_loss(self, loss: float):
        """Update loss tracking"""
        self.loss_history.append(loss)
        self.running_loss += loss
        self.running_count += 1
    
    def get_running_average(self) -> float:
        """Get running average of loss"""
        if self.running_count == 0:
            return 0.0
        return self.running_loss / self.running_count
    
    def reset_running(self):
        """Reset running statistics"""
        self.running_loss = 0.0
        self.running_count = 0


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
        self._log(f"  Total samples: {len(self.samples)}")
        
        if self.normalize and all_u_data and all_v_data:
            self._log("")
            self._log("Computing normalization statistics...")
            
            all_u = np.concatenate([d.flatten() for d in all_u_data])
            all_v = np.concatenate([d.flatten() for d in all_v_data])
            
            self.u_mean = float(np.mean(all_u))
            self.u_std = float(np.std(all_u)) + 1e-8
            self.u_min = float(np.min(all_u))
            self.u_max = float(np.max(all_u))
            
            self.v_mean = float(np.mean(all_v))
            self.v_std = float(np.std(all_v)) + 1e-8
            self.v_min = float(np.min(all_v))
            self.v_max = float(np.max(all_v))
            
            self._log(f"  U: mean={self.u_mean:.6f}, std={self.u_std:.6f}, min={self.u_min:.6f}, max={self.u_max:.6f}")
            self._log(f"  V: mean={self.v_mean:.6f}, std={self.v_std:.6f}, min={self.v_min:.6f}, max={self.v_max:.6f}")
        
        self._log("")
        self._log("=" * 60)
        self._log("DATA LOADING COMPLETE")
        self._log("=" * 60)
        self._log("")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_u = sample['input_u'].copy()
        input_v = sample['input_v'].copy()
        output_u = sample['output_u'].copy()
        output_v = sample['output_v'].copy()
        
        if self.normalize:
            input_u = (input_u - self.u_mean) / self.u_std
            input_v = (input_v - self.v_mean) / self.v_std
            output_u = (output_u - self.u_mean) / self.u_std
            output_v = (output_v - self.v_mean) / self.v_std
        
        input_tensor = np.stack([input_u, input_v], axis=-1)
        output_tensor = np.stack([output_u, output_v], axis=-1)
        
        return {
            'input': input_tensor.astype(np.float32),
            'output': output_tensor.astype(np.float32),
            'case_name': sample['case_name'],
            'time_step': sample['time_step']
        }
    
    def get_normalization_params(self) -> Dict[str, float]:
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


def log_tensor_stats(
    tensor, 
    name: str, 
    logger: EnhancedLogger,
    level: str = "TENSOR"
):
    """Log tensor statistics with detailed information"""
    if isinstance(tensor, paddle.Tensor):
        numpy_tensor = tensor.cpu().numpy()
    else:
        numpy_tensor = tensor
    
    if numpy_tensor.ndim == 0:
        numpy_tensor = np.array([numpy_tensor])
    
    logger.log(f"Tensor Stats - {name}:", level)
    logger.log(f"  shape: {numpy_tensor.shape}", level)
    logger.log(f"  dtype: {numpy_tensor.dtype}", level)
    logger.log(f"  mean: {float(np.mean(numpy_tensor)):.6f}", level)
    logger.log(f"  std: {float(np.std(numpy_tensor)):.6f}", level)
    logger.log(f"  min: {float(np.min(numpy_tensor)):.6f}", level)
    logger.log(f"  max: {float(np.max(numpy_tensor)):.6f}", level)
    logger.log(f"  median: {float(np.median(numpy_tensor)):.6f}", level)


def check_for_anomalies(
    tensor,
    name: str,
    logger: EnhancedLogger
) -> Tuple[bool, str]:
    """Check for NaN or Inf values in tensor"""
    if isinstance(tensor, paddle.Tensor):
        has_nan = paddle.isnan(tensor).any().item()
        has_inf = paddle.isinf(tensor).any().item()
    else:
        has_nan = np.isnan(tensor).any()
        has_inf = np.isinf(tensor).any()
    
    if has_nan:
        return True, f"{name} contains NaN values"
    if has_inf:
        return True, f"{name} contains Inf values"
    
    return False, ""


def train_fno2_paddle(
    data_root: str,
    problems: List[str] = None,
    categories: List[str] = None,
    classNum: int = 3,
    modes: int = 12,
    width: int = 32,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    max_cases_per_category: int = None,
    train_ratio: float = 0.8,
    model_save_path: str = 'fno2_paddle_model.pth',
    log_file: str = 'training_paddle.log',
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    device: str = None,
    print_every: int = 1,
    show_gradient_stats: bool = True,
    show_param_stats: bool = True,
    show_tensor_stats: bool = True,
    show_prediction_samples: bool = True,
) -> Dict[str, Any]:
    """
    Train FNO2 model with enhanced logging (PaddlePaddle version)
    
    All parameters are identical to train_fno2_enhanced.py.
    """
    
    if not FNO2D_AVAILABLE:
        raise RuntimeError("FNO2dPaddle model not available")
    
    logger = EnhancedLogger(
        log_file=log_file if log_to_file else None,
        log_level=log_level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    
    logger.start_phase("TRAINING INITIALIZATION")
    
    if device is None:
        if paddle.device.is_compiled_with_cuda():
            device = 'gpu'
        else:
            device = 'cpu'
    
    paddle.device.set_device(device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"PaddlePaddle version: {paddle.__version__}")
    
    stats_tracker = TrainingStatsTracker()
    
    logger.end_phase("TRAINING INITIALIZATION")
    
    logger.start_phase("DATA LOADING")
    
    dataset = CFDBenchNavierStokesDataset(
        data_root=data_root,
        problems=problems,
        categories=categories,
        classNum=classNum,
        max_cases_per_category=max_cases_per_category,
        normalize=True,
        logger=logger
    )
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    logger.info(f"Dataset split: {n_train} train, {n_test} test")
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    def collate_fn(batch):
        inputs = paddle.to_tensor([item['input'] for item in batch], dtype='float32')
        outputs = paddle.to_tensor([item['output'] for item in batch], dtype='float32')
        case_names = [item['case_name'] for item in batch]
        time_steps = [item['time_step'] for item in batch]
        
        return {
            'input': inputs,
            'output': outputs,
            'case_name': case_names,
            'time_step': time_steps
        }
    
    from paddle.io import Subset
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    n_batches_train = len(train_loader)
    n_batches_test = len(test_loader)
    
    logger.info(f"Train batches: {n_batches_train}")
    logger.info(f"Test batches: {n_batches_test}")
    
    logger.end_phase("DATA LOADING")
    
    logger.start_phase("MODEL CREATION")
    
    model = FNO2dPaddle(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=4,
        out_channels=2
    )
    
    total_params = count_parameters_paddle(model)
    logger.info(f"Model created successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model configuration:")
    logger.info(f"  Fourier modes: {modes}")
    logger.info(f"  Channel width: {width}")
    logger.info(f"  Input channels: 4 (2 data + 2 coords auto-added)")
    logger.info(f"  Output channels: 2 (u, v velocity)")
    
    logger.info(f"\nLayer-wise parameter count:")
    for name, param in model.named_parameters():
        param_count = np.prod(param.shape)
        logger.info(f"    {name}: {int(param_count):,} params")
    
    logger.end_phase("MODEL CREATION")
    
    logger.start_phase("OPTIMIZER AND LOSS SETUP")
    
    criterion = nn.MSELoss()
    
    lr_scheduler = optim.lr.StepDecay(
        learning_rate=learning_rate,
        step_size=50,
        gamma=0.5
    )
    
    optimizer = optim.Adam(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=1e-4
    )
    
    logger.info(f"Optimizer: Adam")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: 1e-4")
    logger.info(f"Loss function: MSELoss")
    logger.info(f"LR scheduler: StepDecay (step_size=50, gamma=0.5)")
    
    logger.end_phase("OPTIMIZER AND LOSS SETUP")
    
    logger.start_phase("TRAINING CONFIGURATION SUMMARY")
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Data root: {data_root}")
    logger.info(f"  Problems: {problems}")
    logger.info(f"  Categories: {dataset.categories}")
    logger.info(f"  classNum: {classNum}")
    logger.info(f"  Max cases per category: {max_cases_per_category}")
    logger.info(f"  Train ratio: {train_ratio}")
    logger.info(f"")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Fourier modes: {modes}")
    logger.info(f"  Channel width: {width}")
    logger.info(f"")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Log level: {log_level}")
    logger.info(f"  Model save path: {model_save_path}")
    logger.info(f"")
    logger.info(f"  Detailed logging:")
    logger.info(f"    Gradient stats: [YES]" if show_gradient_stats else f"    Gradient stats: [NO]")
    logger.info(f"    Parameter stats: [YES]" if show_param_stats else f"    Parameter stats: [NO]")
    logger.info(f"    Tensor stats: [YES]" if show_tensor_stats else f"    Tensor stats: [NO]")
    logger.info(f"    Prediction samples: [YES]" if show_prediction_samples else f"    Prediction samples: [NO]")
    logger.info("=" * 60)
    
    logger.end_phase("TRAINING CONFIGURATION SUMMARY")
    
    logger.start_phase("TRAINING LOOP")
    
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    learning_rates = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[EPOCH {epoch}/{epochs}] START")
        logger.info("=" * 60)
        
        current_lr = optimizer.get_lr()
        learning_rates.append(current_lr)
        logger.info(f"  Current learning rate: {current_lr:.2e}")
        
        if epoch > 1 and current_lr != learning_rates[-2]:
            logger.warning(f"  [LEARNING RATE CHANGE] From {learning_rates[-2]:.2e} to {current_lr:.2e}")
        
        model.train()
        train_loss = 0.0
        train_count = 0
        stats_tracker.reset_running()
        
        n_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input']
            targets = batch['output']
            case_names = batch['case_name']
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            batch_size_actual = inputs.shape[0]
            train_loss += loss.item() * batch_size_actual
            train_count += batch_size_actual
            stats_tracker.update_loss(loss.item())
            
            if batch_idx % print_every == 0 or batch_idx == n_batches - 1:
                logger.info("")
                logger.info(f"  [Batch {batch_idx+1}/{n_batches}]")
                logger.info(f"    Case(s): {', '.join(case_names[:3])}{'...' if len(case_names) > 3 else ''}")
                logger.info(f"    Loss: {loss.item():.8f}")
                logger.info(f"    Running avg loss: {stats_tracker.get_running_average():.8f}")
                
                if show_tensor_stats and batch_idx == 0:
                    logger.info("")
                    logger.info(f"    [Batch {batch_idx+1}/{n_batches}] INPUT DATA STATS")
                    log_tensor_stats(inputs[0], "Input (first sample)", logger, "INFO")
                    
                    logger.info("")
                    logger.info(f"    [Batch {batch_idx+1}/{n_batches}] TARGET DATA STATS")
                    log_tensor_stats(targets[0], "Target (first sample)", logger, "INFO")
                
                if show_gradient_stats and (batch_idx == 0 or batch_idx == n_batches - 1):
                    logger.info("")
                    logger.info(f"    [Batch {batch_idx+1}/{n_batches}] GRADIENT STATISTICS")
                    grad_stats = stats_tracker.update_gradient_stats(model)
                    if grad_stats:
                        logger.info(f"      Gradient norm (mean): {grad_stats['grad_norm_mean']:.6f}")
                        logger.info(f"      Gradient norm (max):  {grad_stats['grad_norm_max']:.6f}")
                        logger.info(f"      Gradient norm (min):  {grad_stats['grad_norm_min']:.6f}")
                        logger.info(f"      Valid gradients:      {grad_stats['valid_gradients']}")
                        logger.info(f"      Invalid gradients:    {grad_stats['invalid_gradients']}")
                        
                        if grad_stats['grad_norm_mean'] > 1e3:
                            logger.warning(f"      [WARNING] Gradient explosion detected: mean norm = {grad_stats['grad_norm_mean']:.2e}")
                        elif grad_stats['grad_norm_mean'] < 1e-10:
                            logger.warning(f"      [WARNING] Gradient vanishing detected: mean norm = {grad_stats['grad_norm_mean']:.2e}")
                
                if show_param_stats and (batch_idx == 0 or batch_idx == n_batches - 1):
                    logger.info("")
                    logger.info(f"    [Batch {batch_idx+1}/{n_batches}] PARAMETER STATISTICS")
                    param_stats = stats_tracker.update_param_stats(model)
                    if param_stats:
                        logger.info(f"      Param norm (mean): {param_stats['param_norm_mean']:.6f}")
                        logger.info(f"      Param norm (max):  {param_stats['param_norm_max']:.6f}")
                        logger.info(f"      Param norm (min):  {param_stats['param_norm_min']:.6f}")
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        with paddle.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch['input']
                targets = batch['output']
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size_actual = inputs.shape[0]
                test_loss += loss.item() * batch_size_actual
                test_count += batch_size_actual
                
                if show_prediction_samples and batch_idx == 0:
                    logger.info("")
                    logger.info(f"  [Test Batch] PREDICTION SAMPLES")
                    
                    pred_sample = outputs[0].cpu().numpy()
                    true_sample = targets[0].cpu().numpy()
                    
                    logger.info(f"    Prediction stats (first sample):")
                    logger.info(f"      Mean: {float(np.mean(pred_sample)):.6f}")
                    logger.info(f"      Std:  {float(np.std(pred_sample)):.6f}")
                    logger.info(f"      Min:  {float(np.min(pred_sample)):.6f}")
                    logger.info(f"      Max:  {float(np.max(pred_sample)):.6f}")
                    
                    logger.info(f"    True values stats (first sample):")
                    logger.info(f"      Mean: {float(np.mean(true_sample)):.6f}")
                    logger.info(f"      Std:  {float(np.std(true_sample)):.6f}")
                    logger.info(f"      Min:  {float(np.min(true_sample)):.6f}")
                    logger.info(f"      Max:  {float(np.max(true_sample)):.6f}")
                    
                    error = np.abs(pred_sample - true_sample)
                    logger.info(f"    Error stats (first sample):")
                    logger.info(f"      Mean error: {float(np.mean(error)):.6f}")
                    logger.info(f"      Max error:  {float(np.max(error)):.6f}")
        
        avg_test_loss = test_loss / test_count if test_count > 0 else 0.0
        test_losses.append(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            logger.info("")
            logger.info(f"  [NEW BEST] Test loss improved: {avg_test_loss:.8f}")
            
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            paddle.save(model.state_dict(), model_save_path)
            logger.info(f"  Model saved to: {os.path.abspath(model_save_path)}")
        
        epoch_duration = time.time() - epoch_start_time
        
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"[EPOCH {epoch}/{epochs}] SUMMARY")
        logger.info("-" * 60)
        logger.info(f"  Train loss:    {avg_train_loss:.8f}")
        logger.info(f"  Test loss:     {avg_test_loss:.8f}")
        logger.info(f"  Best test:     {best_test_loss:.8f}")
        logger.info(f"  Learning rate: {current_lr:.2e}")
        logger.info(f"  Duration:      {epoch_duration:.2f}s")
        logger.info("-" * 60)
        
        lr_scheduler.step()
    
    total_duration = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"  Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info(f"  Final train loss: {train_losses[-1]:.8f}")
    logger.info(f"  Final test loss:  {test_losses[-1]:.8f}")
    logger.info(f"  Best test loss:   {best_test_loss:.8f}")
    logger.info("")
    logger.info("  Loss history (epoch: train_loss, test_loss):")
    for epoch, (tr_l, te_l) in enumerate(zip(train_losses, test_losses), 1):
        logger.info(f"    Epoch {epoch:3d}: train={tr_l:.8f}, test={te_l:.8f}")
    
    logger.end_phase("TRAINING LOOP")
    
    result = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'learning_rates': learning_rates,
        'best_test_loss': best_test_loss,
        'model_path': model_save_path,
        'total_duration_seconds': total_duration,
        'epochs': epochs,
        'normalization_params': dataset.get_normalization_params(),
        'gradient_stats': stats_tracker.gradient_stats,
        'param_stats': stats_tracker.param_stats,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced FNO2 Training with Python Standard Logging Module (PaddlePaddle Version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_root', type=str, default=r'C:\traework\data', help='Dataset root')
    parser.add_argument('--model_output', type=str, default='fno2_paddle_model.pth', help='Model output path')
    parser.add_argument('--log_file', type=str, default='training_paddle.log', help='Log file path')
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
    parser.add_argument('--device', type=str, default=None, help='Device (gpu/cpu)')
    
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
    temp_logger.info("FNO2 PADDLE TRAINING CONFIGURATION")
    temp_logger.info("=" * 80)
    temp_logger.info(f"Framework: PaddlePaddle")
    temp_logger.info(f"Paddle version: {paddle.__version__}")
    temp_logger.info("")
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
    
    result = train_fno2_paddle(
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
