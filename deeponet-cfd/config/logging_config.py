import logging
import sys
from datetime import datetime
from typing import Optional
import os


def setup_logger(
    name: str = "DeepONet",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.epoch_start_time = None
    
    def log_start(self, framework: str, config: dict):
        self.start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("DEEPONET TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Framework: {framework}")
        self.logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("-" * 80)
        self.logger.info("TRAINING CONFIGURATION:")
        
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("-" * 80)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        self.epoch_start_time = datetime.now()
        self.logger.info(f"\nEpoch {epoch}/{total_epochs}")
        self.logger.info("-" * 60)
    
    def log_batch_progress(
        self,
        batch_idx: int,
        total_batches: int,
        loss: float,
        mode: str = "train"
    ):
        progress = (batch_idx + 1) / total_batches * 100
        mode_str = mode.upper()
        self.logger.debug(
            f"[{mode_str}] Batch {batch_idx + 1}/{total_batches} "
            f"({progress:.1f}%) - Loss: {loss:.6f}"
        )
    
    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float = None,
        metrics: dict = None
    ):
        epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        log_msg = f"Epoch {epoch} Summary - "
        log_msg += f"Train Loss: {train_loss:.6f}"
        
        if test_loss is not None:
            log_msg += f", Test Loss: {test_loss:.6f}"
        
        if metrics:
            for key, value in metrics.items():
                log_msg += f", {key}: {value:.6f}"
        
        log_msg += f" (Duration: {epoch_duration:.2f}s)"
        
        self.logger.info(log_msg)
    
    def log_model_info(self, model_name: str, num_params: int, model_structure: str = None):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Total Parameters: {num_params:,}")
        
        if model_structure:
            self.logger.info("\nModel Structure:")
            self.logger.info(model_structure)
        
        self.logger.info("=" * 80)
    
    def log_dataset_info(
        self,
        train_samples: int,
        test_samples: int,
        input_dim: int,
        output_dim: int,
        branch_input_dim: int = None,
        trunk_input_dim: int = None
    ):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DATASET INFORMATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Training Samples: {train_samples:,}")
        self.logger.info(f"Test Samples: {test_samples:,}")
        self.logger.info(f"Input Dimension: {input_dim}")
        self.logger.info(f"Output Dimension: {output_dim}")
        
        if branch_input_dim:
            self.logger.info(f"Branch Input Dimension: {branch_input_dim}")
        if trunk_input_dim:
            self.logger.info(f"Trunk Input Dimension: {trunk_input_dim}")
        
        self.logger.info("=" * 80)
    
    def log_training_complete(
        self,
        best_train_loss: float,
        best_test_loss: float,
        model_save_path: str = None
    ):
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total Duration: {hours}h {minutes}m {seconds}s")
        self.logger.info(f"Best Train Loss: {best_train_loss:.10f}")
        self.logger.info(f"Best Test Loss: {best_test_loss:.10f}")
        
        if model_save_path:
            self.logger.info(f"Model Saved At: {model_save_path}")
        
        self.logger.info("=" * 80)
    
    def log_error(self, error_msg: str, exception: Exception = None):
        self.logger.error("!" * 80)
        self.logger.error("TRAINING ERROR")
        self.logger.error("!" * 80)
        self.logger.error(f"Error Message: {error_msg}")
        
        if exception:
            self.logger.error(f"Exception Type: {type(exception).__name__}")
            import traceback
            self.logger.error(f"Stack Trace:\n{traceback.format_exc()}")
        
        self.logger.error("!" * 80)
    
    def log_checkpoint(self, epoch: int, loss: float, checkpoint_path: str):
        self.logger.info(
            f"\n[CHECKPOINT] Epoch {epoch} - "
            f"Loss: {loss:.6f} - Saved to: {checkpoint_path}"
        )
