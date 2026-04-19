from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logger, TrainingLogger
from data.cfd_bench_dataset import CFDDataGenerator


class TrainingConfig:
    
    def __init__(
        self,
        branch_input_dim: int = 4096,
        trunk_input_dim: int = 2,
        branch_hidden_layers: List[int] = None,
        trunk_hidden_layers: List[int] = None,
        output_dim: int = 4096,
        branch_activation: str = "relu",
        trunk_activation: str = "tanh",
        branch_dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_freqs: int = 10,
        use_bias: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 100,
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.5,
        n_train_samples: int = 1000,
        n_test_samples: int = 200,
        grid_size: int = 64,
        problem_type: str = "poisson",
        normalize_data: bool = True,
        model_save_path: Optional[str] = None,
        log_file: Optional[str] = None,
        verbose: bool = True,
        device: Optional[str] = None
    ):
        if branch_hidden_layers is None:
            branch_hidden_layers = [256, 256, 256]
        if trunk_hidden_layers is None:
            trunk_hidden_layers = [256, 256, 256]
        
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.branch_hidden_layers = branch_hidden_layers
        self.trunk_hidden_layers = trunk_hidden_layers
        self.output_dim = output_dim
        self.branch_activation = branch_activation
        self.trunk_activation = trunk_activation
        self.branch_dropout = branch_dropout
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_freqs = positional_encoding_freqs
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.grid_size = grid_size
        self.problem_type = problem_type
        self.normalize_data = normalize_data
        self.model_save_path = model_save_path
        self.log_file = log_file
        self.verbose = verbose
        self.device = device
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'branch_input_dim': self.branch_input_dim,
            'trunk_input_dim': self.trunk_input_dim,
            'branch_hidden_layers': self.branch_hidden_layers,
            'trunk_hidden_layers': self.trunk_hidden_layers,
            'output_dim': self.output_dim,
            'branch_activation': self.branch_activation,
            'trunk_activation': self.trunk_activation,
            'branch_dropout': self.branch_dropout,
            'use_positional_encoding': self.use_positional_encoding,
            'positional_encoding_freqs': self.positional_encoding_freqs,
            'use_bias': self.use_bias,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'scheduler_step_size': self.scheduler_step_size,
            'scheduler_gamma': self.scheduler_gamma,
            'n_train_samples': self.n_train_samples,
            'n_test_samples': self.n_test_samples,
            'grid_size': self.grid_size,
            'problem_type': self.problem_type,
            'normalize_data': self.normalize_data,
            'model_save_path': self.model_save_path,
            'log_file': self.log_file,
            'verbose': self.verbose,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(**config_dict)


class TrainingResult:
    
    def __init__(self):
        self.model = None
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.final_train_loss: Optional[float] = None
        self.final_test_loss: Optional[float] = None
        self.best_train_loss: float = float('inf')
        self.best_test_loss: float = float('inf')
        self.model_path: Optional[str] = None
        self.config: Optional[TrainingConfig] = None
        self.framework: str = ""
        self.device: str = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'final_train_loss': self.final_train_loss,
            'final_test_loss': self.final_test_loss,
            'best_train_loss': self.best_train_loss,
            'best_test_loss': self.best_test_loss,
            'model_path': self.model_path,
            'config': self.config.to_dict() if self.config else None,
            'framework': self.framework,
            'device': self.device,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds
        }


class BaseDeepONetTrainer(ABC):
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.result = TrainingResult()
        self.result.config = config
        self.result.framework = self.get_framework_name()
        
        self.logger = setup_logger(
            name=f"DeepONet-{self.get_framework_name()}",
            log_file=config.log_file,
            console_output=config.verbose
        )
        self.train_logger = TrainingLogger(self.logger)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.actual_branch_dim = 0
        self.actual_trunk_dim = 0
        self.actual_output_dim = 0
        self.normalization_params: Dict[str, Any] = {}
    
    @abstractmethod
    def get_framework_name(self) -> str:
        pass
    
    @abstractmethod
    def setup_device(self) -> Any:
        pass
    
    @abstractmethod
    def build_model(self) -> Any:
        pass
    
    @abstractmethod
    def build_optimizer(self) -> Any:
        pass
    
    @abstractmethod
    def build_scheduler(self) -> Any:
        pass
    
    @abstractmethod
    def build_criterion(self) -> Any:
        pass
    
    @abstractmethod
    def prepare_data(self) -> Tuple[Any, Any]:
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> float:
        pass
    
    @abstractmethod
    def evaluate(self, epoch: int) -> float:
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_current_learning_rate(self) -> float:
        pass
    
    def _log_start(self):
        self.result.start_time = datetime.now()
        
        config_dict = self.config.to_dict()
        config_dict['framework'] = self.get_framework_name()
        
        self.train_logger.log_start(
            framework=self.get_framework_name(),
            config=config_dict
        )
    
    def _log_dataset_info(self):
        self.train_logger.log_dataset_info(
            train_samples=self.config.n_train_samples,
            test_samples=self.config.n_test_samples,
            input_dim=self.actual_branch_dim + self.actual_trunk_dim,
            output_dim=self.actual_output_dim,
            branch_input_dim=self.actual_branch_dim,
            trunk_input_dim=self.actual_trunk_dim
        )
    
    def _log_model_info(self):
        model_info = self.get_model_info()
        model_structure = str(self.model) if hasattr(self.model, '__str__') else None
        
        self.train_logger.log_model_info(
            model_name=f"DeepONet-{self.get_framework_name()}",
            num_params=model_info.get('total_parameters', 0),
            model_structure=model_structure
        )
    
    def _log_epoch_start(self, epoch: int):
        self.train_logger.log_epoch_start(epoch, self.config.epochs)
    
    def _log_epoch_end(self, epoch: int, train_loss: float, test_loss: float):
        metrics = {
            'learning_rate': self.get_current_learning_rate()
        }
        
        self.train_logger.log_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            metrics=metrics
        )
        
        self.result.train_losses.append(train_loss)
        self.result.test_losses.append(test_loss)
        
        if train_loss < self.result.best_train_loss:
            self.result.best_train_loss = train_loss
        if test_loss < self.result.best_test_loss:
            self.result.best_test_loss = test_loss
    
    def _log_training_complete(self):
        self.result.end_time = datetime.now()
        if self.result.start_time:
            self.result.duration_seconds = (
                self.result.end_time - self.result.start_time
            ).total_seconds()
        
        if self.result.train_losses:
            self.result.final_train_loss = self.result.train_losses[-1]
        if self.result.test_losses:
            self.result.final_test_loss = self.result.test_losses[-1]
        
        model_path = None
        if self.config.model_save_path:
            model_path = self.save_model(self.config.model_save_path)
            self.result.model_path = model_path
        
        self.train_logger.log_training_complete(
            best_train_loss=self.result.best_train_loss,
            best_test_loss=self.result.best_test_loss,
            model_save_path=model_path
        )
    
    def train(self) -> TrainingResult:
        self._log_start()
        
        self.result.device = self.setup_device()
        
        self.logger.info("Preparing dataset...")
        self.prepare_data()
        self._log_dataset_info()
        
        self.logger.info("Building model...")
        self.build_model()
        self._log_model_info()
        
        self.logger.info("Building optimizer and scheduler...")
        self.build_optimizer()
        self.build_scheduler()
        self.build_criterion()
        
        self.logger.info("Starting training loop...")
        
        for epoch in range(1, self.config.epochs + 1):
            self._log_epoch_start(epoch)
            
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(epoch)
            
            self._log_epoch_end(epoch, train_loss, test_loss)
        
        self.result.model = self.model
        self._log_training_complete()
        
        return self.result
    
    def quick_train(self, epochs: int = 10) -> TrainingResult:
        original_epochs = self.config.epochs
        original_n_train = self.config.n_train_samples
        original_n_test = self.config.n_test_samples
        original_grid_size = self.config.grid_size
        
        self.config.epochs = epochs
        self.config.n_train_samples = min(200, self.config.n_train_samples)
        self.config.n_test_samples = min(100, self.config.n_test_samples)
        self.config.grid_size = min(32, self.config.grid_size)
        
        self.logger.info(f"Quick training mode: epochs={epochs}, "
                        f"train_samples={self.config.n_train_samples}, "
                        f"grid_size={self.config.grid_size}")
        
        result = self.train()
        
        self.config.epochs = original_epochs
        self.config.n_train_samples = original_n_train
        self.config.n_test_samples = original_n_test
        self.config.grid_size = original_grid_size
        
        return result
