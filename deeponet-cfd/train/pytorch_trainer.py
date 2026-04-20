import os
import sys
import numpy as np
from typing import Dict, Optional, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.base_trainer import BaseDeepONetTrainer, TrainingConfig, TrainingResult
from models.deeponet_pytorch import DeepONet
from data.cfd_bench_dataset import create_datasets, HAS_TORCH, CFDDatasetPyTorch

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchDeepONetTrainer(BaseDeepONetTrainer):
    
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. "
                "Install it with: pip install torch"
            )
        
        self.device_obj = None
        super().__init__(config)
    
    def get_framework_name(self) -> str:
        return "PyTorch"
    
    def setup_device(self) -> str:
        if self.config.device:
            device = self.config.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device_obj = torch.device(device)
        self.logger.info(f"Using device: {device}")
        
        return device
    
    def build_model(self) -> Any:
        self.model = DeepONet(
            branch_input_dim=self.actual_branch_dim,
            trunk_input_dim=self.actual_trunk_dim,
            branch_hidden_layers=self.config.branch_hidden_layers,
            trunk_hidden_layers=self.config.trunk_hidden_layers,
            output_dim=self.actual_output_dim,
            branch_activation=self.config.branch_activation,
            trunk_activation=self.config.trunk_activation,
            branch_dropout=self.config.branch_dropout,
            use_positional_encoding=self.config.use_positional_encoding,
            positional_encoding_freqs=self.config.positional_encoding_freqs,
            use_bias=self.config.use_bias
        ).to(self.device_obj)
        
        return self.model
    
    def build_optimizer(self) -> Any:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.optimizer
    
    def build_scheduler(self) -> Any:
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        return self.scheduler
    
    def build_criterion(self) -> Any:
        self.criterion = nn.MSELoss()
        return self.criterion
    
    def _create_dataset_from_arrays(
        self,
        branch_inputs: np.ndarray,
        trunk_inputs: np.ndarray,
        outputs: np.ndarray,
        normalize: bool = True
    ) -> Any:
        return CFDDatasetPyTorch(
            branch_inputs,
            trunk_inputs,
            outputs,
            normalize=normalize
        )
    
    def _create_data_loader(self, dataset: Any, shuffle: bool = True) -> Any:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def prepare_data(self) -> Tuple[Any, Any]:
        if self.config.has_external_dataset():
            self.logger.info("Using external dataset...")
            
            dataset_config = self.config.dataset_config
            
            if dataset_config.train_dataset is not None:
                self.set_datasets(
                    dataset_config.train_dataset,
                    dataset_config.test_dataset
                )
                if hasattr(dataset_config.train_dataset, 'branch_inputs'):
                    self.actual_branch_dim = dataset_config.train_dataset.branch_inputs.shape[1]
                    self.actual_trunk_dim = dataset_config.train_dataset.trunk_inputs.shape[-1]
                    self.actual_output_dim = dataset_config.train_dataset.outputs.shape[1]
                elif hasattr(dataset_config.train_dataset, '__len__'):
                    sample = next(iter(DataLoader(dataset_config.train_dataset, batch_size=1)))
                    if len(sample) >= 3:
                        self.actual_branch_dim = sample[0].shape[-1]
                        self.actual_trunk_dim = sample[1].shape[-1]
                        self.actual_output_dim = sample[2].shape[-1]
            
            elif dataset_config.dataset_path is not None:
                self.load_dataset_from_path(
                    dataset_config.dataset_path,
                    format=dataset_config.dataset_format
                )
            
            elif dataset_config.branch_inputs is not None:
                self.load_dataset_from_arrays(
                    branch_inputs=dataset_config.branch_inputs,
                    trunk_inputs=dataset_config.trunk_inputs,
                    outputs=dataset_config.outputs,
                    test_branch_inputs=dataset_config.test_branch_inputs,
                    test_trunk_inputs=dataset_config.test_trunk_inputs,
                    test_outputs=dataset_config.test_outputs
                )
            
            if self.actual_branch_dim > 0:
                self.config.branch_input_dim = self.actual_branch_dim
                self.config.trunk_input_dim = self.actual_trunk_dim
                self.config.output_dim = self.actual_output_dim
        else:
            self.logger.info("Generating synthetic dataset...")
            self.train_dataset, self.test_dataset, self.actual_branch_dim, \
                self.actual_trunk_dim, self.actual_output_dim = create_datasets(
                    n_train=self.config.n_train_samples,
                    n_test=self.config.n_test_samples,
                    grid_size=self.config.grid_size,
                    problem_type=self.config.problem_type,
                    normalize=self.config.normalize_data,
                    framework="pytorch"
                )
            
            self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True)
            self.test_loader = self._create_data_loader(self.test_dataset, shuffle=False)
            
            if self.config.normalize_data:
                self.normalization_params = self.train_dataset.get_normalization_params()
        
        return self.train_loader, self.test_loader
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        total_samples = 0
        
        for batch_idx, (branch_input, trunk_input, target) in enumerate(self.train_loader):
            branch_input = branch_input.to(self.device_obj)
            trunk_input = trunk_input.to(self.device_obj)
            target = target.to(self.device_obj)
            
            self.optimizer.zero_grad()
            
            batch_size = branch_input.size(0)
            n_points = trunk_input.size(1)
            
            output = self.model(
                branch_input,
                trunk_input.reshape(-1, self.actual_trunk_dim)
            )
            output = output.reshape_as(target)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if self.config.verbose and batch_idx % 10 == 0:
                self.train_logger.log_batch_progress(
                    batch_idx=batch_idx,
                    total_batches=len(self.train_loader),
                    loss=loss.item(),
                    mode="train"
                )
        
        self.scheduler.step()
        
        return train_loss / total_samples if total_samples > 0 else 0.0
    
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        test_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (branch_input, trunk_input, target) in enumerate(self.test_loader):
                branch_input = branch_input.to(self.device_obj)
                trunk_input = trunk_input.to(self.device_obj)
                target = target.to(self.device_obj)
                
                batch_size = branch_input.size(0)
                
                output = self.model(
                    branch_input,
                    trunk_input.reshape(-1, self.actual_trunk_dim)
                )
                output = output.reshape_as(target)
                
                loss = self.criterion(output, target)
                test_loss += loss.item() * batch_size
                total_samples += batch_size
                
                if self.config.verbose and batch_idx % 10 == 0:
                    self.train_logger.log_batch_progress(
                        batch_idx=batch_idx,
                        total_batches=len(self.test_loader),
                        loss=loss.item(),
                        mode="test"
                    )
        
        return test_loss / total_samples if total_samples > 0 else 0.0
    
    def save_model(self, filepath: str) -> str:
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        checkpoint = {
            'epoch': self.config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.result.train_losses,
            'test_losses': self.result.test_losses,
            'config': self.config.to_dict(),
            'best_train_loss': self.result.best_train_loss,
            'best_test_loss': self.result.best_test_loss,
            'normalization_params': self.normalization_params,
            'framework': self.get_framework_name()
        }
        
        torch.save(checkpoint, filepath)
        abs_path = os.path.abspath(filepath)
        
        self.train_logger.log_checkpoint(
            epoch=self.config.epochs,
            loss=self.result.best_test_loss,
            checkpoint_path=abs_path
        )
        
        return abs_path
    
    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        
        model_info = self.model.get_model_info()
        model_info['device'] = str(self.device_obj)
        
        return model_info
    
    def get_current_learning_rate(self) -> float:
        if self.optimizer is None:
            return self.config.learning_rate
        
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
        return self.config.learning_rate
    
    @staticmethod
    def load_model(filepath: str, config: Optional[TrainingConfig] = None):
        checkpoint = torch.load(filepath)
        
        if config is None:
            config_dict = checkpoint.get('config', {})
            config = TrainingConfig.from_dict(config_dict)
        
        trainer = PyTorchDeepONetTrainer(config)
        trainer.setup_device()
        trainer.actual_branch_dim = config.branch_input_dim
        trainer.actual_trunk_dim = config.trunk_input_dim
        trainer.actual_output_dim = config.output_dim
        trainer.build_model()
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer.normalization_params = checkpoint.get('normalization_params', {})
        
        return trainer
