import os
import sys
from typing import Dict, Optional, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.base_trainer import BaseDeepONetTrainer, TrainingConfig, TrainingResult
from models.deeponet_paddle import DeepONetPaddle
from data.cfd_bench_dataset import create_datasets, HAS_PADDLE

try:
    import paddle
    import paddle.nn as nn
    import paddle.optimizer as optim
    from paddle.io import DataLoader
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class PaddleDeepONetTrainer(BaseDeepONetTrainer):
    
    def __init__(self, config: TrainingConfig):
        if not PADDLE_AVAILABLE:
            raise ImportError(
                "PaddlePaddle is not installed. "
                "Install it with: pip install paddlepaddle"
            )
        
        super().__init__(config)
    
    def get_framework_name(self) -> str:
        return "PaddlePaddle"
    
    def setup_device(self) -> str:
        if self.config.device:
            device = self.config.device
        else:
            device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        
        paddle.set_device(device)
        self.logger.info(f"Using device: {device}")
        
        return device
    
    def build_model(self) -> Any:
        self.model = DeepONetPaddle(
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
        )
        
        return self.model
    
    def build_optimizer(self) -> Any:
        lr_scheduler = optim.lr.StepDecay(
            learning_rate=self.config.learning_rate,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        
        self.optimizer = optim.Adam(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = lr_scheduler
        
        return self.optimizer
    
    def build_scheduler(self) -> Any:
        return self.scheduler
    
    def build_criterion(self) -> Any:
        self.criterion = nn.MSELoss()
        return self.criterion
    
    def prepare_data(self) -> Tuple[Any, Any]:
        self.train_dataset, self.test_dataset, self.actual_branch_dim, \
            self.actual_trunk_dim, self.actual_output_dim = create_datasets(
                n_train=self.config.n_train_samples,
                n_test=self.config.n_test_samples,
                grid_size=self.config.grid_size,
                problem_type=self.config.problem_type,
                normalize=self.config.normalize_data,
                framework="paddle"
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        if self.config.normalize_data:
            self.normalization_params = self.train_dataset.get_normalization_params()
        
        return self.train_loader, self.test_loader
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        total_samples = 0
        
        for batch_idx, (branch_input, trunk_input, target) in enumerate(self.train_loader):
            batch_size = branch_input.shape[0]
            n_points = trunk_input.shape[1]
            
            output = self.model(
                branch_input,
                trunk_input.reshape((-1, self.actual_trunk_dim))
            )
            output = output.reshape(target.shape)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            
            train_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if self.config.verbose and batch_idx % 10 == 0:
                self.train_logger.log_batch_progress(
                    batch_idx=batch_idx,
                    total_batches=len(self.train_loader),
                    loss=loss.item(),
                    mode="train"
                )
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return train_loss / total_samples if total_samples > 0 else 0.0
    
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        test_loss = 0.0
        total_samples = 0
        
        with paddle.no_grad():
            for batch_idx, (branch_input, trunk_input, target) in enumerate(self.test_loader):
                batch_size = branch_input.shape[0]
                
                output = self.model(
                    branch_input,
                    trunk_input.reshape((-1, self.actual_trunk_dim))
                )
                output = output.reshape(target.shape)
                
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
        
        paddle.save(self.model.state_dict(), filepath)
        
        import json
        config_path = filepath + ".config"
        config_data = {
            'epoch': self.config.epochs,
            'train_losses': self.result.train_losses,
            'test_losses': self.result.test_losses,
            'config': self.config.to_dict(),
            'best_train_loss': float(self.result.best_train_loss),
            'best_test_loss': float(self.result.best_test_loss),
            'normalization_params': {k: v.tolist() if hasattr(v, 'tolist') else v 
                                      for k, v in self.normalization_params.items()},
            'framework': self.get_framework_name()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
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
        model_info['device'] = paddle.get_device()
        
        return model_info
    
    def get_current_learning_rate(self) -> float:
        if self.scheduler is None:
            return self.config.learning_rate
        
        return self.scheduler.get_lr()
    
    @staticmethod
    def load_model(filepath: str, config: Optional[TrainingConfig] = None):
        import json
        
        config_path = filepath + ".config"
        if os.path.exists(config_path) and config is None:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                config_dict = config_data.get('config', {})
                config = TrainingConfig.from_dict(config_dict)
        
        if config is None:
            raise ValueError("No config provided and no config file found")
        
        trainer = PaddleDeepONetTrainer(config)
        trainer.setup_device()
        trainer.actual_branch_dim = config.branch_input_dim
        trainer.actual_trunk_dim = config.trunk_input_dim
        trainer.actual_output_dim = config.output_dim
        trainer.build_model()
        trainer.model.set_state_dict(paddle.load(filepath))
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                trainer.normalization_params = config_data.get('normalization_params', {})
        
        return trainer
