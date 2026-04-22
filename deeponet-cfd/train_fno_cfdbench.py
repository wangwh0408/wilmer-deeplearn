import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cfdbench_loader import CFDBenchDataset, create_cfdbench_dataloaders
from models.fno_cfdbench import AutoRegressiveFNO, count_parameters


class TrainingConfig:
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        epochs: int = 100,
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.5,
        input_steps: int = 1,
        output_steps: int = 1,
        train_ratio: float = 0.8,
        use_coordinates: bool = True,
        use_mask: bool = False,
        normalize: bool = True
    ):
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.train_ratio = train_ratio
        self.use_coordinates = use_coordinates
        self.use_mask = use_mask
        self.normalize = normalize


class TrainingResult:
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.best_train_loss: float = float('inf')
        self.best_test_loss: float = float('inf')
        self.model_path: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_seconds: float = 0.0
        self.normalization_params: Dict[str, float] = {}


class CFDBenchTrainer:
    
    def __init__(
        self,
        config: TrainingConfig,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        max_cases_per_category: int = None,
        device: Optional[str] = None
    ):
        self.config = config
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.max_cases_per_category = max_cases_per_category
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.normalization_params = None
        
        self.result = TrainingResult()
    
    def setup_data(self):
        print(f"Loading data from: {self.data_root}")
        
        self.train_loader, self.test_loader, self.normalization_params = create_cfdbench_dataloaders(
            data_root=self.data_root,
            problems=self.problems,
            categories=self.categories,
            input_steps=self.config.input_steps,
            output_steps=self.config.output_steps,
            batch_size=self.config.batch_size,
            train_ratio=self.config.train_ratio,
            normalize=self.config.normalize,
            max_cases_per_category=self.max_cases_per_category,
            num_workers=0
        )
        
        self.result.normalization_params = self.normalization_params
        
        return self.train_loader, self.test_loader
    
    def build_model(self):
        print(f"Building FNO model...")
        
        in_channels = 2 * self.config.input_steps
        out_channels = 2 * self.config.output_steps
        
        self.model = AutoRegressiveFNO(
            modes1=self.config.modes1,
            modes2=self.config.modes2,
            width=self.config.width,
            in_channels=in_channels if self.config.input_steps > 1 else 2,
            out_channels=out_channels if self.config.output_steps > 1 else 2,
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
            use_coordinates=self.config.use_coordinates,
            use_mask=self.config.use_mask
        ).to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        return self.model
    
    def build_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return self.optimizer
    
    def build_scheduler(self):
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        return self.scheduler
    
    def build_criterion(self):
        self.criterion = nn.MSELoss()
        return self.criterion
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: loss={loss.item():.6f}")
        
        return train_loss / total_samples if total_samples > 0 else 0.0
    
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        test_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                test_loss += loss.item() * batch_size
                total_samples += batch_size
        
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
            'best_train_loss': self.result.best_train_loss,
            'best_test_loss': self.result.best_test_loss,
            'normalization_params': self.normalization_params,
            'config': {
                'modes1': self.config.modes1,
                'modes2': self.config.modes2,
                'width': self.config.width,
                'n_layers': self.config.n_layers,
                'hidden_dim': self.config.hidden_dim,
                'input_steps': self.config.input_steps,
                'output_steps': self.config.output_steps,
                'use_coordinates': self.config.use_coordinates,
                'use_mask': self.config.use_mask
            }
        }
        
        torch.save(checkpoint, filepath)
        abs_path = os.path.abspath(filepath)
        
        print(f"Model saved to: {abs_path}")
        
        return abs_path
    
    def train(self, model_save_path: Optional[str] = None) -> TrainingResult:
        self.result.start_time = datetime.now()
        
        print("=" * 80)
        print("FNO TRAINING STARTED")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Problems: {self.problems}")
        print(f"Categories: {self.categories}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 80)
        
        print("\n[1/6] Setting up data...")
        self.setup_data()
        
        print("\n[2/6] Building model...")
        self.build_model()
        
        print("\n[3/6] Building optimizer...")
        self.build_optimizer()
        
        print("\n[4/6] Building scheduler...")
        self.build_scheduler()
        
        print("\n[5/6] Building criterion...")
        self.build_criterion()
        
        print("\n[6/6] Starting training loop...")
        print("-" * 80)
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(epoch)
            
            self.scheduler.step()
            
            self.result.train_losses.append(train_loss)
            self.result.test_losses.append(test_loss)
            
            if train_loss < self.result.best_train_loss:
                self.result.best_train_loss = train_loss
            if test_loss < self.result.best_test_loss:
                self.result.best_test_loss = test_loss
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Test loss:  {test_loss:.6f}")
            print(f"  Best test:  {self.result.best_test_loss:.6f}")
            print(f"  Learning rate: {current_lr:.2e}")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        
        self.result.end_time = datetime.now()
        if self.result.start_time:
            self.result.duration_seconds = (
                self.result.end_time - self.result.start_time
            ).total_seconds()
        
        if model_save_path:
            self.result.model_path = self.save_model(model_save_path)
        
        print(f"Final train loss: {self.result.train_losses[-1]:.6f}")
        print(f"Final test loss:  {self.result.test_losses[-1]:.6f}")
        print(f"Best train loss:  {self.result.best_train_loss:.6f}")
        print(f"Best test loss:   {self.result.best_test_loss:.6f}")
        print(f"Duration:         {self.result.duration_seconds:.2f} seconds")
        print("=" * 80)
        
        return self.result
    
    @staticmethod
    def load_model(
        filepath: str,
        device: Optional[str] = None
    ) -> Tuple[AutoRegressiveFNO, Dict]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint.get('config', {})
        
        model = AutoRegressiveFNO(
            modes1=config.get('modes1', 12),
            modes2=config.get('modes2', 12),
            width=config.get('width', 32),
            in_channels=2,
            out_channels=2,
            n_layers=config.get('n_layers', 4),
            hidden_dim=config.get('hidden_dim', 128),
            use_coordinates=config.get('use_coordinates', True),
            use_mask=config.get('use_mask', False)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        info = {
            'normalization_params': checkpoint.get('normalization_params', {}),
            'train_losses': checkpoint.get('train_losses', []),
            'test_losses': checkpoint.get('test_losses', []),
            'best_train_loss': checkpoint.get('best_train_loss', float('inf')),
            'best_test_loss': checkpoint.get('best_test_loss', float('inf'))
        }
        
        return model, info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FNO model on CFDBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="Categories to use: bc (boundary conditions), geo (geometry), prop (physical properties) (default: bc)"
    )
    
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Maximum number of cases per category (default: all)"
    )
    
    parser.add_argument(
        "--modes1",
        type=int,
        default=12,
        help="Number of Fourier modes in x direction (default: 12)"
    )
    
    parser.add_argument(
        "--modes2",
        type=int,
        default=12,
        help="Number of Fourier modes in y direction (default: 12)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Hidden dimension width (default: 32)"
    )
    
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of FNO blocks (default: 4)"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension in final layers (default: 128)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=50,
        help="Step size for learning rate scheduler (default: 50)"
    )
    
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Gamma for learning rate scheduler (default: 0.5)"
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
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)"
    )
    
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save trained model (default: None)"
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
    
    config = TrainingConfig(
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        train_ratio=args.train_ratio
    )
    
    trainer = CFDBenchTrainer(
        config=config,
        data_root=args.data_root,
        problems=args.problems,
        categories=args.categories,
        max_cases_per_category=args.max_cases,
        device=args.device
    )
    
    result = trainer.train(model_save_path=args.model_save_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
