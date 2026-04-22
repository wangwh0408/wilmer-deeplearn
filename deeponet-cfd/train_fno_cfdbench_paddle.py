import os
import sys
import argparse
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.optimizer.lr import StepDecay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fno_cfdbench_paddle import AutoRegressiveFNOPaddle, count_parameters_paddle


def setup_logger(log_file: Optional[str] = None, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("FNO_Training")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class CFDBCase:
    
    def __init__(self, case_path: str, name: str = None):
        self.case_path = case_path
        self.name = name or os.path.basename(case_path)
        self.u_data = None
        self.v_data = None
        self.p_data = None
        self.time_steps = 0
        self.grid_size = (0, 0)
        
    def load(self):
        u_path = os.path.join(self.case_path, 'u.npy')
        v_path = os.path.join(self.case_path, 'v.npy')
        p_path = os.path.join(self.case_path, 'p.npy')
        
        if os.path.exists(u_path):
            self.u_data = np.load(u_path).astype(np.float32)
        if os.path.exists(v_path):
            self.v_data = np.load(v_path).astype(np.float32)
        if os.path.exists(p_path):
            self.p_data = np.load(p_path).astype(np.float32)
        
        if self.u_data is not None:
            if self.u_data.ndim == 3:
                self.time_steps = self.u_data.shape[0]
                self.grid_size = (self.u_data.shape[1], self.u_data.shape[2])
            elif self.u_data.ndim == 2:
                self.time_steps = 1
                self.grid_size = self.u_data.shape
            elif self.u_data.ndim == 4:
                self.time_steps = self.u_data.shape[0]
                self.grid_size = (self.u_data.shape[1], self.u_data.shape[2])
        
        return self
    
    def get_velocity_field(self, time_idx: int = 0) -> np.ndarray:
        if self.u_data is None or self.v_data is None:
            return None
        
        if self.u_data.ndim == 3:
            u = self.u_data[time_idx]
            v = self.v_data[time_idx]
        elif self.u_data.ndim == 4:
            u = self.u_data[time_idx, :, :, 0]
            v = self.v_data[time_idx, :, :, 0]
        else:
            u = self.u_data
            v = self.v_data
        
        return np.stack([u, v], axis=-1)


class CFDBCategory:
    
    CATEGORIES = ['bc', 'geo', 'prop']
    
    def __init__(self, problem_path: str, category: str):
        self.problem_path = problem_path
        self.category = category
        self.category_path = os.path.join(problem_path, category)
        self.cases: List[CFDBCase] = []
        
    def load(self, max_cases: int = None, logger: Optional[logging.Logger] = None):
        import glob
        
        if not os.path.exists(self.category_path):
            if logger:
                logger.warning(f"Category directory not found: {self.category_path}")
            return self
        
        case_dirs = sorted(glob.glob(os.path.join(self.category_path, 'case*')))
        
        if max_cases:
            case_dirs = case_dirs[:max_cases]
        
        if logger:
            logger.info(f"  Loading {len(case_dirs)} cases from {self.category_path}")
        
        for i, case_dir in enumerate(case_dirs):
            case = CFDBCase(case_dir).load()
            if case.u_data is not None:
                self.cases.append(case)
                if logger and i % 10 == 0:
                    logger.debug(f"    Loaded case {case.name}: {case.time_steps} time steps, grid={case.grid_size}")
        
        if logger:
            logger.info(f"  Successfully loaded {len(self.cases)} cases")
        
        return self
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        return self.cases[idx]


class CFDBenchDataset:
    
    PROBLEMS = ['cavity', 'tube', 'dam', 'cylinder']
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        input_steps: int = 1,
        output_steps: int = 1,
        step_interval: int = 1,
        include_pressure: bool = False,
        normalize: bool = True,
        max_cases_per_category: int = None,
        logger: Optional[logging.Logger] = None
    ):
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.step_interval = step_interval
        self.include_pressure = include_pressure
        self.normalize = normalize
        self.max_cases_per_category = max_cases_per_category
        self.logger = logger
        
        self.all_cases: List[CFDBCase] = []
        self.samples: List[Dict] = []
        
        self.mean_u = 0.0
        self.std_u = 1.0
        self.mean_v = 0.0
        self.std_v = 1.0
        self.mean_p = 0.0
        self.std_p = 1.0
        
    def load(self):
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("LOADING CFDBENCH DATASET")
            self.logger.info("=" * 80)
            self.logger.info(f"Data root: {self.data_root}")
            self.logger.info(f"Problems: {self.problems}")
            self.logger.info(f"Categories: {self.categories}")
            self.logger.info(f"Input steps: {self.input_steps}")
            self.logger.info(f"Output steps: {self.output_steps}")
            self.logger.info(f"Max cases per category: {self.max_cases_per_category or 'all'}")
            self.logger.info("=" * 80)
        
        for problem in self.problems:
            problem_path = os.path.join(self.data_root, problem)
            if not os.path.exists(problem_path):
                if self.logger:
                    self.logger.warning(f"Problem directory not found: {problem_path}")
                continue
            
            if self.logger:
                self.logger.info(f"\nLoading problem: {problem}")
            
            for category in self.categories:
                if self.logger:
                    self.logger.info(f"\n  Category: {category}")
                
                cat = CFDBCategory(problem_path, category)
                cat.load(max_cases=self.max_cases_per_category, logger=self.logger)
                
                for case in cat.cases:
                    self.all_cases.append(case)
                    
                    if case.time_steps >= self.input_steps + self.output_steps:
                        max_start = case.time_steps - self.input_steps - self.output_steps + 1
                        for start_t in range(0, max_start, self.step_interval):
                            self.samples.append({
                                'case': case,
                                'start_t': start_t
                            })
        
        if self.logger:
            self.logger.info(f"\n" + "=" * 80)
            self.logger.info("DATASET SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total cases loaded: {len(self.all_cases)}")
            self.logger.info(f"Total training samples: {len(self.samples)}")
        
        if self.normalize and len(self.all_cases) > 0:
            self._compute_normalization()
        
        return self
    
    def _compute_normalization(self):
        if self.logger:
            self.logger.info("\nComputing data normalization...")
        
        all_u = []
        all_v = []
        all_p = []
        
        for case in self.all_cases:
            if case.u_data is not None:
                all_u.append(case.u_data.flatten())
            if case.v_data is not None:
                all_v.append(case.v_data.flatten())
            if case.p_data is not None and self.include_pressure:
                all_p.append(case.p_data.flatten())
        
        if all_u:
            all_u = np.concatenate(all_u)
            self.mean_u = np.mean(all_u)
            self.std_u = np.std(all_u) + 1e-8
        
        if all_v:
            all_v = np.concatenate(all_v)
            self.mean_v = np.mean(all_v)
            self.std_v = np.std(all_v) + 1e-8
        
        if all_p:
            all_p = np.concatenate(all_p)
            self.mean_p = np.mean(all_p)
            self.std_p = np.std(all_p) + 1e-8
        
        if self.logger:
            self.logger.info(f"  Normalization parameters:")
            self.logger.info(f"    u: mean={self.mean_u:.6f}, std={self.std_u:.6f}")
            self.logger.info(f"    v: mean={self.mean_v:.6f}, std={self.std_v:.6f}")
            if self.include_pressure:
                self.logger.info(f"    p: mean={self.mean_p:.6f}, std={self.std_p:.6f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        case = sample['case']
        start_t = sample['start_t']
        
        input_fields = []
        output_fields = []
        
        for t in range(self.input_steps):
            vel = case.get_velocity_field(start_t + t)
            if self.normalize:
                u = (vel[..., 0] - self.mean_u) / self.std_u
                v = (vel[..., 1] - self.mean_v) / self.std_v
            else:
                u = vel[..., 0]
                v = vel[..., 1]
            input_fields.append(np.stack([u, v], axis=-1))
        
        for t in range(self.input_steps, self.input_steps + self.output_steps):
            vel = case.get_velocity_field(start_t + t)
            if self.normalize:
                u = (vel[..., 0] - self.mean_u) / self.std_u
                v = (vel[..., 1] - self.mean_v) / self.std_v
            else:
                u = vel[..., 0]
                v = vel[..., 1]
            output_fields.append(np.stack([u, v], axis=-1))
        
        input_tensor = np.stack(input_fields, axis=0)
        output_tensor = np.stack(output_fields, axis=0)
        
        input_tensor = input_tensor.reshape(-1, *input_tensor.shape[-2:])
        output_tensor = output_tensor.reshape(-1, *output_tensor.shape[-2:])
        
        return {
            'input': input_tensor.astype(np.float32),
            'output': output_tensor.astype(np.float32),
            'case_name': case.name,
            'start_t': start_t
        }
    
    def get_normalization_params(self) -> Dict[str, float]:
        return {
            'mean_u': self.mean_u,
            'std_u': self.std_u,
            'mean_v': self.mean_v,
            'std_v': self.std_v,
            'mean_p': self.mean_p,
            'std_p': self.std_p
        }


def create_cfdbench_dataloaders(
    data_root: str,
    problems: List[str] = None,
    categories: List[str] = None,
    input_steps: int = 1,
    output_steps: int = 1,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    normalize: bool = True,
    max_cases_per_category: int = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[paddle.io.DataLoader, paddle.io.DataLoader, Dict[str, float]]:
    
    dataset = CFDBenchDataset(
        data_root=data_root,
        problems=problems,
        categories=categories,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=normalize,
        max_cases_per_category=max_cases_per_category,
        logger=logger
    )
    dataset.load()
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    if logger:
        logger.info(f"\nSplitting dataset: train={n_train}, test={n_test}")
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    from paddle.io import Subset
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    def collate_fn(batch):
        inputs = paddle.to_tensor([item['input'] for item in batch], dtype='float32')
        outputs = paddle.to_tensor([item['output'] for item in batch], dtype='float32')
        case_names = [item['case_name'] for item in batch]
        start_ts = [item['start_t'] for item in batch]
        
        return {
            'input': inputs,
            'output': outputs,
            'case_name': case_names,
            'start_t': start_ts
        }
    
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, dataset.get_normalization_params()


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
        normalize: bool = True,
        seed: int = 42
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
        self.seed = seed


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


class CFDBenchTrainerPaddle:
    
    def __init__(
        self,
        config: TrainingConfig,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        max_cases_per_category: int = None,
        log_file: Optional[str] = None,
        verbose: bool = True
    ):
        self.config = config
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.max_cases_per_category = max_cases_per_category
        
        paddle.seed(config.seed)
        np.random.seed(config.seed)
        
        self.logger = setup_logger(log_file=log_file, verbose=verbose)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None
        self.normalization_params = None
        
        self.result = TrainingResult()
    
    def setup_data(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[1/5] SETTING UP DATA")
        self.logger.info("=" * 80)
        
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
            logger=self.logger
        )
        
        self.result.normalization_params = self.normalization_params
        
        return self.train_loader, self.test_loader
    
    def build_model(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[2/5] BUILDING FNO MODEL")
        self.logger.info("=" * 80)
        
        in_channels = 2 * self.config.input_steps if self.config.input_steps > 1 else 2
        out_channels = 2 * self.config.output_steps if self.config.output_steps > 1 else 2
        
        self.logger.info(f"  Model configuration:")
        self.logger.info(f"    Fourier modes: ({self.config.modes1}, {self.config.modes2})")
        self.logger.info(f"    Width: {self.config.width}")
        self.logger.info(f"    Number of layers: {self.config.n_layers}")
        self.logger.info(f"    Hidden dimension: {self.config.hidden_dim}")
        self.logger.info(f"    Input channels: {in_channels}")
        self.logger.info(f"    Output channels: {out_channels}")
        self.logger.info(f"    Use coordinates: {self.config.use_coordinates}")
        
        self.model = AutoRegressiveFNOPaddle(
            modes1=self.config.modes1,
            modes2=self.config.modes2,
            width=self.config.width,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
            use_coordinates=self.config.use_coordinates,
            use_mask=self.config.use_mask
        )
        
        total_params = count_parameters_paddle(self.model)
        self.logger.info(f"\n  Total parameters: {total_params:,}")
        
        return self.model
    
    def build_optimizer(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[3/5] SETTING UP OPTIMIZER AND SCHEDULER")
        self.logger.info("=" * 80)
        
        self.logger.info(f"  Optimizer: Adam")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Weight decay: {self.config.weight_decay}")
        
        lr_scheduler = StepDecay(
            learning_rate=self.config.learning_rate,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        
        self.optimizer = optim.Adam(
            parameters=self.model.parameters(),
            learning_rate=lr_scheduler,
            weight_decay=self.config.weight_decay
        )
        
        self.logger.info(f"\n  Learning rate scheduler:")
        self.logger.info(f"    Step size: {self.config.scheduler_step_size} epochs")
        self.logger.info(f"    Gamma: {self.config.scheduler_gamma}")
        
        return self.optimizer, lr_scheduler
    
    def build_criterion(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[4/5] SETTING UP LOSS FUNCTION")
        self.logger.info("=" * 80)
        
        self.criterion = nn.MSELoss()
        self.logger.info(f"  Loss function: MSE Loss")
        
        return self.criterion
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        train_count = 0
        
        n_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input']
            targets = batch['output']
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            
            batch_size = inputs.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size
            
            if batch_idx % max(1, n_batches // 10) == 0:
                self.logger.info(
                    f"    Batch {batch_idx}/{n_batches} - "
                    f"Loss: {loss.item():.6f}"
                )
        
        return train_loss / train_count if train_count > 0 else 0.0
    
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        test_loss = 0.0
        test_count = 0
        
        n_batches = len(self.test_loader)
        
        with paddle.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['input']
                targets = batch['output']
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                batch_size = inputs.shape[0]
                test_loss += loss.item() * batch_size
                test_count += batch_size
        
        return test_loss / test_count if test_count > 0 else 0.0
    
    def save_model(self, filepath: str) -> str:
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        paddle.save(self.model.state_dict(), filepath)
        abs_path = os.path.abspath(filepath)
        
        self.logger.info(f"\n  Model saved to: {abs_path}")
        
        return abs_path
    
    def train(self, model_save_path: Optional[str] = None) -> TrainingResult:
        self.result.start_time = datetime.now()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FNO TRAINING STARTED (PaddlePaddle)")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {self.result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Data root: {self.data_root}")
        self.logger.info(f"Problems: {self.problems}")
        self.logger.info(f"Categories: {self.categories}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Random seed: {self.config.seed}")
        self.logger.info("=" * 80)
        
        self.setup_data()
        self.build_model()
        self.build_optimizer()
        self.build_criterion()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[5/5] STARTING TRAINING LOOP")
        self.logger.info("=" * 80)
        
        start_train_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            self.logger.info(f"\n{'='*40} Epoch {epoch}/{self.config.epochs} {'='*40}")
            
            current_lr = self.optimizer.get_lr()
            self.logger.info(f"  Learning rate: {current_lr:.2e}")
            
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(epoch)
            
            self.result.train_losses.append(train_loss)
            self.result.test_losses.append(test_loss)
            
            if train_loss < self.result.best_train_loss:
                self.result.best_train_loss = train_loss
            if test_loss < self.result.best_test_loss:
                self.result.best_test_loss = test_loss
            
            epoch_duration = time.time() - epoch_start_time
            
            self.logger.info(f"\n  Epoch {epoch} Summary:")
            self.logger.info(f"    Train Loss: {train_loss:.6f}")
            self.logger.info(f"    Test Loss:  {test_loss:.6f}")
            self.logger.info(f"    Best Test:  {self.result.best_test_loss:.6f}")
            self.logger.info(f"    Duration:   {epoch_duration:.2f}s")
            
            self.optimizer.step()
        
        total_train_time = time.time() - start_train_time
        
        self.result.end_time = datetime.now()
        self.result.duration_seconds = total_train_time
        
        if model_save_path:
            self.result.model_path = self.save_model(model_save_path)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\nFinal Results:")
        self.logger.info(f"  Final train loss: {self.result.train_losses[-1]:.6f}")
        self.logger.info(f"  Final test loss:  {self.result.test_losses[-1]:.6f}")
        self.logger.info(f"  Best train loss:  {self.result.best_train_loss:.6f}")
        self.logger.info(f"  Best test loss:   {self.result.best_test_loss:.6f}")
        self.logger.info(f"  Total training time: {self.result.duration_seconds:.2f} seconds")
        
        self.logger.info(f"\nLoss History:")
        for epoch, (tr_l, te_l) in enumerate(zip(self.result.train_losses, self.result.test_losses), 1):
            self.logger.info(f"  Epoch {epoch:3d}: train={tr_l:.6f}, test={te_l:.6f}")
        
        self.logger.info("\n" + "=" * 80)
        
        return self.result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FNO model on CFDBench dataset using PaddlePaddle",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"c:\traework\a\wilmer-deeplearn\data",
        help="Root directory containing CFDBench data"
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
        default=["prop"],
        choices=["bc", "geo", "prop"],
        help="Categories to use: bc, geo, prop (default: prop)"
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
        default=8,
        help="Number of Fourier modes in x direction (default: 8)"
    )
    
    parser.add_argument(
        "--modes2",
        type=int,
        default=8,
        help="Number of Fourier modes in y direction (default: 8)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=16,
        help="Hidden dimension width (default: 16)"
    )
    
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of FNO blocks (default: 2)"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension in final layers (default: 64)"
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
        default=4,
        help="Batch size (default: 4)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=20,
        help="Step size for learning rate scheduler (default: 20)"
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save trained model (default: None)"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to save log file (default: None)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("FNO TRAINING WITH PADDLEPADDLE")
    print("=" * 80)
    
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
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    trainer = CFDBenchTrainerPaddle(
        config=config,
        data_root=args.data_root,
        problems=args.problems,
        categories=args.categories,
        max_cases_per_category=args.max_cases,
        log_file=args.log_file,
        verbose=args.verbose
    )
    
    result = trainer.train(model_save_path=args.model_save_path)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
