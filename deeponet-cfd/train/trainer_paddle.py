import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
from typing import Dict, Optional, List, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deeponet_paddle import DeepONetPaddle, DeepONetConfigPaddle
from data.cfd_bench_dataset import create_datasets, CFDDatasetPaddle
from config.logging_config import setup_logger, TrainingLogger


def train_deeponet_paddle(
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
) -> Dict:
    if branch_hidden_layers is None:
        branch_hidden_layers = [256, 256, 256]
    if trunk_hidden_layers is None:
        trunk_hidden_layers = [256, 256, 256]
    
    config = DeepONetConfigPaddle(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_layers=branch_hidden_layers,
        trunk_hidden_layers=trunk_hidden_layers,
        output_dim=output_dim,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
        branch_dropout=branch_dropout,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_freqs=positional_encoding_freqs,
        use_bias=use_bias,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma
    )
    
    logger = setup_logger(
        name="DeepONet-Paddle",
        log_file=log_file,
        console_output=verbose
    )
    train_logger = TrainingLogger(logger)
    
    if device is None:
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    
    paddle.set_device(device)
    
    train_logger.log_start(
        framework="PaddlePaddle",
        config={
            "device": device,
            "branch_input_dim": branch_input_dim,
            "trunk_input_dim": trunk_input_dim,
            "branch_hidden_layers": branch_hidden_layers,
            "trunk_hidden_layers": trunk_hidden_layers,
            "output_dim": output_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "grid_size": grid_size,
            "problem_type": problem_type
        }
    )
    
    if verbose:
        logger.info("Creating dataset...")
    
    train_dataset, test_dataset, actual_branch_dim, actual_trunk_dim, actual_output_dim = create_datasets(
        n_train=n_train_samples,
        n_test=n_test_samples,
        grid_size=grid_size,
        problem_type=problem_type,
        normalize=normalize_data,
        framework="paddle"
    )
    
    train_logger.log_dataset_info(
        train_samples=len(train_dataset),
        test_samples=len(test_dataset),
        input_dim=actual_branch_dim + actual_trunk_dim,
        output_dim=actual_output_dim,
        branch_input_dim=actual_branch_dim,
        trunk_input_dim=actual_trunk_dim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    model = DeepONetPaddle(
        branch_input_dim=actual_branch_dim,
        trunk_input_dim=actual_trunk_dim,
        branch_hidden_layers=branch_hidden_layers,
        trunk_hidden_layers=trunk_hidden_layers,
        output_dim=actual_output_dim,
        branch_activation=branch_activation,
        trunk_activation=trunk_activation,
        branch_dropout=branch_dropout,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_freqs=positional_encoding_freqs,
        use_bias=use_bias
    )
    
    train_logger.log_model_info(
        model_name="DeepONetPaddle",
        num_params=model.count_parameters(),
        model_structure=str(model)
    )
    
    criterion = nn.MSELoss()
    
    lr_scheduler = optim.lr.StepDecay(
        learning_rate=learning_rate,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )
    
    optimizer = optim.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay
    )
    
    train_losses = []
    test_losses = []
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    
    if verbose:
        logger.info("Starting training loop...")
    
    for epoch in range(1, epochs + 1):
        train_logger.log_epoch_start(epoch, epochs)
        
        model.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)
        
        for batch_idx, (branch_input, trunk_input, target) in enumerate(train_loader):
            output = model(branch_input, trunk_input.reshape((-1, actual_trunk_dim)))
            output = output.reshape(target.shape)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            train_loss += loss.item() * branch_input.shape[0]
            
            train_logger.log_batch_progress(
                batch_idx=batch_idx,
                total_batches=n_train_batches,
                loss=loss.item(),
                mode="train"
            )
        
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        
        model.eval()
        test_loss = 0.0
        n_test_batches = len(test_loader)
        
        with paddle.no_grad():
            for batch_idx, (branch_input, trunk_input, target) in enumerate(test_loader):
                output = model(branch_input, trunk_input.reshape((-1, actual_trunk_dim)))
                output = output.reshape(target.shape)
                
                loss = criterion(output, target)
                test_loss += loss.item() * branch_input.shape[0]
                
                train_logger.log_batch_progress(
                    batch_idx=batch_idx,
                    total_batches=n_test_batches,
                    loss=loss.item(),
                    mode="test"
                )
        
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        lr_scheduler.step()
        
        current_lr = lr_scheduler.get_lr()
        metrics = {"learning_rate": current_lr}
        
        train_logger.log_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            metrics=metrics
        )
    
    model_save_path_abs = None
    if model_save_path:
        save_dir = os.path.dirname(model_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        paddle.save(model.state_dict(), model_save_path)
        
        config_path = model_save_path + ".config"
        import json
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        model_save_path_abs = os.path.abspath(model_save_path)
        
        train_logger.log_checkpoint(
            epoch=epochs,
            loss=best_test_loss,
            checkpoint_path=model_save_path_abs
        )
    
    train_logger.log_training_complete(
        best_train_loss=best_train_loss,
        best_test_loss=best_test_loss,
        model_save_path=model_save_path_abs
    )
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_train_loss': best_train_loss,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'model_path': model_save_path_abs,
        'config': config.to_dict(),
        'device': device,
        'framework': 'paddle'
    }


def quick_train_paddle(
    epochs: int = 10,
    batch_size: int = 32,
    n_train_samples: int = 500,
    n_test_samples: int = 100,
    grid_size: int = 32,
    **kwargs
) -> Dict:
    return train_deeponet_paddle(
        epochs=epochs,
        batch_size=batch_size,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        grid_size=grid_size,
        **kwargs
    )
