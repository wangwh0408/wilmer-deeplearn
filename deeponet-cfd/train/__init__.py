from train.base_trainer import (
    TrainingConfig,
    TrainingResult,
    BaseDeepONetTrainer
)

from train.trainer_factory import (
    TrainerFactory,
    train_deeponet,
    quick_train_deeponet
)

try:
    from train.pytorch_trainer import PyTorchDeepONetTrainer
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    PyTorchDeepONetTrainer = None

try:
    from train.paddle_trainer import PaddleDeepONetTrainer
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    PaddleDeepONetTrainer = None

__all__ = [
    'TrainingConfig',
    'TrainingResult',
    'BaseDeepONetTrainer',
    'TrainerFactory',
    'train_deeponet',
    'quick_train_deeponet',
    'PyTorchDeepONetTrainer',
    'PaddleDeepONetTrainer',
    'HAS_PYTORCH',
    'HAS_PADDLE'
]
