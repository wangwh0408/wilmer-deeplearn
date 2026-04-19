from .trainer_pytorch import train_deeponet_pytorch, quick_train_pytorch
from .trainer_paddle import train_deeponet_paddle, quick_train_paddle

__all__ = [
    'train_deeponet_pytorch', 'quick_train_pytorch',
    'train_deeponet_paddle', 'quick_train_paddle'
]
