from .deeponet_pytorch import BranchNet, TrunkNet, DeepONet, DeepONetConfig
from .deeponet_paddle import BranchNetPaddle, TrunkNetPaddle, DeepONetPaddle, DeepONetConfigPaddle

__all__ = [
    'BranchNet', 'TrunkNet', 'DeepONet', 'DeepONetConfig',
    'BranchNetPaddle', 'TrunkNetPaddle', 'DeepONetPaddle', 'DeepONetConfigPaddle'
]
