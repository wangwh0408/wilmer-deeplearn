import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import List, Optional, Tuple


class BranchNetPaddle(nn.Layer):
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super(BranchNetPaddle, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = self._get_activation(activation)
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str) -> nn.Layer:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "selu": nn.SELU()
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.network(x)


class TrunkNetPaddle(nn.Layer):
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "tanh",
        use_positional_encoding: bool = True,
        positional_encoding_freqs: int = 10
    ):
        super(TrunkNetPaddle, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = self._get_activation(activation)
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_freqs = positional_encoding_freqs
        
        if use_positional_encoding:
            self.actual_input_dim = input_dim * (2 * positional_encoding_freqs + 1)
        else:
            self.actual_input_dim = input_dim
        
        layers = []
        prev_dim = self.actual_input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str) -> nn.Layer:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "selu": nn.SELU()
        }
        return activations.get(activation_name.lower(), nn.Tanh())
    
    def _positional_encoding(self, x: paddle.Tensor) -> paddle.Tensor:
        if not self.use_positional_encoding:
            return x
        
        batch_size, coord_dim = x.shape
        encoding = [x]
        
        for i in range(self.positional_encoding_freqs):
            freq = 2 ** i
            encoding.append(paddle.sin(freq * paddle.pi * x))
            encoding.append(paddle.cos(freq * paddle.pi * x))
        
        return paddle.concat(encoding, axis=-1)
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._positional_encoding(x)
        return self.network(x)


class DeepONetPaddle(nn.Layer):
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden_layers: List[int],
        trunk_hidden_layers: List[int],
        output_dim: int,
        branch_activation: str = "relu",
        trunk_activation: str = "tanh",
        branch_dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_freqs: int = 10,
        use_bias: bool = True
    ):
        super(DeepONetPaddle, self).__init__()
        
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.branch_net = BranchNetPaddle(
            input_dim=branch_input_dim,
            hidden_layers=branch_hidden_layers,
            output_dim=output_dim,
            activation=branch_activation,
            dropout=branch_dropout
        )
        
        self.trunk_net = TrunkNetPaddle(
            input_dim=trunk_input_dim,
            hidden_layers=trunk_hidden_layers,
            output_dim=output_dim,
            activation=trunk_activation,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_freqs=positional_encoding_freqs
        )
        
        if use_bias:
            self.bias = self.create_parameter(
                shape=(output_dim,),
                dtype='float32',
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None
    
    def forward(
        self,
        branch_input: paddle.Tensor,
        trunk_input: paddle.Tensor
    ) -> paddle.Tensor:
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        
        if branch_out.ndim == 2 and trunk_out.ndim == 2:
            output = paddle.sum(branch_out * trunk_out, axis=-1, keepdim=True)
        elif branch_out.ndim == 2 and trunk_out.ndim == 3:
            output = paddle.einsum("bd,bnd->bn", branch_out, trunk_out)
            output = output.unsqueeze(-1)
        else:
            output = paddle.sum(branch_out * trunk_out, axis=-1, keepdim=True)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def count_parameters(self) -> int:
        return sum(p.numel().item() for p in self.parameters() if not p.stop_gradient)
    
    def get_model_info(self) -> dict:
        return {
            "branch_input_dim": self.branch_input_dim,
            "trunk_input_dim": self.trunk_input_dim,
            "output_dim": self.output_dim,
            "branch_layers": self.branch_net.hidden_layers,
            "trunk_layers": self.trunk_net.hidden_layers,
            "total_parameters": self.count_parameters(),
            "use_bias": self.use_bias,
            "use_positional_encoding": self.trunk_net.use_positional_encoding
        }
    
    def __str__(self) -> str:
        info = self.get_model_info()
        return (
            f"DeepONetPaddle(\n"
            f"  Branch Input Dim: {info['branch_input_dim']}\n"
            f"  Trunk Input Dim:  {info['trunk_input_dim']}\n"
            f"  Output Dim:       {info['output_dim']}\n"
            f"  Branch Layers:    {info['branch_layers']}\n"
            f"  Trunk Layers:     {info['trunk_layers']}\n"
            f"  Total Parameters: {info['total_parameters']:,}\n"
            f"  Use Bias:         {info['use_bias']}\n"
            f"  Positional Encoding: {info['use_positional_encoding']}\n"
            f")"
        )


class DeepONetConfigPaddle:
    
    def __init__(
        self,
        branch_input_dim: int = 100,
        trunk_input_dim: int = 2,
        branch_hidden_layers: List[int] = None,
        trunk_hidden_layers: List[int] = None,
        output_dim: int = 100,
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
        scheduler_gamma: float = 0.5
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
    
    def to_dict(self) -> dict:
        return {
            "branch_input_dim": self.branch_input_dim,
            "trunk_input_dim": self.trunk_input_dim,
            "branch_hidden_layers": self.branch_hidden_layers,
            "trunk_hidden_layers": self.trunk_hidden_layers,
            "output_dim": self.output_dim,
            "branch_activation": self.branch_activation,
            "trunk_activation": self.trunk_activation,
            "branch_dropout": self.branch_dropout,
            "use_positional_encoding": self.use_positional_encoding,
            "positional_encoding_freqs": self.positional_encoding_freqs,
            "use_bias": self.use_bias,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "DeepONetConfigPaddle":
        return cls(**config_dict)
