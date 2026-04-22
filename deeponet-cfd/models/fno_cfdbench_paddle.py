import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import numpy as np


class SpectralConv2dPaddle(nn.Layer):
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2dPaddle, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        
        self.weights1_real = self.create_parameter(
            shape=(in_channels, out_channels, modes1, modes2),
            dtype='float32',
            default_initializer=nn.initializer.Uniform(
                low=-self.scale,
                high=self.scale
            )
        )
        
        self.weights1_imag = self.create_parameter(
            shape=(in_channels, out_channels, modes1, modes2),
            dtype='float32',
            default_initializer=nn.initializer.Uniform(
                low=-self.scale,
                high=self.scale
            )
        )
        
        self.weights2_real = self.create_parameter(
            shape=(in_channels, out_channels, modes1, modes2),
            dtype='float32',
            default_initializer=nn.initializer.Uniform(
                low=-self.scale,
                high=self.scale
            )
        )
        
        self.weights2_imag = self.create_parameter(
            shape=(in_channels, out_channels, modes1, modes2),
            dtype='float32',
            default_initializer=nn.initializer.Uniform(
                low=-self.scale,
                high=self.scale
            )
        )
    
    def compl_mul2d(self, input: paddle.Tensor, weights: paddle.Tensor) -> paddle.Tensor:
        return paddle.einsum("bixy,ioxy->boxy", input, weights)
    
    def _get_weights1(self) -> paddle.Tensor:
        return paddle.complex(self.weights1_real, self.weights1_imag)
    
    def _get_weights2(self) -> paddle.Tensor:
        return paddle.complex(self.weights2_real, self.weights2_imag)
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        batchsize = x.shape[0]
        
        x_ft = paddle.fft.rfft2(x)
        
        out_ft = paddle.zeros(
            [batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1],
            dtype='complex64'
        )
        
        weights1 = self._get_weights1()
        weights2 = self._get_weights2()
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)
        
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        
        return x


class FNOBlockPaddle(nn.Layer):
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(FNOBlockPaddle, self).__init__()
        
        self.conv = SpectralConv2dPaddle(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2D(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2D(out_channels)
    
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        return x


class FNO2dPaddle(nn.Layer):
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
        hidden_dim: int = 128,
        use_coordinates: bool = True,
        use_mask: bool = False
    ):
        super(FNO2dPaddle, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_coordinates = use_coordinates
        self.use_mask = use_mask
        
        fc0_in_channels = in_channels
        if use_coordinates:
            fc0_in_channels += 2
        if use_mask:
            fc0_in_channels += 1
        
        self.fc0 = nn.Linear(fc0_in_channels, width)
        
        self.blocks = nn.LayerList([
            FNOBlockPaddle(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)
    
    def get_grid(self, shape: Tuple, place: paddle.CPUPlace) -> paddle.Tensor:
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        
        gridx = paddle.linspace(0, 1, size_x, dtype='float32')
        gridx = gridx.reshape([1, 1, size_x, 1])
        gridx = paddle.tile(gridx, [batchsize, 1, 1, size_y])
        
        gridy = paddle.linspace(0, 1, size_y, dtype='float32')
        gridy = gridy.reshape([1, 1, 1, size_y])
        gridy = paddle.tile(gridy, [batchsize, 1, size_x, 1])
        
        return paddle.concat([gridx, gridy], axis=1)
    
    def forward(
        self,
        x: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        if x.ndim == 4:
            x = x.transpose([0, 3, 1, 2])
        
        if self.use_coordinates:
            grid = self.get_grid(x.shape, x.place)
            x = paddle.concat([x, grid], axis=1)
        
        if self.use_mask and mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            x = paddle.concat([x, mask], axis=1)
        
        x = x.transpose([0, 2, 3, 1])
        x = self.fc0(x)
        x = x.transpose([0, 3, 1, 2])
        
        for block in self.blocks:
            x = block(x)
        
        x = x.transpose([0, 2, 3, 1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class AutoRegressiveFNOPaddle(nn.Layer):
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
        hidden_dim: int = 128,
        use_coordinates: bool = True,
        use_mask: bool = False
    ):
        super(AutoRegressiveFNOPaddle, self).__init__()
        
        self.fno = FNO2dPaddle(
            modes1=modes1,
            modes2=modes2,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            use_coordinates=use_coordinates,
            use_mask=use_mask
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        return self.fno(x, mask)
    
    def generate_one(
        self,
        x: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        return self.forward(x, mask)
    
    def generate_many(
        self,
        x: paddle.Tensor,
        n_steps: int,
        mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        predictions = []
        current = x
        
        for _ in range(n_steps):
            next_step = self.generate_one(current, mask)
            predictions.append(next_step)
            current = next_step
        
        return paddle.stack(predictions, axis=1)
    
    def count_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            if not p.stop_gradient:
                total += np.prod(p.shape)
        return int(total)
    
    def get_model_info(self) -> Dict[str, Any]:
        total_params = self.count_parameters()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': total_params,
            'modes1': self.fno.modes1,
            'modes2': self.fno.modes2,
            'width': self.fno.width,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels
        }


def count_parameters_paddle(model: nn.Layer) -> int:
    total = 0
    for p in model.parameters():
        if not p.stop_gradient:
            total += np.prod(p.shape)
    return int(total)


if __name__ == "__main__":
    print("Testing FNO model (PaddlePaddle)...")
    
    paddle.seed(42)
    
    model = AutoRegressiveFNOPaddle(
        modes1=12,
        modes2=12,
        width=32,
        in_channels=2,
        out_channels=2,
        n_layers=4,
        hidden_dim=128
    )
    
    print(f"Model parameters: {count_parameters_paddle(model):,}")
    print(f"Model info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
    
    batch_size = 4
    grid_size = 64
    x = paddle.randn([batch_size, grid_size, grid_size, 2], dtype='float32')
    
    with paddle.no_grad():
        output = model(x)
        print(f"\nSingle-step prediction:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        multi_step = model.generate_many(x, n_steps=5)
        print(f"\nMulti-step autoregressive prediction:")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {multi_step.shape}")
    
    print("\nAll tests passed!")
