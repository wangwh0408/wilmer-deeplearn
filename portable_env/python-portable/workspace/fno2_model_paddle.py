"""
FNO2 Model Implementation for PaddlePaddle

This is the PaddlePaddle version of the FNO2 model,
corresponding to the PyTorch version in fno2_model.py.

Compatible with train_fno2_paddle.py training script.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class SpectralConv2dPaddle(nn.Layer):
    """
    2D Spectral Convolution Layer for PaddlePaddle
    
    Corresponding to PyTorch's SpectralConv2d in fno2_model.py
    """
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dPaddle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        
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
    
    def compl_mul2d(self, input, weights):
        return paddle.einsum("bixy,ioxy->boxy", input, weights)
    
    def _get_weights1(self):
        return paddle.complex(self.weights1_real, self.weights1_imag)
    
    def _get_weights2(self):
        return paddle.complex(self.weights2_real, self.weights2_imag)
    
    def forward(self, x):
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


class FNO2dPaddle(nn.Layer):
    """
    FNO2 Model for PaddlePaddle
    
    Corresponding to PyTorch's FNO2d in fno2_model.py
    4 FNO blocks architecture.
    """
    
    def __init__(self, modes1, modes2, width, in_channels=3, out_channels=1):
        super(FNO2dPaddle, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.fc0 = nn.Linear(in_channels, self.width)
        
        self.conv0 = SpectralConv2dPaddle(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2dPaddle(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2dPaddle(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2dPaddle(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2D(self.width, self.width, 1)
        self.w1 = nn.Conv2D(self.width, self.width, 1)
        self.w2 = nn.Conv2D(self.width, self.width, 1)
        self.w3 = nn.Conv2D(self.width, self.width, 1)
        
        self.bn0 = nn.BatchNorm2D(self.width)
        self.bn1 = nn.BatchNorm2D(self.width)
        self.bn2 = nn.BatchNorm2D(self.width)
        self.bn3 = nn.BatchNorm2D(self.width)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        gridx = paddle.linspace(0, 1, size_x, dtype='float32')
        gridx = gridx.reshape([1, size_x, 1, 1])
        gridx = paddle.tile(gridx, [batchsize, 1, size_y, 1])
        
        gridy = paddle.linspace(0, 1, size_y, dtype='float32')
        gridy = gridy.reshape([1, 1, size_y, 1])
        gridy = paddle.tile(gridy, [batchsize, size_x, 1, 1])
        
        return paddle.concat([gridx, gridy], axis=-1)
    
    def forward(self, x):
        grid = self.get_grid(x.shape)
        x = paddle.concat([x, grid], axis=-1)
        
        x = self.fc0(x)
        x = x.transpose([0, 3, 1, 2])
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.bn0(x)
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.bn1(x)
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.bn2(x)
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x)
        
        x = x.transpose([0, 2, 3, 1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


def count_parameters_paddle(model: nn.Layer) -> int:
    """
    Count the number of trainable parameters in a PaddlePaddle model.
    
    Args:
        model: PaddlePaddle model
        
    Returns:
        Number of trainable parameters
    """
    total = 0
    for p in model.parameters():
        if not p.stop_gradient:
            total += np.prod(p.shape)
    return int(total)


if __name__ == "__main__":
    print("Testing FNO2 model (PaddlePaddle)...")
    
    paddle.seed(42)
    
    modes = 12
    width = 32
    model = FNO2dPaddle(
        modes1=modes,
        modes2=modes,
        width=width,
        in_channels=2,
        out_channels=2
    )
    
    total_params = count_parameters_paddle(model)
    print(f"Model parameters: {total_params:,}")
    print(f"Model configuration:")
    print(f"  modes: ({modes}, {modes})")
    print(f"  width: {width}")
    print(f"  in_channels: 2")
    print(f"  out_channels: 2")
    
    batch_size = 4
    grid_size = 64
    x = paddle.randn([batch_size, grid_size, grid_size, 2], dtype='float32')
    
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nAll tests passed!")
