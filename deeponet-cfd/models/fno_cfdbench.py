import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np


class SpectralConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super(FNOBlock, self).__init__()
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        return x


class FNO2d(nn.Module):
    
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
        super(FNO2d, self).__init__()
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
        
        self.blocks = nn.ModuleList([
            FNOBlock(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)
    
    def get_grid(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        
        if self.use_coordinates:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        
        if self.use_mask and mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            x = torch.cat((x, mask), dim=1)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class AutoRegressiveFNO(nn.Module):
    
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
        super(AutoRegressiveFNO, self).__init__()
        
        self.fno = FNO2d(
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fno(x, mask)
    
    def generate_one(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(x, mask)
    
    def generate_many(
        self,
        x: torch.Tensor,
        n_steps: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        predictions = []
        current = x
        
        for _ in range(n_steps):
            next_step = self.generate_one(current, mask)
            predictions.append(next_step)
            current = next_step
        
        return torch.stack(predictions, dim=1)
    
    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'modes1': self.fno.modes1,
            'modes2': self.fno.modes2,
            'width': self.fno.width,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("Testing FNO model...")
    
    model = AutoRegressiveFNO(
        modes1=12,
        modes2=12,
        width=32,
        in_channels=2,
        out_channels=2,
        n_layers=4,
        hidden_dim=128
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    batch_size = 4
    grid_size = 64
    x = torch.randn(batch_size, grid_size, grid_size, 2)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        
        multi_step = model.generate_many(x, n_steps=5)
        print(f"Multi-step output shape: {multi_step.shape}")
    
    print("\nModel info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
