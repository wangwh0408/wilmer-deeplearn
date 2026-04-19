import numpy as np
from typing import Optional, Tuple, List, Dict
import os
from abc import ABC, abstractmethod

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


class CFDDataGenerator:
    
    def __init__(
        self,
        n_samples: int = 1000,
        grid_size: int = 64,
        domain: Tuple[float, float] = (0.0, 1.0),
        n_sources: Tuple[int, int] = (1, 5),
        source_strength_range: Tuple[float, float] = (0.5, 5.0),
        source_width_range: Tuple[float, float] = (0.05, 0.15)
    ):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.domain = domain
        self.n_sources = n_sources
        self.source_strength_range = source_strength_range
        self.source_width_range = source_width_range
        
        self.x = np.linspace(domain[0], domain[1], grid_size)
        self.y = np.linspace(domain[0], domain[1], grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.coords = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
    
    def generate_poisson_source(self, n_sources: int) -> np.ndarray:
        source = np.zeros_like(self.X)
        
        for _ in range(n_sources):
            x0 = np.random.uniform(0.1, 0.9)
            y0 = np.random.uniform(0.1, 0.9)
            strength = np.random.uniform(
                self.source_strength_range[0],
                self.source_strength_range[1]
            )
            width = np.random.uniform(
                self.source_width_range[0],
                self.source_width_range[1]
            )
            
            source += strength * np.exp(
                -((self.X - x0)**2 + (self.Y - y0)**2) / (2 * width**2)
            )
        
        return source
    
    def solve_poisson_fft(self, source: np.ndarray) -> np.ndarray:
        kx = np.fft.fftfreq(self.grid_size) * self.grid_size
        ky = np.fft.fftfreq(self.grid_size) * self.grid_size
        KX, KY = np.meshgrid(kx, ky)
        
        f_hat = np.fft.fft2(source)
        u_hat = np.zeros_like(f_hat, dtype=np.complex128)
        
        mask = (KX**2 + KY**2) > 0
        u_hat[mask] = -f_hat[mask] / (4 * np.pi**2 * (KX[mask]**2 + KY[mask]**2))
        
        u = np.fft.ifft2(u_hat).real
        u = u - np.mean(u)
        u = (u - np.min(u)) / (np.max(u) - np.min(u) + 1e-8)
        
        return u
    
    def generate_navier_stokes_sample(
        self,
        viscosity: float = 0.01,
        time_steps: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vorticity = np.zeros_like(self.X)
        
        n_eddies = np.random.randint(2, 6)
        for _ in range(n_eddies):
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            strength = np.random.choice([-1, 1]) * np.random.uniform(1.0, 5.0)
            radius = np.random.uniform(0.05, 0.15)
            
            r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
            vorticity += strength * np.exp(-r**2 / (2 * radius**2))
        
        initial_vorticity = vorticity.copy()
        
        for _ in range(time_steps):
            vorticity_hat = np.fft.fft2(vorticity)
            kx = np.fft.fftfreq(self.grid_size) * self.grid_size
            ky = np.fft.fftfreq(self.grid_size) * self.grid_size
            KX, KY = np.meshgrid(kx, ky)
            
            psi_hat = np.zeros_like(vorticity_hat, dtype=np.complex128)
            mask = (KX**2 + KY**2) > 0
            psi_hat[mask] = -vorticity_hat[mask] / (KX[mask]**2 + KY[mask]**2)
            
            u_hat = -1j * KY * psi_hat
            v_hat = 1j * KX * psi_hat
            
            u = np.fft.ifft2(u_hat).real
            v = np.fft.ifft2(v_hat).real
            
            vorticity_x = np.gradient(vorticity, self.x[1] - self.x[0], axis=1)
            vorticity_y = np.gradient(vorticity, self.y[1] - self.y[0], axis=0)
            
            advection = -(u * vorticity_x + v * vorticity_y)
            diffusion = viscosity * (
                np.gradient(np.gradient(vorticity, axis=1), axis=1) +
                np.gradient(np.gradient(vorticity, axis=0), axis=0)
            ) / ((self.x[1] - self.x[0])**2)
            
            dt = 0.001
            vorticity += dt * (advection + diffusion)
        
        final_vorticity = vorticity
        
        return initial_vorticity, final_vorticity, self.coords
    
    def generate_dataset(
        self,
        problem_type: str = "poisson"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        branch_inputs = []
        trunk_inputs = []
        outputs = []
        
        for _ in range(self.n_samples):
            if problem_type == "poisson":
                n_src = np.random.randint(self.n_sources[0], self.n_sources[1] + 1)
                source = self.generate_poisson_source(n_src)
                solution = self.solve_poisson_fft(source)
                
                branch_inputs.append(source.flatten())
                outputs.append(solution.flatten())
                
            elif problem_type == "navier_stokes":
                initial_vort, final_vort, coords = self.generate_navier_stokes_sample()
                
                branch_inputs.append(initial_vort.flatten())
                outputs.append(final_vort.flatten())
        
        branch_inputs = np.array(branch_inputs, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)
        
        trunk_inputs = np.tile(self.coords, (self.n_samples, 1, 1)).astype(np.float32)
        
        return branch_inputs, trunk_inputs, outputs
    
    def save_to_hdf5(
        self,
        filepath: str,
        problem_type: str = "poisson"
    ):
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 operations. "
                "Install it with: pip install h5py"
            )
        
        branch_inputs, trunk_inputs, outputs = self.generate_dataset(problem_type)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('branch_inputs', data=branch_inputs)
            f.create_dataset('trunk_inputs', data=trunk_inputs)
            f.create_dataset('outputs', data=outputs)
            f.attrs['n_samples'] = self.n_samples
            f.attrs['grid_size'] = self.grid_size
            f.attrs['problem_type'] = problem_type
            f.attrs['branch_input_dim'] = branch_inputs.shape[1]
            f.attrs['trunk_input_dim'] = trunk_inputs.shape[-1]
            f.attrs['output_dim'] = outputs.shape[1]
        
        print(f"Dataset saved to {filepath}")
    
    @staticmethod
    def load_from_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 operations. "
                "Install it with: pip install h5py"
            )
        
        with h5py.File(filepath, 'r') as f:
            branch_inputs = f['branch_inputs'][:]
            trunk_inputs = f['trunk_inputs'][:]
            outputs = f['outputs'][:]
            
            attrs = dict(f.attrs)
        
        return branch_inputs, trunk_inputs, outputs, attrs


class BaseCFDDataset(ABC):
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass


try:
    import torch
    from torch.utils.data import Dataset
    
    class CFDDatasetPyTorch(Dataset):
        
        def __init__(
            self,
            branch_inputs: np.ndarray,
            trunk_inputs: np.ndarray,
            outputs: np.ndarray,
            normalize: bool = True
        ):
            self.branch_inputs = branch_inputs.astype(np.float32)
            self.trunk_inputs = trunk_inputs.astype(np.float32)
            self.outputs = outputs.astype(np.float32)
            
            if normalize:
                self.branch_mean = np.mean(self.branch_inputs, axis=(0, 1), keepdims=True)
                self.branch_std = np.std(self.branch_inputs, axis=(0, 1), keepdims=True) + 1e-8
                self.output_mean = np.mean(self.outputs, axis=(0, 1), keepdims=True)
                self.output_std = np.std(self.outputs, axis=(0, 1), keepdims=True) + 1e-8
                
                self.branch_inputs = (self.branch_inputs - self.branch_mean) / self.branch_std
                self.outputs = (self.outputs - self.output_mean) / self.output_std
            
            self.normalize = normalize
        
        def __len__(self):
            return len(self.branch_inputs)
        
        def __getitem__(self, idx):
            branch_input = torch.from_numpy(self.branch_inputs[idx])
            trunk_input = torch.from_numpy(self.trunk_inputs[idx])
            output = torch.from_numpy(self.outputs[idx])
            
            return branch_input, trunk_input, output
        
        def get_normalization_params(self) -> Dict:
            if not self.normalize:
                return {}
            return {
                'branch_mean': self.branch_mean,
                'branch_std': self.branch_std,
                'output_mean': self.output_mean,
                'output_std': self.output_std
            }
    
    HAS_TORCH = True

except ImportError:
    HAS_TORCH = False


try:
    import paddle
    from paddle.io import Dataset as PaddleDataset
    
    class CFDDatasetPaddle(PaddleDataset):
        
        def __init__(
            self,
            branch_inputs: np.ndarray,
            trunk_inputs: np.ndarray,
            outputs: np.ndarray,
            normalize: bool = True
        ):
            self.branch_inputs = branch_inputs.astype(np.float32)
            self.trunk_inputs = trunk_inputs.astype(np.float32)
            self.outputs = outputs.astype(np.float32)
            
            if normalize:
                self.branch_mean = np.mean(self.branch_inputs, axis=(0, 1), keepdims=True)
                self.branch_std = np.std(self.branch_inputs, axis=(0, 1), keepdims=True) + 1e-8
                self.output_mean = np.mean(self.outputs, axis=(0, 1), keepdims=True)
                self.output_std = np.std(self.outputs, axis=(0, 1), keepdims=True) + 1e-8
                
                self.branch_inputs = (self.branch_inputs - self.branch_mean) / self.branch_std
                self.outputs = (self.outputs - self.output_mean) / self.output_std
            
            self.normalize = normalize
        
        def __len__(self):
            return len(self.branch_inputs)
        
        def __getitem__(self, idx):
            branch_input = paddle.to_tensor(self.branch_inputs[idx])
            trunk_input = paddle.to_tensor(self.trunk_inputs[idx])
            output = paddle.to_tensor(self.outputs[idx])
            
            return branch_input, trunk_input, output
        
        def get_normalization_params(self) -> Dict:
            if not self.normalize:
                return {}
            return {
                'branch_mean': self.branch_mean,
                'branch_std': self.branch_std,
                'output_mean': self.output_mean,
                'output_std': self.output_std
            }
    
    HAS_PADDLE = True

except ImportError:
    HAS_PADDLE = False


def create_datasets(
    n_train: int = 1000,
    n_test: int = 200,
    grid_size: int = 64,
    problem_type: str = "poisson",
    normalize: bool = True,
    framework: str = "pytorch"
) -> Tuple:
    generator_train = CFDDataGenerator(n_samples=n_train, grid_size=grid_size)
    generator_test = CFDDataGenerator(n_samples=n_test, grid_size=grid_size)
    
    branch_train, trunk_train, output_train = generator_train.generate_dataset(problem_type)
    branch_test, trunk_test, output_test = generator_test.generate_dataset(problem_type)
    
    if framework.lower() == "pytorch" and HAS_TORCH:
        train_dataset = CFDDatasetPyTorch(branch_train, trunk_train, output_train, normalize)
        test_dataset = CFDDatasetPyTorch(branch_test, trunk_test, output_test, normalize)
    elif framework.lower() == "paddle" and HAS_PADDLE:
        train_dataset = CFDDatasetPaddle(branch_train, trunk_train, output_train, normalize)
        test_dataset = CFDDatasetPaddle(branch_test, trunk_test, output_test, normalize)
    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")
    
    branch_input_dim = branch_train.shape[1]
    trunk_input_dim = trunk_train.shape[-1]
    output_dim = output_train.shape[1]
    
    return train_dataset, test_dataset, branch_input_dim, trunk_input_dim, output_dim
