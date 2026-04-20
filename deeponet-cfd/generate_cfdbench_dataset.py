import numpy as np
import os
import sys
import argparse
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.cfd_bench_dataset import (
    CFDDataGenerator,
    CFDDatasetPyTorch,
    CFDDatasetPaddle,
    HAS_TORCH,
    HAS_PADDLE
)


class CFDBenchDataset:
    
    def __init__(
        self,
        n_train: int = 1000,
        n_test: int = 200,
        grid_size: int = 64,
        problem_type: str = "poisson",
        domain: Tuple[float, float] = (0.0, 1.0),
        n_sources: Tuple[int, int] = (1, 5),
        source_strength_range: Tuple[float, float] = (0.5, 5.0),
        source_width_range: Tuple[float, float] = (0.05, 0.15),
        viscosity: float = 0.01,
        time_steps: int = 10,
        seed: Optional[int] = None
    ):
        self.n_train = n_train
        self.n_test = n_test
        self.grid_size = grid_size
        self.problem_type = problem_type
        self.domain = domain
        self.n_sources = n_sources
        self.source_strength_range = source_strength_range
        self.source_width_range = source_width_range
        self.viscosity = viscosity
        self.time_steps = time_steps
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.train_data = None
        self.test_data = None
        self.metadata = None
    
    def generate(self) -> Dict[str, Any]:
        print(f"Generating CFDBench dataset: {self.problem_type}")
        print(f"  Training samples: {self.n_train}")
        print(f"  Test samples: {self.n_test}")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        
        start_time = datetime.now()
        
        generator_train = CFDDataGenerator(
            n_samples=self.n_train,
            grid_size=self.grid_size,
            domain=self.domain,
            n_sources=self.n_sources,
            source_strength_range=self.source_strength_range,
            source_width_range=self.source_width_range
        )
        
        generator_test = CFDDataGenerator(
            n_samples=self.n_test,
            grid_size=self.grid_size,
            domain=self.domain,
            n_sources=self.n_sources,
            source_strength_range=self.source_strength_range,
            source_width_range=self.source_width_range
        )
        
        print(f"\nGenerating training data...")
        branch_train, trunk_train, output_train = generator_train.generate_dataset(
            self.problem_type
        )
        
        print(f"Generating test data...")
        branch_test, trunk_test, output_test = generator_test.generate_dataset(
            self.problem_type
        )
        
        self.train_data = {
            'branch_inputs': branch_train,
            'trunk_inputs': trunk_train,
            'outputs': output_train
        }
        
        self.test_data = {
            'branch_inputs': branch_test,
            'trunk_inputs': trunk_test,
            'outputs': output_test
        }
        
        self.metadata = {
            'n_train': self.n_train,
            'n_test': self.n_test,
            'grid_size': self.grid_size,
            'problem_type': self.problem_type,
            'domain': self.domain,
            'branch_input_dim': branch_train.shape[1],
            'trunk_input_dim': trunk_train.shape[-1],
            'output_dim': output_train.shape[1],
            'created_at': datetime.now().isoformat(),
            'seed': self.seed,
            'viscosity': self.viscosity if self.problem_type == 'navier_stokes' else None,
            'time_steps': self.time_steps if self.problem_type == 'navier_stokes' else None
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nDataset generation completed in {elapsed:.2f} seconds")
        print(f"  Branch input shape: {branch_train.shape}")
        print(f"  Trunk input shape: {trunk_train.shape}")
        print(f"  Output shape: {output_train.shape}")
        
        return self.train_data, self.test_data, self.metadata
    
    def save(self, filepath: str, format: str = "npz") -> str:
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dataset not generated yet. Call generate() first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format.lower() == "npz":
            np.savez(
                filepath,
                train_branch=self.train_data['branch_inputs'],
                train_trunk=self.train_data['trunk_inputs'],
                train_output=self.train_data['outputs'],
                test_branch=self.test_data['branch_inputs'],
                test_trunk=self.test_data['trunk_inputs'],
                test_output=self.test_data['outputs'],
                **{f'meta_{k}': str(v) if isinstance(v, tuple) else v for k, v in self.metadata.items()}
            )
            print(f"Dataset saved as NPZ: {filepath}")
        
        elif format.lower() == "hdf5":
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
            
            with h5py.File(filepath, 'w') as f:
                train_grp = f.create_group('train')
                train_grp.create_dataset('branch_inputs', data=self.train_data['branch_inputs'])
                train_grp.create_dataset('trunk_inputs', data=self.train_data['trunk_inputs'])
                train_grp.create_dataset('outputs', data=self.train_data['outputs'])
                
                test_grp = f.create_group('test')
                test_grp.create_dataset('branch_inputs', data=self.test_data['branch_inputs'])
                test_grp.create_dataset('trunk_inputs', data=self.test_data['trunk_inputs'])
                test_grp.create_dataset('outputs', data=self.test_data['outputs'])
                
                for key, value in self.metadata.items():
                    if value is not None:
                        if isinstance(value, tuple):
                            f.attrs[key] = str(value)
                        else:
                            f.attrs[key] = value
            
            print(f"Dataset saved as HDF5: {filepath}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'hdf5'.")
        
        return filepath
    
    def get_pytorch_dataset(self, normalize: bool = True) -> Tuple[Any, Any]:
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dataset not generated yet. Call generate() first.")
        
        train_dataset = CFDDatasetPyTorch(
            self.train_data['branch_inputs'],
            self.train_data['trunk_inputs'],
            self.train_data['outputs'],
            normalize=normalize
        )
        
        test_dataset = CFDDatasetPyTorch(
            self.test_data['branch_inputs'],
            self.test_data['trunk_inputs'],
            self.test_data['outputs'],
            normalize=normalize
        )
        
        return train_dataset, test_dataset
    
    def get_paddle_dataset(self, normalize: bool = True) -> Tuple[Any, Any]:
        if not HAS_PADDLE:
            raise ImportError("PaddlePaddle is not installed. Install with: pip install paddlepaddle")
        
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dataset not generated yet. Call generate() first.")
        
        train_dataset = CFDDatasetPaddle(
            self.train_data['branch_inputs'],
            self.train_data['trunk_inputs'],
            self.train_data['outputs'],
            normalize=normalize
        )
        
        test_dataset = CFDDatasetPaddle(
            self.test_data['branch_inputs'],
            self.test_data['trunk_inputs'],
            self.test_data['outputs'],
            normalize=normalize
        )
        
        return train_dataset, test_dataset


def load_cfdbench_dataset(filepath: str, format: Optional[str] = None) -> CFDBenchDataset:
    if format is None:
        if filepath.endswith('.npz'):
            format = 'npz'
        elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
            format = 'hdf5'
        else:
            raise ValueError(f"Cannot determine format from filename: {filepath}")
    
    if format.lower() == "npz":
        data = np.load(filepath, allow_pickle=True)
        
        dataset = CFDBenchDataset(
            n_train=len(data['train_branch']),
            n_test=len(data['test_branch']),
            grid_size=int(np.sqrt(data['train_branch'].shape[1])),
            problem_type=str(data.get('meta_problem_type', 'poisson'))
        )
        
        dataset.train_data = {
            'branch_inputs': data['train_branch'],
            'trunk_inputs': data['train_trunk'],
            'outputs': data['train_output']
        }
        
        dataset.test_data = {
            'branch_inputs': data['test_branch'],
            'trunk_inputs': data['test_trunk'],
            'outputs': data['test_output']
        }
        
        dataset.metadata = {
            'n_train': len(data['train_branch']),
            'n_test': len(data['test_branch']),
            'branch_input_dim': data['train_branch'].shape[1],
            'trunk_input_dim': data['train_trunk'].shape[-1],
            'output_dim': data['train_output'].shape[1]
        }
        
        print(f"Dataset loaded from NPZ: {filepath}")
        
        return dataset
    
    elif format.lower() == "hdf5":
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        
        with h5py.File(filepath, 'r') as f:
            dataset = CFDBenchDataset(
                n_train=len(f['train/branch_inputs']),
                n_test=len(f['test/branch_inputs']),
                grid_size=int(np.sqrt(f['train/branch_inputs'].shape[1])),
                problem_type=str(f.attrs.get('problem_type', 'poisson'))
            )
            
            dataset.train_data = {
                'branch_inputs': f['train/branch_inputs'][:],
                'trunk_inputs': f['train/trunk_inputs'][:],
                'outputs': f['train/outputs'][:]
            }
            
            dataset.test_data = {
                'branch_inputs': f['test/branch_inputs'][:],
                'trunk_inputs': f['test/trunk_inputs'][:],
                'outputs': f['test/outputs'][:]
            }
            
            dataset.metadata = dict(f.attrs)
        
        print(f"Dataset loaded from HDF5: {filepath}")
        
        return dataset
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'hdf5'.")


def generate_and_save_dataset(
    output_path: str,
    n_train: int = 1000,
    n_test: int = 200,
    grid_size: int = 64,
    problem_type: str = "poisson",
    format: str = "npz",
    seed: Optional[int] = None
) -> str:
    dataset = CFDBenchDataset(
        n_train=n_train,
        n_test=n_test,
        grid_size=grid_size,
        problem_type=problem_type,
        seed=seed
    )
    
    dataset.generate()
    
    filepath = dataset.save(output_path, format=format)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate CFDBench dataset')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path (e.g., data/cfd_poisson.npz)')
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid size (default: 64)')
    parser.add_argument('--problem_type', type=str, default='poisson',
                        choices=['poisson', 'navier_stokes'],
                        help='Problem type: poisson or navier_stokes')
    parser.add_argument('--format', type=str, default='npz',
                        choices=['npz', 'hdf5'],
                        help='Output format: npz or hdf5')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--load', '-l', type=str, default=None,
                        help='Load existing dataset instead of generating')
    
    args = parser.parse_args()
    
    if args.load:
        dataset = load_cfdbench_dataset(args.load)
        print("\nDataset Info:")
        print(f"  Train samples: {dataset.n_train}")
        print(f"  Test samples: {dataset.n_test}")
        print(f"  Grid size: {dataset.grid_size}")
        if dataset.metadata:
            print(f"  Metadata: {dataset.metadata}")
    else:
        generate_and_save_dataset(
            output_path=args.output,
            n_train=args.n_train,
            n_test=args.n_test,
            grid_size=args.grid_size,
            problem_type=args.problem_type,
            format=args.format,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
