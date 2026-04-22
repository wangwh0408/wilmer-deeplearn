# FNO Training Environment for CFDBench

## Overview

This environment is specifically configured for training Fourier Neural Operator (FNO) models on the CFDBench dataset.

## Directory Structure

```
deeponet-cfd/
├── env/
│   ├── requirements-fno.txt      # Exact package versions
│   ├── setup_env.bat             # Windows setup script
│   ├── verify_environment.py     # Environment verification
│   └── SETUP_GUIDE.md            # This file
├── data/                          # CFDBench dataset goes here
├── models/
│   └── fno_cfdbench.py           # FNO model implementation
├── data/
│   └── cfdbench_loader.py        # CFDBench data loader
├── train_fno_cfdbench.py         # Training script
├── test_fno_cfdbench.py          # Evaluation script
└── verify_fno_cfdbench.py        # Code verification
```

## Quick Start

### Step 1: Setup Environment

#### Windows (Recommended)

```cmd
cd deeponet-cfd\env
setup_env.bat
```

#### Manual Setup

```cmd
cd deeponet-cfd

# Create virtual environment
python -m venv venv-fno

# Activate environment
venv-fno\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r env/requirements-fno.txt
```

### Step 2: Verify Environment

```cmd
cd deeponet-cfd\env
python verify_environment.py
```

### Step 3: Prepare CFDBench Dataset

Place your CFDBench dataset in the `data/` directory with the following structure:

```
deeponet-cfd/
└── data/
    ├── cavity/
    │   ├── bc/
    │   │   ├── case0000/
    │   │   │   ├── u.npy
    │   │   │   └── v.npy
    │   │   ├── case0001/
    │   │   └── ...
    │   ├── geo/
    │   └── prop/
    ├── tube/
    ├── dam/
    └── cylinder/
```

### Step 4: Run Training

#### Basic Training

```cmd
cd deeponet-cfd

# Activate environment first
venv-fno\Scripts\activate.bat

# Train on cavity dataset (boundary conditions category)
python train_fno_cfdbench.py ^
    --data_root data ^
    --problems cavity ^
    --categories bc ^
    --epochs 100 ^
    --batch_size 8 ^
    --model_save_path models/fno_cavity_bc.pth
```

#### Training with Multiple Categories

```cmd
# Train on cavity with all categories
python train_fno_cfdbench.py ^
    --data_root data ^
    --problems cavity ^
    --categories bc geo prop ^
    --epochs 100 ^
    --model_save_path models/fno_cavity_all.pth
```

#### Training on Different Problems

```cmd
# Train on cylinder flow
python train_fno_cfdbench.py ^
    --data_root data ^
    --problems cylinder ^
    --categories bc ^
    --epochs 100 ^
    --model_save_path models/fno_cylinder.pth
```

### Step 5: Evaluate Model

```cmd
cd deeponet-cfd

# Evaluate trained model
python test_fno_cfdbench.py ^
    --model_path models/fno_cavity_bc.pth ^
    --data_root data ^
    --problems cavity ^
    --categories bc ^
    --results_save_path results/eval_report.json

# Evaluate with multi-step autoregressive prediction
python test_fno_cfdbench.py ^
    --model_path models/fno_cylinder.pth ^
    --data_root data ^
    --problems cylinder ^
    --multi_step 10 ^
    --results_save_path results/multi_step_eval.json
```

## Command Line Reference

### Training Script (`train_fno_cfdbench.py`)

#### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | "data" | Root directory of CFDBench dataset |
| `--problems` | ["cavity"] | Problems to use: cavity, tube, dam, cylinder |
| `--categories` | ["bc"] | Categories: bc (boundary conditions), geo (geometry), prop (properties) |
| `--max_cases` | None | Maximum cases per category (None = all) |
| `--train_ratio` | 0.8 | Ratio of training data |

#### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--modes1` | 12 | Fourier modes in x direction |
| `--modes2` | 12 | Fourier modes in y direction |
| `--width` | 32 | Hidden dimension width |
| `--n_layers` | 4 | Number of FNO blocks |
| `--hidden_dim` | 128 | Hidden dimension in final layers |

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 0.001 | Learning rate |
| `--weight_decay` | 1e-4 | Weight decay (L2 regularization) |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 100 | Number of training epochs |
| `--scheduler_step_size` | 50 | Step size for learning rate scheduler |
| `--scheduler_gamma` | 0.5 | Gamma for learning rate scheduler |
| `--input_steps` | 1 | Number of input time steps |
| `--output_steps` | 1 | Number of output time steps |

#### Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_save_path` | None | Path to save trained model |
| `--device` | None | Device: cuda or cpu (auto-detect if None) |

### Evaluation Script (`test_fno_cfdbench.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | required | Path to trained model checkpoint |
| `--data_root` | "data" | Root directory of CFDBench dataset |
| `--problems` | ["cavity"] | Problems to use |
| `--categories` | ["bc"] | Categories to use |
| `--max_cases` | None | Maximum cases per category |
| `--batch_size` | 8 | Batch size |
| `--train_ratio` | 0.8 | Ratio of training data (for split consistency) |
| `--multi_step` | None | Number of steps for multi-step autoregressive evaluation |
| `--results_save_path` | None | Path to save evaluation results (JSON) |
| `--device` | None | Device: cuda or cpu |

## Python API Reference

### Training

```python
from train_fno_cfdbench import TrainingConfig, CFDBenchTrainer

# Configure training
config = TrainingConfig(
    modes1=12,
    modes2=12,
    width=32,
    n_layers=4,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=8,
    epochs=100,
    input_steps=1,
    output_steps=1
)

# Create trainer
trainer = CFDBenchTrainer(
    config=config,
    data_root="data",
    problems=["cavity"],
    categories=["bc", "geo"]
)

# Start training
result = trainer.train(model_save_path="models/fno_model.pth")

# Access results
print(f"Final train loss: {result.train_losses[-1]}")
print(f"Final test loss: {result.test_losses[-1]}")
print(f"Best test loss: {result.best_test_loss}")
print(f"Training time: {result.duration_seconds:.2f}s")
```

### Model Loading

```python
from train_fno_cfdbench import CFDBenchTrainer

# Load trained model
model, info = CFDBenchTrainer.load_model(
    "models/fno_model.pth",
    device="cuda"
)

# Access training info
print("Normalization params:", info['normalization_params'])
print("Best test loss:", info['best_test_loss'])

# Single-step prediction
import torch
input_tensor = torch.randn(1, 64, 64, 2).cuda()
output = model(input_tensor)

# Multi-step autoregressive prediction
multi_step = model.generate_many(input_tensor, n_steps=10)
# Shape: (1, 10, 64, 64, 2) - (batch, steps, H, W, channels)
```

### Data Loading

```python
from data import CFDBenchDataset, create_cfdbench_dataloaders

# Create dataset
dataset = CFDBenchDataset(
    data_root="data",
    problems=["cavity"],
    categories=["bc"],
    input_steps=1,
    output_steps=1,
    normalize=True
)
dataset.load()

# Get sample
sample = dataset[0]
print(f"Input shape: {sample['input'].shape}")
print(f"Output shape: {sample['output'].shape}")

# Create DataLoaders
train_loader, test_loader, norm_params = create_cfdbench_dataloaders(
    data_root="data",
    problems=["cavity"],
    categories=["bc"],
    batch_size=8,
    train_ratio=0.8
)
```

## Performance Notes

### Hardware Requirements

- **Minimum**: CPU with 16GB RAM
- **Recommended**: 
  - NVIDIA GPU with 8GB+ VRAM
  - CUDA Toolkit 11.8 or 12.1

### CUDA Installation

If CUDA is not detected, install PyTorch with CUDA support:

```cmd
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Training Time Estimates

| Dataset Size | Epochs | CPU Time | GPU Time (RTX 3090) |
|--------------|--------|-----------|----------------------|
| 1000 samples | 100 | ~2 hours | ~5 minutes |
| 10000 samples | 100 | ~20 hours | ~50 minutes |

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate the virtual environment:
```cmd
venv-fno\Scripts\activate.bat
```

#### 2. CUDA is not available

**Solution**: 
1. Install NVIDIA CUDA Toolkit 11.8 or 12.1
2. Reinstall PyTorch with CUDA support (see above)

#### 3. "FileNotFoundError: data/cavity"

**Solution**: 
1. Download CFDBench dataset from HuggingFace
2. Place it in the `data/` directory
3. Verify the directory structure matches the example above

#### 4. Out of Memory Error

**Solution**: Reduce batch size:
```cmd
python train_fno_cfdbench.py --batch_size 4
```

Or reduce model size:
```cmd
python train_fno_cfdbench.py --modes1 8 --modes2 8 --width 16
```

## CFDBench Dataset Information

### Problems

| Problem | Description |
|---------|-------------|
| `cavity` | Lid-driven cavity flow |
| `tube` | Flow in a tube |
| `dam` | Flow over a step/dam |
| `cylinder` | Flow around a cylinder (Kármán vortex street) |

### Categories

| Category | Description |
|----------|-------------|
| `bc` | Boundary conditions variation |
| `geo` | Geometry variation |
| `prop` | Physical properties variation |

### Data Format

Each case contains:
- `u.npy`: x-component velocity field (time_steps, H, W)
- `v.npy`: y-component velocity field (time_steps, H, W)
- `p.npy`: Pressure field (time_steps, H, W) [optional]

## License

This code is provided for academic and research purposes.
Please cite the relevant papers if you use this code in your research.

### References

1. FNO: "Fourier Neural Operator for Parametric Partial Differential Equations" (Li et al., 2020)
2. CFDBench: "CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics" (Luo et al., 2023)
