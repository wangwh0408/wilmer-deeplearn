import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import glob


class CFDBCase:
    
    def __init__(self, case_path: str, name: str = None):
        self.case_path = case_path
        self.name = name or os.path.basename(case_path)
        self.u_data = None
        self.v_data = None
        self.p_data = None
        self.time_steps = 0
        self.grid_size = (0, 0)
        
    def load(self):
        u_path = os.path.join(self.case_path, 'u.npy')
        v_path = os.path.join(self.case_path, 'v.npy')
        p_path = os.path.join(self.case_path, 'p.npy')
        
        if os.path.exists(u_path):
            self.u_data = np.load(u_path).astype(np.float32)
        if os.path.exists(v_path):
            self.v_data = np.load(v_path).astype(np.float32)
        if os.path.exists(p_path):
            self.p_data = np.load(p_path).astype(np.float32)
        
        if self.u_data is not None:
            if self.u_data.ndim == 3:
                self.time_steps = self.u_data.shape[0]
                self.grid_size = (self.u_data.shape[1], self.u_data.shape[2])
            elif self.u_data.ndim == 2:
                self.time_steps = 1
                self.grid_size = self.u_data.shape
            elif self.u_data.ndim == 4:
                self.time_steps = self.u_data.shape[0]
                self.grid_size = (self.u_data.shape[1], self.u_data.shape[2])
        
        return self
    
    def get_velocity_field(self, time_idx: int = 0) -> np.ndarray:
        if self.u_data is None or self.v_data is None:
            return None
        
        if self.u_data.ndim == 3:
            u = self.u_data[time_idx]
            v = self.v_data[time_idx]
        elif self.u_data.ndim == 4:
            u = self.u_data[time_idx, :, :, 0]
            v = self.v_data[time_idx, :, :, 0]
        else:
            u = self.u_data
            v = self.v_data
        
        return np.stack([u, v], axis=-1)
    
    def get_pressure_field(self, time_idx: int = 0) -> np.ndarray:
        if self.p_data is None:
            return None
        
        if self.p_data.ndim == 3:
            return self.p_data[time_idx]
        return self.p_data


class CFDBCategory:
    
    CATEGORIES = ['bc', 'geo', 'prop']
    
    def __init__(self, problem_path: str, category: str):
        self.problem_path = problem_path
        self.category = category
        self.category_path = os.path.join(problem_path, category)
        self.cases: List[CFDBCase] = []
        
    def load(self, max_cases: int = None):
        if not os.path.exists(self.category_path):
            return self
        
        case_dirs = sorted(glob.glob(os.path.join(self.category_path, 'case*')))
        
        if max_cases:
            case_dirs = case_dirs[:max_cases]
        
        for case_dir in case_dirs:
            case = CFDBCase(case_dir).load()
            if case.u_data is not None:
                self.cases.append(case)
        
        return self
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        return self.cases[idx]


class CFDBenchDataset(Dataset):
    
    PROBLEMS = ['cavity', 'tube', 'dam', 'cylinder']
    
    def __init__(
        self,
        data_root: str,
        problems: List[str] = None,
        categories: List[str] = None,
        input_steps: int = 1,
        output_steps: int = 1,
        step_interval: int = 1,
        include_pressure: bool = False,
        normalize: bool = True,
        grid_size: Tuple[int, int] = (64, 64),
        max_cases_per_category: int = None
    ):
        self.data_root = data_root
        self.problems = problems or ['cavity']
        self.categories = categories or ['bc', 'geo', 'prop']
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.step_interval = step_interval
        self.include_pressure = include_pressure
        self.normalize = normalize
        self.grid_size = grid_size
        self.max_cases_per_category = max_cases_per_category
        
        self.all_cases: List[CFDBCase] = []
        self.samples: List[Dict] = []
        
        self.mean_u = 0.0
        self.std_u = 1.0
        self.mean_v = 0.0
        self.std_v = 1.0
        self.mean_p = 0.0
        self.std_p = 1.0
        
    def load(self):
        print(f"Loading CFDBench dataset from: {self.data_root}")
        print(f"Problems: {self.problems}")
        print(f"Categories: {self.categories}")
        
        for problem in self.problems:
            problem_path = os.path.join(self.data_root, problem)
            if not os.path.exists(problem_path):
                print(f"  Warning: Problem directory not found: {problem_path}")
                continue
            
            for category in self.categories:
                cat = CFDBCategory(problem_path, category)
                cat.load(max_cases=self.max_cases_per_category)
                
                for case in cat.cases:
                    self.all_cases.append(case)
                    
                    if case.time_steps >= self.input_steps + self.output_steps:
                        max_start = case.time_steps - self.input_steps - self.output_steps + 1
                        for start_t in range(0, max_start, self.step_interval):
                            self.samples.append({
                                'case': case,
                                'start_t': start_t
                            })
        
        print(f"Loaded {len(self.all_cases)} cases, {len(self.samples)} samples")
        
        if self.normalize and len(self.all_cases) > 0:
            self._compute_normalization()
        
        return self
    
    def _compute_normalization(self):
        all_u = []
        all_v = []
        all_p = []
        
        for case in self.all_cases:
            if case.u_data is not None:
                all_u.append(case.u_data.flatten())
            if case.v_data is not None:
                all_v.append(case.v_data.flatten())
            if case.p_data is not None and self.include_pressure:
                all_p.append(case.p_data.flatten())
        
        if all_u:
            all_u = np.concatenate(all_u)
            self.mean_u = np.mean(all_u)
            self.std_u = np.std(all_u) + 1e-8
        
        if all_v:
            all_v = np.concatenate(all_v)
            self.mean_v = np.mean(all_v)
            self.std_v = np.std(all_v) + 1e-8
        
        if all_p:
            all_p = np.concatenate(all_p)
            self.mean_p = np.mean(all_p)
            self.std_p = np.std(all_p) + 1e-8
        
        print(f"Normalization stats:")
        print(f"  u: mean={self.mean_u:.4f}, std={self.std_u:.4f}")
        print(f"  v: mean={self.mean_v:.4f}, std={self.std_v:.4f}")
        if self.include_pressure:
            print(f"  p: mean={self.mean_p:.4f}, std={self.std_p:.4f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        case = sample['case']
        start_t = sample['start_t']
        
        input_fields = []
        output_fields = []
        
        for t in range(self.input_steps):
            vel = case.get_velocity_field(start_t + t)
            if self.normalize:
                u = (vel[..., 0] - self.mean_u) / self.std_u
                v = (vel[..., 1] - self.mean_v) / self.std_v
            else:
                u = vel[..., 0]
                v = vel[..., 1]
            input_fields.append(np.stack([u, v], axis=-1))
        
        for t in range(self.input_steps, self.input_steps + self.output_steps):
            vel = case.get_velocity_field(start_t + t)
            if self.normalize:
                u = (vel[..., 0] - self.mean_u) / self.std_u
                v = (vel[..., 1] - self.mean_v) / self.std_v
            else:
                u = vel[..., 0]
                v = vel[..., 1]
            output_fields.append(np.stack([u, v], axis=-1))
        
        input_tensor = np.stack(input_fields, axis=0)
        output_tensor = np.stack(output_fields, axis=0)
        
        input_tensor = input_tensor.reshape(-1, *input_tensor.shape[-2:])
        output_tensor = output_tensor.reshape(-1, *output_tensor.shape[-2:])
        
        return {
            'input': torch.from_numpy(input_tensor.astype(np.float32)),
            'output': torch.from_numpy(output_tensor.astype(np.float32)),
            'case_name': case.name,
            'start_t': start_t
        }
    
    def get_normalization_params(self) -> Dict[str, float]:
        return {
            'mean_u': self.mean_u,
            'std_u': self.std_u,
            'mean_v': self.mean_v,
            'std_v': self.std_v,
            'mean_p': self.mean_p,
            'std_p': self.std_p
        }


def create_cfdbench_dataloaders(
    data_root: str,
    problems: List[str] = None,
    categories: List[str] = None,
    input_steps: int = 1,
    output_steps: int = 1,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    normalize: bool = True,
    max_cases_per_category: int = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict]:
    
    dataset = CFDBenchDataset(
        data_root=data_root,
        problems=problems,
        categories=categories,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=normalize,
        max_cases_per_category=max_cases_per_category
    )
    dataset.load()
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Data split: train={n_train}, test={n_test}")
    
    return train_loader, test_loader, dataset.get_normalization_params()


if __name__ == "__main__":
    test_data_root = "data"
    
    if os.path.exists(os.path.join(test_data_root, "cavity")):
        print("Testing CFDBenchDataset...")
        
        dataset = CFDBenchDataset(
            data_root=test_data_root,
            problems=['cavity'],
            categories=['bc'],
            input_steps=1,
            output_steps=1,
            max_cases_per_category=5
        )
        dataset.load()
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample shape:")
            print(f"  Input: {sample['input'].shape}")
            print(f"  Output: {sample['output'].shape}")
            print(f"  Case: {sample['case_name']}")
    else:
        print(f"Data directory not found: {os.path.join(test_data_root, 'cavity')}")
        print("Please download CFDBench dataset and place it in the 'data' directory.")
        print("\nExpected directory structure:")
        print("data/")
        print("├── cavity/")
        print("│   ├── bc/")
        print("│   ├── geo/")
        print("│   └── prop/")
        print("├── tube/")
        print("├── dam/")
        print("└── cylinder/")
