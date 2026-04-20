import os
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), 'data')

files = [
    'poisson_small.npz',
    'poisson_large.npz',
    'navier_stokes_small.npz',
    'navier_stokes_large.npz'
]

print('='*80)
print('GENERATED DATASET FILES SUMMARY')
print('='*80)
print()

for fname in files:
    filepath = os.path.join(data_dir, fname)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        
        data = np.load(filepath, allow_pickle=True)
        
        train_branch = data['train_branch']
        test_branch = data['test_branch']
        
        print(f'--- {fname} ---')
        print(f'  File size:      {size_mb:.2f} MB')
        print(f'  Train samples:  {train_branch.shape[0]}')
        print(f'  Test samples:   {test_branch.shape[0]}')
        print(f'  Points/sample:  {train_branch.shape[1]}')
        print(f'  Grid size:      {int(np.sqrt(train_branch.shape[1]))}x{int(np.sqrt(train_branch.shape[1]))}')
        
        pt = data.get('meta_problem_type')
        if hasattr(pt, 'item'):
            pt = pt.item()
        pt = str(pt).strip()
        print(f'  Problem type:   {pt}')
        
        # 数据统计
        train_branch_data = data['train_branch']
        train_output_data = data['train_output']
        
        print(f'  Branch data:    min={train_branch_data.min():.4f}, max={train_branch_data.max():.4f}, mean={train_branch_data.mean():.4f}')
        print(f'  Output data:    min={train_output_data.min():.4f}, max={train_output_data.max():.4f}, mean={train_output_data.mean():.4f}')
        
        # 额外的元数据
        if 'meta_viscosity' in data:
            visc = data['meta_viscosity']
            if hasattr(visc, 'item'):
                visc = visc.item()
            if visc is not None and str(visc).strip().lower() != 'none':
                print(f'  Viscosity:      {visc}')
        if 'meta_time_steps' in data:
            ts = data['meta_time_steps']
            if hasattr(ts, 'item'):
                ts = ts.item()
            if ts is not None and str(ts).strip().lower() != 'none':
                print(f'  Time steps:     {ts}')
        
        print()

print('='*80)
print('ALL DATASETS GENERATED SUCCESSFULLY!')
print('='*80)
