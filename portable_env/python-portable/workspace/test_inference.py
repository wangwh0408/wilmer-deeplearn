import json
import urllib.request
import numpy as np

# Create 64x64 test data with some variation
np.random.seed(42)
u_data = np.random.rand(64, 64).tolist()
v_data = np.random.rand(64, 64).tolist()

# CFDBender condition parameters
data = {
    'u': u_data,
    'v': v_data,
    'height': 1.0,
    'width': 1.0,
    'vel_top': 1.0,
    'density': 1000.0,
    'viscosity': 0.001,
    'dx': 0.015625,
    'dy': 0.015625,
    'denormalize': True,
    'return_full': False
}

print('=== Inference Request Parameters ===')
print(f'Input shape: U={len(u_data)}x{len(u_data[0])}, V={len(v_data)}x{len(v_data[0])}')
print(f'Cavity height: {data["height"]} m')
print(f'Cavity width: {data["width"]} m')
print(f'Top wall velocity: {data["vel_top"]} m/s')
print(f'Fluid density: {data["density"]} kg/m3')
print(f'Dynamic viscosity: {data["viscosity"]} Pa*s')
print(f'Grid spacing dx: {data["dx"]} m')
print(f'Grid spacing dy: {data["dy"]} m')
print()

req = urllib.request.Request(
    'http://127.0.0.1:9090/predict',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

with urllib.request.urlopen(req, timeout=60) as response:
    result = json.loads(response.read().decode('utf-8'))

print('=== Inference Result ===')
print(f'Success: {result["success"]}')
print(f'Inference time: {result["inference_time_ms"]} ms')
print(f'Output shape: {result["output_shape"]}')

# Analyze prediction results
if 'u' in result and result['u']:
    u_pred = np.array(result['u'])
    v_pred = np.array(result['v'])
    
    print(f'\nPredicted U velocity statistics:')
    print(f'  Mean: {u_pred.mean():.6f}')
    print(f'  Std: {u_pred.std():.6f}')
    print(f'  Min: {u_pred.min():.6f}')
    print(f'  Max: {u_pred.max():.6f}')
    
    print(f'\nPredicted V velocity statistics:')
    print(f'  Mean: {v_pred.mean():.6f}')
    print(f'  Std: {v_pred.std():.6f}')
    print(f'  Min: {v_pred.min():.6f}')
    print(f'  Max: {v_pred.max():.6f}')
    
    print(f'\nSample predictions (center region):')
    center_i, center_j = 32, 32
    print(f'  U[{center_i}][{center_j}] = {u_pred[center_i][center_j]:.6f}')
    print(f'  V[{center_i}][{center_j}] = {v_pred[center_i][center_j]:.6f}')

print('\n=== Inference Completed ===')
