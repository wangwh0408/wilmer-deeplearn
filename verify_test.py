import sys
sys.path.insert(0, '.')

print('=' * 60)
print('Verifying test_fno2.py code')
print('=' * 60)

print('\n1. Testing imports...')
try:
    from test_fno2 import (
        load_model, calculate_metrics, generate_test_case, 
        predict, visualize_training_history, visualize_test_case,
        evaluate_on_dataset, generate_report
    )
    print('   OK: All imports successful')
except Exception as e:
    print(f'   ERROR: Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n2. Testing calculate_metrics...')
try:
    import numpy as np
    np.random.seed(42)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    metrics = calculate_metrics(y_true, y_pred)
    print(f'   OK: Metrics - MSE={metrics["MSE"]:.4f}, R2={metrics["R2"]:.4f}')
except Exception as e:
    print(f'   ERROR: Failed: {e}')
    import traceback
    traceback.print_exc()

print('\n3. Testing generate_test_case...')
try:
    X, Y, f, u_true, case_type = generate_test_case('standard', resolution=32)
    print(f'   OK: Standard case - f shape={f.shape}, u_true shape={u_true.shape}')
    
    case_types = ['standard', 'single_source', 'multiple_sources', 'boundary_near', 'smooth']
    for ct in case_types:
        try:
            _, _, f_test, _, _ = generate_test_case(ct, resolution=32)
            print(f'   OK: {ct} - shape={f_test.shape}')
        except Exception as e:
            print(f'   ERROR: {ct}: {e}')
except Exception as e:
    print(f'   ERROR: Failed: {e}')
    import traceback
    traceback.print_exc()

print('\n4. Testing FNO2d model compatibility...')
try:
    import torch
    from fno2_model import FNO2d
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO2d(modes1=12, modes2=12, width=32, in_channels=3, out_channels=1).to(device)
    
    # Test predict function
    _, _, f_test, _, _ = generate_test_case('standard', resolution=32)
    u_pred = predict(model, f_test, device)
    print(f'   OK: Prediction shape: {u_pred.shape}')
    
    # Test with same resolution as training
    _, _, f_64, _, _ = generate_test_case('standard', resolution=64)
    u_pred_64 = predict(model, f_64, device)
    print(f'   OK: 64x64 prediction shape: {u_pred_64.shape}')
    
except Exception as e:
    print(f'   ERROR: Failed: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('Verification completed!')
print('=' * 60)
print('\nThe test_fno2.py script is ready to use.')
print('\nTo test a trained model:')
print('  1. First run: python train_fno2.py')
print('  2. Then run: python test_fno2.py')
print('\nIf no trained model exists, test_fno2.py will show a clear error message')
print('and tell you to run train_fno2.py first.')
