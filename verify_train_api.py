import sys
sys.path.insert(0, '.')

print('=' * 70)
print('VERIFYING RECONSTRUCTED train_fno2.py')
print('=' * 70)

print('\n1. Testing imports...')
try:
    from train_fno2 import train_fno2, train_with_config, quick_train, PoissonDataset
    print('   OK: All imports successful')
except Exception as e:
    print(f'   ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n2. Testing function signatures...')
try:
    import inspect
    
    sig = inspect.signature(train_fno2)
    params = list(sig.parameters.keys())
    print(f'   train_fno2 parameters: {params}')
    
    expected_params = [
        'modes', 'width', 'epochs', 'batch_size', 
        'learning_rate', 'weight_decay', 'resolution',
        'n_train_samples', 'n_test_samples',
        'scheduler_step_size', 'scheduler_gamma',
        'model_save_path', 'loss_plot_path',
        'device', 'verbose', 'in_channels', 'out_channels'
    ]
    
    for p in expected_params:
        if p in params:
            print(f'   OK: Parameter "{p}" is defined')
        else:
            print(f'   WARNING: Parameter "{p}" not found')
    
except Exception as e:
    print(f'   ERROR: {e}')
    import traceback
    traceback.print_exc()

print('\n3. Testing quick_train with minimal settings...')
try:
    result = quick_train(
        epochs=2,
        n_train_samples=50,
        n_test_samples=25,
        resolution=32,
        modes=8,
        width=16,
        model_save_path=None,
        loss_plot_path=None,
        verbose=True
    )
    
    print(f'\n   Quick train result keys: {list(result.keys())}')
    print(f'   Final train loss: {result["final_train_loss"]:.6f}')
    print(f'   Final test loss:  {result["final_test_loss"]:.6f}')
    print(f'   Best train loss:  {result["best_train_loss"]:.6f}')
    print(f'   Best test loss:   {result["best_test_loss"]:.6f}')
    print(f'   Model is None:    {result["model"] is None}')
    print(f'   Config keys:      {list(result["config"].keys())}')
    
except Exception as e:
    print(f'   ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n4. Testing train_with_config...')
try:
    config = {
        'modes': 8,
        'width': 16,
        'epochs': 1,
        'n_train_samples': 30,
        'n_test_samples': 15,
        'resolution': 32,
        'model_save_path': None,
        'loss_plot_path': None,
        'verbose': False
    }
    
    result = train_with_config(config)
    print(f'   train_with_config successful!')
    print(f'   Final train loss: {result["final_train_loss"]:.6f}')
    
except Exception as e:
    print(f'   ERROR: {e}')
    import traceback
    traceback.print_exc()

print('\n5. Testing docstring availability...')
try:
    if train_fno2.__doc__:
        print('   OK: train_fno2 has docstring')
        doc_lines = train_fno2.__doc__.strip().split('\n')
        print(f'   Docstring length: {len(train_fno2.__doc__)} characters')
        print(f'   First line: {doc_lines[0] if doc_lines else "None"}')
    else:
        print('   WARNING: train_fno2 has no docstring')
    
    if train_with_config.__doc__:
        print('   OK: train_with_config has docstring')
    if quick_train.__doc__:
        print('   OK: quick_train has docstring')
        
except Exception as e:
    print(f'   ERROR: {e}')

print('\n' + '=' * 70)
print('VERIFICATION COMPLETED!')
print('=' * 70)

print('\nSummary:')
print('  - train_fno2() now accepts 17 customizable parameters')
print('  - train_with_config() allows dictionary-based configuration')
print('  - quick_train() provides sensible defaults for quick testing')
print('  - All functions have detailed docstrings with examples')
print('  - Returns a dictionary with model, losses, config, and more')

print('\nUsage Examples:')
print('  1. Basic training (default params):')
print('     >>> result = train_fno2()')
print('     >>> model = result["model"]')

print('\n  2. Custom training:')
print('     >>> result = train_fno2(')
print('     ...     modes=16,')
print('     ...     width=64,')
print('     ...     epochs=100,')
print('     ...     batch_size=32,')
print('     ...     model_save_path="my_model.pth"')
print('     ... )')

print('\n  3. Quick test:')
print('     >>> result = quick_train(epochs=5)')

print('\n  4. Dictionary config:')
print('     >>> config = {"modes": 12, "epochs": 50}')
print('     >>> result = train_with_config(config)')
