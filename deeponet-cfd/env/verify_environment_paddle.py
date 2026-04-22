import os
import sys
import platform
from datetime import datetime


def setup_logger():
    import logging
    logger = logging.getLogger("Paddle_Environment")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def check_paddle_version(logger):
    logger.info("=" * 80)
    logger.info("PADDLEPADDLE ENVIRONMENT CHECK")
    logger.info("=" * 80)
    
    try:
        import paddle
        logger.info(f"\nPaddlePaddle Version: {paddle.__version__}")
        logger.info(f"PaddlePaddle Path: {paddle.__file__}")
        
        logger.info(f"\nPaddlePaddle Devices:")
        logger.info(f"  CPU: Available")
        
        try:
            gpu_available = paddle.is_compiled_with_cuda()
            logger.info(f"  CUDA/GPU: {'Available' if gpu_available else 'Not available'}")
            
            if gpu_available:
                gpu_count = paddle.device.cuda.device_count()
                logger.info(f"  GPU Count: {gpu_count}")
                
                for i in range(gpu_count):
                    props = paddle.device.cuda.get_device_properties(i)
                    logger.info(f"  GPU {i}: {props['name']}")
        except Exception as e:
            logger.info(f"  GPU Info: Could not retrieve - {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"PaddlePaddle import FAILED: {e}")
        logger.info("\nPlease install PaddlePaddle:")
        logger.info("  CPU version: pip install paddlepaddle==2.6.2")
        logger.info("  GPU version (CUDA 11.8): pip install paddlepaddle-gpu==2.6.2.post118 -i https://mirror.baidu.com/pypi/simple")
        return False


def check_common_packages(logger):
    logger.info("\n" + "=" * 80)
    logger.info("COMMON PACKAGE CHECK")
    logger.info("=" * 80)
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("h5py", "h5py"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
    ]
    
    all_ok = True
    for pkg_name, display_name in packages:
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown")
            logger.info(f"  {display_name:15s}: OK (version {version})")
        except ImportError as e:
            logger.error(f"  {display_name:15s}: FAILED - {e}")
            all_ok = False
    
    return all_ok


def check_fno_model(logger):
    logger.info("\n" + "=" * 80)
    logger.info("FNO MODEL CHECK (PaddlePaddle)")
    logger.info("=" * 80)
    
    try:
        import paddle
        
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.fno_cfdbench_paddle import AutoRegressiveFNOPaddle, count_parameters_paddle
        
        logger.info("\nCreating FNO model (small configuration)...")
        model = AutoRegressiveFNOPaddle(
            modes1=8,
            modes2=8,
            width=16,
            in_channels=2,
            out_channels=2,
            n_layers=2,
            hidden_dim=64
        )
        
        total_params = count_parameters_paddle(model)
        logger.info(f"Model parameters: {total_params:,}")
        
        logger.info(f"\nModel info:")
        for k, v in model.get_model_info().items():
            logger.info(f"  {k}: {v}")
        
        logger.info(f"\nTesting forward pass...")
        batch_size = 2
        grid_size = 64
        
        x = paddle.randn([batch_size, grid_size, grid_size, 2], dtype='float32')
        
        with paddle.no_grad():
            output = model(x)
            logger.info(f"  Input shape:  {x.shape}")
            logger.info(f"  Output shape: {output.shape}")
            
            logger.info(f"\nTesting multi-step autoregressive prediction...")
            multi_step = model.generate_many(x, n_steps=5)
            logger.info(f"  Input shape:    {x.shape}")
            logger.info(f"  Output shape:   {multi_step.shape}")
            logger.info(f"    (batch, steps, H, W, channels)")
            
            logger.info(f"\n  Step-wise stats:")
            for step in range(5):
                step_data = multi_step[:, step, ...]
                logger.info(f"    Step {step+1}: min={step_data.min().item():.4f}, max={step_data.max().item():.4f}, mean={step_data.mean().item():.4f}")
        
        logger.info(f"\nFNO Model (PaddlePaddle): OK")
        return True
        
    except ImportError as e:
        logger.error(f"FNO Model import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"FNO Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_loader(logger):
    logger.info("\n" + "=" * 80)
    logger.info("CFDBENCH DATA LOADER CHECK")
    logger.info("=" * 80)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from train_fno_cfdbench_paddle import CFDBenchDataset, CFDBCategory, CFDBCase
        
        logger.info(f"\nAvailable classes:")
        logger.info(f"  - CFDBCase: Single case (u.npy, v.npy, p.npy)")
        logger.info(f"  - CFDBCategory: Category (bc, geo, prop)")
        logger.info(f"  - CFDBenchDataset: Full dataset with multiple problems/categories")
        
        logger.info(f"\nLoader features:")
        logger.info(f"  - load(): Load data from directory")
        logger.info(f"  - get_normalization_params(): Get normalization statistics")
        logger.info(f"  - __len__(): Get number of samples")
        logger.info(f"  - __getitem__(): Get sample by index")
        
        logger.info(f"\nCFDBench Data Loader: OK")
        return True
        
    except ImportError as e:
        logger.error(f"CFDBench Loader import FAILED: {e}")
        return False


def check_training_script(logger):
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SCRIPT CHECK")
    logger.info("=" * 80)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from train_fno_cfdbench_paddle import TrainingConfig, TrainingResult, CFDBenchTrainerPaddle
        
        logger.info(f"\nAvailable classes:")
        logger.info(f"  - TrainingConfig: Training configuration")
        logger.info(f"  - TrainingResult: Training results")
        logger.info(f"  - CFDBenchTrainerPaddle: Main trainer class")
        
        logger.info(f"\nTrainingConfig parameters:")
        config = TrainingConfig()
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    logger.info(f"  - {attr}: {value}")
        
        logger.info(f"\nTrainer methods:")
        logger.info(f"  - setup_data(): Load and prepare dataset")
        logger.info(f"  - build_model(): Build FNO model")
        logger.info(f"  - build_optimizer(): Build optimizer")
        logger.info(f"  - build_scheduler(): Build learning rate scheduler")
        logger.info(f"  - build_criterion(): Build loss function")
        logger.info(f"  - train_epoch(): Train one epoch")
        logger.info(f"  - evaluate(): Evaluate on test set")
        logger.info(f"  - save_model(): Save model checkpoint")
        logger.info(f"  - train(): Full training loop")
        
        logger.info(f"\nTraining Script: OK")
        return True
        
    except ImportError as e:
        logger.error(f"Training Script import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("PADDLEPADDLE FNO TRAINING ENVIRONMENT VERIFICATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    logger = setup_logger()
    
    logger.info(f"\nPython Version: {sys.version}")
    logger.info(f"Python Path: {sys.executable}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    results = []
    
    results.append(("PaddlePaddle", check_paddle_version(logger)))
    results.append(("Common Packages", check_common_packages(logger)))
    results.append(("FNO Model", check_fno_model(logger)))
    results.append(("Data Loader", check_data_loader(logger)))
    results.append(("Training Script", check_training_script(logger)))
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "OK" if result else "FAILED"
        if not result:
            all_passed = False
        logger.info(f"  {name:20s}: {status}")
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("ALL CHECKS PASSED!")
        logger.info("\nYour PaddlePaddle environment is ready for FNO training.")
        logger.info("\nNext steps:")
        logger.info("  1. Ensure CFDBench data is in 'data/' directory")
        logger.info("  2. Run training:")
        logger.info("     python train_fno_cfdbench_paddle.py --help")
        logger.info("     python train_fno_cfdbench_paddle.py --epochs 3 --max_cases 5 --verbose")
    else:
        logger.info("SOME CHECKS FAILED!")
        logger.info("\nPlease fix the issues above before proceeding.")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
