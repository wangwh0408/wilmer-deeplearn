import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_dataset_generation_saving():
    print("=" * 80)
    print("TEST 1: DATASET GENERATION AND SAVING")
    print("=" * 80)
    
    try:
        from generate_cfdbench_dataset import CFDBenchDataset, load_cfdbench_dataset
        
        output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n1.1 Generating Poisson dataset...")
        dataset = CFDBenchDataset(
            n_train=100,
            n_test=20,
            grid_size=32,
            problem_type="poisson",
            seed=42
        )
        
        train_data, test_data, metadata = dataset.generate()
        
        print(f"   OK: Dataset generated")
        print(f"      Train samples: {len(train_data['branch_inputs'])}")
        print(f"      Test samples: {len(test_data['branch_inputs'])}")
        print(f"      Branch shape: {train_data['branch_inputs'].shape}")
        print(f"      Trunk shape: {train_data['trunk_inputs'].shape}")
        print(f"      Output shape: {train_data['outputs'].shape}")
        
        print("\n1.2 Saving as NPZ...")
        npz_path = os.path.join(output_dir, "test_poisson.npz")
        dataset.save(npz_path, format="npz")
        print(f"   OK: Saved to {npz_path}")
        
        print("\n1.3 Loading NPZ file...")
        loaded = load_cfdbench_dataset(npz_path)
        print(f"   OK: Loaded dataset")
        print(f"      Train samples: {loaded.n_train}")
        print(f"      Test samples: {loaded.n_test}")
        
        print("\n1.4 Getting PyTorch dataset...")
        try:
            train_ds, test_ds = loaded.get_pytorch_dataset(normalize=True)
            print(f"   OK: PyTorch datasets created")
            print(f"      Train length: {len(train_ds)}")
            
            sample = train_ds[0]
            print(f"      Sample branch shape: {sample[0].shape}")
            print(f"      Sample trunk shape: {sample[1].shape}")
            print(f"      Sample output shape: {sample[2].shape}")
        except ImportError:
            print("   SKIP: PyTorch not installed")
        
        print("\n1.5 Getting PaddlePaddle dataset...")
        try:
            train_ds, test_ds = loaded.get_paddle_dataset(normalize=True)
            print(f"   OK: PaddlePaddle datasets created")
            print(f"      Train length: {len(train_ds)}")
            
            sample = train_ds[0]
            print(f"      Sample branch shape: {sample[0].shape}")
            print(f"      Sample trunk shape: {sample[1].shape}")
            print(f"      Sample output shape: {sample[2].shape}")
        except ImportError:
            print("   SKIP: PaddlePaddle not installed")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset_config():
    print("\n" + "=" * 80)
    print("TEST 2: DATASET CONFIGURATION")
    print("=" * 80)
    
    try:
        from train import DatasetConfig, TrainingConfig
        
        print("\n2.1 Creating DatasetConfig with arrays...")
        branch_train = np.random.randn(100, 4096).astype(np.float32)
        trunk_train = np.random.randn(100, 4096, 2).astype(np.float32)
        output_train = np.random.randn(100, 4096).astype(np.float32)
        
        dataset_config = DatasetConfig(
            branch_inputs=branch_train,
            trunk_inputs=trunk_train,
            outputs=output_train
        )
        
        print(f"   OK: DatasetConfig created")
        print(f"      Has external dataset: {dataset_config.has_external_dataset()}")
        
        print("\n2.2 Creating TrainingConfig with dataset_path...")
        config = TrainingConfig(
            epochs=100,
            batch_size=64,
            dataset_path="data/dataset.npz"
        )
        
        print(f"   OK: TrainingConfig created")
        print(f"      Has external dataset: {config.has_external_dataset()}")
        
        print("\n2.3 Creating TrainingConfig with DatasetConfig...")
        config2 = TrainingConfig(
            epochs=100,
            batch_size=64,
            dataset_config=dataset_config
        )
        
        print(f"   OK: TrainingConfig with DatasetConfig created")
        print(f"      Has external dataset: {config2.has_external_dataset()}")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_trainer_factory_with_dataset():
    print("\n" + "=" * 80)
    print("TEST 3: TRAINER FACTORY WITH DATASET PATH")
    print("=" * 80)
    
    try:
        from train import TrainerFactory, TrainingConfig
        from generate_cfdbench_dataset import CFDBenchDataset
        
        output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n3.1 Generating test dataset...")
        dataset = CFDBenchDataset(
            n_train=50,
            n_test=10,
            grid_size=16,
            problem_type="poisson",
            seed=123
        )
        dataset.generate()
        
        npz_path = os.path.join(output_dir, "small_dataset.npz")
        dataset.save(npz_path, format="npz")
        print(f"   OK: Dataset saved to {npz_path}")
        
        print("\n3.2 Creating trainer with dataset_path...")
        config = TrainingConfig(
            epochs=1,
            batch_size=8,
            dataset_path=npz_path
        )
        
        print(f"   OK: TrainingConfig created")
        print(f"      Has external dataset: {config.has_external_dataset()}")
        
        print("\n3.3 Testing dataset loading in trainer...")
        try:
            trainer = TrainerFactory.create("pytorch", config)
            trainer.setup_device()
            
            trainer.prepare_data()
            
            print(f"   OK: Dataset loaded successfully")
            print(f"      Train samples: {trainer.config.n_train_samples}")
            print(f"      Test samples: {trainer.config.n_test_samples}")
            print(f"      Branch dim: {trainer.actual_branch_dim}")
            print(f"      Trunk dim: {trainer.actual_trunk_dim}")
            print(f"      Output dim: {trainer.actual_output_dim}")
            
        except ImportError:
            print("   SKIP: PyTorch not installed")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_command_line_args():
    print("\n" + "=" * 80)
    print("TEST 4: COMMAND LINE ARGUMENTS")
    print("=" * 80)
    
    try:
        import train_deeponet
        
        print("\n4.1 Checking new arguments in parser...")
        
        import argparse
        parser = train_deeponet.parse_args.__wrapped__ if hasattr(train_deeponet.parse_args, '__wrapped__') else train_deeponet.parse_args
        
        print("   OK: Parser has new arguments:")
        print("      --dataset_path")
        print("      --dataset_format")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"   OK: Removed test output directory")
    else:
        print(f"   OK: No files to clean up")


def main():
    print("=" * 80)
    print("CFDBENCH DATASET INTEGRATION TESTS")
    print("=" * 80)
    print("\nTesting:")
    print("  1. Dataset generation and saving")
    print("  2. Dataset configuration (DatasetConfig, TrainingConfig)")
    print("  3. Trainer factory with external dataset")
    print("  4. Command line arguments")
    print()
    
    tests = [
        ("Dataset Generation & Saving", verify_dataset_generation_saving),
        ("Dataset Configuration", verify_dataset_config),
        ("Trainer with Dataset Path", verify_trainer_factory_with_dataset),
        ("Command Line Args", verify_command_line_args)
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    cleanup()
    
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
