import sys
import os
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_dataset_generation():
    print("=" * 80)
    print("VERIFYING DATASET GENERATION AND SAVING")
    print("=" * 80)
    
    try:
        from generate_cfdbench_dataset import CFDBenchDataset, load_cfdbench_dataset
        
        print("\n1. Testing dataset generation...")
        dataset = CFDBenchDataset(
            n_train=100,
            n_test=20,
            grid_size=32,
            problem_type="poisson",
            seed=42
        )
        
        train_data, test_data, metadata = dataset.generate()
        print(f"   OK: Dataset generated successfully")
        print(f"      Train samples: {len(train_data['branch_inputs'])}")
        print(f"      Test samples: {len(test_data['branch_inputs'])}")
        print(f"      Branch input shape: {train_data['branch_inputs'].shape}")
        print(f"      Trunk input shape: {train_data['trunk_inputs'].shape}")
        print(f"      Output shape: {train_data['outputs'].shape}")
        
        output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n2. Testing dataset saving (NPZ format)...")
        npz_path = os.path.join(output_dir, "test_dataset.npz")
        dataset.save(npz_path, format="npz")
        print(f"   OK: Dataset saved to {npz_path}")
        
        print("\n3. Testing dataset loading (NPZ format)...")
        loaded_dataset = load_cfdbench_dataset(npz_path)
        print(f"   OK: Dataset loaded from {npz_path}")
        print(f"      Train samples: {loaded_dataset.n_train}")
        print(f"      Test samples: {loaded_dataset.n_test}")
        
        print("\n4. Testing dataset saving (HDF5 format)...")
        hdf5_path = os.path.join(output_dir, "test_dataset.h5")
        try:
            dataset.save(hdf5_path, format="hdf5")
            print(f"   OK: Dataset saved to {hdf5_path}")
            
            print("\n5. Testing dataset loading (HDF5 format)...")
            loaded_dataset_h5 = load_cfdbench_dataset(hdf5_path)
            print(f"   OK: Dataset loaded from {hdf5_path}")
        except ImportError:
            print("   SKIP: h5py not installed, skipping HDF5 tests")
        
        print("\n6. Testing get_pytorch_dataset...")
        try:
            train_ds, test_ds = dataset.get_pytorch_dataset(normalize=True)
            print(f"   OK: PyTorch datasets created")
            print(f"      Train dataset length: {len(train_ds)}")
            
            sample = train_ds[0]
            print(f"      Sample branch shape: {sample[0].shape}")
            print(f"      Sample trunk shape: {sample[1].shape}")
            print(f"      Sample output shape: {sample[2].shape}")
        except ImportError:
            print("   SKIP: PyTorch not installed")
        
        print("\n7. Testing get_paddle_dataset...")
        try:
            train_ds, test_ds = dataset.get_paddle_dataset(normalize=True)
            print(f"   OK: PaddlePaddle datasets created")
            print(f"      Train dataset length: {len(train_ds)}")
            
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


def verify_external_dataset_training():
    print("\n" + "=" * 80)
    print("VERIFYING EXTERNAL DATASET TRAINING")
    print("=" * 80)
    
    try:
        from train import TrainerFactory, TrainingConfig, DatasetConfig
        from data.cfd_bench_dataset import CFDDatasetPyTorch, HAS_TORCH
        
        if not HAS_TORCH:
            print("   SKIP: PyTorch not installed, skipping external dataset training test")
            return True
        
        print("\n1. Generating test data arrays...")
        n_train = 50
        n_test = 10
        grid_size = 32
        
        branch_train = np.random.randn(n_train, grid_size * grid_size).astype(np.float32)
        trunk_train = np.random.randn(n_train, grid_size * grid_size, 2).astype(np.float32)
        output_train = np.random.randn(n_train, grid_size * grid_size).astype(np.float32)
        
        branch_test = np.random.randn(n_test, grid_size * grid_size).astype(np.float32)
        trunk_test = np.random.randn(n_test, grid_size * grid_size, 2).astype(np.float32)
        output_test = np.random.randn(n_test, grid_size * grid_size).astype(np.float32)
        
        print(f"   OK: Generated test arrays")
        print(f"      Branch shape: {branch_train.shape}")
        print(f"      Trunk shape: {trunk_train.shape}")
        print(f"      Output shape: {output_train.shape}")
        
        print("\n2. Creating DatasetConfig with arrays...")
        dataset_config = DatasetConfig(
            branch_inputs=branch_train,
            trunk_inputs=trunk_train,
            outputs=output_train,
            test_branch_inputs=branch_test,
            test_trunk_inputs=trunk_test,
            test_outputs=output_test
        )
        
        print(f"   OK: DatasetConfig created")
        print(f"      Has external dataset: {dataset_config.has_external_dataset()}")
        
        print("\n3. Creating TrainingConfig with dataset_config...")
        config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            dataset_config=dataset_config
        )
        
        print(f"   OK: TrainingConfig created")
        print(f"      Has external dataset: {config.has_external_dataset()}")
        
        print("\n4. Creating trainer via factory...")
        trainer = TrainerFactory.create("pytorch", config)
        
        print(f"   OK: Trainer created")
        print(f"      Framework: {trainer.get_framework_name()}")
        
        print("\n5. Running quick train (testing external dataset loading)...")
        result = trainer.quick_train(epochs=2)
        
        print(f"   OK: Training completed")
        print(f"      Final train loss: {result.final_train_loss}")
        print(f"      Final test loss: {result.final_test_loss}")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset_path_loading():
    print("\n" + "=" * 80)
    print("VERIFYING DATASET PATH LOADING")
    print("=" * 80)
    
    try:
        from train import TrainerFactory, TrainingConfig
        from generate_cfdbench_dataset import CFDBenchDataset
        
        output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n1. Generating and saving dataset...")
        dataset = CFDBenchDataset(
            n_train=100,
            n_test=20,
            grid_size=32,
            problem_type="poisson",
            seed=123
        )
        dataset.generate()
        
        npz_path = os.path.join(output_dir, "train_test_dataset.npz")
        dataset.save(npz_path, format="npz")
        print(f"   OK: Dataset saved to {npz_path}")
        
        print("\n2. Creating config with dataset_path...")
        config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            dataset_path=npz_path
        )
        
        print(f"   OK: TrainingConfig created")
        print(f"      Has external dataset: {config.has_external_dataset()}")
        
        print("\n3. Creating trainer and running quick train...")
        try:
            trainer = TrainerFactory.create("pytorch", config)
            result = trainer.quick_train(epochs=2)
            
            print(f"   OK: Training completed with loaded dataset")
            print(f"      Final train loss: {result.final_train_loss}")
            print(f"      Final test loss: {result.final_test_loss}")
            
        except ImportError:
            print("   SKIP: PyTorch not installed")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_set_datasets_method():
    print("\n" + "=" * 80)
    print("VERIFYING set_datasets() METHOD")
    print("=" * 80)
    
    try:
        from train import TrainerFactory, TrainingConfig
        from data.cfd_bench_dataset import CFDDatasetPyTorch, HAS_TORCH
        
        if not HAS_TORCH:
            print("   SKIP: PyTorch not installed")
            return True
        
        print("\n1. Creating test datasets...")
        n_train = 50
        n_test = 10
        grid_size = 32
        
        branch_train = np.random.randn(n_train, grid_size * grid_size).astype(np.float32)
        trunk_train = np.random.randn(n_train, grid_size * grid_size, 2).astype(np.float32)
        output_train = np.random.randn(n_train, grid_size * grid_size).astype(np.float32)
        
        branch_test = np.random.randn(n_test, grid_size * grid_size).astype(np.float32)
        trunk_test = np.random.randn(n_test, grid_size * grid_size, 2).astype(np.float32)
        output_test = np.random.randn(n_test, grid_size * grid_size).astype(np.float32)
        
        train_dataset = CFDDatasetPyTorch(branch_train, trunk_train, output_train, normalize=True)
        test_dataset = CFDDatasetPyTorch(branch_test, trunk_test, output_test, normalize=True)
        
        print(f"   OK: Datasets created")
        print(f"      Train dataset length: {len(train_dataset)}")
        print(f"      Test dataset length: {len(test_dataset)}")
        
        print("\n2. Creating trainer and setting datasets manually...")
        config = TrainingConfig(
            epochs=2,
            batch_size=16,
            branch_input_dim=grid_size * grid_size,
            trunk_input_dim=2,
            output_dim=grid_size * grid_size
        )
        
        trainer = TrainerFactory.create("pytorch", config)
        trainer.setup_device()
        
        trainer.set_datasets(train_dataset, test_dataset)
        
        print(f"   OK: Datasets set via set_datasets()")
        
        print("\n3. Building model and training...")
        trainer.actual_branch_dim = grid_size * grid_size
        trainer.actual_trunk_dim = 2
        trainer.actual_output_dim = grid_size * grid_size
        trainer.build_model()
        trainer.build_optimizer()
        trainer.build_scheduler()
        trainer.build_criterion()
        
        for epoch in range(1, 3):
            train_loss = trainer.train_epoch(epoch)
            test_loss = trainer.evaluate(epoch)
            print(f"      Epoch {epoch}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        
        print(f"   OK: Training completed")
        
        return True
        
    except Exception as e:
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("""
1. Generate and save dataset:

   from generate_cfdbench_dataset import CFDBenchDataset
   
   # Generate Poisson dataset
   dataset = CFDBenchDataset(
       n_train=1000,
       n_test=200,
       grid_size=64,
       problem_type="poisson",
       seed=42
   )
   
   dataset.generate()
   
   # Save as NPZ
   dataset.save("data/poisson_dataset.npz", format="npz")
   
   # Save as HDF5
   dataset.save("data/poisson_dataset.h5", format="hdf5")

2. Train with external dataset (via dataset path):

   from train import train_deeponet
   
   result = train_deeponet(
       framework="pytorch",
       epochs=100,
       batch_size=64,
       dataset_path="data/poisson_dataset.npz"
   )

3. Train with numpy arrays:

   import numpy as np
   from train import TrainerFactory, TrainingConfig, DatasetConfig
   
   # Prepare your data
   branch_train = np.random.randn(1000, 4096).astype(np.float32)
   trunk_train = np.random.randn(1000, 4096, 2).astype(np.float32)
   output_train = np.random.randn(1000, 4096).astype(np.float32)
   
   branch_test = np.random.randn(200, 4096).astype(np.float32)
   trunk_test = np.random.randn(200, 4096, 2).astype(np.float32)
   output_test = np.random.randn(200, 4096).astype(np.float32)
   
   # Create dataset config
   dataset_config = DatasetConfig(
       branch_inputs=branch_train,
       trunk_inputs=trunk_train,
       outputs=output_train,
       test_branch_inputs=branch_test,
       test_trunk_inputs=trunk_test,
       test_outputs=output_test
   )
   
   # Create trainer
   config = TrainingConfig(
       epochs=100,
       batch_size=64,
       dataset_config=dataset_config
   )
   
   trainer = TrainerFactory.create("pytorch", config)
   result = trainer.train()

4. Train with pre-built dataset objects:

   from train import PyTorchDeepONetTrainer, TrainingConfig
   from data.cfd_bench_dataset import CFDDatasetPyTorch
   
   # Create datasets
   train_dataset = CFDDatasetPyTorch(branch_train, trunk_train, output_train)
   test_dataset = CFDDatasetPyTorch(branch_test, trunk_test, output_test)
   
   # Create trainer
   config = TrainingConfig(
       epochs=100,
       batch_size=64,
       branch_input_dim=4096,
       trunk_input_dim=2,
       output_dim=4096
   )
   
   trainer = PyTorchDeepONetTrainer(config)
   trainer.setup_device()
   
   # Set datasets manually
   trainer.set_datasets(train_dataset, test_dataset)
   
   # Build model and train
   trainer.actual_branch_dim = 4096
   trainer.actual_trunk_dim = 2
   trainer.actual_output_dim = 4096
   trainer.build_model()
   
   result = trainer.train()

5. Command line usage:

   # Generate dataset
   python generate_cfdbench_dataset.py --output data/poisson.npz --n_train 1000 --n_test 200
   
   # Train with external dataset
   python train_deeponet.py --framework pytorch --epochs 100 --dataset_path data/poisson.npz
   
   # Load existing dataset for inspection
   python generate_cfdbench_dataset.py --load data/poisson.npz
    """)
    
    print("=" * 80)


def cleanup_test_files():
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"   OK: Removed test output directory: {output_dir}")
    else:
        print(f"   OK: No test files to clean up")


def main():
    print("=" * 80)
    print("CFDBENCH DATASET AND EXTERNAL DATASET TRAINING VERIFICATION")
    print("=" * 80)
    
    all_passed = True
    
    if not verify_dataset_generation():
        all_passed = False
    
    if not verify_external_dataset_training():
        all_passed = False
    
    if not verify_dataset_path_loading():
        all_passed = False
    
    if not verify_set_datasets_method():
        all_passed = False
    
    show_usage_examples()
    
    cleanup_test_files()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
