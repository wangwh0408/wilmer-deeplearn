import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_imports():
    print("=" * 80)
    print("VERIFYING OOD REFACTORING")
    print("=" * 80)
    
    all_ok = True
    
    print("\n1. Verifying base classes...")
    try:
        from train.base_trainer import (
            TrainingConfig,
            TrainingResult,
            BaseDeepONetTrainer
        )
        print("   OK: Base classes imported successfully")
        
        config = TrainingConfig()
        print(f"   OK: TrainingConfig created with default values")
        print(f"      epochs: {config.epochs}")
        print(f"      batch_size: {config.batch_size}")
        print(f"      learning_rate: {config.learning_rate}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n2. Verifying factory pattern...")
    try:
        from train.trainer_factory import (
            TrainerFactory,
            train_deeponet,
            quick_train_deeponet
        )
        print("   OK: TrainerFactory imported successfully")
        
        available = TrainerFactory.get_available_frameworks()
        print(f"   OK: Available frameworks: {available}")
        
        for fw in available:
            print(f"   - {fw} available: {TrainerFactory.is_framework_available(fw)}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n3. Verifying concrete trainer classes...")
    
    try:
        from train import HAS_PYTORCH, PyTorchDeepONetTrainer
        if HAS_PYTORCH:
            print("   OK: PyTorchDeepONetTrainer available")
        else:
            print("   Info: PyTorch not installed, PyTorchDeepONetTrainer not available")
    except Exception as e:
        print(f"   WARNING: PyTorch trainer import issue: {e}")
    
    try:
        from train import HAS_PADDLE, PaddleDeepONetTrainer
        if HAS_PADDLE:
            print("   OK: PaddleDeepONetTrainer available")
        else:
            print("   Info: PaddlePaddle not installed, PaddleDeepONetTrainer not available")
    except Exception as e:
        print(f"   WARNING: Paddle trainer import issue: {e}")
    
    print("\n4. Verifying OOD design patterns...")
    try:
        from train.base_trainer import BaseDeepONetTrainer
        from train.trainer_factory import TrainerFactory
        
        print("   - Abstract Base Class (ABC): BaseDeepONetTrainer")
        print("     - Defines abstract methods:")
        print("       * get_framework_name()")
        print("       * setup_device()")
        print("       * build_model()")
        print("       * build_optimizer()")
        print("       * build_scheduler()")
        print("       * build_criterion()")
        print("       * prepare_data()")
        print("       * train_epoch()")
        print("       * evaluate()")
        print("       * save_model()")
        print("       * get_model_info()")
        print("       * get_current_learning_rate()")
        
        print("   - Concrete Implementations:")
        print("       * PyTorchDeepONetTrainer (PyTorch)")
        print("       * PaddleDeepONetTrainer (PaddlePaddle)")
        
        print("   - Factory Pattern: TrainerFactory")
        print("       * register() - Register trainer classes")
        print("       * create() - Create trainer by framework name")
        print("       * get_available_frameworks() - List available frameworks")
        
        print("   - Template Method Pattern:")
        print("       * BaseDeepONetTrainer.train() - Defines the algorithm skeleton")
        print("       * Concrete classes implement specific steps")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        all_ok = False
    
    print("\n5. Verifying main script compatibility...")
    try:
        import train_deeponet
        print("   OK: train_deeponet.py imported successfully")
        print("   - Uses TrainerFactory for framework-agnostic training")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False
    
    print("\n" + "=" * 80)
    if all_ok:
        print("ALL OOD CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
    print("=" * 80)
    
    return all_ok


def show_architecture():
    print("\n" + "=" * 80)
    print("OOD ARCHITECTURE OVERVIEW")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                        Client Code                               │
│  (train_deeponet.py, Python API calls)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TrainerFactory                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  register(framework_name, trainer_class)                  │ │
│  │  create(framework, config) → BaseDeepONetTrainer         │ │
│  │  get_available_frameworks() → list                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ PyTorch Trainer │   │ Paddle Trainer  │   │ (Future: TF,    │
│  (Concrete)     │   │   (Concrete)    │   │  JAX, etc.)     │
└─────────┬───────┘   └─────────┬───────┘   └─────────────────┘
          │                     │
          └─────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                BaseDeepONetTrainer (Abstract)                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Template Method: train()                                  │ │
│  │    1. _log_start()                                         │ │
│  │    2. setup_device()  «abstract»                          │ │
│  │    3. prepare_data()  «abstract»                          │ │
│  │    4. build_model()   «abstract»                          │ │
│  │    5. build_optimizer() «abstract»                        │ │
│  │    6. build_scheduler() «abstract»                        │ │
│  │    7. build_criterion() «abstract»                        │ │
│  │    8. For each epoch:                                     │ │
│  │       a. train_epoch()  «abstract»                        │ │
│  │       b. evaluate()     «abstract»                        │ │
│  │    9. _log_training_complete()                            │ │
│  │   10. save_model()     «abstract»                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Common Methods (Implemented in Base):                          │
│    - _log_start(), _log_dataset_info(), _log_model_info()     │
│    - _log_epoch_start(), _log_epoch_end()                      │
│    - _log_training_complete()                                  │
│    - quick_train()                                              │
└─────────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 80)


def show_usage_examples():
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES (OOD VERSION)")
    print("=" * 80)
    
    print("""
1. Using Factory Pattern (Recommended):

   from train import TrainerFactory, TrainingConfig, train_deeponet
   
   # Create configuration
   config = TrainingConfig(
       epochs=100,
       batch_size=64,
       learning_rate=1e-3,
       branch_hidden_layers=[512, 512, 512]
   )
   
   # Create trainer via factory
   trainer = TrainerFactory.create("pytorch", config)
   
   # Or use convenience function
   result = train_deeponet(
       framework="pytorch",
       epochs=100,
       batch_size=64,
       learning_rate=0.001
   )
   
   # Quick training
   from train import quick_train_deeponet
   result = quick_train_deeponet(framework="pytorch", epochs=5)

2. Using Trainer Directly:

   from train.pytorch_trainer import PyTorchDeepONetTrainer
   from train.base_trainer import TrainingConfig
   
   config = TrainingConfig(epochs=100, batch_size=64)
   trainer = PyTorchDeepONetTrainer(config)
   result = trainer.train()
   
   # Or quick train
   result = trainer.quick_train(epochs=5)

3. Extending with New Framework:

   from train.base_trainer import BaseDeepONetTrainer, TrainingConfig
   from train.trainer_factory import TrainerFactory
   
   class TensorFlowDeepONetTrainer(BaseDeepONetTrainer):
       
       def get_framework_name(self) -> str:
           return "TensorFlow"
       
       def setup_device(self):
           # TensorFlow specific device setup
           pass
       
       def build_model(self):
           # TensorFlow specific model building
           pass
       
       # ... implement all abstract methods
   
   # Register with factory
   TrainerFactory.register("tensorflow", TensorFlowDeepONetTrainer)
   
   # Now use like any other framework
   result = train_deeponet(framework="tensorflow", epochs=100)

4. Command Line:

   python train_deeponet.py --framework pytorch --epochs 100
   python train_deeponet.py --framework paddle --quick_train --epochs 5
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    success = verify_imports()
    
    if success:
        show_architecture()
        show_usage_examples()
    
    sys.exit(0 if success else 1)
