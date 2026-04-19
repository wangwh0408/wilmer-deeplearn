import argparse
import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepONet model on CFDBench dataset (OOD Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OOD Design Overview:
  - BaseDeepONetTrainer: Abstract base class defining the training interface
  - PyTorchDeepONetTrainer: PyTorch-specific implementation
  - PaddleDeepONetTrainer: PaddlePaddle-specific implementation  
  - TrainerFactory: Factory class to create appropriate trainer

Examples:
  # PyTorch, default parameters
  python train_deeponet.py --framework pytorch
  
  # PaddlePaddle, custom epochs and batch size
  python train_deeponet.py --framework paddle --epochs 200 --batch_size 128
  
  # Quick training (small dataset, few epochs)
  python train_deeponet.py --framework pytorch --quick_train --epochs 5
  
  # Save model and log to file
  python train_deeponet.py --framework pytorch --model_save_path models/deeponet.pth --log_file logs/train.log
  
  # Custom network architecture
  python train_deeponet.py --framework pytorch --branch_hidden_layers 512 512 512 --trunk_hidden_layers 512 512
        """
    )
    
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        choices=["pytorch", "paddle", "torch", "paddlepaddle"],
        help="Deep learning framework to use (default: pytorch)"
    )
    
    parser.add_argument(
        "--quick_train",
        action="store_true",
        help="Use quick training mode (smaller dataset, fewer epochs)"
    )
    
    parser.add_argument(
        "--branch_input_dim",
        type=int,
        default=4096,
        help="Input dimension for branch network (default: 4096)"
    )
    
    parser.add_argument(
        "--trunk_input_dim",
        type=int,
        default=2,
        help="Input dimension for trunk network (default: 2)"
    )
    
    parser.add_argument(
        "--branch_hidden_layers",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="Hidden layer sizes for branch network (default: 256 256 256)"
    )
    
    parser.add_argument(
        "--trunk_hidden_layers",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="Hidden layer sizes for trunk network (default: 256 256 256)"
    )
    
    parser.add_argument(
        "--output_dim",
        type=int,
        default=4096,
        help="Output dimension (default: 4096)"
    )
    
    parser.add_argument(
        "--branch_activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid", "gelu", "leaky_relu", "elu", "selu"],
        help="Activation function for branch network (default: relu)"
    )
    
    parser.add_argument(
        "--trunk_activation",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "sigmoid", "gelu", "leaky_relu", "elu", "selu"],
        help="Activation function for trunk network (default: tanh)"
    )
    
    parser.add_argument(
        "--branch_dropout",
        type=float,
        default=0.0,
        help="Dropout rate for branch network (default: 0.0)"
    )
    
    parser.add_argument(
        "--no_positional_encoding",
        action="store_true",
        help="Disable positional encoding for trunk network"
    )
    
    parser.add_argument(
        "--positional_encoding_freqs",
        type=int,
        default=10,
        help="Number of frequencies for positional encoding (default: 10)"
    )
    
    parser.add_argument(
        "--no_bias",
        action="store_true",
        help="Disable bias term in final output"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization) (default: 1e-5)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=50,
        help="Step size for learning rate scheduler (default: 50)"
    )
    
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Gamma for learning rate scheduler (default: 0.5)"
    )
    
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)"
    )
    
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=200,
        help="Number of test samples (default: 200)"
    )
    
    parser.add_argument(
        "--grid_size",
        type=int,
        default=64,
        help="Grid size for CFD data (default: 64)"
    )
    
    parser.add_argument(
        "--problem_type",
        type=str,
        default="poisson",
        choices=["poisson", "navier_stokes"],
        help="Type of CFD problem (default: poisson)"
    )
    
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable data normalization"
    )
    
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the trained model (default: None)"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to save log file (default: None)"
    )
    
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Print verbose output (default: True)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu for PyTorch, gpu/cpu for Paddle)"
    )
    
    return parser.parse_args()


def args_to_config(args) -> dict:
    return {
        "branch_input_dim": args.branch_input_dim,
        "trunk_input_dim": args.trunk_input_dim,
        "branch_hidden_layers": args.branch_hidden_layers,
        "trunk_hidden_layers": args.trunk_hidden_layers,
        "output_dim": args.output_dim,
        "branch_activation": args.branch_activation,
        "trunk_activation": args.trunk_activation,
        "branch_dropout": args.branch_dropout,
        "use_positional_encoding": not args.no_positional_encoding,
        "positional_encoding_freqs": args.positional_encoding_freqs,
        "use_bias": not args.no_bias,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "n_train_samples": args.n_train_samples,
        "n_test_samples": args.n_test_samples,
        "grid_size": args.grid_size,
        "problem_type": args.problem_type,
        "normalize_data": not args.no_normalize,
        "model_save_path": args.model_save_path,
        "log_file": args.log_file,
        "verbose": args.verbose,
        "device": args.device
    }


def main():
    args = parse_args()
    
    print("=" * 80)
    print("DEEPONET TRAINING STARTING (OOD VERSION)")
    print("=" * 80)
    print(f"Framework: {args.framework}")
    print(f"Quick Train: {args.quick_train}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Design Pattern: Factory + Template Method")
    print("=" * 80)
    
    config_dict = args_to_config(args)
    
    try:
        from train import TrainerFactory, train_deeponet, quick_train_deeponet
        
        available_frameworks = TrainerFactory.get_available_frameworks()
        print(f"Available frameworks: {available_frameworks}")
        
        if args.quick_train:
            print("\nUsing quick_train_deeponet() via factory...")
            result = quick_train_deeponet(
                framework=args.framework,
                epochs=args.epochs,
                **config_dict
            )
        else:
            print("\nUsing train_deeponet() via factory...")
            result = train_deeponet(
                framework=args.framework,
                **config_dict
            )
        
        print("\n" + "=" * 80)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 80)
        print(f"Framework: {result.framework}")
        print(f"Device: {result.device}")
        print(f"Final Train Loss: {result.final_train_loss}")
        print(f"Final Test Loss:  {result.final_test_loss}")
        print(f"Best Train Loss:  {result.best_train_loss}")
        print(f"Best Test Loss:   {result.best_test_loss}")
        print(f"Model Saved At:   {result.model_path or 'Not saved'}")
        print(f"Duration:         {result.duration_seconds:.2f} seconds")
        print("=" * 80)
        
        return result
        
    except ValueError as e:
        print(f"\nError: {e}")
        print(f"Please install the required framework first.")
        sys.exit(1)
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print(f"Available frameworks: {TrainerFactory.get_available_frameworks() if 'TrainerFactory' in dir() else 'None'}")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
