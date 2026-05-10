"""
Quick training script for FNO2 on CFDBench (Navier-Stokes)

Usage:
  python run_training_quick.py
  python run_training_quick.py --data_root D:\data --model_output D:\models\my_model.pth
  python run_training_quick.py --epochs 50 --batch_size 16
"""
import sys
import os
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_fno2_cfdbench import train_fno2_cfdbench


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='FNO2 Training on CFDBench (Navier-Stokes)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default=r'C:\traework\data',
        help='Root directory of CFDBench dataset'
    )
    
    parser.add_argument(
        '--model_output',
        type=str,
        default='fno2_cfdbench_quick_model.pth',
        help='Output path for trained model file'
    )
    
    parser.add_argument(
        '--problems',
        type=str,
        default='cavity',
        help='Comma-separated list of problems (e.g., cavity,tube,dam,cylinder)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        default='bc,geo,prop',
        help='Comma-separated list of categories (e.g., bc,geo,prop)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--modes',
        type=int,
        default=12,
        help='Number of Fourier modes (FNO hyperparameter)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=32,
        help='Channel width (FNO hyperparameter)'
    )
    
    parser.add_argument(
        '--max_cases_per_category',
        type=int,
        default=10,
        help='Maximum number of cases per category (use small for quick tests)'
    )
    
    parser.add_argument(
        '--loss_plot',
        type=str,
        default='training_loss_quick.png',
        help='Output path for loss plot'
    )
    
    parser.add_argument(
        '--sample_plot',
        type=str,
        default='prediction_samples_quick.png',
        help='Output path for prediction sample plot'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Suppress verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("FNO2 TRAINING ON CFD BENCH (NAVIER-STOKES)")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if args.device:
        print(f"Using specified device: {args.device}")
    elif torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU mode")
    
    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    
    problems = [p.strip() for p in args.problems.split(',')]
    categories = [c.strip() for c in args.categories.split(',')]
    
    config = {
        'Data Root': args.data_root,
        'Problems': problems,
        'Categories': categories,
        'Epochs': args.epochs,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'FNO Modes': args.modes,
        'FNO Width': args.width,
        'Max Cases/Category': args.max_cases_per_category,
        'Model Output': args.model_output,
        'Loss Plot': args.loss_plot,
        'Sample Plot': args.sample_plot,
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("CHECKING DATASET...")
    print("=" * 70)
    
    if not os.path.exists(args.data_root):
        print(f"  ERROR: Data root not found: {args.data_root}")
        print()
        print("Please set the correct data path using --data_root option:")
        print(f"  python {os.path.basename(__file__)} --data_root YOUR_DATA_PATH")
        sys.exit(1)
    
    for problem in problems:
        problem_path = os.path.join(args.data_root, problem)
        if os.path.exists(problem_path):
            print(f"  [OK] Found problem: {problem}")
            for category in categories:
                cat_path = os.path.join(problem_path, category)
                if os.path.exists(cat_path):
                    cases = len([d for d in os.listdir(cat_path) if d.startswith('case')])
                    print(f"    [OK] {category}: {cases} cases")
                else:
                    print(f"    [WARN] {category} not found")
        else:
            print(f"  [WARN] Problem not found: {problem}")
    
    print()
    print("=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70)
    
    result = train_fno2_cfdbench(
        data_root=args.data_root,
        problems=problems,
        categories=categories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        modes=args.modes,
        width=args.width,
        max_cases_per_category=args.max_cases_per_category,
        model_save_path=args.model_output,
        loss_plot_path=args.loss_plot,
        sample_plot_path=args.sample_plot,
        verbose=not args.quiet
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    
    if result.get('model_path'):
        print(f"Model saved to: {result['model_path']}")
    if result.get('loss_plot_path'):
        print(f"Loss plot saved to: {result['loss_plot_path']}")
    if result.get('sample_plot_path'):
        print(f"Sample plot saved to: {result['sample_plot_path']}")
    
    if result.get('final_test_metrics'):
        m = result['final_test_metrics']
        print(f"\nFinal Test Metrics:")
        print(f"  MSE:  {m.get('mse', 0):.6f}")
        print(f"  RMSE: {m.get('rmse', 0):.6f}")
        print(f"  MAE:  {m.get('mae', 0):.6f}")
        print(f"  R2:   {m.get('r2', 0):.4f}")
    
    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("To evaluate the trained model:")
    print(f"  python evaluate_navier_stokes_model.py --model_path {args.model_output}")
    print()
    print("To train with different parameters:")
    print(f"  python {os.path.basename(__file__)} --help")
    print()


if __name__ == '__main__':
    main()
