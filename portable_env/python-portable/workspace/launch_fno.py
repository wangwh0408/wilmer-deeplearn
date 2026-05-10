"""
FNO Model Training/Evaluation Interactive Launcher

This script provides an interactive interface for:
1. Training FNO models on CFDBench data
2. Evaluating trained FNO models

Features:
- Interactive parameter input
- Configuration saving/loading
- Path validation
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class FNOLauncher:
    """Interactive launcher for FNO model training and evaluation"""
    
    DEFAULT_CONFIG = {
        'data_root': r'C:\traework\data',
        'model_path': 'fno2_cfdbench_quick_model.pth',
        'model_output': 'fno2_trained_model.pth',
        'problems': ['cavity'],
        'categories': ['bc', 'geo', 'prop'],
        'epochs': 30,
        'batch_size': 8,
        'learning_rate': 0.001,
        'modes': 12,
        'width': 32,
        'max_cases_per_category': 10,
        'output_report': 'evaluation_report.json'
    }
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.script_dir, 'launcher_config.json')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    return {**self.DEFAULT_CONFIG, **saved}
            except:
                pass
        return self.DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def _print_header(self, title: str):
        """Print section header"""
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print()
    
    def _get_input(self, prompt: str, default: Any, allow_empty: bool = True) -> str:
        """Get user input with default value"""
        default_str = str(default) if default is not None else ''
        if allow_empty:
            prompt = f"{prompt} [{default_str}]: "
        else:
            prompt = f"{prompt} (required): "
        
        user_input = input(prompt).strip()
        
        if allow_empty and user_input == '':
            return default_str
        return user_input
    
    def _get_path_input(self, prompt: str, default: str, path_type: str = 'directory') -> str:
        """Get path input with validation"""
        while True:
            path = self._get_input(prompt, default)
            
            if path_type == 'directory' and os.path.isdir(path):
                return path
            elif path_type == 'file' and os.path.isfile(path):
                return path
            elif path_type == 'output':
                return path
            
            print(f"  [Warning] Path not found: {path}")
            print(f"  Please enter a valid {path_type} path")
            print()
    
    def _print_config(self, title: str, config_items: Dict[str, Any]):
        """Print configuration summary"""
        self._print_header(title)
        for key, value in config_items.items():
            print(f"  {key}: {value}")
    
    def main_menu(self) -> str:
        """Display main menu and get user choice"""
        self._print_header("FNO MODEL TRAINING/EVALUATION LAUNCHER")
        
        print("  What would you like to do?")
        print()
        print("  1. Train a new FNO model")
        print("  2. Evaluate a trained FNO model")
        print("  3. View/Edit configuration")
        print("  4. Save current configuration")
        print("  5. Reset to defaults")
        print("  0. Exit")
        print()
        
        while True:
            choice = input("  Enter choice [1]: ").strip()
            if choice == '' or choice == '1':
                return 'train'
            elif choice == '2':
                return 'evaluate'
            elif choice == '3':
                return 'config'
            elif choice == '4':
                return 'save'
            elif choice == '5':
                return 'reset'
            elif choice == '0':
                return 'exit'
            else:
                print("  Invalid choice. Please try again.")
    
    def train_menu(self):
        """Interactive training configuration"""
        self._print_header("TRAINING CONFIGURATION")
        
        print("  Current settings (press Enter to use default):")
        print()
        
        print("  --- Data Settings ---")
        self.config['data_root'] = self._get_path_input(
            "  Dataset root directory", 
            self.config['data_root'],
            'directory'
        )
        
        problems_input = self._get_input(
            "  Problems (comma-separated)",
            ','.join(self.config['problems'])
        )
        self.config['problems'] = [p.strip() for p in problems_input.split(',')]
        
        categories_input = self._get_input(
            "  Categories (comma-separated)",
            ','.join(self.config['categories'])
        )
        self.config['categories'] = [c.strip() for c in categories_input.split(',')]
        
        self.config['max_cases_per_category'] = int(self._get_input(
            "  Max cases per category",
            self.config['max_cases_per_category']
        ))
        
        print()
        print("  --- Training Hyperparameters ---")
        self.config['epochs'] = int(self._get_input(
            "  Number of epochs",
            self.config['epochs']
        ))
        
        self.config['batch_size'] = int(self._get_input(
            "  Batch size",
            self.config['batch_size']
        ))
        
        self.config['learning_rate'] = float(self._get_input(
            "  Learning rate",
            self.config['learning_rate']
        ))
        
        print()
        print("  --- FNO Architecture ---")
        self.config['modes'] = int(self._get_input(
            "  Number of Fourier modes",
            self.config['modes']
        ))
        
        self.config['width'] = int(self._get_input(
            "  Channel width",
            self.config['width']
        ))
        
        print()
        print("  --- Output Settings ---")
        self.config['model_output'] = self._get_path_input(
            "  Output model file path",
            self.config['model_output'],
            'output'
        )
        
        self._print_config("TRAINING SETTINGS SUMMARY", {
            'Dataset Root': self.config['data_root'],
            'Problems': self.config['problems'],
            'Categories': self.config['categories'],
            'Max Cases/Category': self.config['max_cases_per_category'],
            'Epochs': self.config['epochs'],
            'Batch Size': self.config['batch_size'],
            'Learning Rate': self.config['learning_rate'],
            'Fourier Modes': self.config['modes'],
            'Channel Width': self.config['width'],
            'Output Model': self.config['model_output']
        })
        
        confirm = input("\n  Start training? (y/n) [y]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            self._run_training()
        else:
            print("  Training cancelled.")
    
    def evaluate_menu(self):
        """Interactive evaluation configuration"""
        self._print_header("EVALUATION CONFIGURATION")
        
        print("  Current settings (press Enter to use default):")
        print()
        
        print("  --- Model Settings ---")
        self.config['model_path'] = self._get_path_input(
            "  Path to trained model file",
            self.config['model_path'],
            'file'
        )
        
        print()
        print("  --- Data Settings ---")
        self.config['data_root'] = self._get_path_input(
            "  Dataset root directory",
            self.config['data_root'],
            'directory'
        )
        
        problems_input = self._get_input(
            "  Problems (comma-separated)",
            ','.join(self.config['problems'])
        )
        self.config['problems'] = [p.strip() for p in problems_input.split(',')]
        
        categories_input = self._get_input(
            "  Categories (comma-separated)",
            ','.join(self.config['categories'])
        )
        self.config['categories'] = [c.strip() for c in categories_input.split(',')]
        
        self.config['max_cases_per_category'] = int(self._get_input(
            "  Max cases per category",
            self.config['max_cases_per_category']
        ))
        
        print()
        print("  --- Output Settings ---")
        self.config['output_report'] = self._get_path_input(
            "  Output report file path",
            self.config['output_report'],
            'output'
        )
        
        self._print_config("EVALUATION SETTINGS SUMMARY", {
            'Model File': self.config['model_path'],
            'Dataset Root': self.config['data_root'],
            'Problems': self.config['problems'],
            'Categories': self.config['categories'],
            'Max Cases/Category': self.config['max_cases_per_category'],
            'Output Report': self.config['output_report']
        })
        
        confirm = input("\n  Start evaluation? (y/n) [y]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            self._run_evaluation()
        else:
            print("  Evaluation cancelled.")
    
    def _run_training(self):
        """Execute training script"""
        self._print_header("STARTING TRAINING")
        
        train_script = os.path.join(self.script_dir, 'run_training_quick.py')
        
        if not os.path.exists(train_script):
            print(f"  Error: Training script not found: {train_script}")
            return
        
        cmd = [
            sys.executable,
            train_script,
            '--data_root', self.config['data_root'],
            '--model_output', self.config['model_output'],
            '--problems', ','.join(self.config['problems']),
            '--categories', ','.join(self.config['categories']),
            '--epochs', str(self.config['epochs']),
            '--batch_size', str(self.config['batch_size']),
            '--learning_rate', str(self.config['learning_rate']),
            '--modes', str(self.config['modes']),
            '--width', str(self.config['width']),
            '--max_cases_per_category', str(self.config['max_cases_per_category'])
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        print()
        print("  Training in progress...")
        print("=" * 70)
        print()
        
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print()
                print("=" * 70)
                print("  TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"  Model saved to: {self.config['model_output']}")
            else:
                print()
                print(f"  Training exited with code: {result.returncode}")
        except Exception as e:
            print(f"  Error running training: {e}")
        
        input("\n  Press Enter to continue...")
    
    def _run_evaluation(self):
        """Execute evaluation script"""
        self._print_header("STARTING EVALUATION")
        
        eval_script = os.path.join(self.script_dir, 'evaluate_navier_stokes_model.py')
        
        if not os.path.exists(eval_script):
            print(f"  Error: Evaluation script not found: {eval_script}")
            return
        
        cmd = [
            sys.executable,
            eval_script,
            '--model_path', self.config['model_path'],
            '--data_root', self.config['data_root'],
            '--problems', ','.join(self.config['problems']),
            '--categories', ','.join(self.config['categories']),
            '--max_cases_per_category', str(self.config['max_cases_per_category']),
            '--output_report', self.config['output_report']
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        print()
        print("  Evaluation in progress...")
        print("=" * 70)
        print()
        
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                print()
                print("=" * 70)
                print("  EVALUATION COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"  Report saved to: {self.config['output_report']}")
            else:
                print()
                print(f"  Evaluation exited with code: {result.returncode}")
        except Exception as e:
            print(f"  Error running evaluation: {e}")
        
        input("\n  Press Enter to continue...")
    
    def view_config(self):
        """View and edit current configuration"""
        self._print_header("CURRENT CONFIGURATION")
        
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        print()
        print("  To modify specific settings, use the Training or Evaluation menu.")
        input("\n  Press Enter to continue...")
    
    def run(self):
        """Main launcher loop"""
        while True:
            choice = self.main_menu()
            
            if choice == 'train':
                self.train_menu()
            elif choice == 'evaluate':
                self.evaluate_menu()
            elif choice == 'config':
                self.view_config()
            elif choice == 'save':
                self._save_config()
                print()
                print(f"  Configuration saved to: {self.config_file}")
                input("\n  Press Enter to continue...")
            elif choice == 'reset':
                self.config = self.DEFAULT_CONFIG.copy()
                print()
                print("  Configuration reset to defaults.")
                input("\n  Press Enter to continue...")
            elif choice == 'exit':
                print()
                print("  Goodbye!")
                break


def main():
    """Entry point"""
    launcher = FNOLauncher()
    launcher.run()


if __name__ == '__main__':
    main()
