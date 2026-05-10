"""
验证可移植Python环境
运行此脚本检查环境是否正确配置
"""
import sys
import os

print("=" * 60)
print("FNO模型训练/评测 - 环境验证")
print("=" * 60)
print(f"\nPython版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")

print("\n" + "=" * 60)
print("检查核心依赖...")
print("=" * 60)

dependencies = [
    ('torch', 'PyTorch - 深度学习框架'),
    ('torchvision', 'TorchVision - 计算机视觉工具'),
    ('torchaudio', 'TorchAudio - 音频处理工具'),
    ('numpy', 'NumPy - 数值计算'),
    ('scipy', 'SciPy - 科学计算'),
    ('matplotlib', 'Matplotlib - 绘图'),
    ('h5py', 'H5Py - HDF5数据处理'),
    ('yaml', 'PyYAML - YAML配置文件'),
    ('tqdm', 'TQDM - 进度条'),
]

all_ok = True
for module_name, description in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"  [OK] {module_name:15s} v{version:15s} - {description}")
    except ImportError as e:
        print(f"  [ERROR] {module_name:15s} 未安装 - {description}")
        print(f"          错误: {e}")
        all_ok = False

print("\n" + "=" * 60)
print("检查PyTorch详细信息...")
print("=" * 60)

try:
    import torch
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  使用CPU模式")
    print(f"  [OK] PyTorch正常工作")
except Exception as e:
    print(f"  [ERROR] PyTorch检查失败: {e}")
    all_ok = False

print("\n" + "=" * 60)
print("检查FNO模型模块...")
print("=" * 60)

try:
    import fno2_model
    print(f"  [OK] fno2_model模块导入成功")
    print(f"  FNO2d类可用: {hasattr(fno2_model, 'FNO2d')}")
except ImportError as e:
    print(f"  [WARNING] fno2_model模块未找到 - 确保工作目录正确")
    print(f"  当前目录: {os.getcwd()}")

print("\n" + "=" * 60)
print("验证结果总结")
print("=" * 60)

if all_ok:
    print("\n  [成功] 所有核心依赖已正确安装!")
    print("\n  可以执行以下任务:")
    print("    1. 训练模型: python train_fno2_cfdbench.py")
    print("    2. 评测模型: python evaluate_navier_stokes_model.py")
    print("    3. 快速训练: python run_training_quick.py")
else:
    print("\n  [失败] 部分依赖未安装，请检查环境")
    print("  使用以下命令安装缺失的依赖:")
    print("    pip install -r requirements-fno-train.txt")

print("\n" + "=" * 60)
print("环境信息")
print("=" * 60)
print(f"  Python可执行文件: {sys.executable}")
print(f"  Python版本: {sys.version}")
print(f"  平台: {sys.platform}")
print(f"  路径: {sys.path[0] if sys.path else 'N/A'}")

print("\n验证完成!")
