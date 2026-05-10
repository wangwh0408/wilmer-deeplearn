# FNO模型训练/评测 - 可移植Python环境

## 概述

此目录包含一个完全可移植的Python环境，可以直接复制到其他Windows机器上运行，无需安装Python。

## 目录结构

```
python-portable/
├── python/              # Python解释器和所有依赖
│   ├── python.exe
│   ├── python310.dll
│   ├── DLLs/
│   ├── Lib/
│   │   └── site-packages/  # 所有第三方库 (torch, numpy等)
│   ├── Scripts/
│   └── ...
├── workspace/           # 工作目录
│   ├── train_fno2_cfdbench.py      # 训练脚本
│   ├── run_training_quick.py        # 快速训练脚本
│   ├── evaluate_navier_stokes_model.py  # 评测脚本
│   ├── fno2_model.py                 # FNO模型定义
│   ├── fno2_cfdbench_quick_model.pth # 训练好的模型
│   └── requirements-fno-train.txt    # 依赖列表
├── run.bat             # 运行Python脚本
├── start_cmd.bat       # 打开命令行环境
├── 快速训练.bat         # 一键开始训练
├── 模型评测.bat         # 一键评测模型
└── verify_environment.py  # 环境验证脚本
```

## 快速开始

### 方法一：使用批处理脚本（推荐）

1. **验证环境**
   ```
   双击 start_cmd.bat 打开命令行
   执行: python verify_environment.py
   ```

2. **快速训练**
   ```
   双击 快速训练.bat
   ```
   或者在命令行中执行:
   ```
   python run_training_quick.py
   ```

3. **评测模型**
   ```
   双击 模型评测.bat
   ```
   或者在命令行中执行:
   ```
   python evaluate_navier_stokes_model.py --model_path fno2_cfdbench_quick_model.pth
   ```

### 方法二：使用命令行

1. 双击 `start_cmd.bat` 打开配置好的命令行环境
2. 执行Python命令

## 移植到其他机器

### 步骤

1. **复制整个目录**
   将 `python-portable` 目录复制到目标机器的任意位置

2. **准备数据集**
   确保目标机器有相同的数据集路径:
   ```
   C:\traework\data\
   ├── cavity\
   │   ├── bc\
   │   ├── geo\
   │   └── prop\
   ```

3. **运行**
   双击 `快速训练.bat` 或 `模型评测.bat`

### 注意事项

- ✅ 支持: Windows 10/11 64位
- ❌ 不支持: Windows 32位、Linux、macOS
- 目标机器不需要安装Python
- 数据集路径必须匹配: `C:\traework\data\`

## 依赖包列表

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.2.2 | PyTorch深度学习框架 |
| torchvision | 0.17.2 | 计算机视觉工具 |
| torchaudio | 2.2.2 | 音频处理工具 |
| numpy | 1.24.4 | 数值计算 |
| scipy | 1.11.4 | 科学计算 |
| matplotlib | 3.7.5 | 数据可视化 |
| h5py | 3.10.0 | HDF5数据处理 |
| pyyaml | 6.0.1 | YAML配置文件 |
| tqdm | 4.66.2 | 进度条 |

## 脚本说明

### 训练脚本

**train_fno2_cfdbench.py**
- 完整的训练脚本
- 支持自定义参数: epochs, batch_size, learning_rate等

**run_training_quick.py**
- 快速训练脚本
- 使用预设参数快速开始训练
- 每类取10个样本，30个epoch

### 评测脚本

**evaluate_navier_stokes_model.py**
- 评测已训练模型
- 输出指标: MSE, RMSE, MAE, R2, MAPE
- 支持命令行参数

## 命令行参数

### 训练参数

```bash
python train_fno2_cfdbench.py \
    --data_root "C:\traework\data" \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --modes 12 \
    --width 32
```

### 评测参数

```bash
python evaluate_navier_stokes_model.py \
    --model_path "fno2_cfdbench_quick_model.pth" \
    --data_root "C:\traework\data" \
    --max_cases_per_category 10
```

## 故障排除

### 问题1: 找不到数据集

**错误信息:**
```
RuntimeError: No data loaded from C:\traework\data
```

**解决方案:**
1. 检查数据集路径是否存在: `C:\traework\data\`
2. 确保目录结构正确:
   ```
   C:\traework\data\cavity\bc\case0000\u.npy
   C:\traework\data\cavity\bc\case0000\v.npy
   ```

### 问题2: 找不到fno2_model模块

**错误信息:**
```
ModuleNotFoundError: No module named 'fno2_model'
```

**解决方案:**
1. 确保在 `workspace` 目录下运行
2. 或者设置PYTHONPATH环境变量

### 问题3: PyTorch CUDA错误

**错误信息:**
```
CUDA error: ...
```

**解决方案:**
1. 检查GPU驱动是否正确安装
2. 或者使用CPU模式（默认）
3. 代码会自动检测CUDA是否可用

### 问题4: 依赖缺失

**解决方案:**
在命令行中执行:
```bash
pip install -r requirements-fno-train.txt
```

## 性能说明

### 训练时间预估

| 配置 | 时间预估 |
|------|----------|
| CPU (i7) | 每epoch约1-2分钟 |
| GPU (RTX 3090) | 每epoch约10-30秒 |

### 内存需求

- Python运行时: ~500MB
- PyTorch + 依赖: ~2GB
- 训练时: ~4-8GB (取决于batch size)

## 更新日志

### v1.0 (2026-04-30)
- 初始版本
- 包含Python 3.10.11可移植环境
- 包含所有必要依赖
- 包含训练和评测脚本
- 包含预训练模型

## 技术支持

如有问题，请检查:
1. 数据集路径是否正确
2. Python环境是否完整
3. 依赖是否全部安装

## 高级用法

### 创建新的可移植环境

如果需要在新机器上创建完整环境:

1. 安装Python 3.10.11
2. 安装依赖:
   ```bash
   pip install -r requirements-fno-train.txt
   ```
3. 复制整个Python目录

### 使用run.bat运行脚本

```bash
run.bat train_fno2_cfdbench.py
run.bat evaluate_navier_stokes_model.py --model_path model.pth
```

---

**注意:** 此环境专为Windows 64位系统设计，不可跨平台使用。
