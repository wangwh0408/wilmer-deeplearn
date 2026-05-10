@echo off
chcp 65001 >nul
echo ============================================================
echo FNO模型训练/评测 - 可移植Python环境搭建脚本
echo ============================================================
echo.
echo 此脚本将创建一个完全可移植的Python环境
echo 可以直接复制到其他Windows机器上运行，无需安装Python
echo.

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PORTABLE_DIR=%SCRIPT_DIR%\python-portable
set PYTHON_VERSION=3.10.11
set PYTHON_EMBED_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

echo [步骤 1/6] 检查环境...
if exist "%PORTABLE_DIR%" (
    echo 警告: 可移植环境目录已存在: %PORTABLE_DIR%
    set /p OVERWRITE="是否删除并重建? (y/n): "
    if /i "%OVERWRITE%"=="y" (
        echo 删除现有目录...
        rmdir /s /q "%PORTABLE_DIR%"
    ) else (
        echo 取消操作
        pause
        exit /b 0
    )
)

echo [步骤 2/6] 创建目录结构...
mkdir "%PORTABLE_DIR%"
mkdir "%PORTABLE_DIR%\python"
mkdir "%PORTABLE_DIR%\workspace"
mkdir "%PORTABLE_DIR%\scripts"

echo.
echo ============================================================
echo 重要说明:
echo ============================================================
echo.
echo 由于PyTorch等依赖非常大（约1GB+），且需要正确的版本匹配，
echo 推荐以下两种方式之一来创建可移植环境：
echo.
echo 方式一：使用现有的Python安装（推荐，最快）
echo   1. 复制 C:\Program Files\Python310 目录到 %PORTABLE_DIR%\python
echo   2. 复制所有依赖到 Lib\site-packages
echo   3. 运行此脚本的剩余部分
echo.
echo 方式二：使用嵌入式Python（需要手动安装依赖）
echo   1. 下载Python嵌入式版本
echo   2. 配置pip
echo   3. 手动安装所有依赖
echo.
echo ============================================================
echo.

set /p CHOICE="请选择方式 (1=使用现有Python, 2=下载嵌入式Python, q=退出): "

if /i "%CHOICE%"=="1" goto USE_EXISTING
if /i "%CHOICE%"=="2" goto DOWNLOAD_EMBED
if /i "%CHOICE%"=="q" exit /b 0

echo 无效选择，退出
pause
exit /b 1


:USE_EXISTING
echo.
echo [方式一] 使用现有Python安装
echo.

set EXISTING_PYTHON=C:\Program Files\Python310
if not exist "%EXISTING_PYTHON%\python.exe" (
    echo 错误: 未找到Python安装在: %EXISTING_PYTHON%
    set /p EXISTING_PYTHON="请输入Python安装目录路径: "
    if not exist "%EXISTING_PYTHON%\python.exe" (
        echo 错误: 无效的Python目录
        pause
        exit /b 1
    )
)

echo 正在复制Python文件...
echo 源: %EXISTING_PYTHON%
echo 目标: %PORTABLE_DIR%\python

xcopy "%EXISTING_PYTHON%" "%PORTABLE_DIR%\python\" /E /I /H /Y >nul
if errorlevel 1 (
    echo 警告: xcopy可能遇到权限问题
    echo 请手动复制:
    echo   复制 %EXISTING_PYTHON%\*.* 到 %PORTABLE_DIR%\python\
    echo   复制 %EXISTING_PYTHON%\DLLs 到 %PORTABLE_DIR%\python\DLLs
    echo   复制 %EXISTING_PYTHON%\Lib 到 %PORTABLE_DIR%\python\Lib
    echo   复制 %EXISTING_PYTHON%\Scripts 到 %PORTABLE_DIR%\python\Scripts
    echo   复制 %EXISTING_PYTHON%\tcl 到 %PORTABLE_DIR%\python\tcl
) else (
    echo Python文件复制完成
)

goto FINAL_SETUP


:DOWNLOAD_EMBED
echo.
echo [方式二] 下载Python嵌入式版本
echo.

echo 正在下载Python %PYTHON_VERSION% 嵌入式版本...
echo URL: %PYTHON_EMBED_URL%

powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_EMBED_URL%' -OutFile '%SCRIPT_DIR%\python-embed.zip' }"
if errorlevel 1 (
    echo 错误: 下载失败
    echo 请手动下载: %PYTHON_EMBED_URL%
    echo 并保存到: %SCRIPT_DIR%\python-embed.zip
    pause
    exit /b 1
)

echo 正在解压...
powershell -Command "& { Expand-Archive -Path '%SCRIPT_DIR%\python-embed.zip' -DestinationPath '%PORTABLE_DIR%\python' -Force }"
if errorlevel 1 (
    echo 错误: 解压失败
    pause
    exit /b 1
)

echo 正在配置pip...
echo 下载get-pip.py...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GET_PIP_URL%' -OutFile '%PORTABLE_DIR%\python\get-pip.py' }"

echo 配置python310._pth...
(
echo python310.zip
echo .
echo Lib\site-packages
echo import site
) > "%PORTABLE_DIR%\python\python310._pth"

echo 安装pip...
"%PORTABLE_DIR%\python\python.exe" "%PORTABLE_DIR%\python\get-pip.py"

echo 安装依赖...
"%PORTABLE_DIR%\python\python.exe" -m pip install --no-warn-script-location -r "%SCRIPT_DIR%\requirements-fno-train.txt"

goto FINAL_SETUP


:FINAL_SETUP
echo.
echo [步骤 5/6] 创建启动脚本...

echo 创建 run.bat...
(
echo @echo off
echo chcp 65001 ^>nul
echo set SCRIPT_DIR=%%~dp0
echo set SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%
echo set PYTHON_DIR=%%SCRIPT_DIR%%\python
echo set WORKSPACE=%%SCRIPT_DIR%%\workspace
echo.
echo set PATH=%%PYTHON_DIR%%;%%PYTHON_DIR%%\Scripts;%%PATH%%
echo set PYTHONPATH=%%WORKSPACE%%
echo.
echo echo ============================================================
echo echo FNO模型训练/评测 - 可移植Python环境
echo echo ============================================================
echo echo Python路径: %%PYTHON_DIR%%
echo echo工作目录: %%WORKSPACE%%
echo echo ============================================================
echo.
echo if "%%~1"=="" (
echo     echo 用法: run.bat ^<脚本文件^> [参数...]
echo     echo 示例:
echo     echo   run.bat train_fno2_cfdbench.py
echo     echo   run.bat evaluate_navier_stokes_model.py --model_path model.pth
echo     echo.
echo     echo 或者直接进入Python交互模式:
echo     "%%PYTHON_DIR%%\python.exe"
echo     pause
echo     exit /b 0
echo ^)
echo.
echo "%%PYTHON_DIR%%\python.exe" %%*
) > "%PORTABLE_DIR%\run.bat"

echo 创建 start_cmd.bat...
(
echo @echo off
echo chcp 65001 ^>nul
echo set SCRIPT_DIR=%%~dp0
echo set SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%
echo set PYTHON_DIR=%%SCRIPT_DIR%%\python
echo set WORKSPACE=%%SCRIPT_DIR%%\workspace
echo.
echo set PATH=%%PYTHON_DIR%%;%%PYTHON_DIR%%\Scripts;%%PATH%%
echo set PYTHONPATH=%%WORKSPACE%%
echo.
echo title FNO Python环境
echo cd /d "%%WORKSPACE%%"
echo echo ============================================================
echo echo FNO模型训练/评测 - 可移植Python环境
echo echo ============================================================
echo echo Python路径: %%PYTHON_DIR%%
echo echo工作目录: %%WORKSPACE%%
echo echo ============================================================
echo echo.
echo echo 可用命令:
echo echo   python train_fno2_cfdbench.py      - 训练模型
echo echo   python evaluate_navier_stokes_model.py  - 评测模型
echo echo   pip list                             - 查看已安装包
echo echo.
echo cmd /k
) > "%PORTABLE_DIR%\start_cmd.bat"

echo 创建 快速训练.bat...
(
echo @echo off
echo chcp 65001 ^>nul
echo set SCRIPT_DIR=%%~dp0
echo set SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%
echo set PYTHON_DIR=%%SCRIPT_DIR%%\python
echo.
echo set PATH=%%PYTHON_DIR%%;%%PYTHON_DIR%%\Scripts;%%PATH%%
echo.
echo echo ============================================================
echo echo 开始训练FNO模型...
echo echo ============================================================
echo.
echo "%%PYTHON_DIR%%\python.exe" "%%SCRIPT_DIR%%\workspace\run_training_quick.py"
echo.
echo echo 训练完成!
echo pause
) > "%PORTABLE_DIR%\快速训练.bat"

echo 创建 模型评测.bat...
(
echo @echo off
echo chcp 65001 ^>nul
echo set SCRIPT_DIR=%%~dp0
echo set SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%
echo set PYTHON_DIR=%%SCRIPT_DIR%%\python
echo.
echo set PATH=%%PYTHON_DIR%%;%%PYTHON_DIR%%\Scripts;%%PATH%%
echo.
echo echo ============================================================
echo echo 开始评测FNO模型...
echo echo ============================================================
echo.
echo "%%PYTHON_DIR%%\python.exe" "%%SCRIPT_DIR%%\workspace\evaluate_navier_stokes_model.py" --model_path "%%SCRIPT_DIR%%\workspace\fno2_cfdbench_quick_model.pth"
echo.
echo echo 评测完成!
echo pause
) > "%PORTABLE_DIR%\模型评测.bat"

echo.
echo [步骤 6/6] 复制工作文件...

echo 复制训练和评测脚本到workspace...
copy "%SCRIPT_DIR%\..\train_fno2_cfdbench.py" "%PORTABLE_DIR%\workspace\" >nul
copy "%SCRIPT_DIR%\..\run_training_quick.py" "%PORTABLE_DIR%\workspace\" >nul
copy "%SCRIPT_DIR%\..\evaluate_navier_stokes_model.py" "%PORTABLE_DIR%\workspace\" >nul
copy "%SCRIPT_DIR%\..\fno2_model.py" "%PORTABLE_DIR%\workspace\" >nul

if exist "%SCRIPT_DIR%\..\fno2_cfdbench_quick_model.pth" (
    copy "%SCRIPT_DIR%\..\fno2_cfdbench_quick_model.pth" "%PORTABLE_DIR%\workspace\" >nul
)

echo 复制requirements...
copy "%SCRIPT_DIR%\requirements-fno-train.txt" "%PORTABLE_DIR%\workspace\" >nul

echo.
echo ============================================================
echo 可移植Python环境创建完成!
echo ============================================================
echo.
echo 环境位置: %PORTABLE_DIR%
echo.
echo 目录结构:
echo   %PORTABLE_DIR%\
echo   +-- python\          ^(Python解释器和所有依赖^)
echo   +-- workspace\       ^(工作目录，训练/评测脚本^)
echo   +-- run.bat          ^(运行脚本^)
echo   +-- start_cmd.bat    ^(打开命令行^)
echo   +-- 快速训练.bat      ^(一键训练^)
echo   +-- 模型评测.bat      ^(一键评测^)
echo.
echo ============================================================
echo 移植方法:
echo ============================================================
echo 1. 将整个 "%PORTABLE_DIR%" 目录复制到目标机器
echo 2. 确保目标机器有相同的数据集路径 (C:\traework\data\)
echo 3. 双击 start_cmd.bat 打开命令行
echo 4. 或者双击 快速训练.bat / 模型评测.bat 直接运行
echo.
echo ============================================================
echo 注意事项:
echo ============================================================
echo 1. 此环境只能在Windows 64位系统上运行
echo 2. 目标机器需要有相同的数据集路径
echo 3. 第一次运行可能需要配置Windows防火墙
echo 4. 如果使用方式二（嵌入式Python），依赖可能需要重新安装
echo.
pause
