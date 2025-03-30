#!/bin/bash

# SHAP依赖安装脚本
# 该脚本安装运行SHAP鲁棒性测试所需的所有依赖

set -e  # 任何命令失败时停止执行

echo "=== 开始安装SHAP测试所需的依赖 ==="

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

echo "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 检测是否在虚拟环境中
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "警告: 没有检测到虚拟环境。建议在虚拟环境中安装依赖。"
    echo "您可以使用以下命令创建并激活虚拟环境:"
    echo "python -m venv .venv"
    echo "source .venv/bin/activate (Linux/Mac)"
    echo "或 .venv\\Scripts\\activate (Windows)"
    
    read -p "是否继续安装? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

# 安装Python依赖
echo "=== 安装Python依赖 ==="
pip install -r "$SCRIPT_DIR/requirements.txt"

# 安装RobustBench
echo "=== 安装RobustBench (用于鲁棒模型) ==="
pip install git+https://github.com/RobustBench/robustbench.git

# 检测操作系统并安装系统依赖
echo "=== 检测操作系统类型 ==="
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
    echo "检测到Linux系统，尝试安装OpenCV系统依赖..."
    
    # 检测Linux发行版
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        echo "检测到Debian/Ubuntu系统"
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    elif [ -f /etc/redhat-release ]; then
        # RedHat/CentOS/Fedora
        echo "检测到RedHat/CentOS/Fedora系统"
        sudo yum install -y mesa-libGL glib2
    else
        echo "未能识别Linux发行版，请手动安装OpenCV依赖"
        echo "Debian/Ubuntu: sudo apt-get install -y libgl1-mesa-glx libglib2.0-0"
        echo "RedHat/CentOS: sudo yum install -y mesa-libGL glib2"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    echo "检测到MacOS系统"
    if command -v brew &> /dev/null; then
        echo "正在使用Homebrew安装OpenCV..."
        brew install opencv
    else
        echo "未检测到Homebrew，请先安装Homebrew: https://brew.sh/"
        echo "然后运行: brew install opencv"
    fi
else
    echo "未识别的操作系统: $OS"
    echo "请手动安装OpenCV依赖"
fi

# 验证安装
echo "=== 验证安装 ==="
echo "尝试导入关键库..."

# 验证PyTorch
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ PyTorch安装成功"
else
    echo "✗ PyTorch导入失败"
fi

# 验证SHAP
python -c "import shap; print(f'SHAP版本: {shap.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ SHAP安装成功"
else
    echo "✗ SHAP导入失败"
fi

# 验证OpenCV
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
if [ $? -eq 0 ]; then
    echo "✓ OpenCV安装成功"
else
    echo "✗ OpenCV导入失败，可能缺少系统依赖"
fi

# 验证RobustBench
python -c "from robustbench.utils import load_model; print('RobustBench可用')"
if [ $? -eq 0 ]; then
    echo "✓ RobustBench安装成功"
else
    echo "✗ RobustBench导入失败"
fi

echo "=== 依赖安装和验证完成 ==="
echo "现在您可以运行SHAP测试:"
echo "bash experiments/scripts/shap/run_shap_test.sh --test" 