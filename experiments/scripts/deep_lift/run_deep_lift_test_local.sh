#!/bin/bash
# DeepLIFT本地测试脚本 - 用于验证设置并在少量图像上测试

# 设置路径和文件名
TEST_DIR="experiments/test_data/tiny-imagenet-200/val/n01443537"
RESULTS_FILE="experiments/results/deep_lift_local_test_results.json"
TEMP_FILE="experiments/results/deep_lift_local_temp.json"
VIZ_DIR="experiments/results/deep_lift_viz_local_test"
FIGURES_DIR="experiments/results/figures/deep_lift_local_test"
REPORT_PATH="experiments/results/deep_lift_local_test_analysis_report.md"

# 创建必要的目录
mkdir -p experiments/results
mkdir -p "$VIZ_DIR"
mkdir -p "$FIGURES_DIR"

# 安装依赖
echo "安装依赖..."
pip install -r experiments/scripts/deep_lift/requirements.txt

# 检查系统依赖
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "检测到macOS系统，检查OpenCV依赖..."
    if ! brew list opencv &>/dev/null; then
        echo "正在安装OpenCV..."
        brew install opencv
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "检测到Linux系统，检查OpenCV依赖..."
    if ! dpkg -l | grep -q libopencv; then
        echo "正在安装OpenCV依赖..."
        sudo apt-get update
        sudo apt-get install -y libopencv-dev python3-opencv
    fi
fi

# 执行DeepLIFT鲁棒性测试
echo "开始执行DeepLIFT鲁棒性测试..."
python experiments/scripts/deep_lift/test_deep_lift_robustness.py \
    --image_dir "$TEST_DIR" \
    --output_file "$RESULTS_FILE" \
    --temp_file "$TEMP_FILE" \
    --model "standard" \
    --save_viz \
    --viz_dir "$VIZ_DIR"

# 分析结果
echo "分析结果..."
python experiments/scripts/deep_lift/analyze_deep_lift_robustness_results.py \
    --results_path "$RESULTS_FILE" \
    --figures_dir "$FIGURES_DIR" \
    --report_path "$REPORT_PATH" \
    --severity_level 3

echo "======================================================"
echo "DeepLIFT本地测试完成!"
echo "结果保存在: $RESULTS_FILE"
echo "分析报告: $REPORT_PATH"
echo "热图保存在: $FIGURES_DIR"
echo "======================================================" 