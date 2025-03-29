#!/bin/bash

# SHAP Robustness Test Local Script
# 该脚本在本地测试SHAP鲁棒性，只处理少量图片，用于验证脚本功能正常

# 设置变量
TEST_DIR="experiments/data/tiny-imagenet-200/val"
RESULTS_FILE="experiments/results/shap_robustness_local_results.json"
TEMP_FILE="experiments/results/shap_robustness_local_temp.json"
VIZ_DIR="experiments/results/shap_viz_local"
FIGURES_DIR="experiments/results/figures/shap_local"
REPORT_PATH="experiments/results/shap_local_analysis_report.md"

# 创建必要的目录
mkdir -p "experiments/results"
mkdir -p "$VIZ_DIR"
mkdir -p "$FIGURES_DIR"

# 安装依赖
echo "Installing dependencies..."
pip install -r experiments/scripts/shap/requirements.txt

# 检查是否安装OpenCV相关的系统依赖
if ! python -c "import cv2" &> /dev/null; then
    echo "OpenCV可能缺少系统依赖。尝试安装系统依赖..."
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        brew install opencv
    else
        # Linux
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    fi
fi

# 创建临时脚本使用测试模式
SCRIPT_PATH="experiments/scripts/shap/test_shap_robustness.py"

echo "Starting local test with 10 images..."
python $SCRIPT_PATH \
  --image_dir $TEST_DIR \
  --output_file $RESULTS_FILE \
  --temp_file $TEMP_FILE \
  --model_type standard \
  --save_viz \
  --viz_dir $VIZ_DIR \
  --test_mode

# Analyze results
python experiments/scripts/shap/analyze_shap_robustness_results.py \
  --results_path $RESULTS_FILE \
  --figures_dir $FIGURES_DIR \
  --report_path $REPORT_PATH \
  --severity_level 3

echo "SHAP local test completed!"
echo "Results saved to $RESULTS_FILE"
echo "Analysis report saved to $REPORT_PATH"
echo "Heatmaps saved to $FIGURES_DIR" 