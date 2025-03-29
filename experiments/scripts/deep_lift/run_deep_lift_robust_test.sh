#!/bin/bash
# DeepLIFT鲁棒性测试脚本 - 用于在Tiny-ImageNet-200验证集上测试鲁棒ResNet模型

# 处理命令行参数
OPTIMIZE=false
STEPS=50
TEST_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --optimize) OPTIMIZE=true; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --test) TEST_MODE=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 设置基本变量
SCRIPT_PATH="experiments/scripts/deep_lift/test_deep_lift_robustness.py"
SCRIPT_DIR="experiments/scripts/deep_lift"

# 根据测试模式设置文件和目录路径
if [ "$TEST_MODE" = true ]; then
    echo "运行测试模式: 处理20张图像..."
    RESULTS_FILE="experiments/results/deep_lift_robustness_robust_test_results.json"
    TEMP_FILE="experiments/results/deep_lift_robust_test_temp.json"
    VIZ_DIR="experiments/results/deep_lift_viz_robust_test"
    FIGURES_DIR="experiments/results/figures/deep_lift_robust_test"
    REPORT_PATH="experiments/results/deep_lift_robust_test_analysis_report.md"
    # 测试模式使用的图像目录限制为20张图像
    IMAGE_DIR="experiments/test_data/tiny-imagenet-200/val/sample"
else
    echo "运行完整测试: 处理全部验证集图像..."
    RESULTS_FILE="experiments/results/deep_lift_robustness_robust_results.json"
    TEMP_FILE="experiments/results/deep_lift_robust_temp.json"
    VIZ_DIR="experiments/results/deep_lift_viz_robust"
    FIGURES_DIR="experiments/results/figures/deep_lift_robust"
    REPORT_PATH="experiments/results/deep_lift_robust_analysis_report.md"
    # 完整模式使用的图像目录
    IMAGE_DIR="experiments/test_data/tiny-imagenet-200/val"
fi

# 创建必要的目录
mkdir -p $(dirname "$RESULTS_FILE")
mkdir -p "$VIZ_DIR"
mkdir -p "$FIGURES_DIR"

# 优化处理（如果请求）
if [ "$OPTIMIZE" = true ]; then
    echo "正在优化DeepLIFT性能设置..."
    # 创建临时脚本副本并修改样本数参数
    TEMP_SCRIPT="${SCRIPT_PATH%.py}_temp.py"
    cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
    
    # 使用sed替换样本数值
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS使用不同的sed语法
        sed -i '' "s/n_samples = 5/n_samples = $STEPS/g" "$TEMP_SCRIPT"
    else
        # Linux sed语法
        sed -i "s/n_samples = 5/n_samples = $STEPS/g" "$TEMP_SCRIPT"
    fi
    
    echo "已将DeepLIFT稳定性样本数设置为 $STEPS (原默认值: 5)"
    SCRIPT_PATH="$TEMP_SCRIPT"
fi

# 执行DeepLIFT鲁棒性测试
echo "开始执行DeepLIFT鲁棒性测试 (鲁棒模型)..."
python $SCRIPT_PATH \
    --image_dir "$IMAGE_DIR" \
    --output_file "$RESULTS_FILE" \
    --temp_file "$TEMP_FILE" \
    --model "robust" \
    --save_viz \
    --viz_dir "$VIZ_DIR"

# 分析结果
echo "分析结果..."
python "$SCRIPT_DIR/analyze_deep_lift_robustness_results.py" \
    --results_path "$RESULTS_FILE" \
    --figures_dir "$FIGURES_DIR" \
    --report_path "$REPORT_PATH" \
    --severity_level 3

# 清理临时文件
if [ "$OPTIMIZE" = true ]; then
    echo "清理临时文件..."
    rm "$TEMP_SCRIPT"
fi

echo "======================================================"
echo "DeepLIFT鲁棒性测试完成 (鲁棒模型)!"
echo "结果保存在: $RESULTS_FILE"
echo "分析报告: $REPORT_PATH"
echo "热图保存在: $FIGURES_DIR"
echo "======================================================" 