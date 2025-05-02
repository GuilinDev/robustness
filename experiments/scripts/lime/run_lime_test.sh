#!/bin/bash
# LIME鲁棒性测试脚本 - 用于在Tiny-ImageNet-200验证集上测试标准ResNet模型

# 添加错误处理，任何命令失败时脚本停止执行
set -e

# 脚本配置 - 取消注释并修改这些选项以进行优化
OPTIMIZE=true      # 设置为true启用优化
SAMPLE_SIZE=1000    # 要处理的图像数量（设为0处理所有图像）
STEPS=50           # LIME采样数量（原始为1000，数值越低性能越好）
RANDOM_SEED=42     # 固定随机种子以保证可复现性
# 结束配置

# 处理命令行参数
TEST_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --optimize) OPTIMIZE=true; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --test) TEST_MODE=true; shift ;;
        --samples) SAMPLE_SIZE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; echo "用法: $0 [--optimize] [--steps N] [--test] [--samples N]"; exit 1 ;;
    esac
done

# 确保目录结构存在
mkdir -p experiments/results/lime_viz
mkdir -p experiments/results/lime_standard_figures
mkdir -p experiments/data/samples

# 设置基本变量
SCRIPT_PATH="experiments/scripts/lime/test_lime_robustness.py"
SCRIPT_DIR="experiments/scripts/lime"

# 根据测试模式设置文件和目录路径
if [ "$TEST_MODE" = true ]; then
    echo "运行测试模式: 处理20张图像..."
    RESULTS_FILE="experiments/results/lime_robustness_test_results.json"
    TEMP_FILE="experiments/results/lime_test_temp.json"
    VIZ_DIR="experiments/results/lime_viz_test"
    FIGURES_DIR="experiments/results/figures/lime_test"
    REPORT_PATH="experiments/results/lime_test_analysis_report.md"
    
    # 测试模式使用的图像目录限制为20张图像
    TEST_DIR="experiments/data/lime_test_subset"
    mkdir -p "$TEST_DIR"
    find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | head -n 20 | xargs -I{} cp {} "$TEST_DIR/"
    IMAGE_DIR="$TEST_DIR"
elif [ "$OPTIMIZE" = true ] && [ $SAMPLE_SIZE -gt 0 ]; then
    echo "运行优化模式: 处理 $SAMPLE_SIZE 张图像..."
    RESULTS_FILE="experiments/results/lime_robustness_standard_results.json"
    TEMP_FILE="experiments/results/lime_robustness_standard_temp.json"
    VIZ_DIR="experiments/results/lime_viz_standard"
    FIGURES_DIR="experiments/results/figures/lime_standard"
    REPORT_PATH="experiments/results/lime_standard_analysis_report.md"
    
    # 优化模式：创建图像子集
    echo "准备优化数据集，使用 $SAMPLE_SIZE 张图片..."
    TEST_DIR="experiments/data/lime_test_subset"
    mkdir -p ${TEST_DIR}
    
    # 定义样本列表文件，确保标准模型和鲁棒模型使用相同样本
    SAMPLE_LIST="experiments/data/samples/lime_sample_list.txt"
    
    # 如果子集目录为空或样本列表不存在，则创建包含随机图像的子集
    if [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ] || [ ! -f "$SAMPLE_LIST" ]; then
        echo "创建测试子集，使用 $SAMPLE_SIZE 张随机图片和种子 $RANDOM_SEED..."
        # 设置随机种子以确保可重现性
        export RANDOM=$RANDOM_SEED
        
        # 查找所有图像文件
        find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | sort | shuf -n $SAMPLE_SIZE > "$SAMPLE_LIST"
        
        # 根据样本列表创建子集
        cat "$SAMPLE_LIST" | while read img; do
            # 创建与原始路径相同的目标目录结构
            REL_PATH=$(echo $img | sed "s|experiments/data/tiny-imagenet-200/val/||")
            DIR_NAME=$(dirname ${REL_PATH})
            mkdir -p "${TEST_DIR}/${DIR_NAME}"
            # 将图像复制到子集目录
            cp "$img" "${TEST_DIR}/${REL_PATH}"
        done
        echo "测试子集已创建，包含 $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) 张图片"
        echo "样本列表已保存到 $SAMPLE_LIST 以便复现"
    else
        echo "使用现有测试子集，其中包含 $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) 张图片"
        echo "样本列表已存在于 $SAMPLE_LIST"
    fi
    
    # 使用子集进行测试
    IMAGE_DIR=$TEST_DIR
else
    echo "运行完整测试: 处理全部验证集图像..."
    RESULTS_FILE="experiments/results/lime_robustness_standard_results.json"
    TEMP_FILE="experiments/results/lime_robustness_standard_temp.json"
    VIZ_DIR="experiments/results/lime_viz_standard"
    FIGURES_DIR="experiments/results/figures/lime_standard"
    REPORT_PATH="experiments/results/lime_standard_analysis_report.md"
    # 完整模式使用的图像目录
    IMAGE_DIR="experiments/data/tiny-imagenet-200/val"
fi

# 创建必要的目录
mkdir -p $(dirname "$RESULTS_FILE")
mkdir -p "$VIZ_DIR"
mkdir -p "$FIGURES_DIR"

# 优化处理（如果请求）
if [ "$OPTIMIZE" = true ]; then
    echo "创建优化脚本，使用 $STEPS 个样本..."
    # 创建临时脚本副本并修改nsamples参数
    TEMP_SCRIPT="${SCRIPT_PATH%.py}_temp.py"
    cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
    
    # 使用sed替换nsamples值
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS使用不同的sed语法
        sed -i '' "s/nsamples=1000/nsamples=$STEPS/g" "$TEMP_SCRIPT"
    else
        # Linux sed语法
        sed -i "s/nsamples=1000/nsamples=$STEPS/g" "$TEMP_SCRIPT"
    fi
    
    echo "已将LIME采样数从1000减少到 $STEPS 以提高性能"
    SCRIPT_PATH="$TEMP_SCRIPT"
fi

# 显示预计的处理时间和图像数量
IMAGE_COUNT=$(find $IMAGE_DIR -type f -name "*.JPEG" | wc -l)
if [ $IMAGE_COUNT -gt 0 ]; then
    echo "将处理 $IMAGE_COUNT 张图像"
    # 估算每张图像处理时间 (基于200秒/图像/1000样本的粗略估计)
    EST_TIME_PER_IMAGE=$(echo "scale=2; 200 * $STEPS / 1000" | bc)
    EST_TOTAL_TIME=$(echo "scale=2; $EST_TIME_PER_IMAGE * $IMAGE_COUNT / 3600" | bc)
    echo "基于每张图像 $EST_TIME_PER_IMAGE 秒的估计，总处理时间约为 $EST_TOTAL_TIME 小时"
    echo "实际时间可能因硬件性能和图像复杂度而异"
fi

# 显示启动时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "启动时间: $START_TIME"
echo "----------------------------------------"

# 执行LIME鲁棒性测试
echo "开始执行LIME鲁棒性测试（标准模型）..."
python $SCRIPT_PATH \
    --image_dir "$IMAGE_DIR" \
    --output_file "$RESULTS_FILE" \
    --temp_file "$TEMP_FILE" \
    --model "standard" \
    --save_viz \
    --viz_dir "$VIZ_DIR"

# 检查测试脚本是否成功完成并创建了结果文件
if [ ! -f "$RESULTS_FILE" ]; then
    echo "错误: LIME测试未能生成结果文件: $RESULTS_FILE"
    exit 1
fi

# 显示完成时间和总持续时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
HOURS=$(( DURATION / 3600 ))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$(( DURATION % 60 ))

echo "----------------------------------------"
echo "处理完成!"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"

# 分析结果
echo "分析结果..."
python "$SCRIPT_DIR/analyze_lime_robustness_results.py" \
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
echo "LIME鲁棒性测试（标准模型）完成!"
echo "结果保存在: $RESULTS_FILE"
echo "分析报告: $REPORT_PATH"
echo "热图保存在: $FIGURES_DIR"
echo "======================================================" 