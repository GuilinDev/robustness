#!/bin/bash
# 标准模型测试脚本 - 使用标准ResNet50模型测试Occlusion Sensitivity鲁棒性

# 添加错误处理，任何命令失败时脚本停止执行
set -e

# 脚本配置 - 取消注释并修改这些选项以进行优化
OPTIMIZE=true      # 设置为true启用优化
SAMPLE_SIZE=1000   # 要处理的图像数量（设为0处理所有图像）
PATCH_SIZE=16      # 遮挡区域大小（默认16，越小效果越好但越慢）
STRIDE=8           # 滑动步长（默认8，越大越快）
RANDOM_SEED=42     # 固定随机种子以保证可复现性
# 结束配置

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
IMAGE_DIR="${ROOT_DIR}/experiments/data/tiny-imagenet-200/val"
OUTPUT_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/standard"
TEMP_FILE="${OUTPUT_DIR}/occlusion_sensitivity_standard_temp.json"
OUTPUT_FILE="${OUTPUT_DIR}/occlusion_sensitivity_standard_results.json"
VIZ_DIR="${OUTPUT_DIR}/visualizations"

# 解析命令行参数
SAVE_VIZ=false
DEVICE="cuda:0"
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --save_viz)
            SAVE_VIZ=true
            shift
            ;;
        --patch_size)
            PATCH_SIZE="$2"
            shift
            shift
            ;;
        --stride)
            STRIDE="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --samples)
            SAMPLE_SIZE="$2"
            shift
            shift
            ;;
        --optimize)
            OPTIMIZE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--save_viz] [--patch_size SIZE] [--stride STRIDE] [--device DEVICE] [--test] [--samples N] [--optimize]"
            exit 1
            ;;
    esac
done

# 确保目录结构存在
mkdir -p experiments/results/occlusion_sensitivity/visualizations
mkdir -p experiments/results/occlusion_sensitivity/standard
mkdir -p experiments/data/samples

# 样本列表路径
SAMPLE_LIST="${ROOT_DIR}/experiments/data/samples/occlusion_sensitivity_sample_list.txt"

# 根据测试模式设置文件和目录路径
if [ "$TEST_MODE" = true ]; then
    echo "运行测试模式: 处理20张图像..."
    OUTPUT_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_test_results.json"
    TEMP_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_test_temp.json"
    VIZ_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/visualizations/standard_test"
    ANALYSIS_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/analysis/standard_test"
    
    # 测试模式使用的图像目录限制为20张图像
    TEST_DIR="${ROOT_DIR}/experiments/data/occlusion_sensitivity_test_subset"
    mkdir -p "$TEST_DIR"
    find ${IMAGE_DIR} -type f -name "*.JPEG" | head -n 20 | xargs -I{} cp {} "$TEST_DIR/"
    IMAGE_DIR="$TEST_DIR"
elif [ "$OPTIMIZE" = true ] && [ $SAMPLE_SIZE -gt 0 ]; then
    echo "运行优化模式: 处理 $SAMPLE_SIZE 张图像..."
    OUTPUT_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_results.json"
    TEMP_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_temp.json"
    VIZ_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/visualizations/standard"
    ANALYSIS_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/analysis/standard"
    
    # 创建测试子集
    TEST_DIR="${ROOT_DIR}/experiments/data/occlusion_sensitivity_test_subset"
    mkdir -p "$TEST_DIR"
    
    # 如果测试子集为空或样本列表不存在，创建新的样本列表和测试子集
    if [ ! -f "$SAMPLE_LIST" ] || [ -z "$(ls -A ${TEST_DIR} 2>/dev/null)" ]; then
        echo "创建新的样本列表和测试子集..."
        # 清空并重新创建目录
        rm -rf "$TEST_DIR"
        mkdir -p "$TEST_DIR"
        
        # 随机选择图像并创建样本列表
        find ${IMAGE_DIR} -type f -name "*.JPEG" | shuf -n $SAMPLE_SIZE --random-source=<(yes $RANDOM_SEED | head -n 1) > "$SAMPLE_LIST"
        
        # 将选中的图像复制到测试子集目录
        cat "$SAMPLE_LIST" | while read img; do
            # 创建与原始路径相同的目标目录结构
            REL_PATH=$(echo $img | sed "s|${IMAGE_DIR}/||")
            DIR_NAME=$(dirname ${REL_PATH})
            mkdir -p "${TEST_DIR}/${DIR_NAME}"
            # 将图像复制到子集目录
            cp "$img" "${TEST_DIR}/${REL_PATH}"
        done
        
        echo "创建了样本列表: $SAMPLE_LIST"
        echo "创建了测试子集: $TEST_DIR 包含 $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) 张图片"
    else
        echo "使用现有的测试子集，包含 $(find ${TEST_DIR} -type f -name "*.JPEG" | wc -l) 张图片"
        echo "样本列表位于 $SAMPLE_LIST"
    fi
    
    # 使用子集进行测试
    IMAGE_DIR=$TEST_DIR
else
    echo "运行完整测试: 处理全部验证集图像..."
    OUTPUT_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_results.json"
    TEMP_FILE="${ROOT_DIR}/experiments/results/occlusion_sensitivity/occlusion_sensitivity_standard_temp.json"
    VIZ_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/visualizations/standard"
    ANALYSIS_DIR="${ROOT_DIR}/experiments/results/occlusion_sensitivity/analysis/standard"
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ANALYSIS_DIR}"
if [ "$SAVE_VIZ" = true ]; then
    mkdir -p "${VIZ_DIR}"
    VIZ_ARGS="--save_viz --viz_dir ${VIZ_DIR}"
else
    VIZ_ARGS=""
fi

# 显示预计的处理时间和图像数量
IMAGE_COUNT=$(find $IMAGE_DIR -type f -name "*.JPEG" | wc -l)
if [ $IMAGE_COUNT -gt 0 ]; then
    echo "将处理 $IMAGE_COUNT 张图像"
    # 估算每张图像处理时间（粗略估计基于遮挡区域大小和滑动步长）
    # 处理时间与 (224/STRIDE)^2 成正比
    BASE_TIME=10  # 基础时间(秒)：标准设置下(PATCH_SIZE=16,STRIDE=8)每张图片耗时
    EST_TIME_PER_IMAGE=$(echo "scale=2; $BASE_TIME * (224 / $STRIDE)^2 / (224 / 8)^2" | bc)
    EST_TOTAL_TIME=$(echo "scale=2; $EST_TIME_PER_IMAGE * $IMAGE_COUNT / 3600" | bc)
    echo "基于每张图像 $EST_TIME_PER_IMAGE 秒的估计，总处理时间约为 $EST_TOTAL_TIME 小时"
    echo "实际时间可能因硬件性能和图像复杂度而异"
fi

# 打印信息
echo "===== Occlusion Sensitivity 鲁棒性测试 - 标准模型 ====="
echo "图片目录: ${IMAGE_DIR}"
echo "结果目录: ${OUTPUT_DIR}"
echo "模型类型: standard (ResNet50)"
echo "参数:"
echo "  - 遮挡区域大小: ${PATCH_SIZE}"
echo "  - 滑动步长: ${STRIDE}"
echo "  - 设备: ${DEVICE}"
echo "  - 保存可视化: ${SAVE_VIZ}"
if [ "$SAVE_VIZ" = true ]; then
    echo "  - 可视化目录: ${VIZ_DIR}"
fi

# 显示启动时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "启动时间: $START_TIME"
echo "----------------------------------------"

# 运行测试脚本
echo "开始运行测试..."
/workspace/robustness/.venv/bin/python "${SCRIPT_DIR}/test_occlusion_sensitivity_robustness.py" \
    --image_dir "${IMAGE_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --temp_file "${TEMP_FILE}" \
    --model_type "standard" \
    --device "${DEVICE}" \
    --patch_size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    ${VIZ_ARGS}

# 检查测试脚本是否成功完成并创建了结果文件
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "错误: Occlusion Sensitivity测试未能生成结果文件: $OUTPUT_FILE"
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
echo "正在分析结果..."
    
/workspace/robustness/.venv/bin/python "${SCRIPT_DIR}/analyze_occlusion_sensitivity_robustness_results.py" \
    --results_file "${OUTPUT_FILE}" \
    --output_dir "${ANALYSIS_DIR}"

echo "====================================================="
echo "Occlusion Sensitivity鲁棒性测试（标准模型）完成!"
echo "结果保存在: ${OUTPUT_FILE}"
echo "分析报告保存在: ${ANALYSIS_DIR}/summary_report.md"
echo "可视化保存在: ${VIZ_DIR}" 
echo "=====================================================" 