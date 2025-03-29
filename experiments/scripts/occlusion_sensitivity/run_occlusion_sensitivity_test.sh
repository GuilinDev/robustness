#!/bin/bash
# 标准模型测试脚本 - 使用标准ResNet50模型测试Occlusion Sensitivity鲁棒性

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
IMAGE_DIR="${ROOT_DIR}/data/images/imagenet"
OUTPUT_DIR="${ROOT_DIR}/results/occlusion_sensitivity/standard"
TEMP_FILE="${OUTPUT_DIR}/temp_results.json"
OUTPUT_FILE="${OUTPUT_DIR}/occlusion_sensitivity_results.json"
VIZ_DIR="${OUTPUT_DIR}/visualizations"

# 解析命令行参数
SAVE_VIZ=false
PATCH_SIZE=16
STRIDE=8
DEVICE="cuda:0"

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
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--save_viz] [--patch_size SIZE] [--stride STRIDE] [--device DEVICE]"
            exit 1
            ;;
    esac
done

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"
if [ "$SAVE_VIZ" = true ]; then
    mkdir -p "${VIZ_DIR}"
    VIZ_ARGS="--save_viz --viz_dir ${VIZ_DIR}"
else
    VIZ_ARGS=""
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

# 运行测试脚本
echo "开始运行测试..."
python "${SCRIPT_DIR}/test_occlusion_sensitivity_robustness.py" \
    --image_dir "${IMAGE_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --temp_file "${TEMP_FILE}" \
    --model_type "standard" \
    --device "${DEVICE}" \
    --patch_size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    ${VIZ_ARGS}

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo "测试完成！结果保存在: ${OUTPUT_FILE}"
    
    # 分析结果
    echo "正在分析结果..."
    ANALYSIS_DIR="${OUTPUT_DIR}/analysis"
    mkdir -p "${ANALYSIS_DIR}"
    
    python "${SCRIPT_DIR}/analyze_occlusion_sensitivity_robustness_results.py" \
        --results_file "${OUTPUT_FILE}" \
        --output_dir "${ANALYSIS_DIR}"
    
    echo "分析完成！结果保存在: ${ANALYSIS_DIR}"
    echo "可以查看 ${ANALYSIS_DIR}/summary_report.md 获取摘要报告"
else
    echo "测试失败！请检查错误信息。"
    exit 1
fi 