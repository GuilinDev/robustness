#!/bin/bash
# 本地测试脚本 - 使用少量图片测试Occlusion Sensitivity鲁棒性评估

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
IMAGE_DIR="${ROOT_DIR}/data/images/sample"
OUTPUT_DIR="${ROOT_DIR}/results/occlusion_sensitivity"
TEMP_FILE="${OUTPUT_DIR}/temp_test_results.json"
OUTPUT_FILE="${OUTPUT_DIR}/occlusion_sensitivity_test_results.json"
VIZ_DIR="${OUTPUT_DIR}/visualizations"

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${VIZ_DIR}"

# 打印信息
echo "===== Occlusion Sensitivity 鲁棒性测试 - 本地测试模式 ====="
echo "图片目录: ${IMAGE_DIR}"
echo "结果目录: ${OUTPUT_DIR}"
echo "可视化目录: ${VIZ_DIR}"

# 运行测试脚本（测试模式，只处理少量图片）
python "${SCRIPT_DIR}/test_occlusion_sensitivity_robustness.py" \
    --image_dir "${IMAGE_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --temp_file "${TEMP_FILE}" \
    --save_viz \
    --viz_dir "${VIZ_DIR}" \
    --test_mode \
    --test_samples 5 \
    --patch_size 16 \
    --stride 8

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo "本地测试完成！结果保存在: ${OUTPUT_FILE}"
    
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