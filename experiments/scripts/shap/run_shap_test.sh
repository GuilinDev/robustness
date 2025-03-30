#!/bin/bash

# SHAP Robustness Test Script
# 该脚本运行SHAP鲁棒性测试，处理Tiny-ImageNet-200验证集中的图像

# 添加错误处理，任何命令失败时脚本停止执行
set -e

# 脚本配置 - 取消注释并修改这些选项以进行优化
OPTIMIZE=true      # 设置为true启用优化
SAMPLE_SIZE=1000    # 要处理的图像数量（设为0处理所有图像）
N_STEPS=5          # SHAP样本数量（原始为10，数值越低性能越好）
RANDOM_SEED=42     # 固定随机种子以保证可复现性
# 结束配置

# 命令行参数处理
TEST_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --optimize)
      OPTIMIZE=true
      shift
      ;;
    --steps)
      N_STEPS="$2"
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
    *)
      echo "Unknown option: $key"
      echo "Usage: $0 [--optimize] [--steps N] [--test] [--samples N]"
      exit 1
      ;;
  esac
done

# 确保目录结构存在
mkdir -p experiments/results/shap_viz
mkdir -p experiments/results/shap_standard_figures
mkdir -p experiments/data/samples

# 设置变量
if [ "$TEST_MODE" = true ]; then
  RESULTS_FILE="experiments/results/shap_robustness_test_results.json"
  TEMP_FILE="experiments/results/shap_robustness_test_temp.json"
  VIZ_DIR="experiments/results/shap_viz_test"
  FIGURES_DIR="experiments/results/figures/shap_test"
  REPORT_PATH="experiments/results/shap_test_analysis_report.md"
  
  # 只使用少数图片进行测试
  TEST_DIR="experiments/data/shap_test_subset"
  mkdir -p "$TEST_DIR"
  find experiments/data/tiny-imagenet-200/val -type f -name "*.JPEG" | head -n 20 | xargs -I{} cp {} "$TEST_DIR/"
  IMAGE_DIR="$TEST_DIR"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
elif [ "$OPTIMIZE" = true ] && [ $SAMPLE_SIZE -gt 0 ]; then
  RESULTS_FILE="experiments/results/shap_robustness_standard_results.json"
  TEMP_FILE="experiments/results/shap_robustness_standard_temp.json"
  VIZ_DIR="experiments/results/shap_viz_standard"
  FIGURES_DIR="experiments/results/figures/shap_standard"
  REPORT_PATH="experiments/results/shap_standard_analysis_report.md"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
  
  # 优化模式：创建图像子集
  echo "准备优化数据集，使用 $SAMPLE_SIZE 张图片..."
  TEST_DIR="experiments/data/shap_test_subset"
  mkdir -p ${TEST_DIR}
  
  # 定义样本列表文件，确保标准模型和鲁棒模型使用相同样本
  SAMPLE_LIST="experiments/data/samples/shap_sample_list.txt"
  
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
  RESULTS_FILE="experiments/results/shap_robustness_standard_results.json"
  TEMP_FILE="experiments/results/shap_robustness_standard_temp.json"
  VIZ_DIR="experiments/results/shap_viz_standard"
  FIGURES_DIR="experiments/results/figures/shap_standard"
  REPORT_PATH="experiments/results/shap_standard_analysis_report.md"
  IMAGE_DIR="experiments/data/tiny-imagenet-200/val"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
fi

# 如果优化，创建具有修改后的样本数的临时脚本
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  echo "创建优化脚本，使用 $N_STEPS 个样本..."
  SCRIPT_PATH="experiments/scripts/shap/test_shap_robustness_temp.py"
  cp experiments/scripts/shap/test_shap_robustness.py $SCRIPT_PATH
  
  # 使用sed替换nsamples值
  sed -i "s/nsamples=10/nsamples=$N_STEPS/g" $SCRIPT_PATH
  echo "已将SHAP样本从10减少到 $N_STEPS 以提高性能"
else
  SCRIPT_PATH="experiments/scripts/shap/test_shap_robustness.py"
fi

# 运行SHAP测试
echo "开始SHAP鲁棒性测试，使用标准模型..."
echo "图像目录: $IMAGE_DIR"
echo "结果将保存到: $RESULTS_FILE"

# 显示预计的处理时间和图像数量
IMAGE_COUNT=$(find $IMAGE_DIR -type f -name "*.JPEG" | wc -l)
if [ $IMAGE_COUNT -gt 0 ]; then
  echo "将处理 $IMAGE_COUNT 张图像"
  # 估算每张图像处理时间 (基于20秒/图像/样本的粗略估计，每个图像15种污染x5个严重程度)
  EST_TIME_PER_IMAGE=$(echo "scale=2; 20 * $N_STEPS / 10" | bc)
  EST_TOTAL_TIME=$(echo "scale=2; $EST_TIME_PER_IMAGE * $IMAGE_COUNT / 3600" | bc)
  echo "基于每张图像 $EST_TIME_PER_IMAGE 秒的估计，总处理时间约为 $EST_TOTAL_TIME 小时"
  echo "实际时间可能因硬件性能和图像复杂度而异"
fi

# 显示启动时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "启动时间: $START_TIME"
echo "----------------------------------------"

# 运行脚本
/workspace/robustness/.venv/bin/python $SCRIPT_PATH \
  --image_dir $IMAGE_DIR \
  --output_file $RESULTS_FILE \
  --temp_file $TEMP_FILE \
  --model_type standard \
  --save_viz \
  --viz_dir $VIZ_DIR

# 检查测试脚本是否成功完成并创建了结果文件
if [ ! -f "$RESULTS_FILE" ]; then
  echo "错误: SHAP测试未能生成结果文件: $RESULTS_FILE"
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
echo "开始分析结果..."
/workspace/robustness/.venv/bin/python experiments/scripts/shap/analyze_shap_robustness_results.py \
  --results_path $RESULTS_FILE \
  --figures_dir $FIGURES_DIR \
  --report_path $REPORT_PATH \
  --severity_level 3

# 清理临时脚本（如果已创建）
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  rm $SCRIPT_PATH
fi

echo "SHAP鲁棒性测试（标准模型）已完成!"
echo "结果已保存到 $RESULTS_FILE"
echo "分析报告已保存到 $REPORT_PATH"
echo "热图已保存到 $FIGURES_DIR" 