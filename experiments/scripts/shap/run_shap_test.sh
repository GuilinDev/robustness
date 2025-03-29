#!/bin/bash

# SHAP Robustness Test Script
# 该脚本运行SHAP鲁棒性测试，处理Tiny-ImageNet-200验证集中的图像

# 命令行参数处理
OPTIMIZE=false
N_STEPS=0
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
    *)
      echo "Unknown option: $key"
      echo "Usage: $0 [--optimize] [--steps N] [--test]"
      exit 1
      ;;
  esac
done

# 设置变量
if [ "$TEST_MODE" = true ]; then
  RESULTS_FILE="experiments/results/shap_robustness_test_results.json"
  TEMP_FILE="experiments/results/shap_robustness_test_temp.json"
  VIZ_DIR="experiments/results/shap_viz_test"
  FIGURES_DIR="experiments/results/figures/shap_test"
  REPORT_PATH="experiments/results/shap_test_analysis_report.md"
  IMAGE_DIR="experiments/data/tiny-imagenet-200/val"
  
  # 只使用少数图片进行测试
  mkdir -p "experiments/data/shap_test_subset"
  find $IMAGE_DIR -type f -name "*.JPEG" | head -n 20 | xargs -I{} cp {} "experiments/data/shap_test_subset/"
  IMAGE_DIR="experiments/data/shap_test_subset"
  
  mkdir -p $VIZ_DIR
  mkdir -p $FIGURES_DIR
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

# If optimizing, create temporary script with modified n_samples
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  echo "Creating optimized script with $N_STEPS samples..."
  SCRIPT_PATH="experiments/scripts/shap/test_shap_robustness_temp.py"
  cp experiments/scripts/shap/test_shap_robustness.py $SCRIPT_PATH
  
  # Use sed to replace nsamples value
  sed -i "s/nsamples=10/nsamples=$N_STEPS/g" $SCRIPT_PATH
  echo "Reduced SHAP samples from 10 to $N_STEPS for better performance"
else
  SCRIPT_PATH="experiments/scripts/shap/test_shap_robustness.py"
fi

# Run the SHAP test
echo "Starting SHAP robustness test with standard model..."
echo "Image directory: $IMAGE_DIR"
echo "Results will be saved to: $RESULTS_FILE"

python $SCRIPT_PATH \
  --image_dir $IMAGE_DIR \
  --output_file $RESULTS_FILE \
  --temp_file $TEMP_FILE \
  --model_type standard \
  --save_viz \
  --viz_dir $VIZ_DIR

# Analyze results
python experiments/scripts/shap/analyze_shap_robustness_results.py \
  --results_path $RESULTS_FILE \
  --figures_dir $FIGURES_DIR \
  --report_path $REPORT_PATH \
  --severity_level 3

# Clean up temporary script if created
if [ "$OPTIMIZE" = true ] && [ $N_STEPS -gt 0 ]; then
  rm $SCRIPT_PATH
fi

echo "SHAP robustness test on standard model completed!"
echo "Results saved to $RESULTS_FILE"
echo "Analysis report saved to $REPORT_PATH"
echo "Heatmaps saved to $FIGURES_DIR" 