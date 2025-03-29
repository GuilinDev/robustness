#!/bin/bash

# 确保目录结构存在
mkdir -p experiments/results/ig_viz_robust
mkdir -p experiments/results/figures/ig_robust

# 安装所需依赖
pip install -r experiments/scripts/ig/requirements.txt

# 运行完整的IG测试 (robust模型)
python experiments/scripts/ig/test_ig_robustness.py \
  --image_dir experiments/data/tiny-imagenet-200/val \
  --output_file experiments/results/ig_robustness_robust_results.json \
  --temp_file experiments/results/ig_robustness_robust_results_temp.json \
  --model_type robust \
  --save_viz \
  --viz_dir experiments/results/ig_viz_robust

# 分析结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_robust_results.json \
  --figures_dir experiments/results/figures/ig_robust \
  --report_path experiments/results/ig_robust_analysis_report.md \
  --severity_level 3

echo "Robust模型IG鲁棒性测试完成！" 