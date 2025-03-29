#!/bin/bash

# 确保目录结构存在
mkdir -p experiments/results/ig_viz
mkdir -p experiments/results/figures/ig

# 安装所需依赖
pip install -r experiments/scripts/ig/requirements.txt

# 运行完整的IG测试 (非测试模式)
python experiments/scripts/ig/test_ig_robustness.py \
  --image_dir experiments/data/tiny-imagenet-200/val \
  --output_file experiments/results/ig_robustness_results.json \
  --temp_file experiments/results/ig_robustness_results_temp.json \
  --model_type standard \
  --save_viz

# 分析结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_results.json \
  --figures_dir experiments/results/figures/ig \
  --report_path experiments/results/ig_analysis_report.md \
  --severity_level 3

echo "IG鲁棒性测试完成！" 