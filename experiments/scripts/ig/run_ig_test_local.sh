#!/bin/bash

# 确保目录结构存在
mkdir -p experiments/results/ig_viz_test
mkdir -p experiments/results/figures/ig_test

# 安装所需依赖
pip install -r experiments/scripts/ig/requirements.txt

# 运行测试模式的IG测试（只处理10张图片）
python experiments/scripts/ig/test_ig_robustness.py \
  --image_dir experiments/data/tiny-imagenet-200/val \
  --output_file experiments/results/ig_robustness_test_results.json \
  --temp_file experiments/results/ig_robustness_test_results_temp.json \
  --model_type standard \
  --save_viz \
  --viz_dir experiments/results/ig_viz_test \
  --test_mode \
  --test_samples 10

# 分析结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_test_results.json \
  --figures_dir experiments/results/figures/ig_test \
  --report_path experiments/results/ig_test_analysis_report.md \
  --severity_level 3

echo "IG鲁棒性本地测试完成！" 