# GradCAM Robustness Testing

This directory contains scripts for testing and analyzing the robustness of GradCAM explanations against various corruptions.

## Scripts

### `test_gradcam_robustness.py`
Python script that implements the GradCAM robustness testing.

Usage:
```bash
python test_gradcam_robustness.py --image_dir <image_directory> --output_file <output_file> --model_type <model_type>
```

### `analyze_gradcam_robustness_results.py`
Script for analyzing the results of GradCAM robustness tests.

Usage:
```bash
python analyze_gradcam_robustness_results.py --results_path <results_file> --figures_dir <figures_directory> --report_path <report_file> --model_type <model_type>
```

Arguments:
- `--results_path`: Path to the results JSON file
- `--figures_dir`: Directory to save generated figures
- `--report_path`: Path to save the analysis report
- `--severity_level`: Corruption severity level to analyze (default: 3)
- `--model_type`: Type of model used ('standard' or 'robust')

## 图表标识说明

生成的热图包含两种重要的标识信息：

1. **标题标识**：每个热图的标题会显示"GradCAM: [指标名称]"，清晰标识这些结果是使用GradCAM方法生成的。

2. **模型类型水印**：每个热图的右下角有一个水印标识：
   - **S**: 表示使用标准模型 (Standard model) 生成的结果
   - **R**: 表示使用鲁棒模型 (Robust model) 生成的结果

这样的标识方式能够在比较不同解释方法和不同模型的结果时，快速识别图表来源。

## 运行示例

### 标准模型分析
```bash
python analyze_gradcam_robustness_results.py \
  --results_path experiments/results/gradcam_robustness_results.json \
  --figures_dir experiments/results/figures/gradcam/standard \
  --report_path experiments/results/gradcam/standard/analysis_report.md \
  --model_type standard
```

### 鲁棒模型分析
```bash
python analyze_gradcam_robustness_results.py \
  --results_path experiments/results/gradcam_robustness_robustbench_results.json \
  --figures_dir experiments/results/figures/gradcam/robust \
  --report_path experiments/results/gradcam/robust/analysis_report.md \
  --model_type robust
```

## 结果比较

比较标准模型和鲁棒模型的结果可以帮助理解模型鲁棒性与解释鲁棒性之间的关系。通过对比相同指标在不同模型下的热图，可以发现:

1. 鲁棒模型是否同时也产生更鲁棒的解释
2. 特定类型的腐蚀对不同模型解释的影响是否一致
3. 模型准确率与解释质量之间的相关性 