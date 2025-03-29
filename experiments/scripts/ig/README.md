# Integrated Gradients (IG) Robustness Testing

This directory contains scripts for testing and analyzing the robustness of Integrated Gradients (IG) explanations against various corruptions.

## Requirements

### Python Dependencies
Install the required Python packages with:
```
pip install -r requirements.txt
```

### System Dependencies for OpenCV
If you encounter OpenCV related errors (`ImportError: libGL.so.1: cannot open shared object file`), install the necessary system packages:

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

## Scripts

### `run_ig_test.sh`
Main script to run the IG robustness test on a standard ResNet model.

Usage:
```bash
./run_ig_test.sh
```

#### Optimization Options
The script includes optimization options at the top:

```bash
# Script configuration - Uncomment and modify these options for optimization
OPTIMIZE=true      # Set to true to enable optimizations
SAMPLE_SIZE=1000   # Number of images to process (set to 0 for all images)
N_STEPS=20         # IG integration steps (original is 50, lower is faster)
RANDOM_SEED=42     # Fixed random seed for reproducibility
```

- Set `OPTIMIZE=true` to enable optimizations
- Adjust `SAMPLE_SIZE` to control the number of test images (1000 is recommended for quick testing)
- Modify `N_STEPS` to change the number of integration steps (20 is recommended for faster results)
- The `RANDOM_SEED` ensures the same images are selected for both standard and robust tests

When running with optimizations, the script will:
1. Create a subset of random images for testing
2. Use fewer integration steps for faster computation
3. Save a list of selected images to ensure reproducibility between standard and robust model tests

### `run_ig_robust_test.sh`
Script to run the IG robustness test on a robust (adversarially trained) ResNet model.

Usage:
```bash
./run_ig_robust_test.sh
```

**Note:** Always run the standard model test first (`run_ig_test.sh`), as the robust model test uses the same sample list created by the standard test.

### `test_ig_robustness.py`
Python script that implements the IG robustness testing.

Usage:
```bash
python test_ig_robustness.py --image_dir <image_directory> --output_file <output_file> --model_type <model_type>
```

Arguments:
- `--image_dir`: Directory containing images to test
- `--output_file`: Path to save the results
- `--temp_file`: Path to save intermediate results
- `--model_type`: Type of model to use ('standard' or 'robust')
- `--save_viz`: Flag to save visualizations
- `--viz_dir`: Directory to save visualizations

### `analyze_ig_robustness_results.py`
Script for analyzing the results of IG robustness tests.

Usage:
```bash
python analyze_ig_robustness_results.py --results_path <results_file> --figures_dir <figures_directory> --report_path <report_file>
```

Arguments:
- `--results_path`: Path to the results JSON file
- `--figures_dir`: Directory to save generated figures
- `--report_path`: Path to save the analysis report
- `--severity_level`: Corruption severity level to analyze (default: 3)

## Performance Considerations

- **Processing Time**: IG is computationally intensive. The full test on all TinyImageNet validation images can take several hours or even days on a CPU.
- **GPU Acceleration**: Running on a GPU is highly recommended and will significantly speed up processing.
- **Memory Usage**: Processing and storing results for large datasets requires significant memory. Use the optimization options for development and testing.
- **Storage Space**: Visualization files can consume significant disk space (multiple GB for the full dataset).

## Running on GPU Instance

For best performance on a GPU instance:

1. Ensure CUDA is properly configured.
2. Verify PyTorch is using the GPU:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   print(torch.cuda.device_count())  # Number of available GPUs
   ```
3. Run the standard model test first, then the robust model test:
   ```bash
   # First, run the standard model test
   nohup ./run_ig_test.sh > ig_standard_log.log 2>&1 &
   
   # After the standard test is complete, run the robust model test
   nohup ./run_ig_robust_test.sh > ig_robust_log.log 2>&1 &
   ```
4. Monitor the process:
   ```bash
   tail -f ig_standard_log.log
   tail -f ig_robust_log.log
   ```

## Common Issues and Solutions

- **Gray or blank stability heatmap**: This may occur if the stability values are too uniform. The script now automatically adjusts the color mapping to improve visualization.
- **Missing OpenCV dependencies**: Install the system packages as noted in the System Dependencies section.
- **Process killed**: This typically occurs due to out-of-memory errors. Use the optimization options to reduce memory usage.
- **Sample list errors**: If you see an error about missing sample list when running the robust test, ensure you've run the standard model test first.

## 文件结构

- `test_ig_robustness.py`: 主测试脚本，处理图像并计算指标
- `analyze_ig_robustness_results.py`: 结果分析脚本，生成报告和可视化
- `requirements.txt`: 依赖项列表
- `run_ig_test.sh`: 运行标准模型完整测试的脚本
- `run_ig_test_robust.sh`: 运行鲁棒模型完整测试的脚本
- `run_ig_test_local.sh`: 本地测试脚本（仅处理10张图片）

## 准备工作

在运行测试前，确保数据集已正确设置：

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/tinyimageset.sh
```

## 在GPU实例上运行测试

### 1. 先运行本地测试脚本确认设置正确

在GPU实例上，首先运行测试模式验证代码是否正常工作：

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/ig/run_ig_test_local.sh
```

本地测试成功后，可以进行完整的测试运行。

### 2. 使用nohup运行标准模型测试（后台运行）

```bash
# 确保在项目根目录运行
cd /path/to/robustness
nohup bash experiments/scripts/ig/run_ig_test.sh > ig_standard_log.out 2>&1 &

# 记录进程ID
echo $!
```

### 3. 使用nohup运行鲁棒模型测试（后台运行）

```bash
# 确保在项目根目录运行
cd /path/to/robustness
nohup bash experiments/scripts/ig/run_ig_robust_test.sh > ig_robust_log.out 2>&1 &

# 记录进程ID
echo $!
```

### 4. 检查测试进度

可以通过以下方式检查测试进度：

```bash
# 查看标准模型测试的日志
tail -f ig_standard_log.out

# 查看鲁棒模型测试的日志
tail -f ig_robust_log.out

# 查看临时结果文件大小
ls -lh experiments/results/ig_robustness_standard_temp.json
ls -lh experiments/results/ig_robustness_robust_temp.json
```

### 5. 手动运行分析脚本（如果测试脚本中没有自动运行）

等测试完成后，可以手动运行分析脚本：

```bash
# 分析标准模型结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_standard_results.json \
  --figures_dir experiments/results/ig_standard_figures \
  --report_path experiments/results/ig_standard_analysis_report.md

# 分析鲁棒模型结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_robust_results.json \
  --figures_dir experiments/results/ig_robust_figures \
  --report_path experiments/results/ig_robust_analysis_report.md
```

## 结果文件

测试完成后，结果将保存在不同目录，确保不会相互覆盖：

- 标准模型：
  - 结果JSON：`experiments/results/ig_robustness_standard_results.json`
  - 可视化：`experiments/results/ig_viz_standard/`
  - 分析报告：`experiments/results/ig_standard_analysis_report.md`
  - 热图：`experiments/results/ig_standard_figures/`

- 鲁棒模型：
  - 结果JSON：`experiments/results/ig_robustness_robust_results.json`
  - 可视化：`experiments/results/ig_viz_robust/`
  - 分析报告：`experiments/results/ig_robust_analysis_report.md`
  - 热图：`experiments/results/ig_robust_figures/`

- 本地测试：
  - 结果JSON：`experiments/results/ig_robustness_local_results.json`
  - 可视化：`experiments/results/ig_viz_local/`
  - 分析报告：`experiments/results/ig_local_analysis_report.md`
  - 热图：`experiments/results/ig_local_figures/`

## 比较分析

为了比较标准模型和鲁棒模型的结果，可以手动比较两个报告:

```bash
# 查看两个报告
cat experiments/results/ig_standard_analysis_report.md
cat experiments/results/ig_robust_analysis_report.md

# 比较热图
ls experiments/results/ig_standard_figures/
ls experiments/results/ig_robust_figures/
```

也可以编写额外的Python脚本来直接比较两个JSON结果文件中的指标。 