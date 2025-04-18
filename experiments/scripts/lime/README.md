# LIME解释方法鲁棒性测试

此目录包含用于测试LIME解释方法对各种图像腐蚀（噪声、模糊、天气效果等）的鲁棒性的脚本。

## 系统依赖

除了Python依赖项之外，这些脚本还需要OpenCV系统依赖：

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev python3-opencv
```

### CentOS/RHEL
```bash
sudo yum install -y opencv opencv-devel
```

### MacOS
```bash
brew install opencv
```

## 性能考虑

LIME解释生成在计算上比GradCAM更密集。处理大量图像可能需要很长时间。可以通过以下方式优化性能：

1. 使用`--optimize`选项减少超参数(例如样本数量)
2. 使用`--steps`选项手动设置样本数(默认为1000，建议10-200用于快速测试)
3. 使用`--test`选项运行测试模式，仅处理20张图像

示例:
```bash
bash experiments/scripts/lime/run_lime_test.sh --optimize --steps 50
```

## 本地测试

要在少量图像上运行快速测试以验证设置是否正确，请使用:

```bash
bash experiments/scripts/lime/run_lime_test_local.sh
```

这将处理一个特定类别的几张图像。

## 运行完整测试

要在Tiny-ImageNet-200验证集上使用标准模型运行完整测试：

```bash
bash experiments/scripts/lime/run_lime_test.sh
```

要使用鲁棒模型运行测试：

```bash
bash experiments/scripts/lime/run_lime_robust_test.sh
```

## 手动运行分析

如果您已经有了测试结果，可以单独运行分析脚本:

```bash
python experiments/scripts/lime/analyze_lime_robustness_results.py \
    --results_path experiments/results/lime_robustness_results.json \
    --figures_dir experiments/results/figures/lime \
    --report_path experiments/results/lime_analysis_report.md \
    --severity_level 3
```

## 故障排除

### 灰色稳定性热图

如果生成的稳定性热图在所有单元格中显示相同的值(例如都为1.000)，热图可能会显示为灰色。这是正常的，表示该解释方法在不同的腐蚀和严重级别上一致地表现。不需要任何更改。

### 缺少OpenCV依赖

如果您在运行脚本时看到与OpenCV相关的错误，请确保已安装系统依赖项（见上文）。

### 内存问题

如果您在处理图像时遇到内存问题，请考虑：
1. 使用`--optimize --steps 10`选项减少LIME样本数量
2. 使用`--test`选项运行测试模式，仅处理20张图像
3. 在具有更多RAM的计算机上运行测试

### `bc: command not found` 错误

运行 `run_lime_test.sh` 或 `run_lime_robust_test.sh` 时，您可能会看到类似 `bc: command not found` 的错误。`bc` 是一个用于脚本中估算运行时间的命令行计算器工具。此错误表示您的系统上未安装 `bc`。该错误会阻止脚本打印预计运行时间，并且由于脚本中的 `set -e`，可能会导致脚本提前终止。

**解决方法：** 使用系统的包管理器安装 `bc`。例如：
*   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install bc`
*   CentOS/RHEL: `sudo yum install bc`

安装 `bc` 后，脚本应该能够正确运行。

## 在GPU实例上运行

建议在GPU实例上运行这些测试，因为LIME解释计算可能非常密集。要在后台运行测试，使用：

```bash
nohup bash experiments/scripts/lime/run_lime_test.sh > lime_standard_log.out 2>&1 &
```
```bash
nohup bash experiments/scripts/lime/run_lime_robust_test.sh > lime_robust_log.out 2>&1 &
```

要检查测试进度，可以查看日志文件：

```bash
tail -f lime_standard_log.out
```

## 图表标识说明

生成的热图包含两种重要的标识信息：

1. **标题标识**：每个热图的标题会显示"LIME: [指标名称]"，清晰标识这些结果是使用LIME方法生成的。

2. **模型类型水印**：每个热图的右下角有一个水印标识：
   - **S**: 表示使用标准模型 (Standard model) 生成的结果
   - **R**: 表示使用鲁棒模型 (Robust model) 生成的结果

这样的标识方式能够在比较不同解释方法和不同模型的结果时，快速识别图表来源。水印使用白色填充和黑色边框设计，确保在任何背景颜色下都清晰可见。

### 运行分析脚本示例

```bash
# 标准模型分析
python experiments/scripts/lime/analyze_lime_robustness_results.py \
  --results_path experiments/results/lime_robustness_results.json \
  --figures_dir experiments/results/figures/lime/standard \
  --report_path experiments/results/lime/standard/analysis_report.md \
  --model_type standard

# 鲁棒模型分析
python experiments/scripts/lime/analyze_lime_robustness_results.py \
  --results_path experiments/results/lime_robustness_robust_results.json \
  --figures_dir experiments/results/figures/lime/robust \
  --report_path experiments/results/lime/robust/analysis_report.md \
  --model_type robust
``` 