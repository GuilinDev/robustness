# LIME解释方法鲁棒性测试

此目录包含用于测试LIME解释方法对各种图像腐蚀（噪声、模糊、天气效果等）的鲁棒性的脚本。

## 依赖项

1.  **Python:** 安装 `requirements.txt` 中的依赖项:
    ```bash
    pip install -r experiments/scripts/lime/requirements.txt
    ```
2.  **OpenCV:** 需要系统级的OpenCV库。根据您的系统安装：
    *   **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y libopencv-dev python3-opencv`
    *   **CentOS/RHEL:** `sudo yum install -y opencv opencv-devel`
    *   **MacOS:** `brew install opencv`
3.  **bc (可选):** 脚本使用 `bc` 估算运行时间。如果未安装，会看到 `bc: command not found` 错误，但这不会影响核心计算。安装方法:
    *   Debian/Ubuntu: `sudo apt-get install bc`
    *   CentOS/RHEL: `sudo yum install bc`

## 性能与配置

LIME计算量较大。脚本默认启用优化 (`OPTIMIZE=true`)，使用 **1000张图像样本** (`SAMPLE_SIZE=1000`) 和 **50个LIME采样步数** (`STEPS=50`)，这是推荐的平衡性能和结果质量的配置。您可以在脚本顶部修改这些默认值，或通过命令行参数覆盖 (例如 `--samples 500 --steps 100`)。

## 运行测试

**重要:** 在运行任何脚本之前，请确保已激活您的项目虚拟环境 (e.g., `source .venv/bin/activate`).

### 本地快速测试

使用 `run_lime_test.sh` 或 `run_lime_robust_test.sh` 的 `--test` 标志可在20张图像上快速验证脚本是否正常工作：

```bash
# 测试标准模型
bash experiments/scripts/lime/run_lime_test.sh --test

# 测试鲁棒模型
bash experiments/scripts/lime/run_lime_robust_test.sh --test
```

### 完整测试 (推荐：后台运行)

建议在GPU服务器上运行完整测试，并使用 `nohup` 在后台执行，以防连接中断。这将使用默认的1000张图像和50个LIME步骤。

**1. 运行标准模型:**
```bash
nohup bash experiments/scripts/lime/run_lime_test.sh > lime_standard_run.log 2>&1 &
```

**2. 运行鲁棒模型:**
*(请在标准模型运行完成后再运行此命令，因为它依赖标准模型创建的样本列表)*
```bash
nohup bash experiments/scripts/lime/run_lime_robust_test.sh > lime_robust_run.log 2>&1 &
```

**3. 监控进度:**
您可以使用 `tail` 命令查看日志文件以监控进度或检查错误：
```bash
tail -f lime_standard_run.log
# (完成后，使用 Ctrl+C 退出 tail)

tail -f lime_robust_run.log
# (完成后，使用 Ctrl+C 退出 tail)
```

运行完成后，结果JSON文件将保存在 `experiments/results/` 目录中，分析报告和图表会自动生成。

## 手动运行分析

如果计算和分析分开进行，或需要重新生成报告/图表，可以手动运行分析脚本。请确保为 `standard` 和 `robust` 模型分别指定正确的路径和 `--model_type`。

```bash
# 标准模型分析示例
python experiments/scripts/lime/analyze_lime_robustness_results.py \
  --results_path experiments/results/lime_robustness_standard_results.json \
  --figures_dir experiments/results/figures/lime_standard \
  --report_path experiments/results/lime_standard_analysis_report.md \
  --model_type standard

# 鲁棒模型分析示例
python experiments/scripts/lime/analyze_lime_robustness_results.py \
  --results_path experiments/results/lime_robustness_robust_results.json \
  --figures_dir experiments/results/figures/lime_robust \
  --report_path experiments/results/lime_robust_analysis_report.md \
  --model_type robust
```

## 图表标识说明

生成的热图包含两种重要的标识信息：

1. **标题标识**: 每个热图的标题会显示"LIME: [指标名称]"。
2. **模型类型水印**: 每个热图的右下角有一个水印标识：
   - **S**: 标准模型 (Standard model)
   - **R**: 鲁棒模型 (Robust model)

## 故障排除

*   **缺少OpenCV依赖:** 参见上面的依赖项部分。
*   **内存问题:** 尝试减少LIME采样步数 (例如 `--steps 20`) 或减少处理的图像数量 (例如 `--samples 100`)。
*   **`bc: command not found`:** 参见上面的依赖项部分 (安装`bc`是可选的)。