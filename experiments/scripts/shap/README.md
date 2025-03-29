# SHAP鲁棒性测试

本目录包含用于测试SHAP解释方法鲁棒性的脚本和工具。SHAP（SHapley Additive exPlanations）是一种基于博弈论的模型解释方法，用于解释模型的预测结果。本测试框架评估SHAP解释在图像受到各种腐蚀（如噪声、模糊、天气效果等）时的鲁棒性。

## 系统依赖

在运行脚本前，确保已安装以下系统依赖：

### OpenCV依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install -y mesa-libGL glib2

# MacOS
brew install opencv
```

## 性能考虑

SHAP计算计算量较大，尤其是在处理大量图像时。为了优化性能，可以考虑：

1. 使用`--optimize`和`--steps`选项减少SHAP解释计算的样本数
2. 使用TEST模式（`--test`）处理少量图像进行调试
3. 降低处理的图像数量
4. 在具有足够GPU内存的机器上运行测试

注意：标准的SHAP测试在图像数量较多时可能需要几天时间完成，建议在GPU实例上运行。

## 运行本地测试

为了快速验证设置是否正确，首先运行本地测试脚本，该脚本只处理少量图像：

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/shap/run_shap_test_local.sh
```

## 运行完整测试

### 标准模型测试

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/shap/run_shap_test.sh
```

可选参数：
- `--optimize`: 启用优化模式
- `--steps N`: 设置SHAP解释计算的样本数（默认为10）
- `--test`: 仅使用少量图像进行测试

例如：
```bash
bash experiments/scripts/shap/run_shap_test.sh --optimize --steps 5
```

### 鲁棒模型测试

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/shap/run_shap_robust_test.sh
```

此脚本接受与标准模型测试相同的可选参数。

## 手动运行分析脚本

如果想要单独运行分析脚本，可以使用：

```bash
python experiments/scripts/shap/analyze_shap_robustness_results.py \
  --results_path experiments/results/shap_robustness_standard_results.json \
  --figures_dir experiments/results/figures/shap_standard \
  --report_path experiments/results/shap_standard_analysis_report.md \
  --severity_level 3
```

## 常见问题解决

- **灰色或空白的稳定性热图**：如果稳定性值非常一致，热图可能显示为灰色。这是正常现象，表示SHAP解释的稳定性在不同情况下非常一致。
- **缺少OpenCV依赖**：安装系统包，如上述系统依赖部分所示。
- **进程被终止**：通常是由于内存不足。使用优化选项减少内存使用量。
- **样本列表错误**：如果在运行鲁棒测试时出现关于缺失样本列表的错误，确保先运行标准模型测试。

## 在GPU实例上运行测试

### 1. 先运行本地测试脚本确认设置正确

在GPU实例上，首先运行测试模式验证代码是否正常工作：

```bash
# 在项目根目录运行
cd /path/to/robustness
bash experiments/scripts/shap/run_shap_test_local.sh
```

本地测试成功后，可以进行完整的测试运行。

### 2. 使用nohup运行标准模型测试（后台运行）

```bash
# 确保在项目根目录运行
cd /path/to/robustness
nohup bash experiments/scripts/shap/run_shap_test.sh > shap_standard_log.out 2>&1 &

# 记录进程ID
echo $!
```

### 3. 使用nohup运行鲁棒模型测试（后台运行）

```bash
# 确保在项目根目录运行
cd /path/to/robustness
nohup bash experiments/scripts/shap/run_shap_robust_test.sh > shap_robust_log.out 2>&1 &

# 记录进程ID
echo $!
```

### 4. 检查测试进度

可以通过以下方式检查测试进度：

```bash
# 查看标准模型测试的日志
tail -f shap_standard_log.out

# 查看鲁棒模型测试的日志
tail -f shap_robust_log.out

# 查看临时结果文件大小
ls -lh experiments/results/shap_robustness_standard_temp.json
ls -lh experiments/results/shap_robustness_robust_temp.json
``` 