# Integrated Gradients (IG) 鲁棒性测试

本目录包含用于测试 Integrated Gradients 解释方法对图像腐蚀的鲁棒性的脚本。

## 文件结构

- `test_ig_robustness.py`: 主测试脚本，处理图像并计算指标
- `analyze_ig_robustness_results.py`: 结果分析脚本，生成报告和可视化
- `requirements.txt`: 依赖项列表
- `run_ig_test.sh`: 运行标准模型完整测试的脚本
- `run_ig_test_robust.sh`: 运行健壮模型完整测试的脚本
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

### 3. 使用nohup运行健壮模型测试（后台运行）

```bash
# 确保在项目根目录运行
cd /path/to/robustness
nohup bash experiments/scripts/ig/run_ig_test_robust.sh > ig_robust_log.out 2>&1 &

# 记录进程ID
echo $!
```

### 4. 检查测试进度

您可以通过以下方式检查测试进度：

```bash
# 查看标准模型测试的日志
tail -f ig_standard_log.out

# 查看健壮模型测试的日志
tail -f ig_robust_log.out

# 查看临时结果文件大小
ls -lh experiments/results/ig_robustness_results_temp.json
ls -lh experiments/results/ig_robustness_robust_results_temp.json
```

### 5. 手动运行分析脚本（如果测试脚本中没有自动运行）

等测试完成后，您可以手动运行分析脚本：

```bash
# 分析标准模型结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_results.json \
  --figures_dir experiments/results/figures/ig \
  --report_path experiments/results/ig_analysis_report.md

# 分析健壮模型结果
python experiments/scripts/ig/analyze_ig_robustness_results.py \
  --results_path experiments/results/ig_robustness_robust_results.json \
  --figures_dir experiments/results/figures/ig_robust \
  --report_path experiments/results/ig_robust_analysis_report.md
```

## 结果文件

测试完成后，结果将保存在：

- 标准模型：
  - 结果JSON：`experiments/results/ig_robustness_results.json`
  - 可视化：`experiments/results/ig_viz/`
  - 分析报告：`experiments/results/ig_analysis_report.md`
  - 热图：`experiments/results/figures/ig/`

- 健壮模型：
  - 结果JSON：`experiments/results/ig_robustness_robust_results.json`
  - 可视化：`experiments/results/ig_viz_robust/`
  - 分析报告：`experiments/results/ig_robust_analysis_report.md`
  - 热图：`experiments/results/figures/ig_robust/`

## 命令行参数

测试脚本 `test_ig_robustness.py` 支持以下参数：

- `--image_dir`: 图像目录路径
- `--output_file`: 结果JSON输出路径
- `--temp_file`: 临时结果文件路径（每处理一张图片保存一次）
- `--model_type`: 模型类型，可选 'standard' 或 'robust'
- `--save_viz`: 是否保存可视化
- `--viz_dir`: 可视化保存目录
- `--test_mode`: 是否启用测试模式
- `--test_samples`: 测试模式下处理的图片数量

## 常见问题及解决方案

1. **SSH连接断开**：使用nohup命令可以确保即使SSH连接断开，测试也会继续在后台运行。

2. **内存不足**：如果遇到内存问题，可以修改`test_ig_robustness.py`中的批处理大小或减少同时生成的解释数量。

3. **磁盘空间不足**：结果JSON文件和可视化可能会占用大量磁盘空间。如果空间有限，可以考虑禁用可视化保存（删除`--save_viz`参数）。

4. **查看进程是否还在运行**：
   ```bash
   ps aux | grep test_ig_robustness
   ``` 