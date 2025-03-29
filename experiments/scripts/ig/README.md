# Integrated Gradients (IG) 鲁棒性测试

本目录包含用于测试 Integrated Gradients 解释方法对图像腐蚀的鲁棒性的脚本。

## 文件结构

- `test_ig_robustness.py`: 主测试脚本，处理图像并计算指标
- `analyze_ig_robustness_results.py`: 结果分析脚本，生成报告和可视化
- `requirements.txt`: 依赖项列表
- `run_ig_test.sh`: 运行标准模型完整测试的脚本
- `run_ig_test_robust.sh`: 运行健壮模型完整测试的脚本
- `run_ig_test_local.sh`: 本地测试脚本（仅处理10张图片）

## 如何使用

### 本地测试

在将代码上传到GPU服务器之前，可以在本地运行测试模式验证代码是否正常工作：

```bash
bash experiments/scripts/ig/run_ig_test_local.sh
```

### 在GPU服务器上运行

1. 将整个项目上传到GPU服务器
2. 确保数据集位于 `experiments/data/tiny-imagenet-200` 目录
3. 确保运行目录是项目根目录

运行标准模型测试：
```bash
bash experiments/scripts/ig/run_ig_test.sh
```

运行Robust模型测试：
```bash
bash experiments/scripts/ig/run_ig_test_robust.sh
```

## 结果

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