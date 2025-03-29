# Occlusion Sensitivity 鲁棒性测试框架

这个框架用于测试Occlusion Sensitivity解释方法对图像腐蚀的鲁棒性。Occlusion Sensitivity是一种通过系统地遮挡图像的不同部分并观察模型输出变化来生成解释的方法。

## 概述

该测试框架包含以下组件：

1. **主测试脚本** (`test_occlusion_sensitivity_robustness.py`): 这个脚本实现了Occlusion Sensitivity解释方法，并测试其对各种图像腐蚀的鲁棒性。

2. **分析脚本** (`analyze_occlusion_sensitivity_robustness_results.py`): 用于分析测试结果，生成热图和汇总报告。

3. **运行脚本**:
   - `run_occlusion_sensitivity_test_local.sh`: 本地测试脚本，使用少量图片进行快速测试。
   - `run_occlusion_sensitivity_test.sh`: 标准模型测试脚本，使用ResNet50模型进行测试。
   - `run_occlusion_sensitivity_robust_test.sh`: 鲁棒模型测试脚本，使用经过对抗训练的鲁棒ResNet50模型。

## 环境要求

- Python 3.8或更高版本
- PyTorch 2.0或更高版本
- torchvision
- NumPy
- Pillow
- OpenCV
- scikit-image
- scikit-learn
- SciPy
- Matplotlib
- Pandas
- Seaborn
- RobustBench (用于鲁棒模型)
- tqdm

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## Occlusion Sensitivity方法

Occlusion Sensitivity解释方法的工作原理是：

1. 使用滑动窗口沿图像移动，对每个位置的图像区域进行遮挡。
2. 记录每个遮挡位置模型输出的变化。
3. 生成一个敏感度图，其中每个像素值表示遮挡该位置对模型预测的影响程度。
4. 重要的区域被遮挡后会导致模型预测概率的显著下降，因此敏感度图的高值区域对应着模型认为重要的区域。

与其他解释方法相比，Occlusion Sensitivity的优点：

- 概念简单、易于理解。
- 完全模型无关，可以应用于任何黑盒模型。
- 不依赖于模型的梯度，适用于非可微模型。

## 使用指南

### 本地测试

运行本地测试脚本来快速验证框架：

```bash
bash experiments/scripts/occlusion_sensitivity/run_occlusion_sensitivity_test_local.sh
```

这将在少量样本图像上运行测试，并在`results/occlusion_sensitivity/`目录下生成结果。

### 标准模型测试

使用标准ResNet50模型进行完整测试：

```bash
bash experiments/scripts/occlusion_sensitivity/run_occlusion_sensitivity_test.sh
```

可选参数：
- `--save_viz`: 保存可视化结果
- `--patch_size SIZE`: 设置遮挡区域大小（默认16）
- `--stride STRIDE`: 设置滑动步长（默认8）
- `--device DEVICE`: 指定设备（例如"cuda:0"或"cpu"）

例如：

```bash
bash experiments/scripts/occlusion_sensitivity/run_occlusion_sensitivity_test.sh --save_viz --patch_size 24 --stride 12 --device "cuda:0"
```

### 鲁棒模型测试

使用鲁棒ResNet50模型进行测试：

```bash
bash experiments/scripts/occlusion_sensitivity/run_occlusion_sensitivity_robust_test.sh
```

可选参数与标准模型测试相同。

## 参数调整

- **patch_size**: 遮挡区域的大小。较大的值会使解释更粗糙但计算更快。
- **stride**: 滑动窗口的步长。较小的值会产生更精细的解释，但计算成本更高。
- **baseline_value**: 用于遮挡的基准值。默认为0。

## 输出说明

测试脚本将生成一个JSON文件，包含以下指标：

- **cosine_similarity**: 原始和腐蚀后解释之间的余弦相似度。
- **mutual_information**: 解释之间的互信息。
- **iou**: 解释之间的交并比。
- **prediction_change**: 预测类别是否改变的指示符。
- **confidence_diff**: 预测概率的差异。
- **kl_divergence**: 原始和腐蚀后预测分布的KL散度。
- **top5_distance**: 前5个预测类别之间的差异。
- **corruption_error**: 腐蚀图像上的错误率。
- **stability**: 解释的稳定性度量。

分析脚本将生成：

1. 热图，显示不同腐蚀类型和严重程度下的指标变化。
2. 折线图，显示指标如何随腐蚀严重程度变化。
3. 摘要报告，包含关键统计数据。

## 故障排除

### 常见问题

1. **内存错误**: 
   - 减小`patch_size`或增大`stride`值。
   - 减少批处理大小或使用更小的图像。

2. **CUDA错误**:
   - 确保指定的CUDA设备可用。
   - 检查CUDA驱动程序和PyTorch版本是否兼容。
   - 使用`--device cpu`切换到CPU处理。

3. **运行时间过长**:
   - 增大`patch_size`和`stride`值以减少计算量。
   - 减少测试图像数量。
   - 考虑使用更强大的GPU。

4. **结果不一致**:
   - 不同的`patch_size`和`stride`值可能产生不同的解释。
   - 确保对比实验使用相同的参数设置。

### 中断恢复

如果测试中断，可以使用相同的输出文件和临时文件路径重新运行脚本，它将从上次中断的地方继续。

```bash
bash experiments/scripts/occlusion_sensitivity/run_occlusion_sensitivity_test.sh
```

## 与其他解释方法的比较

与其他解释方法（如LIME、SHAP、DeepLIFT等）相比，Occlusion Sensitivity具有以下特点：

1. **简单直观**: 概念上更容易理解，不涉及复杂的数学推导。
2. **通用性**: 可以应用于任何黑盒模型，不需要访问模型内部结构或梯度。
3. **计算开销**: 通常计算成本较高，尤其是对于高分辨率图像和小步长。
4. **解释质量**: 通常能够捕捉模型关注的区域，但可能不如基于梯度的方法精细。

推荐根据不同的应用场景和需求选择适合的解释方法。 