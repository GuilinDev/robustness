# SHAP Robustness Analysis Report

## Results by Corruption Type (Severity Level 3)

*表格按照准确率从高到低排序*

| Corruption Type | Accuracy | Cosine Similarity | Mutual Information | IoU Score | Mean Flip Rate | Confidence Difference | KL Divergence | Mean Top-5 Distance | Mean Corruption Error | Stability Score |
|---------------|----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| contrast | 0.900 | 0.582 | 2.365 | 0.024 | 0.100 | 0.095 | 0.213 | 1.450 | 0.100 | 0.877 |
| zoom_blur | 0.800 | 0.555 | 2.466 | 0.011 | 0.200 | 0.089 | 0.402 | 1.750 | 0.200 | 0.907 |
| brightness | 0.750 | 0.514 | 2.407 | 0.016 | 0.250 | 0.125 | 0.465 | 1.850 | 0.250 | 0.880 |
| jpeg | 0.700 | 0.540 | 2.386 | 0.007 | 0.300 | 0.179 | 0.411 | 1.950 | 0.300 | 0.907 |
| motion_blur | 0.650 | 0.470 | 2.403 | 0.000 | 0.350 | 0.222 | 0.942 | 3.050 | 0.350 | 0.922 |
| fog | 0.650 | 0.559 | 2.437 | 0.029 | 0.350 | 0.236 | 0.706 | 2.000 | 0.350 | 0.929 |
| defocus_blur | 0.600 | 0.469 | 2.387 | 0.001 | 0.400 | 0.236 | 1.023 | 2.950 | 0.400 | 0.909 |
| glass_blur | 0.600 | 0.475 | 2.413 | 0.002 | 0.400 | 0.229 | 0.969 | 2.950 | 0.400 | 0.912 |
| shot_noise | 0.550 | 0.505 | 2.454 | 0.008 | 0.450 | 0.407 | 1.601 | 2.550 | 0.450 | 0.864 |
| pixelate | 0.450 | 0.490 | 2.382 | 0.000 | 0.550 | 0.456 | 2.170 | 3.700 | 0.550 | 0.890 |
| gaussian_noise | 0.100 | 0.474 | 2.376 | 0.000 | 0.900 | 0.612 | 4.209 | 4.050 | 0.900 | 0.780 |
| snow | 0.050 | 0.466 | 2.460 | 0.001 | 0.950 | 0.498 | 5.305 | 4.250 | 0.950 | 0.853 |
| frost | 0.050 | 0.464 | 2.393 | 0.002 | 0.950 | 0.523 | 3.674 | 3.450 | 0.950 | 0.878 |
| impulse_noise | 0.000 | 0.470 | 2.470 | 0.001 | 1.000 | 0.629 | 5.075 | 4.650 | 1.000 | 0.814 |
| elastic_transform | 0.000 | 0.426 | 2.385 | 0.000 | 1.000 | 0.592 | 5.801 | 4.850 | 1.000 | 0.914 |

## Correlation Between Accuracy and Metrics

This section analyzes the correlation between model accuracy and explanation metrics across different corruption types.

| Metric | Correlation with Accuracy |
|--------|----------------------------|
| Mean Flip Rate | -1.000 (strong negative) |
| Mean Corruption Error | -1.000 (strong negative) |
| KL Divergence | -0.972 (strong negative) |
| Confidence Difference | -0.955 (strong negative) |
| Mean Top-5 Distance | -0.925 (strong negative) |
| Cosine Similarity | 0.767 (strong positive) |
| IoU Score | 0.606 (moderate positive) |
| Stability Score | 0.541 (moderate positive) |
| Mutual Information | -0.118 (weak negative) |

## Analysis of Corruption Effects

### Most and Least Robust Corruptions

- **Most Robust**: contrast (Accuracy: 0.900)
- **Least Robust**: elastic_transform (Accuracy: 0.000)

### Corruption Category Analysis

Average accuracy by corruption category:

- **Blur**: 0.663
- **Digital**: 0.560
- **Weather**: 0.250
- **Noise**: 0.217

## Explanation Robustness Analysis

### Explanation Robustness by Corruption Type

- **Most Robust Explanations**: fog (Avg. metrics: 0.989)
- **Least Robust Explanations**: gaussian_noise (Avg. metrics: 0.908)


## Summary

This analysis evaluated the robustness of SHAP explanations across 15 different corruption types. The results show how different corruptions affect both model predictions and explanation quality.

Key findings:

1. Model accuracy varies significantly across corruption types
2. Explanation robustness is highest for fog corruptions
3. Explanation robustness is lowest for gaussian_noise corruptions
4. The metric most correlated with accuracy is Mean Flip Rate (correlation: -1.000)
