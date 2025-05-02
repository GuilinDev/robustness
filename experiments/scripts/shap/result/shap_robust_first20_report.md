# SHAP Robustness Analysis Report

## Results by Corruption Type (Severity Level 3)

*表格按照准确率从高到低排序*

| Corruption Type | Accuracy | Cosine Similarity | Mutual Information | IoU Score | Mean Flip Rate | Confidence Difference | KL Divergence | Mean Top-5 Distance | Mean Corruption Error | Stability Score |
|---------------|----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| zoom_blur | 0.950 | 0.528 | 1.469 | 0.112 | 0.050 | 0.072 | 0.126 | 1.000 | 0.050 | 0.955 |
| fog | 0.750 | 0.625 | 1.339 | 0.281 | 0.250 | 0.276 | 1.803 | 3.400 | 0.250 | 0.965 |
| jpeg | 0.750 | 0.431 | 1.481 | 0.037 | 0.250 | 0.096 | 0.167 | 1.350 | 0.250 | 0.959 |
| motion_blur | 0.600 | 0.258 | 1.460 | 0.007 | 0.400 | 0.268 | 1.109 | 2.650 | 0.400 | 0.948 |
| contrast | 0.550 | 0.412 | 1.538 | 0.037 | 0.450 | 0.233 | 0.715 | 2.000 | 0.450 | 0.955 |
| glass_blur | 0.500 | 0.310 | 1.503 | 0.022 | 0.500 | 0.239 | 1.432 | 2.950 | 0.500 | 0.948 |
| defocus_blur | 0.450 | 0.301 | 1.548 | 0.020 | 0.550 | 0.233 | 1.525 | 2.950 | 0.550 | 0.945 |
| brightness | 0.450 | 0.258 | 1.538 | 0.024 | 0.550 | 0.342 | 1.685 | 3.450 | 0.550 | 0.953 |
| pixelate | 0.150 | 0.248 | 1.500 | 0.003 | 0.850 | 0.390 | 2.185 | 2.650 | 0.850 | 0.956 |
| gaussian_noise | 0.000 | 0.200 | 1.599 | 0.000 | 1.000 | 0.347 | 8.661 | 4.300 | 1.000 | 0.928 |
| shot_noise | 0.000 | 0.140 | 1.523 | 0.000 | 1.000 | 0.352 | 7.173 | 4.250 | 1.000 | 0.919 |
| impulse_noise | 0.000 | 0.106 | 1.706 | 0.000 | 1.000 | 0.370 | 13.338 | 4.400 | 1.000 | 0.935 |
| snow | 0.000 | 0.221 | 1.684 | 0.001 | 1.000 | 0.442 | 4.098 | 4.350 | 1.000 | 0.904 |
| frost | 0.000 | 0.249 | 1.430 | 0.027 | 1.000 | 0.377 | 4.783 | 3.950 | 1.000 | 0.962 |
| elastic_transform | 0.000 | 0.087 | 1.536 | 0.002 | 1.000 | 0.350 | 9.572 | 4.350 | 1.000 | 0.949 |

## Correlation Between Accuracy and Metrics

This section analyzes the correlation between model accuracy and explanation metrics across different corruption types.

| Metric | Correlation with Accuracy |
|--------|----------------------------|
| Mean Flip Rate | -1.000 (strong negative) |
| Mean Corruption Error | -1.000 (strong negative) |
| Confidence Difference | -0.869 (strong negative) |
| Cosine Similarity | 0.863 (strong positive) |
| Mean Top-5 Distance | -0.855 (strong negative) |
| KL Divergence | -0.784 (strong negative) |
| IoU Score | 0.591 (moderate positive) |
| Stability Score | 0.588 (moderate positive) |
| Mutual Information | -0.572 (moderate negative) |

## Analysis of Corruption Effects

### Most and Least Robust Corruptions

- **Most Robust**: zoom_blur (Accuracy: 0.950)
- **Least Robust**: elastic_transform (Accuracy: 0.000)

### Corruption Category Analysis

Average accuracy by corruption category:

- **Blur**: 0.625
- **Digital**: 0.380
- **Weather**: 0.250
- **Noise**: 0.000

## Explanation Robustness Analysis

### Explanation Robustness by Corruption Type

- **Most Robust Explanations**: fog (Avg. metrics: 0.803)
- **Least Robust Explanations**: elastic_transform (Avg. metrics: 0.644)


## Summary

This analysis evaluated the robustness of SHAP explanations across 15 different corruption types. The results show how different corruptions affect both model predictions and explanation quality.

Key findings:

1. Model accuracy varies significantly across corruption types
2. Explanation robustness is highest for fog corruptions
3. Explanation robustness is lowest for elastic_transform corruptions
4. The metric most correlated with accuracy is Mean Flip Rate (correlation: -1.000)
