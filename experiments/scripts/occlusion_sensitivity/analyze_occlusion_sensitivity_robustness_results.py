#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from collections import defaultdict
from typing import Dict

# 设置matplotlib和seaborn样式 (Keep existing style settings)
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

class OcclusionSensitivityRobustnessAnalyzer:
    """分析Occlusion Sensitivity鲁棒性实验结果的类"""

    def __init__(self, results_path: str):
        """初始化分析器
        Args:
            results_path: 主结果文件的路径
        """
        self.results_path = results_path
        self.results = self._load_results()
        # Use metric names consistent with the results JSON produced by test script
        self.metrics = [
            'similarity', 'consistency', 'localization',
            'prediction_change', 'confidence_diff', 'kl_divergence',
            'top5_distance', 'corruption_error', 'stability'
        ]
        # Define display names similar to IG script
        self.metric_names = {
            'similarity': 'Cosine Similarity',
            'consistency': 'Mutual Information',
            'localization': 'IoU Score',
            'prediction_change': 'Mean Flip Rate',
            'confidence_diff': 'Confidence Difference',
            'kl_divergence': 'KL Divergence',
            'top5_distance': 'Mean Top-5 Distance',
            'corruption_error': 'Mean Corruption Error',
            'stability': 'Stability Score'
        }
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression' # Match occlusion script's corruption name
        ]

    def _load_results(self) -> Dict:
        """加载主结果文件 (Adapted from old load_results)"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.results_path}")

    def _create_dataframe(self) -> pd.DataFrame:
        """将结果转换为pandas DataFrame格式 (Adapted from old extract_metrics)"""
        data = []
        image_count = 0
        max_images = None # Set to a number for debugging/testing if needed

        for image_path, image_results in self.results.items():
            if max_images is not None and image_count >= max_images:
                break
            for corruption_type, corruption_data in image_results.items():
                # Ensure corruption type from JSON matches expected list
                if corruption_type not in self.corruption_types:
                     # Handle potential mismatch like 'jpeg' vs 'jpeg_compression'
                     if corruption_type == 'jpeg' and 'jpeg_compression' in self.corruption_types:
                         corruption_type = 'jpeg_compression'
                     else:
                         print(f"Warning: Skipping unexpected corruption type '{corruption_type}' in results.")
                         continue

                if 'results' in corruption_data:
                    for result in corruption_data['results']:
                        if 'severity' not in result:
                            print(f"Warning: Skipping result missing 'severity' for {image_path} / {corruption_type}")
                            continue
                        row = {
                            'image': image_path,
                            'corruption_type': corruption_type,
                            'severity': result['severity'],
                        }
                        # 添加所有指标
                        valid_metric_found = False
                        for metric in self.metrics:
                            if metric in result:
                                row[metric] = result.get(metric, None)
                                valid_metric_found = True
                            else:
                                row[metric] = None # Ensure column exists even if metric is missing
                        # 添加是否预测正确的标志 (using prediction_change as proxy)
                        row['correct_prediction'] = 1 if result.get('prediction_change', 1) == 0 else 0

                        # Only add row if it contains at least one valid metric value and severity
                        if valid_metric_found:
                            data.append(row)
                else:
                    print(f"Warning: 'results' key missing for {image_path} / {corruption_type}")
            image_count += 1

        if not data:
             print("Warning: No valid data loaded into DataFrame. Check results file structure and content.")
             # Return empty DataFrame with expected columns to avoid downstream errors
             cols = ['image', 'corruption_type', 'severity', 'correct_prediction'] + self.metrics
             return pd.DataFrame(columns=cols)

        return pd.DataFrame(data)

    def plot_metric_heatmaps(self, output_dir: str, model_type: str = "standard"):
        """为每个指标生成热图 (Style adapted from IG script)"""
        os.makedirs(output_dir, exist_ok=True)
        df = self._create_dataframe()

        if df.empty or 'severity' not in df.columns:
            print("Skipping heatmap generation due to empty or invalid DataFrame (missing 'severity').")
            return

        # 设置水印标记
        watermark = "S" if model_type.lower() == "standard" else "R"

        # 打印当前使用的水印标记，用于调试
        print(f"Using watermark: {watermark} for model_type: {model_type}")

        # 为每个指标生成热图
        for metric in self.metrics:
            if metric not in df.columns:
                print(f"Skipping heatmap for metric '{metric}' as it's not in the DataFrame.")
                continue

            # 计算每个噪声类型和严重程度的平均值
            heatmap_data = []
            severities_found = sorted(df['severity'].unique())
            if not severities_found:
                print(f"Skipping heatmap for metric '{metric}' as no severity levels were found.")
                continue

            for severity in severities_found: # Use actual severities found in data
                row = []
                # Ensure we filter the original DataFrame 'df' here
                severity_df = df[df['severity'] == severity]
                for corruption in self.corruption_types:
                    # Filter the severity-specific DataFrame 'severity_df'
                    corruption_df = severity_df[severity_df['corruption_type'] == corruption]
                    if not corruption_df.empty and not corruption_df[metric].isnull().all():
                        avg_value = corruption_df[metric].mean()
                        row.append(avg_value)
                    else:
                        row.append(np.nan) # Use NaN if no data for this combo
                heatmap_data.append(row)

            if not heatmap_data or all(all(np.isnan(x) for x in r) for r in heatmap_data):
                print(f"Skipping heatmap for metric '{metric}' as no valid data points were found.")
                continue

            # 创建热图数据框架 (Index=Severity, Columns=Corruption Type)
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=[f"Severity {s}" for s in severities_found], # Use actual severities
                columns=self.corruption_types
            )

            # 设置热图参数 (Matching IG script)
            plt.figure(figsize=(12, 6)) # Match IG size

            # 绘制热图 (Use coolwarm, match other IG params)
            sns.heatmap(
                heatmap_df,
                annot=True,
                cmap='coolwarm', # Standardize cmap
                fmt='.3f',
                linewidths=.5,
                annot_kws={"size": 8},
                cbar_kws={'label': self.metric_names.get(metric, metric), 'shrink': 0.5} # Use display name
            )

            # 添加清晰的方法标识 (Match IG script title format)
            metric_title = self.metric_names.get(metric, metric.replace("_", " ").title())
            plt.title(f"Occlusion Sensitivity: {metric_title} by Corruption Type and Severity", fontsize=14, fontweight='bold')

            # 添加水印标识模型类型 (Match IG script)
            plt.text(0.99, 0.01, watermark, transform=plt.gca().transAxes,
                     fontsize=20, color='white', fontweight='bold',
                     ha='right', va='bottom',
                     path_effects=[
                         path_effects.withStroke(linewidth=3, foreground='black')
                     ])

            plt.xticks(rotation=45, ha='right', fontsize=8) # Match IG ticks
            plt.yticks(fontsize=10) # Match IG ticks
            plt.ylabel("Severity Level") # Correct label since severity is index
            plt.xlabel("Corruption Type") # Correct label since corruption is columns

            # 调整布局 (Match IG script)
            plt.tight_layout()

            # 保存为PNG (Match IG script)
            plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), bbox_inches='tight', dpi=300)
            plt.close()

    def generate_report_table(self, output_path: str, severity_level: int = 3):
        """生成包含准确率和指标的报告表格 (Logic adapted from IG script)"""
        df = self._create_dataframe()

        if df.empty or 'severity' not in df.columns:
            print(f"Skipping report generation due to empty or invalid DataFrame (missing 'severity').")
            # Create empty report file
            with open(output_path, 'w') as f:
                 f.write('# Occlusion Sensitivity Robustness Analysis Report\n\n')
                 f.write(f'## Results by Corruption Type (Severity Level {severity_level})\n\n')
                 f.write('*No data available to generate report.*\n')
            return

        # 筛选特定严重程度的数据
        severity_df = df[df['severity'] == severity_level].copy() # Use .copy() to avoid SettingWithCopyWarning

        if severity_df.empty:
             print(f"No data found for severity level {severity_level}. Report will be empty for this level.")
             # Create empty report file
             with open(output_path, 'w') as f:
                 f.write('# Occlusion Sensitivity Robustness Analysis Report\n\n')
                 f.write(f'## Results by Corruption Type (Severity Level {severity_level})\n\n')
                 f.write(f'*No data found for severity level {severity_level}.*\n')
             return

        # 准备表格数据
        table_data = []
        for corruption in self.corruption_types:
            corruption_df = severity_df[severity_df['corruption_type'] == corruption]
            if not corruption_df.empty:
                row = {'Corruption': corruption}

                # 添加准确率 (确保 'correct_prediction' 列存在)
                if 'correct_prediction' in corruption_df.columns:
                    row['Accuracy'] = corruption_df['correct_prediction'].mean()
                else:
                    row['Accuracy'] = np.nan # Or handle as needed

                # 添加每个指标的平均值
                for metric in self.metrics:
                    col_name = self.metric_names.get(metric, metric)
                    if metric in corruption_df.columns and not corruption_df[metric].isnull().all():
                         row[col_name] = corruption_df[metric].mean()
                    else:
                         row[col_name] = np.nan # Metric missing or all NaN for this group

                table_data.append(row)

        if not table_data:
             print(f"No table data generated for severity level {severity_level}.")
             # Create empty report file (redundant with check above, but safe)
             with open(output_path, 'w') as f:
                 f.write('# Occlusion Sensitivity Robustness Analysis Report\n\n')
                 f.write(f'## Results by Corruption Type (Severity Level {severity_level})\n\n')
                 f.write(f'*No data found for severity level {severity_level}.*\n')
             return

        # 创建DataFrame并按Accuracy降序排序
        table_df = pd.DataFrame(table_data)
        # Handle cases where Accuracy might be NaN if 'correct_prediction' was missing
        if 'Accuracy' in table_df.columns:
            table_df = table_df.sort_values(by='Accuracy', ascending=False, na_position='last')
        else:
            print("Warning: 'Accuracy' column missing in table_df for sorting.")


        # 生成Markdown表格 (Matching IG script format)
        with open(output_path, 'w') as f:
            f.write('# Occlusion Sensitivity Robustness Analysis Report\n\n') # Update title
            f.write(f'## Results by Corruption Type (Severity Level {severity_level})\n\n')
            f.write('*表格按照准确率从高到低排序*\n\n')

            # 添加表格头
            header = "| Corruption Type | Accuracy |"
            separator = "|---------------|----------|"
            display_metrics = [self.metric_names.get(m, m) for m in self.metrics] # Get display names
            for metric_name in display_metrics:
                 # Ensure metric name is in table_df columns before adding
                 if metric_name in table_df.columns:
                    header += f" {metric_name} |"
                    separator += "------------|"
                 else:
                     print(f"Warning: Metric display name '{metric_name}' not found in table_df columns for header.")

            f.write(header + "\n")
            f.write(separator + "\n")

            # 添加表格数据
            for _, row in table_df.iterrows():
                acc_str = f"{row.get('Accuracy', np.nan):.3f}" if pd.notna(row.get('Accuracy', np.nan)) else "N/A"
                line = f"| {row['Corruption']} | {acc_str} |"
                for metric_name in display_metrics:
                    if metric_name in row:
                        val_str = f"{row[metric_name]:.3f}" if pd.notna(row[metric_name]) else "N/A"
                        line += f" {val_str} |"
                    else:
                        line += f" N/A |" # Handle missing metric in row
                f.write(line + "\n")

            # --- Start: Optional sections from IG script ---
            # Check if required columns exist before attempting analysis
            analysis_possible = 'Accuracy' in table_df.columns and any(m in table_df.columns for m in display_metrics)

            if analysis_possible:
                # 添加准确率与指标相关性分析
                f.write("\n## Correlation Between Accuracy and Metrics\n\n")
                f.write("This section analyzes the correlation between model accuracy and explanation metrics across different corruption types.\n\n")

                # 计算相关性 (Handle potential NaNs)
                correlations = {}
                valid_metrics_for_corr = [m for m in display_metrics if m in table_df.columns and table_df[m].notna().any()]

                if table_df['Accuracy'].notna().any():
                    for metric_name in valid_metrics_for_corr:
                        # Calculate correlation only on non-NaN pairs
                         corr_df = table_df[['Accuracy', metric_name]].dropna()
                         if len(corr_df) > 1: # Need at least 2 points for correlation
                             correlation = corr_df['Accuracy'].corr(corr_df[metric_name])
                             correlations[metric_name] = correlation
                         else:
                             correlations[metric_name] = np.nan
                else:
                     print("Skipping correlation analysis as 'Accuracy' column has no valid values.")


                if correlations:
                    # 按相关性绝对值排序 (Handle NaNs in sorting)
                    sorted_correlations = sorted(
                        [(k, v) for k, v in correlations.items() if pd.notna(v)],
                        key=lambda x: abs(x[1]), reverse=True
                    )

                    # 添加相关性表格
                    f.write("| Metric | Correlation with Accuracy |\n")
                    f.write("|--------|----------------------------|\n")
                    for metric_name, corr_value in sorted_correlations:
                        direction = "positive" if corr_value >= 0 else "negative"
                        strength = "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.4 else "weak"
                        f.write(f"| {metric_name} | {corr_value:.3f} ({strength} {direction}) |\n")
                    # Report metrics with NaN correlation if any
                    nan_corr_metrics = [k for k, v in correlations.items() if pd.isna(v)]
                    if nan_corr_metrics:
                         f.write(f"| *({', '.join(nan_corr_metrics)})* | *Not Available (Insufficient Data)* |\n")

                else:
                     f.write("*Correlation analysis could not be performed (insufficient data).*\n")


                # 添加关于各种腐蚀类型的影响分析
                f.write("\n## Analysis of Corruption Effects\n\n")

                # Get most/least robust based on Accuracy (handle NaNs)
                table_df_valid_acc = table_df.dropna(subset=['Accuracy'])
                if not table_df_valid_acc.empty:
                    most_robust = table_df_valid_acc.iloc[0] # Already sorted
                    least_robust = table_df_valid_acc.iloc[-1]

                    f.write(f"### Most and Least Robust Corruptions (based on Accuracy)\n\n")
                    f.write(f"- **Most Robust**: {most_robust['Corruption']} (Accuracy: {most_robust['Accuracy']:.3f})\n")
                    f.write(f"- **Least Robust**: {least_robust['Corruption']} (Accuracy: {least_robust['Accuracy']:.3f})\n\n")
                else:
                     f.write("*(Could not determine most/least robust corruptions due to missing accuracy data)*\n\n")

                # 分类腐蚀类型 (Use full list, check if present in table_df)
                noise_types = ['gaussian_noise', 'shot_noise', 'impulse_noise']
                blur_types = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
                weather_types = ['snow', 'frost', 'fog']
                digital_types = ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'] # Use correct name

                categories = {'Noise': noise_types, 'Blur': blur_types, 'Weather': weather_types, 'Digital': digital_types}
                avg_acc = {}

                for category, types in categories.items():
                     # Filter table_df for relevant corruption types present in the table
                     cat_df = table_df[table_df['Corruption'].isin(types)].dropna(subset=['Accuracy'])
                     if not cat_df.empty:
                         avg_acc[category] = cat_df['Accuracy'].mean()

                if avg_acc:
                    f.write("### Corruption Category Analysis (based on Average Accuracy)\n\n")
                    # f.write("Average accuracy by corruption category:\n\n") # Redundant title
                    for category, acc in sorted(avg_acc.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- **{category}**: {acc:.3f}\n")
                else:
                     f.write("*(Corruption category analysis could not be performed due to missing accuracy data)*\n")


                # 添加关于解释方法在不同腐蚀下的表现分析
                f.write("\n## Explanation Robustness Analysis\n\n")

                # Analyze explanation-related metrics
                explanation_metric_keys = ['similarity', 'consistency', 'localization', 'stability'] # Use internal keys
                explanation_metric_names_present = [self.metric_names.get(m) for m in explanation_metric_keys if self.metric_names.get(m) in table_df.columns]

                if explanation_metric_names_present:
                    explanation_robustness = {}
                    for corruption in table_df['Corruption'].values: # Iterate only over corruptions present
                        row = table_df[table_df['Corruption'] == corruption].iloc[0]
                        # Calculate mean of *available* explanation metrics for this row (handle NaNs)
                        metric_values = [row[name] for name in explanation_metric_names_present if pd.notna(row.get(name))]
                        if metric_values:
                            metrics_avg = np.mean(metric_values)
                            explanation_robustness[corruption] = metrics_avg

                    if explanation_robustness:
                        # Find min/max based on the calculated average explanation metric score
                        most_robust_exp = max(explanation_robustness.items(), key=lambda x: x[1])
                        least_robust_exp = min(explanation_robustness.items(), key=lambda x: x[1])

                        f.write("### Explanation Robustness by Corruption Type (based on avg. similarity/consistency/IoU/stability)\n\n")
                        f.write(f"- **Most Robust Explanations**: {most_robust_exp[0]} (Avg. metrics: {most_robust_exp[1]:.3f})\n")
                        f.write(f"- **Least Robust Explanations**: {least_robust_exp[0]} (Avg. metrics: {least_robust_exp[1]:.3f})\n\n")
                    else:
                         f.write("*(Explanation robustness analysis could not be performed due to missing relevant metric data)*\n\n")
                else:
                    f.write("*(Explanation robustness analysis requires 'Cosine Similarity', 'Mutual Information', 'IoU Score', or 'Stability Score' data)*\n\n")

                # 总结发现
                f.write("\n## Summary\n\n")
                f.write(f"This analysis evaluated the robustness of Occlusion Sensitivity explanations across 15 different corruption types at severity level {severity_level}. ") # Update method name
                f.write("The results show how different corruptions affect both model predictions and explanation quality.\n\n")

                # 根据实际结果给出结论
                f.write("Key findings:\n\n")
                if 'most_robust' in locals(): # Check if analysis was possible
                     f.write(f"1. Model accuracy at this severity is highest for **{most_robust['Corruption']}** and lowest for **{least_robust['Corruption']}**.\n")
                else:
                     f.write("1. Model accuracy variation could not be determined.\n")

                if 'most_robust_exp' in locals():
                     f.write(f"2. Explanation robustness (avg. metrics) is highest for **{most_robust_exp[0]}** and lowest for **{least_robust_exp[0]}**.\n")
                else:
                     f.write("2. Explanation robustness variation could not be determined.\n")

                if 'sorted_correlations' in locals() and sorted_correlations:
                     most_correlated = sorted_correlations[0]
                     f.write(f"3. The metric most correlated with accuracy is **{most_correlated[0]}** (correlation: {most_correlated[1]:.3f}).\n")
                else:
                     f.write("3. Correlation between metrics and accuracy could not be determined.\n")

            else: # Analysis not possible
                f.write("\n*Further analysis (correlation, effects, summary) could not be performed due to missing Accuracy or metric data.*\n")
            # --- End: Optional sections ---


    def run_analysis(self, figures_dir: str, report_path: str, severity_level: int = 3, model_type: str = "standard"):
        """运行分析流程"""
        # 打印开始信息 (Add some logging)
        print("-" * 50)
        print(f"Starting Occlusion Sensitivity Analysis for Model: {model_type}")
        print(f"Results File: {self.results_path}")
        print(f"Figures Dir: {figures_dir}")
        print(f"Report Path: {report_path}")
        print(f"Report Severity Level: {severity_level}")
        print("-" * 50)

        # 生成热图 (Pass model_type)
        self.plot_metric_heatmaps(figures_dir, model_type)

        # 生成报告表格 (Pass severity_level)
        self.generate_report_table(report_path, severity_level)

        # 打印完成信息
        print("-" * 50)
        print(f"Analysis completed.")
        print(f"Heatmaps saved in: {figures_dir}")
        print(f"Report for severity {severity_level} saved as: {report_path}")
        print("-" * 50)


def main():
    """主函数 (Adapted from IG script)"""
    parser = argparse.ArgumentParser(description='Analyze Occlusion Sensitivity robustness test results')
    # Use arguments consistent with other scripts
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to the Occlusion Sensitivity robustness results JSON file')
    parser.add_argument('--figures_dir', type=str, required=True,
                       help='Directory to save the generated figures (heatmaps)')
    parser.add_argument('--report_path', type=str, required=True,
                       help='Path to save the analysis report (Markdown format)')
    parser.add_argument('--severity_level', type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help='Severity level to use for the report table (1-5)')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'robust'],
                       help='Model type (standard or robust) for proper labeling')
    # Add optional limit for debugging
    parser.add_argument('--limit_images', type=int, default=None,
                       help='Limit analysis to the first N images (optional, for testing)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.figures_dir, exist_ok=True)
    # Ensure report directory exists
    report_dir = os.path.dirname(args.report_path)
    if report_dir: # Only create if path includes a directory
        os.makedirs(report_dir, exist_ok=True)

    # 运行分析 using the new class structure
    analyzer = OcclusionSensitivityRobustnessAnalyzer(args.results_path)
    # If limit_images is used, it's handled inside _create_dataframe now
    analyzer.run_analysis(args.figures_dir, args.report_path, args.severity_level, args.model_type)

if __name__ == "__main__":
    main() 