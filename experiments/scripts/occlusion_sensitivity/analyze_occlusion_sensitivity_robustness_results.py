#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_results(results_file):
    """加载测试结果文件"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def extract_metrics(results):
    """从结果中提取指标"""
    # 初始化数据结构来存储指标
    metrics_data = defaultdict(lambda: defaultdict(list))
    stability_data = defaultdict(lambda: defaultdict(list))
    
    # 遍历所有图像
    for image_path, image_results in results.items():
        # 遍历所有腐蚀类型
        for corruption_type, corruption_results in image_results.items():
            # 遍历所有严重程度
            for result in corruption_results["results"]:
                severity = result["severity"]
                
                # 提取稳定性指标（如果存在）
                if "stability" in result:
                    stability_data[corruption_type][severity].append(result["stability"])
                
                # 提取其他指标
                for metric in [
                    "cosine_similarity", "mutual_information", "iou", 
                    "prediction_change", "confidence_diff", "kl_divergence",
                    "top5_distance", "corruption_error"
                ]:
                    if metric in result:
                        metrics_data[metric][f"{corruption_type}_{severity}"].append(result[metric])
    
    return metrics_data, stability_data

def calculate_statistics(metrics_data, stability_data):
    """计算每个指标的统计数据"""
    stats = {}
    
    # 处理常规指标
    for metric, corruption_data in metrics_data.items():
        stats[metric] = {}
        for corruption_severity, values in corruption_data.items():
            if values:  # 确保不是空列表
                stats[metric][corruption_severity] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
    
    # 处理稳定性指标
    stats["stability"] = {}
    for corruption_type, severity_data in stability_data.items():
        for severity, values in severity_data.items():
            if values:  # 确保不是空列表
                key = f"{corruption_type}_{severity}"
                stats["stability"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
    
    return stats

def generate_heatmaps(stats, output_dir):
    """生成热图来可视化指标随腐蚀类型和严重程度的变化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义腐蚀类型和指标
    corruption_types = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", 
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", 
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]
    
    metrics = [
        "cosine_similarity", "mutual_information", "iou", "prediction_change", 
        "confidence_diff", "kl_divergence", "top5_distance", "corruption_error",
        "stability"
    ]
    
    # 创建每个指标的热图
    for metric in metrics:
        if metric not in stats:
            continue
            
        # 创建数据框
        data = []
        for corruption_type in corruption_types:
            row = []
            for severity in range(1, 6):
                key = f"{corruption_type}_{severity}"
                if key in stats[metric]:
                    row.append(stats[metric][key]["mean"])
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, index=corruption_types, columns=[f"Severity {i}" for i in range(1, 6)])
        
        # 设置热图颜色
        if metric in ["cosine_similarity", "mutual_information", "iou"]:
            # 这些指标越高越好，使用红色表示低值
            cmap = "RdYlGn"
        else:
            # 这些指标越低越好，使用绿色表示低值
            cmap = "RdYlGn_r"
        
        # 创建热图
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(df, annot=True, cmap=cmap, fmt=".3f", linewidths=.5)
        
        # 设置标题和标签
        metric_name = metric.replace("_", " ").title()
        plt.title(f"Occlusion Sensitivity: {metric_name} vs Corruption Type and Severity")
        plt.ylabel("Corruption Type")
        plt.xlabel("Severity Level")
        
        # 保存热图
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{metric}.png"), dpi=300)
        plt.close()

def generate_severity_plots(stats, output_dir):
    """生成折线图显示指标如何随严重程度变化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义腐蚀类型和指标
    corruption_types = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", 
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", 
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]
    
    metrics = [
        "cosine_similarity", "mutual_information", "iou", "prediction_change", 
        "confidence_diff", "kl_divergence", "top5_distance", "corruption_error",
        "stability"
    ]
    
    # 为每个指标创建折线图
    for metric in metrics:
        if metric not in stats:
            continue
            
        plt.figure(figsize=(12, 8))
        
        for corruption_type in corruption_types:
            severities = []
            values = []
            
            for severity in range(1, 6):
                key = f"{corruption_type}_{severity}"
                if key in stats[metric]:
                    severities.append(severity)
                    values.append(stats[metric][key]["mean"])
            
            if severities:
                plt.plot(severities, values, marker='o', label=corruption_type)
        
        # 设置图表
        metric_name = metric.replace("_", " ").title()
        plt.title(f"Occlusion Sensitivity: {metric_name} vs Severity Level")
        plt.xlabel("Severity Level")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"severity_plot_{metric}.png"), dpi=300)
        plt.close()

def generate_summary_report(stats, output_file):
    """生成摘要报告"""
    with open(output_file, 'w') as f:
        f.write("# Occlusion Sensitivity Robustness Analysis Report\n\n")
        
        # 计算所有腐蚀类型和严重程度的平均值
        f.write("## Overall Metrics Summary\n\n")
        f.write("| Metric | Mean | Standard Deviation | Median | Min | Max |\n")
        f.write("|--------|------|--------------------|--------|-----|-----|\n")
        
        for metric in [
            "cosine_similarity", "mutual_information", "iou", "prediction_change", 
            "confidence_diff", "kl_divergence", "top5_distance", "corruption_error",
            "stability"
        ]:
            if metric not in stats:
                continue
                
            all_means = [v["mean"] for v in stats[metric].values()]
            if all_means:
                f.write(f"| {metric.replace('_', ' ').title()} | {np.mean(all_means):.4f} | {np.std(all_means):.4f} | {np.median(all_means):.4f} | {np.min(all_means):.4f} | {np.max(all_means):.4f} |\n")
        
        # 按腐蚀类型分析
        f.write("\n## Analysis by Corruption Type\n\n")
        
        corruption_types = [
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", 
            "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", 
            "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
        ]
        
        for corruption_type in corruption_types:
            f.write(f"\n### {corruption_type.replace('_', ' ').title()}\n\n")
            f.write("| Metric | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |\n")
            f.write("|--------|------------|------------|------------|------------|------------|\n")
            
            for metric in [
                "cosine_similarity", "mutual_information", "iou", "prediction_change", 
                "confidence_diff", "kl_divergence", "top5_distance", "corruption_error",
                "stability"
            ]:
                if metric not in stats:
                    continue
                    
                f.write(f"| {metric.replace('_', ' ').title()} |")
                
                for severity in range(1, 6):
                    key = f"{corruption_type}_{severity}"
                    if key in stats[metric]:
                        f.write(f" {stats[metric][key]['mean']:.4f} |")
                    else:
                        f.write(" N/A |")
                
                f.write("\n")
        
        # 可视化引用
        f.write("\n## Visualizations\n\n")
        f.write("请参阅输出目录中的热图和折线图，以获取更详细的指标可视化。\n")

def main():
    parser = argparse.ArgumentParser(description="分析Occlusion Sensitivity鲁棒性测试结果")
    parser.add_argument('--results_file', type=str, required=True, help="测试结果文件的路径")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录的路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    print(f"加载结果文件: {args.results_file}")
    results = load_results(args.results_file)
    
    # 提取指标
    print("提取指标...")
    metrics_data, stability_data = extract_metrics(results)
    
    # 计算统计数据
    print("计算统计数据...")
    stats = calculate_statistics(metrics_data, stability_data)
    
    # 生成热图
    print("生成热图...")
    generate_heatmaps(stats, args.output_dir)
    
    # 生成严重程度折线图
    print("生成严重程度折线图...")
    generate_severity_plots(stats, args.output_dir)
    
    # 生成摘要报告
    print("生成摘要报告...")
    generate_summary_report(stats, os.path.join(args.output_dir, "summary_report.md"))
    
    print(f"分析完成! 结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 