import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
import os
import argparse

# 设置matplotlib和seaborn样式
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

class GradCAMRobustnessAnalyzer:
    """分析GradCAM鲁棒性实验结果的类"""
    
    def __init__(self, results_path: str):
        """初始化分析器
    Args:
            results_path: 主结果文件的路径 (gradcam_robustness_results.json)
        """
        self.results_path = results_path
        self.results = self._load_results()
        self.metrics = [
            'similarity', 'consistency', 'localization',
            'prediction_change', 'confidence_diff', 'kl_divergence',
            'top5_distance', 'corruption_error', 'stability'
        ]
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
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'
        ]
        
    def _load_results(self) -> Dict:
        """加载主结果文件"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.results_path}")

    def _create_dataframe(self) -> pd.DataFrame:
        """将结果转换为pandas DataFrame格式"""
        data = []
        for image_path, image_results in self.results.items():
            for corruption_type, corruption_data in image_results.items():
                if corruption_type in self.corruption_types:
                    for result in corruption_data.get('results', []):
                        row = {
                            'image': image_path,
                            'corruption_type': corruption_type,
                            'severity': result['severity'],
                        }
                        # 添加所有指标
                        for metric in self.metrics:
                            row[metric] = result.get(metric, None)
                        # 添加是否预测正确的标志
                        row['correct_prediction'] = 1 if result.get('prediction_change', 1) == 0 else 0
                        data.append(row)
        return pd.DataFrame(data)

    def plot_metric_heatmaps(self, output_dir: str):
        """为每个指标生成热图，横轴为噪声类型，纵轴为严重程度
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self._create_dataframe()
        
        # 为每个指标生成热图
        for metric in self.metrics:
            # 如果是stability指标，使用专门的方法处理
            if metric == 'stability':
                self._plot_stability_chart(output_dir)
                continue
                
            # 计算每个噪声类型和严重程度的平均值
            heatmap_data = []
            for severity in range(1, 6):
                row = []
                severity_df = df[df['severity'] == severity]
                for corruption in self.corruption_types:
                    corruption_df = severity_df[severity_df['corruption_type'] == corruption]
                    if len(corruption_df) > 0:
                        avg_value = corruption_df[metric].mean()
                        row.append(avg_value)
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            # 创建热图数据框架
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=[f"Severity {s}" for s in range(1, 6)],
                columns=self.corruption_types
            )
            
            # 设置热图参数
            plt.figure(figsize=(12, 6))
            
            # 绘制热图
            sns.heatmap(
                heatmap_df,
                annot=True,
                cmap='coolwarm',
                fmt='.3f',
                linewidths=.5,
                annot_kws={"size": 8},
                cbar_kws={'label': metric, 'shrink': 0.5}
            )
            
            # 添加清晰的方法标识
            metric_title = self.metric_names[metric]
            plt.title(f"GradCAM: {metric_title} by Corruption Type and Severity", fontsize=14, fontweight='bold')
            
            # 添加水印标识
            plt.text(0.98, 0.02, "GC", transform=plt.gca().transAxes, 
                     fontsize=18, color='gray', alpha=0.3, 
                     ha='right', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=10)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存为PNG
            plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), bbox_inches='tight', dpi=300)
            plt.close()

    def generate_report_table(self, output_path: str, severity_level: int = 3):
        """生成包含准确率和九个指标的报告表格
        
        Args:
            output_path: 输出文件路径
            severity_level: 要报告的严重程度级别（1-5）
        """
        df = self._create_dataframe()
        
        # 筛选特定严重程度的数据
        severity_df = df[df['severity'] == severity_level]
        
        # 准备表格数据
        table_data = []
        for corruption in self.corruption_types:
            corruption_df = severity_df[severity_df['corruption_type'] == corruption]
            if len(corruption_df) > 0:
                row = {'Corruption': corruption}
                
                # 添加准确率
                row['Accuracy'] = corruption_df['correct_prediction'].mean()
                
                # 添加每个指标的平均值
                for metric in self.metrics:
                    row[self.metric_names[metric]] = corruption_df[metric].mean()
                
                table_data.append(row)
        
        # 创建DataFrame并按Accuracy降序排序
        table_df = pd.DataFrame(table_data)
        table_df = table_df.sort_values(by='Accuracy', ascending=False)
        
        # 生成Markdown表格
        with open(output_path, 'w') as f:
            f.write('# GradCAM Robustness Analysis Report\n\n')
            f.write(f'## Results by Corruption Type (Severity Level {severity_level})\n\n')
            f.write('*表格按照准确率从高到低排序*\n\n')
            
            # 添加表格头
            header = "| Corruption Type | Accuracy |"
            separator = "|---------------|----------|"
            for metric in self.metrics:
                header += f" {self.metric_names[metric]} |"
                separator += "------------|"
            f.write(header + "\n")
            f.write(separator + "\n")
            
            # 添加表格数据
            for _, row in table_df.iterrows():
                line = f"| {row['Corruption']} | {row['Accuracy']:.3f} |"
                for metric in self.metrics:
                    col_name = self.metric_names[metric]
                    line += f" {row[col_name]:.3f} |"
                f.write(line + "\n")

    def _plot_stability_chart(self, output_dir: str):
        """为稳定性指标生成热图
        
        Args:
            output_dir: 输出目录
        """
        # 收集稳定性数据 - 直接从主结果文件中获取
        stability_data = {}
        processed_images = set()
        
        # 按照corruption_type和severity收集
        for image_path, image_results in self.results.items():
            processed_images.add(image_path)
            for corruption_type, corruption_data in image_results.items():
                if corruption_type in self.corruption_types:
                    if corruption_type not in stability_data:
                        stability_data[corruption_type] = {1: [], 2: [], 3: [], 4: [], 5: []}
                    
                    # 从results数组中收集每个severity的stability值
                    for result in corruption_data.get('results', []):
                        severity = result.get('severity')
                        stability = result.get('stability')
                        if severity and stability is not None:
                            stability_data[corruption_type][severity].append(stability)
        
        # 检查是否有稳定性数据
        if not stability_data or all(not any(values) for values in stability_data.values()):
            print("警告: 未找到任何稳定性数据，跳过稳定性可视化。")
            return
            
        # 计算每个corruption_type每个severity的平均稳定性
        avg_stability = {}
        for corruption_type, severity_data in stability_data.items():
            avg_stability[corruption_type] = {}
            for severity, values in severity_data.items():
                if values:
                    avg_stability[corruption_type][severity] = np.mean(values)
                else:
                    avg_stability[corruption_type][severity] = np.nan
        
        # 打印调试信息 - 显示实际处理的图片数量
        print(f"已处理 {len(processed_images)} 张图片")
        print("稳定性数据计数 (每个腐蚀/严重程度):")
        stability_counts = {k: {s: len(v) for s, v in sv.items()} for k, sv in stability_data.items()}
        # 打印部分计数以避免过长输出
        count_summary = {k: stability_counts[k] for k in list(stability_counts.keys())[:3]} 
        print(count_summary, "...")
        
        # 生成热图 - 显示所有severity级别
        plt.figure(figsize=(12, 6))
        
        # 创建热图数据
        heatmap_data = []
        for severity in range(1, 6):
            row = []
            for corruption in self.corruption_types:
                value = avg_stability.get(corruption, {}).get(severity, np.nan)
                row.append(value)
            heatmap_data.append(row)
        
        # 创建热图数据框架
        heatmap_df = pd.DataFrame(
            heatmap_data, 
            index=[f"Severity {s}" for s in range(1, 6)],
            columns=self.corruption_types
        )
        
        # 绘制热图
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='coolwarm',  # 改为与其他指标相同的颜色映射
            fmt='.3f',
            linewidths=.5,
            annot_kws={"size": 8},
            cbar_kws={'label': 'stability', 'shrink': 0.5}
        )
        
        plt.title(f"{self.metric_names['stability']} by Corruption Type and Severity", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_heatmap.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def run_analysis(self, figures_dir: str, report_path: str, severity_level: int = 3):
        """运行分析流程
        
        Args:
            figures_dir: 图表输出目录
            report_path: 报告输出路径
            severity_level: 报告中使用的严重程度级别
        """
        # 生成热图
        self.plot_metric_heatmaps(figures_dir)
        
        # 生成报告表格
        self.generate_report_table(report_path, severity_level)
        
        # 生成稳定性热图 (现在由plot_metric_heatmaps统一处理，如果需要单独逻辑再取消注释)
        # self._plot_stability_chart(figures_dir) # 之前 plot_metric_heatmaps 会跳过 stability
        
        print(f"分析完成。热图保存在: {figures_dir}")
        print(f"报告保存为: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Analyze GradCAM robustness results.')
    parser.add_argument('--results_path', type=str, 
                        default='experiments/results/gradcam_robustness_results.json', 
                        help='Path to the GradCAM results JSON file.')
    parser.add_argument('--figures_dir', type=str, 
                        default='experiments/results/figures', 
                        help='Directory to save the heatmap figures.')
    parser.add_argument('--report_path', type=str, 
                        default='experiments/results/analysis_report.md', 
                        help='Path to save the analysis report.')
    parser.add_argument('--severity_level', type=int, default=3, 
                        help='Severity level to focus on in the report (1-5).')

    args = parser.parse_args()
    
    # 使用命令行参数创建分析器实例
    analyzer = GradCAMRobustnessAnalyzer(args.results_path)
    
    # 使用命令行参数运行分析
    analyzer.run_analysis(args.figures_dir, args.report_path, args.severity_level)

if __name__ == "__main__":
    main() 