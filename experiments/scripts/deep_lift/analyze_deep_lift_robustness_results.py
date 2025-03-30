import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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

class DeepLIFTRobustnessAnalyzer:
    """分析DeepLIFT鲁棒性实验结果的类"""
    
    def __init__(self, results_path: str):
        """初始化分析器
        Args:
            results_path: 主结果文件的路径 (deep_lift_robustness_results.json)
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

    def plot_metric_heatmaps(self, output_dir: str, model_type: str = "standard"):
        """为每个指标生成热图，横轴为噪声类型，纵轴为严重程度
        
        Args:
            output_dir: 输出目录
            model_type: 模型类型，'standard'或'robust'
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self._create_dataframe()
        
        # 设置水印标记
        watermark = "S" if model_type.lower() == "standard" else "R"
        
        # 为每个指标生成热图
        for metric in self.metrics:
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
            plt.title(f"DeepLIFT: {metric_title} by Corruption Type and Severity", fontsize=14, fontweight='bold')
            
            # 添加水印标识模型类型
            plt.text(0.99, 0.01, watermark, transform=plt.gca().transAxes, 
                     fontsize=20, color='white', fontweight='bold',
                     ha='right', va='bottom', 
                     path_effects=[
                         path_effects.withStroke(linewidth=3, foreground='black')
                     ])
            
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
            f.write('# DeepLIFT Robustness Analysis Report\n\n')
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
                
            # 添加准确率与指标相关性分析
            f.write("\n## Correlation Between Accuracy and Metrics\n\n")
            f.write("This section analyzes the correlation between model accuracy and explanation metrics across different corruption types.\n\n")
            
            # 计算相关性
            correlations = {}
            for metric in self.metrics:
                col_name = self.metric_names[metric]
                correlation = table_df['Accuracy'].corr(table_df[col_name])
                correlations[col_name] = correlation
            
            # 按相关性绝对值排序
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # 添加相关性表格
            f.write("| Metric | Correlation with Accuracy |\n")
            f.write("|--------|----------------------------|\n")
            for metric_name, corr_value in sorted_correlations:
                direction = "positive" if corr_value >= 0 else "negative"
                strength = "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.4 else "weak"
                f.write(f"| {metric_name} | {corr_value:.3f} ({strength} {direction}) |\n")
            
            # 添加关于各种腐蚀类型的影响分析
            f.write("\n## Analysis of Corruption Effects\n\n")
            
            # 获取最低和最高准确率的腐蚀类型
            most_robust = table_df.iloc[0]
            least_robust = table_df.iloc[-1]
            
            f.write(f"### Most and Least Robust Corruptions\n\n")
            f.write(f"- **Most Robust**: {most_robust['Corruption']} (Accuracy: {most_robust['Accuracy']:.3f})\n")
            f.write(f"- **Least Robust**: {least_robust['Corruption']} (Accuracy: {least_robust['Accuracy']:.3f})\n\n")
            
            # 分类腐蚀类型
            noise_types = ['gaussian_noise', 'shot_noise', 'impulse_noise']
            blur_types = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
            weather_types = ['snow', 'frost', 'fog']
            digital_types = ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg']
            
            # 计算各类腐蚀的平均准确率
            avg_acc = {}
            if any(c in table_df['Corruption'].values for c in noise_types):
                noise_df = table_df[table_df['Corruption'].isin(noise_types)]
                if not noise_df.empty:
                    avg_acc['Noise'] = noise_df['Accuracy'].mean()
            
            if any(c in table_df['Corruption'].values for c in blur_types):
                blur_df = table_df[table_df['Corruption'].isin(blur_types)]
                if not blur_df.empty:
                    avg_acc['Blur'] = blur_df['Accuracy'].mean()
            
            if any(c in table_df['Corruption'].values for c in weather_types):
                weather_df = table_df[table_df['Corruption'].isin(weather_types)]
                if not weather_df.empty:
                    avg_acc['Weather'] = weather_df['Accuracy'].mean()
            
            if any(c in table_df['Corruption'].values for c in digital_types):
                digital_df = table_df[table_df['Corruption'].isin(digital_types)]
                if not digital_df.empty:
                    avg_acc['Digital'] = digital_df['Accuracy'].mean()
            
            # 添加腐蚀类别分析
            if avg_acc:
                f.write("### Corruption Category Analysis\n\n")
                f.write("Average accuracy by corruption category:\n\n")
                for category, acc in sorted(avg_acc.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{category}**: {acc:.3f}\n")
            
            # 添加关于解释方法在不同腐蚀下的表现分析
            f.write("\n## Explanation Robustness Analysis\n\n")
            
            # 分析解释相关指标
            explanation_metrics = ['similarity', 'consistency', 'localization', 'stability']
            explanation_metric_names = [self.metric_names[m] for m in explanation_metrics if m in self.metrics]
            
            # 找出解释最鲁棒和最不鲁棒的腐蚀类型
            explanation_robustness = {}
            for corruption in self.corruption_types:
                if corruption in table_df['Corruption'].values:
                    row = table_df[table_df['Corruption'] == corruption].iloc[0]
                    # 计算解释指标的平均值
                    metrics_avg = np.mean([row[self.metric_names[m]] for m in explanation_metrics if m in self.metrics])
                    explanation_robustness[corruption] = metrics_avg
            
            if explanation_robustness:
                most_robust_exp = max(explanation_robustness.items(), key=lambda x: x[1])
                least_robust_exp = min(explanation_robustness.items(), key=lambda x: x[1])
                
                f.write("### Explanation Robustness by Corruption Type\n\n")
                f.write(f"- **Most Robust Explanations**: {most_robust_exp[0]} (Avg. metrics: {most_robust_exp[1]:.3f})\n")
                f.write(f"- **Least Robust Explanations**: {least_robust_exp[0]} (Avg. metrics: {least_robust_exp[1]:.3f})\n\n")
            
            # 总结发现
            f.write("\n## Summary\n\n")
            f.write("This analysis evaluated the robustness of DeepLIFT explanations across 15 different corruption types. ")
            f.write("The results show how different corruptions affect both model predictions and explanation quality.\n\n")
            
            # 根据实际结果给出结论，实际应用中可能需要修改
            f.write("Key findings:\n\n")
            f.write("1. Model accuracy varies significantly across corruption types\n")
            if explanation_robustness:
                f.write(f"2. Explanation robustness is highest for {most_robust_exp[0]} corruptions\n")
                f.write(f"3. Explanation robustness is lowest for {least_robust_exp[0]} corruptions\n")
            if correlations:
                most_correlated = sorted_correlations[0]
                f.write(f"4. The metric most correlated with accuracy is {most_correlated[0]} (correlation: {most_correlated[1]:.3f})\n")

    def run_analysis(self, figures_dir: str, report_path: str, severity_level: int = 3, model_type: str = "standard"):
        """运行分析流程
        
        Args:
            figures_dir: 图表输出目录
            report_path: 报告输出路径
            severity_level: 报告中使用的严重程度级别
            model_type: 模型类型，'standard'或'robust'
        """
        # 生成热图
        self.plot_metric_heatmaps(figures_dir, model_type)
        
        # 生成报告表格
        self.generate_report_table(report_path, severity_level)
        
        # 打印完成信息
        print(f"Analysis completed. Heatmaps saved in: {figures_dir}")
        print(f"Report saved as: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Analyze DeepLIFT robustness test results')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to the DeepLIFT robustness results JSON file')
    parser.add_argument('--figures_dir', type=str, required=True,
                       help='Directory to save the generated figures')
    parser.add_argument('--report_path', type=str, required=True,
                       help='Path to save the analysis report')
    parser.add_argument('--severity_level', type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help='Severity level to use for the report (1-5)')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'robust'],
                       help='Model type (standard or robust) for proper labeling')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    
    # 运行分析
    analyzer = DeepLIFTRobustnessAnalyzer(args.results_path)
    analyzer.run_analysis(args.figures_dir, args.report_path, args.severity_level, args.model_type)

if __name__ == "__main__":
    main() 