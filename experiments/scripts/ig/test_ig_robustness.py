import torch
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
import cv2
import time
import argparse
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, map_coordinates
from skimage.filters import gaussian
from skimage import util as sk_util
from io import BytesIO
from sklearn.metrics import mutual_info_score
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# 添加RobustBench相关导入
try:
    from robustbench.utils import load_model
    ROBUSTBENCH_AVAILABLE = True
except ImportError:
    ROBUSTBENCH_AVAILABLE = False
    print("RobustBench not installed. Only standard models will be available.")

def clipped_zoom(img, zoom_factor):
    """Center-crop an image after zooming"""
    h, w = img.shape[:2]
    zh = int(np.round(h * zoom_factor))
    zw = int(np.round(w * zoom_factor))
    zh = max(zh, 1)
    zw = max(zw, 1)
    zoomed = zoom(img, [zoom_factor, zoom_factor, 1], order=1, mode='reflect')
    if zoom_factor > 1:
        trim_h = ((zoomed.shape[0] - h) // 2)
        trim_w = ((zoomed.shape[1] - w) // 2)
        zoomed = zoomed[trim_h:trim_h+h, trim_w:trim_w+w]
    elif zoom_factor < 1:
        pad_h = ((h - zoomed.shape[0]) // 2)
        pad_w = ((w - zoomed.shape[1]) // 2)
        out = np.zeros_like(img)
        out[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
        zoomed = out
    if zoomed.shape[:2] != (h, w):
        zoomed = cv2.resize(zoomed, (w, h))
    return zoomed

class IGRobustnessTest:
    """Integrated Gradients鲁棒性测试类"""
    
    def __init__(self, model_type="standard", device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 根据model_type选择不同的模型
        self.model_type = model_type
        if model_type == "standard":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("Loaded standard ResNet50 model")
        elif model_type == "robust" and ROBUSTBENCH_AVAILABLE:
            self.model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')
            print("Loaded RobustBench Salman2020Do_50_2 model")
        else:
            raise ValueError(f"Invalid model_type {model_type} or RobustBench not available")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 创建Integrated Gradients解释器
        self.ig = IntegratedGradients(self.model)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 图像反归一化转换，用于展示
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'
        ]
        self.alexnet_baseline = 0.7  # AlexNet在Tiny-ImageNet上的错误率

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """加载图像并转换为模型输入格式"""
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
        """应用图像腐蚀"""
        img = np.array(image) / 255.0
        np.random.seed(1)
        severity = float(severity) / 5.0

        if corruption_type == 'gaussian_noise':
            noise = np.random.normal(loc=0, scale=severity * 0.5, size=img.shape)
            corrupted = np.clip(img + noise, 0, 1)
        elif corruption_type == 'shot_noise':
            corrupted = np.random.poisson(img * 255.0 * (1-severity)) / 255.0
            corrupted = np.clip(corrupted, 0, 1)
        elif corruption_type == 'impulse_noise':
            corrupted = sk_util.random_noise(img, mode='s&p', amount=severity)
        elif corruption_type == 'defocus_blur':
            corrupted = gaussian(img, sigma=severity * 4, channel_axis=-1)
        elif corruption_type == 'glass_blur':
            kernel_size = int(severity * 10) * 2 + 1
            corrupted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif corruption_type == 'motion_blur':
            kernel_size = int(severity * 20)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            corrupted = cv2.filter2D(img, -1, kernel)
        elif corruption_type == 'zoom_blur':
            out = np.zeros_like(img)
            zoom_factors = [1-severity*0.1, 1-severity*0.05, 1, 1+severity*0.05, 1+severity*0.1]
            for zoom_factor in zoom_factors:
                zoomed = clipped_zoom(img, zoom_factor)
                out += zoomed
            corrupted = np.clip(out / len(zoom_factors), 0, 1)
        elif corruption_type == 'snow':
            snow_layer = np.random.normal(size=img.shape[:2], loc=0.5, scale=severity * 1.5) 
            snow_layer = np.clip(snow_layer, 0, 1)
            snow_layer = np.expand_dims(snow_layer, axis=2)
            corrupted = np.clip(img + snow_layer, 0, 1)
        elif corruption_type == 'frost':
            frost_layer = np.random.uniform(size=img.shape[:2]) * severity
            frost_layer = np.expand_dims(frost_layer, axis=2)
            corrupted = np.clip(img * (1 - frost_layer), 0, 1)
        elif corruption_type == 'fog':
            fog_layer = severity * np.ones_like(img)
            corrupted = np.clip(img * (1 - severity) + fog_layer * severity, 0, 1)
        elif corruption_type == 'brightness':
            corrupted = np.clip(img * (1 + severity), 0, 1)
        elif corruption_type == 'contrast':
            mean = np.mean(img, axis=(0,1), keepdims=True)
            corrupted = np.clip((img - mean) * (1 + severity) + mean, 0, 1)
        elif corruption_type == 'elastic_transform':
            shape = img.shape[:2]
            dx = np.random.uniform(-1, 1, shape) * severity * 30
            dy = np.random.uniform(-1, 1, shape) * severity * 30
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            corrupted = np.zeros_like(img)
            for c in range(3):
                corrupted[:,:,c] = np.reshape(map_coordinates(img[:,:,c], indices, order=1), shape)
        elif corruption_type == 'pixelate':
            h, w = img.shape[:2]
            size = max(int((1-severity) * min(h,w)), 1)
            corrupted = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
            corrupted = cv2.resize(corrupted, (w,h), interpolation=cv2.INTER_NEAREST)
        elif corruption_type == 'jpeg':
            quality = int((1-severity) * 100)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            corrupted = np.array(Image.open(buffer)) / 255.0
        else:
            return image

        corrupted = np.clip(corrupted * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted)

    def generate_ig(self, input_tensor: torch.Tensor) -> np.ndarray:
        """生成Integrated Gradients解释"""
        # 创建梯度的目标
        if self.model_type == "standard":
            # 对于标准ResNet50，我们使用预测的类别作为目标
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        else:
            # 对于健壮模型，同样使用预测的类别
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # 计算IG归因
        attributions = self.ig.attribute(
            input_tensor, 
            target=target_class, 
            n_steps=50  # 积分步数，可以根据需要调整
        )
        
        # 转换为numpy数组并计算绝对值之和(跨通道)
        attr_sum = torch.abs(attributions).sum(dim=1).squeeze(0).cpu().detach().numpy()
        
        # 归一化到[0, 1]范围
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        
        # 调整大小到224x224并返回
        return cv2.resize(attr_norm, (224, 224))

    def generate_multiple_igs(self, input_tensor: torch.Tensor, n_samples: int = 5) -> List[np.ndarray]:
        """生成多次IG解释以计算稳定性"""
        explanations = []
        for _ in range(n_samples):
            explanations.append(self.generate_ig(input_tensor))
        return explanations

    def compute_metrics(self, img_explanation: np.ndarray, corrupted_explanation: np.ndarray, 
                      img_pred: int, corrupted_pred: int, img_probs: np.ndarray, 
                      corrupted_probs: np.ndarray, stability_list: List[np.ndarray] = None) -> Dict:
        """计算解释方法的鲁棒性指标"""
        
        # 1. 相似度指标 (余弦相似度)
        cosine_sim = np.sum(img_explanation * corrupted_explanation) / (
            np.sqrt(np.sum(img_explanation**2)) * np.sqrt(np.sum(corrupted_explanation**2)) + 1e-8)
        
        # 2. 一致性指标 (使用互信息)
        # 将解释转换为直方图以计算互信息
        img_hist = np.histogram(img_explanation.flatten(), bins=20)[0]
        corrupted_hist = np.histogram(corrupted_explanation.flatten(), bins=20)[0]
        mutual_info = mutual_info_score(img_hist, corrupted_hist)

        # 3. 定位指标 (IoU - 首先二值化解释)
        threshold = 0.5
        binary_img_explanation = (img_explanation > threshold).astype(int)
        binary_corrupted_explanation = (corrupted_explanation > threshold).astype(int)
        
        intersection = np.logical_and(binary_img_explanation, binary_corrupted_explanation).sum()
        union = np.logical_or(binary_img_explanation, binary_corrupted_explanation).sum()
        iou = intersection / (union + 1e-8)
        
        # 4. 预测变化指标 (预测翻转)
        prediction_change = int(img_pred != corrupted_pred)
        
        # 5. 置信度差异
        confidence_diff = np.abs(img_probs[img_pred] - corrupted_probs[corrupted_pred])
        
        # 6. KL散度
        # 添加平滑以避免零概率
        img_probs_smooth = img_probs + 1e-10
        corrupted_probs_smooth = corrupted_probs + 1e-10
        
        # 归一化
        img_probs_smooth = img_probs_smooth / img_probs_smooth.sum()
        corrupted_probs_smooth = corrupted_probs_smooth / corrupted_probs_smooth.sum()
        
        kl_div = np.sum(img_probs_smooth * np.log(img_probs_smooth / corrupted_probs_smooth))
        
        # 7. Top-5距离
        img_top5 = np.argsort(img_probs)[-5:]
        corrupted_top5 = np.argsort(corrupted_probs)[-5:]
        
        # 计算top5中不同位置的数量
        top5_diff = len(set(img_top5) - set(corrupted_top5))
        
        # 8. 腐蚀错误率
        ce = 1.0 if img_pred != corrupted_pred else 0.0
        
        # 9. 稳定性指标（如果提供了多个解释）
        stability = 0.0
        if stability_list and len(stability_list) > 1:
            avg_exp = np.mean(stability_list, axis=0)
            stability = np.mean([np.corrcoef(avg_exp.flatten(), exp.flatten())[0, 1] for exp in stability_list])
        
        return {
            "similarity": float(cosine_sim),
            "consistency": float(mutual_info),
            "localization": float(iou),
            "prediction_change": prediction_change,
            "confidence_diff": float(confidence_diff),
            "kl_divergence": float(kl_div),
            "top5_distance": int(top5_diff),
            "corruption_error": float(ce),
            "stability": float(stability)
        }

    def save_visualization(self, original_img, corrupted_img, original_attr, corrupted_attr, 
                         output_path, corruption_type, severity):
        """保存解释的可视化图像"""
        plt.figure(figsize=(10, 5))
        
        # 显示原始图像
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(original_img))
        plt.title("Original Image")
        plt.axis('off')
        
        # 显示原始图像的IG解释
        plt.subplot(2, 2, 2)
        plt.imshow(original_attr, cmap='jet')
        plt.title("Original IG")
        plt.axis('off')
        
        # 显示腐蚀后的图像
        plt.subplot(2, 2, 3)
        plt.imshow(np.array(corrupted_img))
        plt.title(f"{corruption_type} (Severity {severity})")
        plt.axis('off')
        
        # 显示腐蚀后图像的IG解释
        plt.subplot(2, 2, 4)
        plt.imshow(corrupted_attr, cmap='jet')
        plt.title("Corrupted IG")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def test_robustness(self, image_dir: str, output_file: str, temp_file: str = None, 
                       save_viz: bool = False, viz_dir: str = None, test_mode: bool = False,
                       test_samples: int = 10):
        """测试IG解释的鲁棒性"""
        results = {}
        
        # 如果提供了临时文件且存在，则加载它
        if temp_file and os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                results = json.load(f)
                print(f"Loaded {len(results)} previous results from {temp_file}")
                
        # 获取图像列表
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files.append(os.path.join(root, file))
        
        # 如果是测试模式，则只使用指定数量的样本
        if test_mode:
            if len(image_files) > test_samples:
                image_files = image_files[:test_samples]
            print(f"Test mode: using {len(image_files)} images")
                
        # 创建可视化目录（如果需要）
        if save_viz and viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
            
        total_images = len(image_files)
        start_time = time.time()
        
        # 处理每个图像
        for idx, image_path in enumerate(image_files):
            if image_path in results:
                print(f"Skipping already processed image {idx+1}/{total_images}: {image_path}")
                continue
                
            print(f"Processing image {idx+1}/{total_images}: {image_path}")
            try:
                # 初始化该图像的结果
                results[image_path] = {}
                
                # 加载原始图像
                input_tensor, original_image = self.load_image(image_path)
                
                # 获取模型预测
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                # 获取预测类别和概率
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output, dim=1).item()
                
                # 生成原始图像的IG解释
                original_explanation = self.generate_ig(input_tensor)
                
                # 生成多个IG解释用于稳定性计算
                original_stability_explanations = self.generate_multiple_igs(input_tensor)
                
                # 处理每种腐蚀类型
                for corruption_type in self.corruption_types:
                    results[image_path][corruption_type] = {"results": []}
                    
                    # 对每个严重程度级别
                    for severity in range(1, 6):
                        # 应用腐蚀
                        corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                        corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                        
                        # 获取腐蚀后图像的预测
                        with torch.no_grad():
                            corrupted_output = self.model(corrupted_tensor)
                            
                        corrupted_probs = torch.nn.functional.softmax(corrupted_output, dim=1).cpu().numpy()[0]
                        corrupted_pred_class = torch.argmax(corrupted_output, dim=1).item()
                        
                        # 生成腐蚀后图像的IG解释
                        corrupted_explanation = self.generate_ig(corrupted_tensor)
                        
                        # 生成多个IG解释用于稳定性计算
                        corrupted_stability_explanations = self.generate_multiple_igs(corrupted_tensor)
                        
                        # 计算指标
                        metrics = self.compute_metrics(
                            original_explanation, 
                            corrupted_explanation,
                            pred_class, 
                            corrupted_pred_class,
                            probs, 
                            corrupted_probs,
                            corrupted_stability_explanations
                        )
                        
                        # 添加严重程度
                        metrics["severity"] = severity
                        
                        # 保存到结果
                        results[image_path][corruption_type]["results"].append(metrics)
                        
                        # 保存可视化（如果需要）
                        if save_viz and viz_dir:
                            viz_filename = f"{os.path.basename(image_path).split('.')[0]}_{corruption_type}_s{severity}.png"
                            viz_path = os.path.join(viz_dir, viz_filename)
                            self.save_visualization(
                                original_image, corrupted_image,
                                original_explanation, corrupted_explanation,
                                viz_path, corruption_type, severity
                            )
                
                # 每完成一张图像保存一次临时结果
                if temp_file:
                    with open(temp_file, 'w') as f:
                        json.dump(results, f)
                        
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (idx + 1)
                remaining_images = total_images - (idx + 1)
                est_remaining_time = avg_time_per_image * remaining_images
                
                print(f"Completed {idx+1}/{total_images} images.")
                print(f"Avg time per image: {avg_time_per_image:.2f}s. Est. remaining time: {est_remaining_time/60:.2f} minutes")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
                
        # 保存最终结果
        with open(output_file, 'w') as f:
            json.dump(results, f)
            
        print(f"Testing completed. Results saved to {output_file}")
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test robustness of Integrated Gradients explanations')
    parser.add_argument('--image_dir', type=str, default='experiments/data/tiny-imagenet-200/val',
                        help='Directory containing validation images')
    parser.add_argument('--output_file', type=str, default='experiments/results/ig_robustness_results.json',
                        help='Output JSON file path')
    parser.add_argument('--temp_file', type=str, default='experiments/results/ig_robustness_results_temp.json',
                        help='Temporary output file path for incremental saving')
    parser.add_argument('--model_type', type=str, choices=['standard', 'robust'], default='standard',
                        help='Model type to use (standard ResNet50 or robust Salman2020Do_50_2)')
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations of explanations')
    parser.add_argument('--viz_dir', type=str, default='experiments/results/ig_viz',
                        help='Directory to save visualizations')
    parser.add_argument('--test_mode', action='store_true', 
                        help='Run in test mode with limited samples')
    parser.add_argument('--test_samples', type=int, default=10,
                        help='Number of samples to use in test mode')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.temp_file:
        os.makedirs(os.path.dirname(args.temp_file), exist_ok=True)
    
    # 创建测试实例
    tester = IGRobustnessTest(model_type=args.model_type)
    
    # 运行测试
    tester.test_robustness(
        image_dir=args.image_dir,
        output_file=args.output_file,
        temp_file=args.temp_file,
        save_viz=args.save_viz,
        viz_dir=args.viz_dir,
        test_mode=args.test_mode,
        test_samples=args.test_samples
    )

if __name__ == "__main__":
    main() 