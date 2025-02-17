import torch
import torchvision.models as models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
import cv2
import time
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, map_coordinates
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO

def clipped_zoom(img, zoom_factor):
    """Center-crop an image after zooming
    
    Args:
        img: Input image (HxWxC)
        zoom_factor: Amount to zoom in/out
        
    Returns:
        Zoomed and cropped image of the same size as input
    """
    h, w = img.shape[:2]
    
    # 新的尺寸
    zh = int(np.round(h * zoom_factor))
    zw = int(np.round(w * zoom_factor))
    
    # 确保至少有1个像素
    zh = max(zh, 1)
    zw = max(zw, 1)
    
    # 缩放图像
    zoomed = zoom(img, [zoom_factor, zoom_factor, 1], order=1, mode='reflect')
    
    # 如果放大
    if zoom_factor > 1:
        # 计算裁剪区域
        trim_h = ((zoomed.shape[0] - h) // 2)
        trim_w = ((zoomed.shape[1] - w) // 2)
        zoomed = zoomed[trim_h:trim_h+h, trim_w:trim_w+w]
    
    # 如果缩小
    elif zoom_factor < 1:
        # 计算填充区域
        pad_h = ((h - zoomed.shape[0]) // 2)
        pad_w = ((w - zoomed.shape[1]) // 2)
        # 创建输出图像
        out = np.zeros_like(img)
        # 将缩小的图像放在中心
        out[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
        zoomed = out
    
    # 确保输出大小完全匹配
    if zoomed.shape[:2] != (h, w):
        # 如果大小不匹配，进行调整
        zoomed = cv2.resize(zoomed, (w, h))
    
    return zoomed

class GradCAMRobustnessTest:
    """GradCAM鲁棒性测试类
    
    用于评估GradCAM在不同corruption条件下的解释稳定性
    
    Attributes:
        model: 预训练的ResNet50模型
        device: 计算设备（CPU/GPU）
        transform: 图像预处理转换
        gradcam: GradCAM实例
    """
    
    def __init__(self, device: torch.device = None):
        """初始化GradCAM鲁棒性测试
        
        Args:
            device: 计算设备，如果为None则自动选择
        """
        # 设置设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练的ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化GradCAM
        self.gradcam = GradCAM(
            model=self.model,
            target_layers=[self.model.layer4[-1]]  # 使用最后一个残差块
        )
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # 定义所有corruption类型
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur
            'snow', 'frost', 'fog', 'brightness',  # Weather
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'  # Digital
        ]

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """加载并预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像张量和原始PIL图像
        """
        # 加载原始图像
        original_image = Image.open(image_path).convert('RGB')
        # 预处理图像
        input_tensor = self.transform(original_image)
        # 添加batch维度
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
        """应用corruption到图像
        
        Args:
            image: 原始PIL图像
            corruption_type: corruption类型
            severity: corruption严重程度（1-5）
            
        Returns:
            应用corruption后的PIL图像
        """
        # 将PIL图像转换为numpy数组
        img = np.array(image) / 255.0  # 归一化到[0,1]

        # 设置随机种子以确保可重复性
        np.random.seed(1)

        # 根据severity调整参数
        # 这些参数是基于论文中的建议值
        severity = float(severity) / 5.0  # 归一化到[0,1]

        if corruption_type == 'gaussian_noise':
            noise = np.random.normal(loc=0, scale=severity * 0.5, size=img.shape)
            corrupted = np.clip(img + noise, 0, 1)
        
        elif corruption_type == 'shot_noise':
            corrupted = np.random.poisson(img * 255.0 * (1-severity)) / 255.0
            corrupted = np.clip(corrupted, 0, 1)
        
        elif corruption_type == 'impulse_noise':
            corrupted = sk.util.random_noise(img, mode='s&p', amount=severity)
        
        elif corruption_type == 'defocus_blur':
            corrupted = gaussian(img, sigma=severity * 4, channel_axis=-1)
        
        elif corruption_type == 'glass_blur':
            # 简化版本的玻璃模糊
            kernel_size = int(severity * 10) * 2 + 1
            corrupted = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        elif corruption_type == 'motion_blur':
            # 简化版本的运动模糊
            kernel_size = int(severity * 20)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            corrupted = cv2.filter2D(img, -1, kernel)
        
        elif corruption_type == 'zoom_blur':
            # 改进的zoom blur实现
            out = np.zeros_like(img)
            zoom_factors = [1-severity*0.1, 1-severity*0.05, 1, 1+severity*0.05, 1+severity*0.1]
            
            for zoom_factor in zoom_factors:
                zoomed = clipped_zoom(img, zoom_factor)
                out += zoomed
            
            corrupted = np.clip(out / len(zoom_factors), 0, 1)
        
        elif corruption_type == 'snow':
            # 简化版本的雪效果
            snow_layer = np.random.normal(size=img.shape[:2], loc=0.5, scale=severity)
            snow_layer = np.clip(snow_layer, 0, 1)
            snow_layer = np.expand_dims(snow_layer, axis=2)
            corrupted = np.clip(img + snow_layer, 0, 1)
        
        elif corruption_type == 'frost':
            # 简化版本的霜冻效果
            frost_layer = np.random.uniform(size=img.shape[:2]) * severity
            frost_layer = np.expand_dims(frost_layer, axis=2)
            corrupted = np.clip(img * (1 - frost_layer), 0, 1)
        
        elif corruption_type == 'fog':
            # 简化版本的雾效果
            fog_layer = severity * np.ones_like(img)
            corrupted = np.clip(img * (1 - severity) + fog_layer * severity, 0, 1)
        
        elif corruption_type == 'brightness':
            # 亮度调整
            corrupted = np.clip(img * (1 + severity), 0, 1)
        
        elif corruption_type == 'contrast':
            # 对比度调整
            mean = np.mean(img, axis=(0,1), keepdims=True)
            corrupted = np.clip((img - mean) * (1 + severity) + mean, 0, 1)
        
        elif corruption_type == 'elastic_transform':
            # 简化版本的弹性变换
            shape = img.shape[:2]
            dx = np.random.uniform(-1, 1, shape) * severity * 30
            dy = np.random.uniform(-1, 1, shape) * severity * 30
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            corrupted = np.zeros_like(img)
            for c in range(3):
                corrupted[:,:,c] = np.reshape(
                    map_coordinates(img[:,:,c], indices, order=1),
                    shape)
        
        elif corruption_type == 'pixelate':
            # 像素化
            h, w = img.shape[:2]
            size = max(int((1-severity) * min(h,w)), 1)
            corrupted = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
            corrupted = cv2.resize(corrupted, (w,h), interpolation=cv2.INTER_NEAREST)
        
        elif corruption_type == 'jpeg':
            # JPEG压缩
            quality = int((1-severity) * 100)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            corrupted = np.array(Image.open(buffer)) / 255.0

        else:
            return image

        # 将numpy数组转换回PIL图像
        corrupted = np.clip(corrupted * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted)

    def generate_gradcam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """生成GradCAM解释
        
        Args:
            input_tensor: 输入图像张量
            
        Returns:
            GradCAM热力图
        """
        # 确保启用梯度
        input_tensor.requires_grad = True
        
        # 生成GradCAM
        gradcam_mask = self.gradcam(input_tensor=input_tensor)
        
        return gradcam_mask[0]  # 返回第一个样本的mask

    def calculate_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个GradCAM mask之间的相似度
        
        Args:
            mask1: 第一个GradCAM mask
            mask2: 第二个GradCAM mask
            
        Returns:
            相似度分数（0-1之间）
        """
        # 展平并计算余弦相似度
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        
        similarity = np.dot(mask1_flat, mask2_flat) / (
            np.linalg.norm(mask1_flat) * np.linalg.norm(mask2_flat)
        )
        return float(similarity)

    def calculate_additional_metrics(self, original_tensor: torch.Tensor, 
                                  corrupted_tensor: torch.Tensor) -> Dict:
        """计算额外的评估指标
        
        Args:
            original_tensor: 原始图像张量
            corrupted_tensor: 受corrupted的图像张量
            
        Returns:
            包含额外评估指标的字典
        """
        with torch.no_grad():
            # 获取原始预测
            original_output = self.model(original_tensor)
            original_pred = torch.argmax(original_output, dim=1)
            original_probs = torch.softmax(original_output, dim=1)
            
            # 获取corrupted预测
            corrupted_output = self.model(corrupted_tensor)
            corrupted_pred = torch.argmax(corrupted_output, dim=1)
            corrupted_probs = torch.softmax(corrupted_output, dim=1)
            
            # 计算指标
            prediction_change = (original_pred != corrupted_pred).float().mean().item()
            confidence_original = original_probs.max(dim=1)[0].item()
            confidence_corrupted = corrupted_probs.max(dim=1)[0].item()
            confidence_diff = abs(confidence_original - confidence_corrupted)
            
            # 计算KL散度
            kl_div = torch.nn.functional.kl_div(
                corrupted_probs.log(),
                original_probs,
                reduction='batchmean'
            ).item()
            
            return {
                'prediction_change': prediction_change,
                'confidence_diff': confidence_diff,
                'kl_divergence': kl_div,
                'original_confidence': confidence_original,
                'corrupted_confidence': confidence_corrupted
            }

    def evaluate_single_image(self, image_path: str) -> Dict:
        """评估单张图像的GradCAM鲁棒性
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含评估结果的字典
        """
        # 加载原始图像
        input_tensor, original_image = self.load_image(image_path)
        
        # 生成原始GradCAM
        original_mask = self.generate_gradcam(input_tensor)
        
        results = {}
        # 对每种corruption类型
        for corruption_type in self.corruption_types:
            corruption_results = []
            # 对每个严重程度
            for severity in range(1, 6):
                # 应用corruption
                corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                
                # 生成corrupted图像的GradCAM
                corrupted_mask = self.generate_gradcam(corrupted_tensor)
                
                # 计算相似度
                similarity = self.calculate_similarity(original_mask, corrupted_mask)
                
                # 计算额外指标
                additional_metrics = self.calculate_additional_metrics(input_tensor, corrupted_tensor)
                
                corruption_results.append({
                    'severity': severity,
                    'similarity': similarity,
                    **additional_metrics
                })
            
            results[corruption_type] = corruption_results
            
        return results

def main():
    """主函数"""
    # 初始化测试器
    tester = GradCAMRobustnessTest()
    
    # 设置数据路径
    val_dir = "experiments/data/tiny-imagenet-200/val"
    
    # 结果存储
    all_results = {}
    
    # 处理验证集图像
    for root, _, files in os.walk(val_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}")
                
                try:
                    # 评估单张图像
                    results = tester.evaluate_single_image(image_path)
                    all_results[image_path] = results
                    
                    # 定期保存结果
                    if len(all_results) % 10 == 0:
                        temp_output_path = "experiments/results/gradcam_robustness_results_temp.json"
                        with open(temp_output_path, 'w') as f:
                            json.dump(all_results, f, indent=4)
                        print(f"Temporary results saved to {temp_output_path}")
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
    
    # 保存最终结果
    output_path = "experiments/results/gradcam_robustness_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Final results saved to {output_path}")

if __name__ == "__main__":
    main() 
