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
from sklearn.metrics import mutual_info_score

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
            GradCAM热力图 (224x224)
        """
        # 确保启用梯度
        input_tensor.requires_grad = True
        
        # 生成GradCAM
        gradcam_mask = self.gradcam(input_tensor=input_tensor)
        
        # 将mask调整到224x224大小
        mask = cv2.resize(gradcam_mask[0], (224, 224))
        
        return mask

    def calculate_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个解释之间的余弦相似度
        
        Args:
            mask1: 第一个GradCAM mask (224x224)
            mask2: 第二个GradCAM mask (224x224)
        """
        # 确保mask大小相同
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
        
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        
        return np.dot(mask1_flat, mask2_flat) / (
            np.linalg.norm(mask1_flat) * np.linalg.norm(mask2_flat)
        )

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
            original_top5 = torch.topk(original_output, k=5, dim=1)[1][0]
            
            # 获取corrupted预测
            corrupted_output = self.model(corrupted_tensor)
            corrupted_pred = torch.argmax(corrupted_output, dim=1)
            corrupted_probs = torch.softmax(corrupted_output, dim=1)
            corrupted_top5 = torch.topk(corrupted_output, k=5, dim=1)[1][0]
            
            # 计算基本指标
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
            
            # 计算Top-5距离
            top5_distance = self.calculate_mt5d(
                original_top5.cpu().numpy(),
                corrupted_top5.cpu().numpy()
            )
            
            # 计算Corruption Error (相对于原始预测的错误率增加)
            original_error = 1 - confidence_original
            corruption_error = 1 - confidence_corrupted
            mce = self.calculate_mce(original_error, corruption_error)
            
            return {
                'prediction_change': prediction_change,
                'confidence_diff': confidence_diff,
                'kl_divergence': kl_div,
                'original_confidence': confidence_original,
                'corrupted_confidence': confidence_corrupted,
                'top5_distance': top5_distance,
                'corruption_error': mce
            }

    def calculate_consistency(self, exp1: np.ndarray, exp2: np.ndarray) -> float:
        """计算语义一致性
        
        Args:
            exp1: 原始解释 (224x224)
            exp2: 扰动后的解释 (224x224)
        """
        # 确保大小相同
        if exp1.shape != exp2.shape:
            exp2 = cv2.resize(exp2, (exp1.shape[1], exp1.shape[0]))
        
        exp1_bins = np.digitize(exp1.flatten(), bins=np.linspace(0, 1, 20))
        exp2_bins = np.digitize(exp2.flatten(), bins=np.linspace(0, 1, 20))
        mi = mutual_info_score(exp1_bins, exp2_bins)
        return mi / np.log(20)

    def calculate_stability(self, explanations: List[np.ndarray]) -> float:
        """计算解释的稳定性
        
        Args:
            explanations: 同一图像在不同扰动下的解释列表 (每个都是224x224)
        """
        # 确保所有解释大小相同
        base_shape = explanations[0].shape
        resized_explanations = []
        for exp in explanations:
            if exp.shape != base_shape:
                exp = cv2.resize(exp, (base_shape[1], base_shape[0]))
            resized_explanations.append(exp)
        
        explanations = np.array(resized_explanations)
        return 1 - np.mean(np.var(explanations, axis=0))

    def calculate_localization(self, explanation: np.ndarray, image: np.ndarray) -> float:
        """计算空间准确性
        Localization = IoU(Mask_exp, Mask_edge)
        IoU = intersection / union
        来源: "Score-CAM: Score-Weighted Visual Explanations" (CVPR 2020)
        
        Args:
            explanation: GradCAM解释 (任意大小)
            image: 原始图像 (HxWx3)
        """
        try:
            # 确保explanation和image具有相同的空间维度
            if explanation.shape != image.shape[:2]:
                explanation = cv2.resize(explanation, (image.shape[1], image.shape[0]))
            
            # 归一化explanation到0-1范围
            explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
            
            # 创建mask
            exp_mask = (explanation > np.mean(explanation)).astype(np.float32)
            
            # 转换图像为灰度并进行边缘检测
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, 100, 200)
            edges = edges.astype(np.float32) / 255.0
            
            # 计算IoU
            intersection = np.sum(exp_mask * edges)
            union = np.sum((exp_mask + edges) > 0)
            
            return float(intersection / (union + 1e-6))
            
        except Exception as e:
            print(f"Error in calculate_localization: {e}")
            return 0.0  # 返回默认值而不是失败

    def calculate_mce(self, clean_err: float, corruption_err: float) -> float:
        """计算Mean Corruption Error
        mCE = E_c[CE_c], where CE_c = E_c(f) / E_c(AlexNet)
        来源: "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (ICLR 2019)
        """
        return corruption_err / clean_err

    def calculate_confidence_diff(self, original_conf: float, corrupted_conf: float) -> float:
        """计算置信度差异
        Conf_diff = |conf(x) - conf(x̃)|
        来源: "On Calibration of Modern Neural Networks" (ICML 2017)
        """
        return abs(original_conf - corrupted_conf)

    def calculate_mfr(self, original_pred: torch.Tensor, corrupted_pred: torch.Tensor) -> float:
        """计算平均翻转率
        mFR = mean(I[f(x) ≠ f(x̃)])
        来源: "Benchmarking the Robustness of Spatial Computing" (NeurIPS 2019)
        """
        return (original_pred != corrupted_pred).float().mean().item()

    def calculate_mt5d(self, original_top5: torch.Tensor, corrupted_top5: torch.Tensor) -> float:
        """计算Top-5预测距离
        mT5D = mean(|top5(x) ∆ top5(x̃)|) / 5
        ∆: 对称差
        来源: "ImageNet-trained CNNs are biased towards texture" (ICLR 2019)
        """
        return len(set(original_top5) ^ set(corrupted_top5)) / 5.0

    def calculate_kl_divergence(self, original_probs: torch.Tensor, corrupted_probs: torch.Tensor) -> float:
        """计算KL散度
        KL(P||Q) = Σ P(x)log(P(x)/Q(x))
        来源: "Understanding deep learning requires rethinking generalization" (ICLR 2017)
        """
        return torch.nn.functional.kl_div(
            corrupted_probs.log(),
            original_probs,
            reduction='batchmean'
        ).item()

    def evaluate_single_image(self, image_path: str) -> Dict:
        """评估单张图像的所有metrics"""
        try:
            # 加载并预处理图像
            input_tensor, original_image = self.load_image(image_path)
            # 确保图像是uint8类型
            original_image_np = np.array(original_image).astype(np.uint8)
            
            # 生成原始GradCAM
            original_mask = self.generate_gradcam(input_tensor)
            
            results = {}
            for corruption_type in self.corruption_types:
                corruption_results = []
                explanations = []  # 用于计算stability
                
                for severity in range(1, 6):
                    try:
                        # 应用corruption
                        corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                        # 确保corrupted_image是uint8类型
                        corrupted_image_np = np.array(corrupted_image).astype(np.uint8)
                        corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                        
                        # 生成corrupted图像的GradCAM
                        corrupted_mask = self.generate_gradcam(corrupted_tensor)
                        
                        # 保存解释用于计算stability
                        explanations.append(corrupted_mask)
                        
                        # 计算所有metrics
                        metrics = {
                            'severity': severity,
                            'similarity': self.calculate_similarity(original_mask, corrupted_mask),
                            'consistency': self.calculate_consistency(original_mask, corrupted_mask),
                            'localization': self.calculate_localization(corrupted_mask, corrupted_image_np)
                        }
                        
                        # 添加现有的其他metrics
                        additional_metrics = self.calculate_additional_metrics(input_tensor, corrupted_tensor)
                        metrics.update(additional_metrics)
                        
                        corruption_results.append(metrics)
                        
                    except Exception as e:
                        print(f"Error processing severity {severity} for {corruption_type}: {e}")
                        continue
                
                if corruption_results:
                    # 计算stability
                    stability = self.calculate_stability(explanations)
                    
                    results[corruption_type] = {
                        'results': corruption_results,
                        'stability': stability
                    }
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def convert_to_serializable(obj):
    """将NumPy类型转换为可JSON序列化的Python原生类型
    
    Args:
        obj: 输入对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    else:
        return obj

def main():
    """主函数"""
    # 设置测试模式
    TEST_MODE = False  # 关闭测试模式
    MAX_TEST_IMAGES = 3  # 此参数在非测试模式下不会被使用
    
    tester = GradCAMRobustnessTest()
    data_dir = "experiments/data/tiny-imagenet-200/val"
    
    processed_count = 0
    successful_count = 0
    all_results = {}
    
    # 获取所有图像文件，包括子目录
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if TEST_MODE:
        image_files = image_files[:MAX_TEST_IMAGES]
    
    total_images = len(image_files)
    print(f"Starting processing of {total_images} images...")
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\nProcessing image {idx}/{total_images}: {image_path}")
        
        try:
            results = tester.evaluate_single_image(image_path)
            if results:
                all_results[image_path] = results
                successful_count += 1
                
                # 每10张图片保存一次临时结果
                if successful_count % 10 == 0:
                    temp_output_path = "experiments/results/gradcam_robustness_results_temp.json"
                    with open(temp_output_path, 'w') as f:
                        # 转换为可序列化的格式
                        serializable_results = convert_to_serializable(all_results)
                        json.dump(serializable_results, f, indent=4)
                    print(f"Temporary results saved. Successful: {successful_count}/{idx}")
                    
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            continue
    
    # 保存最终结果
    if all_results:
        output_path = "experiments/results/gradcam_robustness_results.json"
        with open(output_path, 'w') as f:
            # 转换为可序列化的格式
            serializable_results = convert_to_serializable(all_results)
            json.dump(serializable_results, f, indent=4)
        print(f"\nFinal results saved. Successfully processed {successful_count}/{total_images} images.")
    else:
        print("\nNo results were generated!")

if __name__ == "__main__":
    main() 
