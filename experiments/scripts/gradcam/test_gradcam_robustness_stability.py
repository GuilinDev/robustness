import torch
import torchvision.models as models
from pytorch_grad_cam import GradCAM
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
import cv2
from typing import Dict, List, Tuple
from scipy.ndimage import zoom, map_coordinates  # 显式导入 map_coordinates
from skimage.filters import gaussian  # 显式导入 gaussian
from skimage import util as sk_util  # 导入 skimage.util 作为 sk_util
from io import BytesIO  # 导入 BytesIO

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
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'
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
        severity = float(severity) / 5.0  # 归一化到[0,1]

        if corruption_type == 'gaussian_noise':
            noise = np.random.normal(loc=0, scale=severity * 0.5, size=img.shape)
            corrupted = np.clip(img + noise, 0, 1)
        
        elif corruption_type == 'shot_noise':
            corrupted = np.random.poisson(img * 255.0 * (1-severity)) / 255.0
            corrupted = np.clip(corrupted, 0, 1)
        
        elif corruption_type == 'impulse_noise':
            corrupted = sk_util.random_noise(img, mode='s&p', amount=severity)  # 使用 sk_util
        
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
            snow_layer = np.random.normal(size=img.shape[:2], loc=0.5, scale=severity)
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
                corrupted[:,:,c] = np.reshape(
                    map_coordinates(img[:,:,c], indices, order=1),
                    shape)
        
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

    def evaluate_single_image(self, image_path: str) -> Dict:
        """评估单张图像的Stability Score"""
        try:
            # 加载并预处理图像
            input_tensor, original_image = self.load_image(image_path)
            # 确保图像是uint8类型
            original_image_np = np.array(original_image).astype(np.uint8)
            
            # 生成原始GradCAM
            original_mask = self.generate_gradcam(input_tensor)
            
            results = {}
            for corruption_type in self.corruption_types:
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
                        
                    except Exception as e:
                        print(f"Error processing severity {severity} for {corruption_type}: {e}")
                        continue
                
                if explanations:
                    # 计算stability
                    stability = self.calculate_stability(explanations)
                    
                    results[corruption_type] = {
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
                    temp_output_path = "experiments/results/stability_results_temp.json"
                    with open(temp_output_path, 'w') as f:
                        serializable_results = convert_to_serializable(all_results)
                        json.dump(serializable_results, f, indent=4)
                    print(f"Temporary results saved. Successful: {successful_count}/{idx}")
                    
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            continue
    
    # 保存最终结果
    if all_results:
        output_path = "experiments/results/stability_results.json"
        with open(output_path, 'w') as f:
            serializable_results = convert_to_serializable(all_results)
            json.dump(serializable_results, f, indent=4)
        print(f"\nFinal results saved. Successfully processed {successful_count}/{total_images} images.")
    else:
        print("\nNo results were generated!")

if __name__ == "__main__":
    main()
    