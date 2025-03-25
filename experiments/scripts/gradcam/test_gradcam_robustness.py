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
import argparse
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, map_coordinates
from skimage.filters import gaussian
from skimage import util as sk_util
from io import BytesIO
from sklearn.metrics import mutual_info_score
from torch.cuda.amp import autocast, GradScaler

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

class GradCAMRobustnessTest:
    """GradCAM鲁棒性测试类"""
    
    def __init__(self, model_type="standard", device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 根据model_type选择不同的模型
        self.model_type = model_type
        if model_type == "standard":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("Loaded standard ResNet50 model")
            # 为标准ResNet50设置目标层
            target_layers = [self.model.layer4[-1]]
        elif model_type == "robust" and ROBUSTBENCH_AVAILABLE:
            self.model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')
            print("Loaded RobustBench Salman2020Do_50_2 model")
            # 检查模型结构和层
            print(f"Model type: {type(self.model).__name__}")
            # 打印模型的子模块名称，帮助找到正确的target layer
            for name, module in self.model.named_children():
                print(f"Module name: {name}, Type: {type(module).__name__}")
            
            # 尝试为Salman2020Do模型设置合适的目标层
            # 由于出现AttributeError，这里我们需要检查模型结构并调整
            if hasattr(self.model, 'layer4'):
                target_layers = [self.model.layer4[-1]]
            elif hasattr(self.model, 'features') and isinstance(self.model.features, torch.nn.Sequential):
                # 如果模型使用了Sequential结构的features，选择最后一个卷积层
                conv_layers = [m for m in self.model.features.modules() if isinstance(m, torch.nn.Conv2d)]
                if conv_layers:
                    target_layers = [conv_layers[-1]]
                else:
                    raise ValueError("No convolutional layers found in the model")
            else:
                # 尝试寻找任何卷积层作为备选
                conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
                if conv_layers:
                    target_layers = [conv_layers[-1]]
                else:
                    raise ValueError("Cannot determine appropriate target layers for GradCAM")
        else:
            raise ValueError(f"Invalid model_type {model_type} or RobustBench not available")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        self.gradcam = GradCAM(model=self.model, target_layers=target_layers)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg'
        ]
        self.alexnet_baseline = 0.7  # AlexNet 在 Tiny-ImageNet 上的实际错误率

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
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
            snow_layer = np.random.normal(size=img.shape[:2], loc=0.5, scale=severity * 1.5)  # 增加强度
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

    def generate_gradcam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """生成GradCAM解释"""
        # 创建新的叶节点张量，并设置 requires_grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True).to(self.device)
        gradcam_mask = self.gradcam(input_tensor=input_tensor)
        mask = cv2.resize(gradcam_mask[0], (224, 224))
        return mask

    def generate_multiple_gradcams(self, input_tensor: torch.Tensor, n_samples: int = 5) -> List[np.ndarray]:
        """生成多次GradCAM解释以计算稳定性"""
        explanations = []
        for _ in range(n_samples):
            noisy_tensor = input_tensor + torch.randn_like(input_tensor) * 0.01
            mask = self.generate_gradcam(noisy_tensor)
            explanations.append(mask)
        return explanations

    def calculate_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
        mask1_flat = mask1.flatten()
        mask2_flat = mask2.flatten()
        return np.dot(mask1_flat, mask2_flat) / (np.linalg.norm(mask1_flat) * np.linalg.norm(mask2_flat) + 1e-8)

    def calculate_additional_metrics(self, original_tensor: torch.Tensor, corrupted_tensor: torch.Tensor) -> Dict:
        with torch.no_grad():
            with autocast():
                original_output = self.model(original_tensor)
            original_pred = torch.argmax(original_output, dim=1)
            original_probs = torch.softmax(original_output, dim=1)
            original_top5 = torch.topk(original_output, k=5, dim=1)[1][0]
            
            with autocast():
                corrupted_output = self.model(corrupted_tensor)
            corrupted_pred = torch.argmax(corrupted_output, dim=1)
            corrupted_probs = torch.softmax(corrupted_output, dim=1)
            corrupted_top5 = torch.topk(corrupted_output, k=5, dim=1)[1][0]
            
            prediction_change = (original_pred != corrupted_pred).float().mean().item()
            confidence_original = original_probs.max(dim=1)[0].item()
            confidence_corrupted = corrupted_probs.max(dim=1)[0].item()
            confidence_diff = abs(confidence_original - confidence_corrupted)
            
            eps = 1e-8
            original_probs = original_probs + eps
            corrupted_probs = corrupted_probs + eps
            kl_div = torch.nn.functional.kl_div(original_probs.log(), corrupted_probs, reduction='batchmean').item()
            
            top5_distance = self.calculate_mt5d(original_top5.cpu().numpy(), corrupted_top5.cpu().numpy())
            original_error = 1 - confidence_original
            corruption_error = 1 - confidence_corrupted
            mce = self.calculate_mce(original_error, corruption_error, self.alexnet_baseline)
            
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
        if exp1.shape != exp2.shape:
            exp2 = cv2.resize(exp2, (exp1.shape[1], exp1.shape[0]))
        exp1 = (exp1 - exp1.min()) / (exp1.max() - exp1.min() + 1e-8)
        exp2 = (exp2 - exp2.min()) / (exp2.max() - exp2.min() + 1e-8)
        n_bins = max(10, int(np.sqrt(exp1.size)))
        exp1_bins = np.digitize(exp1.flatten(), bins=np.linspace(0, 1, n_bins))
        exp2_bins = np.digitize(exp2.flatten(), bins=np.linspace(0, 1, n_bins))
        mi = mutual_info_score(exp1_bins, exp2_bins)
        return mi / np.log(n_bins)

    def calculate_stability(self, explanations: List[np.ndarray]) -> float:
        base_shape = explanations[0].shape
        resized_explanations = []
        for exp in explanations:
            if exp.shape != base_shape:
                exp = cv2.resize(exp, (base_shape[1], base_shape[0]))
            resized_explanations.append(exp)
        explanations = np.array(resized_explanations)
        return 1 - np.mean(np.var(explanations, axis=0))

    def calculate_localization(self, explanation: np.ndarray, image: np.ndarray) -> float:
        try:
            if explanation.shape != image.shape[:2]:
                explanation = cv2.resize(explanation, (image.shape[1], image.shape[0]))
            explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
            exp_mask = (explanation > 0.5).astype(np.float32)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            edges = edges.astype(np.float32) / 255.0
            intersection = np.sum(exp_mask * edges)
            union = np.sum((exp_mask + edges) > 0)
            return float(intersection / (union + 1e-6))
        except Exception as e:
            print(f"Error in calculate_localization: {e}")
            return 0.0

    def calculate_mce(self, clean_err: float, corruption_err: float, alexnet_baseline: float) -> float:
        return corruption_err / alexnet_baseline if alexnet_baseline > 0 else corruption_err

    def calculate_confidence_diff(self, original_conf: float, corrupted_conf: float) -> float:
        return abs(original_conf - corrupted_conf)

    def calculate_mfr(self, original_pred: torch.Tensor, corrupted_pred: torch.Tensor) -> float:
        return (original_pred != corrupted_pred).float().mean().item()

    def calculate_mt5d(self, original_top5: np.ndarray, corrupted_top5: np.ndarray) -> float:
        return len(set(original_top5) ^ set(corrupted_top5)) / 5.0

    def calculate_kl_divergence(self, original_probs: torch.Tensor, corrupted_probs: torch.Tensor) -> float:
        eps = 1e-8
        original_probs = original_probs + eps
        corrupted_probs = corrupted_probs + eps
        return torch.nn.functional.kl_div(original_probs.log(), corrupted_probs, reduction='batchmean').item()

    def evaluate_single_image(self, image_path: str) -> Dict:
        try:
            input_tensor, original_image = self.load_image(image_path)
            original_image_np = np.array(original_image).astype(np.uint8)
            original_mask = self.generate_gradcam(input_tensor)
            
            results = {}
            for corruption_type in self.corruption_types:
                corruption_results = []
                explanations = self.generate_multiple_gradcams(input_tensor, n_samples=5)
                
                for severity in range(1, 6):
                    try:
                        corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                        corrupted_image_np = np.array(corrupted_image).astype(np.uint8)
                        corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                        corrupted_mask = self.generate_gradcam(corrupted_tensor)
                        
                        metrics = {
                            'severity': severity,
                            'similarity': self.calculate_similarity(original_mask, corrupted_mask),
                            'consistency': self.calculate_consistency(original_mask, corrupted_mask),
                            'localization': self.calculate_localization(corrupted_mask, original_image_np),
                            'stability': self.calculate_stability(explanations + [corrupted_mask])
                        }
                        additional_metrics = self.calculate_additional_metrics(input_tensor, corrupted_tensor)
                        metrics.update(additional_metrics)
                        corruption_results.append(metrics)
                    except Exception as e:
                        print(f"Error processing severity {severity} for {corruption_type}: {e}")
                        continue
                
                if corruption_results:
                    results[corruption_type] = {'results': corruption_results}
            
            return results
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def convert_to_serializable(obj):
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
    parser = argparse.ArgumentParser(description='GradCAM Robustness Test')
    parser.add_argument('--model', type=str, choices=['standard', 'robust'], default='robust',
                        help='Model type: standard (ResNet50) or robust (RobustBench Salman2020Do_50_2)')
    parser.add_argument('--test', action='store_true', help='Run in test mode with only 10 images')
    args = parser.parse_args()
    
    TEST_MODE = args.test
    MAX_TEST_IMAGES = 10
    
    # 根据选择的模型类型设置输出文件路径
    if args.model == "standard":
        temp_output_path = "/workspace/experiments/results/gradcam_robustness_results_temp.json"
        output_path = "/workspace/experiments/results/gradcam_robustness_results.json"
    else:
        temp_output_path = "/workspace/experiments/results/gradcam_robustness_robustbench_results_temp.json"
        output_path = "/workspace/experiments/results/gradcam_robustness_robustbench_results.json"
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    tester = GradCAMRobustnessTest(model_type=args.model)
    data_dir = "/workspace/experiments/data/tiny-imagenet-200/val"
    
    processed_count = 0
    successful_count = 0
    all_results = {}
    
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if TEST_MODE:
        image_files = image_files[:MAX_TEST_IMAGES]
    
    total_images = len(image_files)
    print(f"Starting processing of {total_images} images with {args.model} model...")
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\nProcessing image {idx}/{total_images}: {image_path}")
        try:
            results = tester.evaluate_single_image(image_path)
            if results:
                all_results[image_path] = results
                successful_count += 1
                if successful_count % 10 == 0:  # 每10张图片保存一次临时结果
                    with open(temp_output_path, 'w') as f:
                        serializable_results = convert_to_serializable(all_results)
                        json.dump(serializable_results, f, indent=4)
                    print(f"Temporary results saved to {temp_output_path}. Successful: {successful_count}/{idx}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            continue
    
    if all_results:
        with open(output_path, 'w') as f:
            serializable_results = convert_to_serializable(all_results)
            json.dump(serializable_results, f, indent=4)
        print(f"\nFinal results saved to {output_path}. Successfully processed {successful_count}/{total_images} images.")
    else:
        print("\nNo results were generated!")

if __name__ == "__main__":
    main()
