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

# add RobustBench related imports
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
        
        # select different models based on model_type
        self.model_type = model_type
        if model_type == "standard":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("Loaded standard ResNet50 model")
        elif model_type == "robust":
            if ROBUSTBENCH_AVAILABLE:
                try:
                    self.model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', threat_model='Linf')
                    print("Loaded RobustBench Salman2020Do_50_2 model")
                except Exception as e:
                    print(f"Warning: Unable to load robust model, falling back to standard model. Error: {str(e)}")
                    self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                    print("Falling back to standard ResNet50 model")
                    self.model_type = "standard"  # update model type
            else:
                print("Warning: RobustBench not installed or not available, falling back to standard model")
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                print("Falling back to standard ResNet50 model")
                self.model_type = "standard"  # update model type
        else:
            print(f"Warning: Unknown model type '{model_type}', falling back to standard model")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("Falling back to standard ResNet50 model")
            self.model_type = "standard"  # update model type
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # create Integrated Gradients explainer
        self.ig = IntegratedGradients(self.model)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # image denormalization for visualization
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
        self.alexnet_baseline = 0.7  # error rate of AlexNet on Tiny-ImageNet

    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """load image and convert to model input format"""
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(original_image).unsqueeze(0)
        return input_tensor.to(self.device), original_image

    def apply_corruption(self, image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
        """apply image corruption"""
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
        """generate Integrated Gradients explanation"""
        # create gradient target
        if self.model_type == "standard":
            # for standard ResNet50, we use the predicted class as target
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        else:
            # for robust model, use the predicted class as well
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # calculate IG attribution
        attributions = self.ig.attribute(
            input_tensor, 
            target=target_class, 
            n_steps=50  # number of integration steps, can be adjusted
        )
        
        # convert to numpy array and calculate absolute sum (across channels)
        attr_sum = torch.abs(attributions).sum(dim=1).squeeze(0).cpu().detach().numpy()
        
        # normalize to [0, 1] range
        attr_norm = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        
        # resize to 224x224 and return
        return cv2.resize(attr_norm, (224, 224))

    def generate_multiple_igs(self, input_tensor: torch.Tensor, n_samples: int = 5) -> List[np.ndarray]:
        """generate multiple IG explanations to calculate stability"""
        explanations = []
        for _ in range(n_samples):
            explanations.append(self.generate_ig(input_tensor))
        return explanations

    def compute_metrics(self, img_explanation: np.ndarray, corrupted_explanation: np.ndarray, 
                      img_pred: int, corrupted_pred: int, img_probs: np.ndarray, 
                      corrupted_probs: np.ndarray, stability_list: List[np.ndarray] = None) -> Dict:
        """calculate robustness metrics for explanation methods"""
        
        # 1. similarity metric (cosine similarity)
        cosine_sim = np.sum(img_explanation * corrupted_explanation) / (
            np.sqrt(np.sum(img_explanation**2)) * np.sqrt(np.sum(corrupted_explanation**2)) + 1e-8)
        
        # 2. consistency metric (using mutual information)
        # convert explanation to histogram to calculate mutual information
        img_hist = np.histogram(img_explanation.flatten(), bins=20)[0]
        corrupted_hist = np.histogram(corrupted_explanation.flatten(), bins=20)[0]
        mutual_info = mutual_info_score(img_hist, corrupted_hist)

        # 3. localization metric (IoU - first binarize explanation)
        threshold = 0.5
        binary_img_explanation = (img_explanation > threshold).astype(int)
        binary_corrupted_explanation = (corrupted_explanation > threshold).astype(int)
        
        intersection = np.logical_and(binary_img_explanation, binary_corrupted_explanation).sum()
        union = np.logical_or(binary_img_explanation, binary_corrupted_explanation).sum()
        iou = intersection / (union + 1e-8)
        
        # 4. prediction change metric (prediction flip)
        prediction_change = int(img_pred != corrupted_pred)
        
        # 5. confidence difference
        confidence_diff = np.abs(img_probs[img_pred] - corrupted_probs[corrupted_pred])
        
        # 6. KL divergence
        # add smoothing to avoid zero probability
        img_probs_smooth = img_probs + 1e-10
        corrupted_probs_smooth = corrupted_probs + 1e-10
        
        # normalize
        img_probs_smooth = img_probs_smooth / img_probs_smooth.sum()
        corrupted_probs_smooth = corrupted_probs_smooth / corrupted_probs_smooth.sum()
        
        kl_div = np.sum(img_probs_smooth * np.log(img_probs_smooth / corrupted_probs_smooth))
        
        # 7. Top-5 distance
        img_top5 = np.argsort(img_probs)[-5:]
        corrupted_top5 = np.argsort(corrupted_probs)[-5:]
        
        # calculate number of different positions in top5
        top5_diff = len(set(img_top5) - set(corrupted_top5))
        
        # 8. corruption error rate
        ce = 1.0 if img_pred != corrupted_pred else 0.0
        
        # 9. stability metric (if multiple explanations are provided)
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
        """save explanation visualization"""
        plt.figure(figsize=(10, 5))
        
        # show original image
        plt.subplot(2, 2, 1)
        plt.imshow(np.array(original_img))
        plt.title("Original Image")
        plt.axis('off')
        
        # show original image's IG explanation
        plt.subplot(2, 2, 2)
        plt.imshow(original_attr, cmap='jet')
        plt.title("Original IG")
        plt.axis('off')
        
        # show corrupted image
        plt.subplot(2, 2, 3)
        plt.imshow(np.array(corrupted_img))
        plt.title(f"{corruption_type} (Severity {severity})")
        plt.axis('off')
        
        # show corrupted image's IG explanation
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
        """test robustness of IG explanation"""
        results = {}
        
        # if temp file is provided and exists, load it
        if temp_file and os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                results = json.load(f)
                print(f"已加载 {len(results)} 个之前的结果从 {temp_file}")
                
        # get image list
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_files.append(os.path.join(root, file))
        
        # if test mode, only use specified number of samples
        if test_mode:
            if len(image_files) > test_samples:
                image_files = image_files[:test_samples]
            print(f"测试模式：使用 {len(image_files)} 张图片")
                
        # create visualization directory (if needed)
        if save_viz and viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
            
        total_images = len(image_files)
        start_time = time.time()
        total_corruptions = len(self.corruption_types) * 5  # 5 severity levels for each corruption type
        
        print(f"Start processing {total_images} images, each will be tested with {len(self.corruption_types)} corruption types, each with 5 severity levels")
        print(f"Total {total_images * total_corruptions} samples to process")
        
        # process each image
        for idx, image_path in enumerate(image_files):
            image_start_time = time.time()
            if image_path in results:
                print(f"Skipping processed image [{idx+1}/{total_images}] ({(idx+1)/total_images*100:.1f}%): {os.path.basename(image_path)}")
                continue
                
            print(f"\nProcessing image [{idx+1}/{total_images}] ({(idx+1)/total_images*100:.1f}%): {os.path.basename(image_path)}")
            try:
                # initialize results for this image
                results[image_path] = {}
                
                # load original image
                input_tensor, original_image = self.load_image(image_path)
                
                # get model prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                # get prediction class and probability
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = torch.argmax(output, dim=1).item()
                
                print(f"  Original image prediction class: {pred_class}")
                
                # generate original image's IG explanation
                print(f"  Generating original image's IG explanation...")
                original_explanation = self.generate_ig(input_tensor)
                
                # generate multiple IG explanations for stability calculation
                print(f"  Generating multiple explanations for stability calculation...")
                original_stability_explanations = self.generate_multiple_igs(input_tensor)
                
                # process each corruption type
                corruption_count = 0
                for c_idx, corruption_type in enumerate(self.corruption_types):
                    results[image_path][corruption_type] = {"results": []}
                    print(f"  Processing corruption type [{c_idx+1}/{len(self.corruption_types)}]: {corruption_type}")
                    
                    # process each severity level
                    for severity in range(1, 6):
                        corruption_count += 1
                        progress = corruption_count / total_corruptions * 100
                        print(f"    Severity {severity}/5 - Current image progress: {corruption_count}/{total_corruptions} ({progress:.1f}%)")
                        
                        # apply corruption
                        corrupted_image = self.apply_corruption(original_image, corruption_type, severity)
                        corrupted_tensor = self.transform(corrupted_image).unsqueeze(0).to(self.device)
                        
                        # get prediction of corrupted image
                        with torch.no_grad():
                            corrupted_output = self.model(corrupted_tensor)
                            
                        corrupted_probs = torch.nn.functional.softmax(corrupted_output, dim=1).cpu().numpy()[0]
                        corrupted_pred_class = torch.argmax(corrupted_output, dim=1).item()
                        
                        # generate IG explanation of corrupted image
                        corrupted_explanation = self.generate_ig(corrupted_tensor)
                        
                        # generate multiple IG explanations for stability calculation
                        corrupted_stability_explanations = self.generate_multiple_igs(corrupted_tensor)
                        
                        # compute metrics
                        metrics = self.compute_metrics(
                            original_explanation, 
                            corrupted_explanation,
                            pred_class, 
                            corrupted_pred_class,
                            probs, 
                            corrupted_probs,
                            corrupted_stability_explanations
                        )
                        
                        # add severity level
                        metrics["severity"] = severity
                        
                        # save to results
                        results[image_path][corruption_type]["results"].append(metrics)
                        
                        # save visualization (if needed)
                        if save_viz and viz_dir:
                            viz_filename = f"{os.path.basename(image_path).split('.')[0]}_{corruption_type}_s{severity}.png"
                            viz_path = os.path.join(viz_dir, viz_filename)
                            self.save_visualization(
                                original_image, corrupted_image,
                                original_explanation, corrupted_explanation,
                                viz_path, corruption_type, severity
                            )
                
                # save temporary results after each image
                if temp_file:
                    with open(temp_file, 'w') as f:
                        json.dump(results, f)
                        
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (idx + 1)
                remaining_images = total_images - (idx + 1)
                est_remaining_time = avg_time_per_image * remaining_images
                
                image_time = time.time() - image_start_time
                
                print(f"\nImage [{idx+1}/{total_images}] processed, time: {image_time:.2f} seconds")
                print(f"Total progress: {idx+1}/{total_images} images ({(idx+1)/total_images*100:.1f}%)")
                print(f"Average time per image: {avg_time_per_image:.2f} seconds")
                print(f"Estimated remaining time: {est_remaining_time/60:.1f} minutes ({est_remaining_time/3600:.1f} hours)")
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
                
        # save final results
        with open(output_file, 'w') as f:
            json.dump(results, f)
            
        total_time = time.time() - start_time
        print(f"\nTest completed. Results saved to {output_file}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"Average time per image: {total_time/total_images:.1f} seconds")

def main():
    """Main function"""
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
    
    # create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.temp_file:
        os.makedirs(os.path.dirname(args.temp_file), exist_ok=True)
    
    # create test instance
    tester = IGRobustnessTest(model_type=args.model_type)
    
    # run test
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