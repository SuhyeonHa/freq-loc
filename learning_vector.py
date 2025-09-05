from pyexpat import features
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Tuple, List, Optional
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
import os
import warnings 
import torchvision
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
from diffusers.utils import logging

class Params:
    """Hyperparameters and configuration settings for FreqMark."""
    def __init__(self):
        # --- System & Paths ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_dir = '/mnt/nas5/suhyeon/datasets/DIV2K_train_HR'
        self.image_path = '/mnt/nas5/suhyeon/datasets/DIV2K_train_HR/0002.png'
        self.output_dir = '/mnt/nas5/suhyeon/projects/freq-loc/random_vec'
        self.cache_dir = '/mnt/nas5/suhyeon/caches'
        self.nsfw_list_path = './nsfw_list.txt'

        # --- Model Configurations ---
        self.vae_model_name = "stabilityai/stable-diffusion-2-1"
        self.vae_subfolder = "vae"
        self.dino_model_repo = 'facebookresearch/dinov2'
        self.dino_model_name = 'dinov2_vits14'
        
        # --- Image Size Parameters ---
        self.vae_image_size = 512
        self.dino_image_size = 224
        self.transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(self.dino_image_size),
        transforms.ToTensor(),
        ])

        # --- FreqMark Core Parameters ---
        self.message_bits = 48
        self.feature_dim = 384
        self.margin = 1.0
        # self.grid_size = 14
        # self.num_patches = self.grid_size*self.grid_size
        self.mask_percentage = 0.
        self.num_masks = 1

        # --- Optimization Parameters ---
        self.lr = 5e-5
        self.steps = 400
        self.epochs = 10
        self.lambda_p = 0.04#0.05
        self.lambda_i = 0.20#0.25

        # --- Robustness Parameters ---
        self.eps1_std = 0.25 
        self.eps2_std = 0.06
        self.lambda_reg = 0.01
        
        # --- Demo/Evaluation Parameters ---
        self.batch_size = 8
        self.num_workers = 2
        self.num_test_images = 800
        self.seed = 42

class FreqMarkRandom:
    def __init__(self, args):
        self.args = args

        # Initialize networks
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.args.device)
        self.image_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.args.device)
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained("sd-legacy/stable-diffusion-inpainting", torch_dtype=torch.float16, cache_dir=self.args.cache_dir).to(self.args.device)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained("sd-legacy/stable-diffusion-inpainting", cache_dir=self.args.cache_dir).to(self.args.device)
        # Freeze all networks
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Pre-define direction vectors
        self._init_direction_vectors()
    
        self.mu = self.args.margin      # Hinge loss margin
        
        # Noise parameters for robustness
        self.args.eps1_std = 0.25  # Latent noise
        self.args.eps2_std = 0.06  # Pixel noise

    def _init_direction_vectors(self) -> torch.Tensor:
        """Initialize direction vectors as learnable"""
        initial_vector = torch.randn(1, self.args.feature_dim, device=self.args.device)
        initial_vector = F.normalize(initial_vector, p=2, dim=1)
        self.direction_vectors = nn.Parameter(initial_vector)
    
    def _create_random_mask(self, img_pt, num_masks=1, mask_percentage=0.1, max_attempts=100):
        _, _, height, width = img_pt.shape
        mask_area = int(height * width * mask_percentage)
        masks = torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype)

        if mask_percentage >= 0.999:
            # Full mask for entire image
            return torch.ones((num_masks, 1, height, width), dtype=img_pt.dtype).to(img_pt.device)
        if mask_percentage < 0.001:
            return torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype).to(img_pt.device)

        for ii in range(num_masks):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1

                max_dim = int(mask_area ** 0.5)
                mask_width = random.randint(1, max_dim)
                mask_height = mask_area // mask_width

                # Allow broader aspect ratios for larger masks
                aspect_ratio = mask_width / mask_height if mask_height != 0 else 0
                if 0.25 <= aspect_ratio <= 4:  # Looser ratio constraint
                    if mask_height <= height and mask_width <= width:
                        x_start = random.randint(0, width - mask_width)
                        y_start = random.randint(0, height - mask_height)
                        overlap = False
                        for jj in range(ii):
                            if torch.sum(masks[jj, :, y_start:y_start + mask_height, x_start:x_start + mask_width]) > 0:
                                overlap = True
                                break
                        if not overlap:
                            masks[ii, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1
                            placed = True

            if not placed:
                # Fallback: just fill a central region if all attempts fail
                print(f"Warning: Failed to place mask {ii}, using fallback.")
                center_h = height // 2
                center_w = width // 2
                half_area = int((mask_area // 2) ** 0.5)
                h_half = min(center_h, half_area)
                w_half = min(center_w, half_area)
                masks[ii, :, center_h - h_half:center_h + h_half, center_w - w_half:center_w + w_half] = 1

        return masks.to(img_pt.device)

    def vae_recon(self, image: torch.Tensor, iter: int):
        """VAE reconstruction. Inputs are outputs are 512x512"""
        latent = self.vae.encode(2*image-1).latent_dist.sample()
        reconstructed = self.vae.decode(latent).sample
        reconstructed = (reconstructed + 1) / 2
        for _ in range(iter-1):
            latent = self.vae.encode(2*reconstructed-1).latent_dist.sample()
            reconstructed = self.vae.decode(latent).sample
            reconstructed = (reconstructed + 1) / 2
        return reconstructed

    def embed_watermark(self, original: torch.Tensor) -> torch.Tensor:
        """
        Embed watermark in image using latent frequency space optimization
        
        Args:
            image: Input image tensor [B, C, H, W]
            message: Binary message {-1, 1} [B, message_bits]
        
        Returns:
            Watermarked image tensor
        """
        original = original.to(self.args.device)
        # message = message.to(self.device)
        
        # Step 1: Encode image to latent space
        image = F.interpolate(original, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)
        latent = self.vae.encode(2*image-1).latent_dist.sample() # [-1, 1], [B,4,64,64]
        
        # Step 2: Transform to frequency domain
        latent_fft = torch.fft.fft2(latent, dim=(-2, -1))
        
        # Step 3: Initialize perturbation (trainable parameter)
        delta_m = torch.rand_like(latent_fft)
        # mask = self._create_random_mask(image, num_masks=1, mask_percentage=1.0)
        image = F.interpolate(original, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)

        final_fft = latent_fft + delta_m # using fixed noise
        final_latent = torch.fft.ifft2(final_fft, dim=(-2, -1)).real
        final_watermarked = self.vae.decode(final_latent).sample
        final_watermarked = (final_watermarked + 1) / 2
        
        return final_watermarked.detach()
    
    def decode_watermark(self, watermarked_image: torch.Tensor) -> torch.Tensor:
        """
        Decode watermark from image using pre-trained image encoder
        
        Args:
            watermarked_image: Watermarked image tensor [B, C, H, W]
        
        Returns:
            Decoded message {-1, 1} [B, message_bits]
        """
        watermarked_image = watermarked_image.to(self.args.device)
        
        with torch.no_grad():
            # Extract features using image encoder
            # features = self.image_encoder(watermarked_image) # [1, 256, 384]
            features = self.image_encoder.get_intermediate_layers(watermarked_image)[0] # [1, 256, 384]
            
        # Compute dot products with direction vectors
        dot_products = torch.matmul(features, self.direction_vectors.T) # [1, 256, 384]*[1, 384, 256] -> [1, 256, 1]
        
        B = dot_products.shape[0]
        H = W = int(dot_products.shape[1] ** 0.5)
        grid = dot_products.squeeze(-1).view(B, H, W) # [1, 256, 1] -> [1, 16, 16]
        grid = grid.unsqueeze(1)
        grid = F.interpolate(grid, size=self.args.dino_image_size, mode='bilinear', align_corners=False)
        return grid
    
    def _message_loss(self, watermarked_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Hinge loss for message embedding (Equation 7)"""
        features = self.image_encoder(watermarked_image)
        dot_products = torch.matmul(features, self.direction_vectors.T)
        
        # Hinge loss with margin
        projections = dot_products * message
        loss = torch.clamp(self.mu - projections, min=0).mean()
        
        return loss
    
    def _mask_loss(self, watermarked_image: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss based on patch-wise watermark detection to enforce a global watermark presence.
        The ground truth mask is implicitly all-ones, meaning the loss is minimized when all patches
        correctly embed the watermark.
        """
        image_for_dino = F.interpolate(watermarked_image, 
                                       size=(self.args.dino_image_size, self.args.dino_image_size), 
                                       mode="bilinear", align_corners=False)

        features = self.image_encoder.get_intermediate_layers(image_for_dino, n=1)[0] # [B, Num_Patches, Feature_Dim]
        dot_products = torch.matmul(features, self.direction_vectors.T)
        B = dot_products.shape[0]
        H = W = int(dot_products.shape[1] ** 0.5)
        grid = dot_products.view(B, H, W).unsqueeze(0)
        # grid = dot_products.view(self.args.grid_size, self.args.grid_size).unsqueeze(0).unsqueeze(0) # [1, 256, 1] -> [1, 1, 14, 14]
        grid = F.interpolate(grid, size=self.args.dino_image_size, mode='bilinear', align_corners=False) # [B, Num_Patches, Feature_Dim]*[B, Feature_Dim, 1] = [B, Num_Patches, 1]
        loss = F.binary_cross_entropy_with_logits(grid, gt_mask)
        return loss
    
    def _psnr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative PSNR loss (Equation 5)"""
        mse = F.mse_loss(pred, target)
        psnr = -10 * torch.log10(mse + 1e-8)
        return -psnr  # Negative for minimization
    
    def _lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simplified LPIPS-like loss"""
        # Simplified perceptual loss using L2 in feature space
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        return F.mse_loss(pred_gray, target_gray)
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between images"""
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return 100.0
        return 20 * np.log10(1.0 / np.sqrt(mse))



def load_images_from_path(path: str, num_images=None, transform=None, nsfw_list_path=None) -> torch.Tensor:
    images = []
    file_names = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    nsfw_filenames = set()
    if nsfw_list_path and os.path.exists(nsfw_list_path):
        with open(nsfw_list_path, 'r') as f:
            nsfw_filenames = {line.strip() for line in f if line.strip()}
    
    # file_list = sorted(os.listdir(path))
    # print(f"Scanning {len(file_list)} files in '{path}'...")
    
    for filename in sorted(os.listdir(path)):
        if filename in nsfw_filenames:
            # print(f"Skipping NSFW image: {filename}")
            continue

        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(path, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(transform(image))
                file_names.append(filename)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")

    if not images:
        raise FileNotFoundError(f"No valid images found in the specified path: {path}")
    
    if num_images:
        return torch.stack(images[:num_images]), file_names[:num_images]
    return torch.stack(images), file_names


def load_image(path: str, transform=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def visualize_results(original_images: torch.Tensor, watermarked_images: torch.Tensor, 
                     results: dict, save_path: str = None):
    """Visualize watermarking results"""
    
    # Plot image quality comparison
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('FreqMark Watermarking Results', fontsize=16)
    
    for i in range(4):
        # Original image
        axes[0, i].imshow(original_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')
        
        # Watermarked image
        axes[1, i].imshow(watermarked_images[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f'Watermarked Image {i+1}')
        axes[1, i].axis('off')
        
        # Difference (amplified by 10 as in paper)
        diff = torch.abs(watermarked_images[i] - original_images[i]) * 10
        diff = torch.clamp(diff, 0, 1)
        axes[2, i].imshow(diff.permute(1, 2, 0).cpu().numpy())
        axes[2, i].set_title(f'Difference ×10')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_visual.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Image quality metrics
    quality_metrics = ['psnr', 'ssim']
    quality_values = [results[metric] for metric in quality_metrics]
    
    ax1.bar(quality_metrics, quality_values)
    ax1.set_title('Image Quality Metrics') 
    ax1.set_ylabel('Value')
    
    # Add text annotations
    for i, v in enumerate(quality_values):
        ax1.text(i, v + max(quality_values)*0.01, f'{v:.3f}', 
                ha='center', va='bottom')
    
    # Bit accuracy under different attacks
    attack_names = [key.replace('bit_acc_', '').replace('_', ' ').title() 
                   for key in results.keys() if key.startswith('bit_acc_')]
    bit_accuracies = [results[key] for key in results.keys() if key.startswith('bit_acc_')]
    
    ax2.bar(attack_names, bit_accuracies)
    ax2.set_title('Bit Accuracy Under Different Attacks')
    ax2.set_ylabel('Bit Accuracy')
    ax2.set_ylim([0, 1.1])
    plt.xticks(rotation=45)
    
    # Add text annotations
    for i, v in enumerate(bit_accuracies):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def learn_weight_vec(args=None):
    images, filenames = load_images_from_path(args.image_dir, transform=args.transform, nsfw_list_path=args.nsfw_list_path)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded {len(images)} images.")
    freqmark = FreqMarkRandom(args=args)
    freqmark.pipe.set_progress_bar_config(disable=True)
    to_tensor = transforms.ToTensor()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    optimizer = optim.Adam([freqmark.direction_vectors], lr=args.lr)
    psnrs = []  
    pbar = tqdm(dataloader)

    best_loss = float('inf')
    best_vector = None

    for batch_idx, batch in enumerate(pbar):
        # batch = images[i:i+1].to(args.device)
        batch = batch[0].to(args.device)
        # filename = filenames[i]

        start_idx = batch_idx * args.batch_size
        current_filenames = filenames[start_idx : start_idx + args.batch_size]

        # print("Embedding watermarks...")
        watermarked_512 = freqmark.embed_watermark(batch)

        inp_list = []

        watermarked = F.interpolate(watermarked_512, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)

        for i in range(watermarked_512.size(0)):
            current_filename = current_filenames[i]
            with torch.no_grad():
                img_norm, min_norm, max_norm = norm_tensor(watermarked_512[i:i+1])
                zero_mask = freqmark._create_random_mask(img_norm, num_masks=args.num_masks, mask_percentage=args.mask_percentage)
                pipe_output = freqmark.pipe(prompt="", image=img_norm, mask_image=zero_mask, generator=generator)

                if pipe_output.nsfw_content_detected[0]:
                    print(f"\n🚨 NSFW content detected in file: {current_filename}")

                img_edit_pil = pipe_output.images[0]
                img_edit = to_tensor(img_edit_pil)
                img_edit = img_edit.unsqueeze(0).to(args.device)
                img_edit = denorm_tensor(img_edit, min_norm, max_norm)  # [1, 3, H, W]
            # img_edit_m = img_edit * mask + watermarked * (1 - mask)
                img_edit = F.interpolate(img_edit, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
                inp_list.append(img_edit)

        inp_batch = torch.cat(inp_list, dim=0)
        ori_features = freqmark.image_encoder.get_intermediate_layers(batch)[0]
        wm_features = freqmark.image_encoder.get_intermediate_layers(watermarked)[0] # [1, 256, 384]
        inp_features = freqmark.image_encoder.get_intermediate_layers(inp_batch)[0] # [1, 256, 384]

        ori_dots = torch.matmul(ori_features, freqmark.direction_vectors.T)
        wm_dots = torch.matmul(wm_features, freqmark.direction_vectors.T)
        inp_dots = torch.matmul(inp_features, freqmark.direction_vectors.T)

        loss_diff = F.mse_loss(wm_dots, inp_dots)
        loss_var = -torch.var(ori_dots)

        loss = loss_diff + loss_var

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            freqmark.direction_vectors.data = F.normalize(freqmark.direction_vectors.data, p=2, dim=1)

        # with torch.no_grad():
        #     freqmark.direction_vectors.data = F.normalize(freqmark.direction_vectors.data, p=2, dim=1)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_vector = freqmark.direction_vectors.detach().clone()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}", "Best Loss": f"{best_loss:.6f}"})
        else:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        for i in range(inp_batch.size(0)):
            psnr = freqmark._compute_psnr(
                watermarked[i:i+1],
                batch[i:i+1]
            )
        psnrs.append(psnr)

        print((f"Loss: {loss.item():.6f}, Best Loss: {best_loss:.6f}"))

    print(f"PSNR: {np.mean(psnrs):.2f} dB")
    torch.save(best_vector, "learned_directional_vector_best.pt")
    print(f"Learned vector saved")
    return best_vector

def norm_tensor(tensor):
    t = tensor.clone().detach()
    
    min_val = t.min()
    max_val = t.max()

    tensor_norm = (tensor - min_val) / (max_val - min_val + 1e-6)

    # print(f"Tensor normalized: min={tensor_norm.min()}, max={tensor_norm.max()}")
    
    return tensor_norm, min_val, max_val

def denorm_tensor(tensor, original_min=None, original_max=None):
    t = tensor.clone().detach()

    return t * (original_max - original_min) + original_min

if __name__ == "__main__":
    print("Learning a weight vector insensitive to distorted watermarks")

    args = Params()
    os.makedirs(args.output_dir, exist_ok=True)

    vector = learn_weight_vec(args=args)

    print(vector)