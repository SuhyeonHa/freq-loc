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
from diffusers import AutoencoderKL
import os
import warnings 
import torchvision
import random
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Denormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean_rev = -mean / std
        self.std_rev = 1 / std
        super().__init__(mean=self.mean_rev, std=self.std_rev)

norm_dino = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
denorm_dino = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

class Params:
    """Hyperparameters and configuration settings for FreqMark."""
    def __init__(self):
        # --- System & Paths ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_path = '/mnt/nas5/suhyeon/datasets/DIV2K_train_HR/0002.png'
        self.output_dir = '/mnt/nas5/suhyeon/projects/freq-loc/secret_code'

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
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # --- FreqMark Core Parameters ---
        self.message_bits = 48
        self.feature_dim = 384 # DINOv2 small
        self.margin = 1.0
        # self.grid_size = 14
        # self.num_patches = self.grid_size*self.grid_size
        self.mask_percentage = 0.3
        self.num_masks = 1

        # --- Optimization Parameters ---
        self.lr = 2.0
        self.steps = 400
        self.lambda_p = 0.1#0.05
        self.lambda_i = 0.25#0.25

        # --- Robustness Parameters --- 
        # self.eps1_std = [0.2, 0.6] # Latent noise
        # self.eps2_std = [0.06, 0.2] # Pixel noise
        # self.eps1_std = 0.25
        # self.eps2_std = 0.06
        # self.eps0_std = 0.01
        self.eps0_std = [0.0, 0.8] # Latent noise
        self.eps1_std = [1.5, 3.0] # Latent noise
        self.eps2_std = 0.06
        self.lambda_reg = 0.01

        self.dino_layer = 1
        
        # --- Demo/Evaluation Parameters ---
        self.batch_size = 1
        self.num_test_images = 1

class FreqMark:
    def __init__(self, args):
        self.args = args

        # Initialize networks
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.args.device)
        self.image_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.args.device)
        # Freeze all networks
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Pre-define direction vectors
        # self.direction_vectors = torch.load('./sensitive_vec.pt')
        self.direction_vectors = torch.randn(self.args.feature_dim)
        self.direction_vectors = F.normalize(self.direction_vectors, p=2, dim=0)
        self.direction_vectors = self.direction_vectors.unsqueeze(0).to(self.args.device)
        torch.save(self.direction_vectors, './random_vec.pt')
    
        self.mu = self.args.margin      # Hinge loss margin
        self.num_patches = (self.args.dino_image_size // 14) ** 2
        self.secret_key = torch.randint(0, 2, (1, self.num_patches, 1), device=self.args.device) * 2 - 1 # {-1, 1}

        torch.save(self.secret_key, './secret_key.pt')


    # def _init_direction_vectors(self) -> torch.Tensor:
    #     """Initialize direction vectors as described in paper"""
    #     # binary bit for each patch
    #     # vectors = torch.zeros(1, self.args.feature_dim)
    #     # for i in range(1):
    #     #     vectors[i, self.args.feature_dim-1] = 1.0  # One-hot encoding
    #     # print(vectors)
    #     # return vectors.to(self.args.device)
    #     # random_vector = torch.randn(self.args.feature_dim, device=self.args.device)
    #     # normalized_vector = F.normalize(random_vector, p=2, dim=0)
    #     # return normalized_vector.unsqueeze(0)
    #     self.direction_vectors = torch.load('./insensitive_vec.pt')

    def _create_random_mask(self, img_pt, num_masks=1, mask_percentage=0.1, max_attempts=100):
        _, _, height, width = img_pt.shape
        mask_area = int(height * width * mask_percentage)
        masks = torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype)

        if mask_percentage >= 0.999:
            # Full mask for entire image
            return torch.ones((num_masks, 1, height, width), dtype=img_pt.dtype).to(img_pt.device)

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

    def embed_watermark(self, original: torch.Tensor, img_size: int) -> torch.Tensor:
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
        delta_m = torch.zeros_like(latent_fft, requires_grad=True)
        optimizer = optim.Adam([delta_m], lr=self.args.lr)

        # Training loop
        # for step in range(self.args.steps):
        for step in tqdm(range(self.args.steps), desc="Embedding Watermark"):
            optimizer.zero_grad()

            mask = self._create_random_mask(image, num_masks=1, mask_percentage=self.args.mask_percentage)
            mask = mask.to(self.args.device)

            if random.random() < 0.5:
                mask = 1 - mask

            image = F.interpolate(original, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=(self.args.vae_image_size, self.args.vae_image_size), mode="bilinear", align_corners=False)
            
            perturbed_fft = latent_fft + delta_m
            
            perturbed_latent = torch.fft.ifft2(perturbed_fft, dim=(-2, -1)).real
            
            watermarked_image = self.vae.decode(perturbed_latent).sample
            watermarked_image = (watermarked_image + 1) / 2
            
            masked = watermarked_image * mask + (1 - mask) * image

            # inpaint
            latent_mask = F.interpolate(mask, size=(64, 64), mode="bilinear", align_corners=False)
            
            std_val_0 = random.uniform(self.args.eps0_std[0], self.args.eps0_std[1])
            eps0 = torch.randn_like(perturbed_latent) * std_val_0

            perturbed_latent_1 = (perturbed_latent + eps0)*latent_mask + perturbed_latent*(1-latent_mask)

            watermarked_image_1 = self.vae.decode(perturbed_latent_1).sample
            masked_1 = (watermarked_image_1 + 1) / 2
            masked_1 = masked_1 * mask + (1 - mask) * image ################################

            # Compute losses
            image = F.interpolate(original, size=(img_size, img_size), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=(img_size, img_size), mode="bilinear", align_corners=False)
            masked = F.interpolate(masked, size=(img_size, img_size), mode="bilinear", align_corners=False)
            masked_1 = F.interpolate(masked_1, size=(img_size, img_size), mode="bilinear", align_corners=False)
            # masked_2 = F.interpolate(masked_2, size=(img_size, img_size), mode="bilinear", align_corners=False)
            # masked_3 = F.interpolate(masked_3, size=(img_size, img_size), mode="bilinear", align_corners=False)
            watermarked_image = F.interpolate(watermarked_image, size=(img_size, img_size), mode="bilinear", align_corners=False)
            
            watermarked_image = norm_dino(watermarked_image)
            masked = norm_dino(masked)
            masked_1 = norm_dino(masked_1)

            loss_m = self._mask_loss(masked, mask)
            loss_d = self._dice_loss(masked, mask)
            loss_m1 = self._mask_loss(masked_1, mask)
            loss_d1 = self._dice_loss(masked_1, mask)

            gt_mask_patch = F.avg_pool2d(mask, kernel_size=14, stride=14).view(1, 16*16, 1)

            features = self.image_encoder.get_intermediate_layers(watermarked_image)[0]
            dot_products = torch.matmul(features, self.direction_vectors.T)
            loss_auth = self._auth_loss(dot_products, self.secret_key, gt_mask_patch)

            features_1 = self.image_encoder.get_intermediate_layers(masked_1)[0]
            dot_products_1 = torch.matmul(features_1, self.direction_vectors.T)
            loss_auth_1 = self._auth_loss(dot_products_1, self.secret_key, gt_mask_patch)


            watermarked_image = denorm_dino(watermarked_image)
            masked = denorm_dino(masked)
            masked_1 = denorm_dino(masked_1)

            loss_psnr = self._psnr_loss(watermarked_image, image)
            loss_lpips = self._lpips_loss(watermarked_image, image)

            # if step < 200:
            #     auth_loss_weight = 1.0
            #     shape_loss_weight = 0.0
            # else:  # Stage 2: Key Embedding
            #     auth_loss_weight = 1.0
            #     shape_loss_weight = 1.0

            auth_loss_weight = 1.0
            shape_loss_weight = 1.0

            # Combined loss (Equation 10 from paper)
            total_loss = auth_loss_weight * (loss_auth + loss_auth_1) + \
                         shape_loss_weight * (loss_m + loss_m1 + loss_d + loss_d1) + \
                         self.args.lambda_p * loss_psnr + \
                         self.args.lambda_i * loss_lpips
            
            total_loss.backward()
            optimizer.step()

            if step == 0 or (step+1) % 100 == 0:
                psnr_val = self._compute_psnr(watermarked_image, image)
                print(f"Step {step+1}, Loss: {total_loss.item():.4f}, PSNR: {psnr_val:.2f}")
                print(f"Mask Loss: {loss_m.item():.4f}, DICE Loss: {loss_d.item():.4f}")
                print(f"Mask1 Loss: {(loss_m1).item():.4f}, DICE1 Loss: {loss_d1.item():.4f}")
                print(f"Auth Loss: {(loss_auth).item():.4f}, Auth1 Loss: {loss_auth_1.item():.4f}")
                print(f"PSNR Loss: {loss_psnr.item():.4f}, LPIPS Loss: {loss_lpips.item():.4f}")

        # Final watermarked image
        final_fft = latent_fft + delta_m
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
            watermarked_image = norm_dino(watermarked_image)
            features = self.image_encoder.get_intermediate_layers(watermarked_image)[0] # [1, 256, 384]
            
            # Compute dot products with direction vectors
            dot_products = torch.matmul(features, self.direction_vectors.T) # [1, 256, 384]*[1, 384, 256] -> [1, 256, 1]
            agreement_scores = dot_products * self.secret_key

            B = dot_products.shape[0]
            H = W = int(dot_products.shape[1] ** 0.5)
            grid = agreement_scores.view(B, H, W).unsqueeze(0) # [1, 256, 1] -> [1, 1, 16, 16]
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
        # image_for_dino = F.interpolate(watermarked_image, 
        #                                size=(self.args.dino_image_size, self.args.dino_image_size), 
        #                                mode="bilinear", align_corners=False)

        features = self.image_encoder.get_intermediate_layers(watermarked_image)[0] # [B, Num_Patches, Feature_Dim]
        dot_products = torch.matmul(features, self.direction_vectors.T)
        agreement_scores = dot_products * self.secret_key
        B = dot_products.shape[0]
        H = W = int(dot_products.shape[1] ** 0.5)
        grid = agreement_scores.view(B, H, W).unsqueeze(0)
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
    
    def _dice_loss(self, watermarked_image, gt_mask, smooth=1e-5):
        image_for_dino = F.interpolate(watermarked_image, 
                                       size=(self.args.dino_image_size, self.args.dino_image_size), 
                                       mode="bilinear", align_corners=False)

        features = self.image_encoder.get_intermediate_layers(image_for_dino)[0] # [B, Num_Patches, Feature_Dim]
        dot_products = torch.matmul(features, self.direction_vectors.T)
        agreement_scores = dot_products * self.secret_key
        B = dot_products.shape[0]
        H = W = int(dot_products.shape[1] ** 0.5)
        grid = agreement_scores.view(B, H, W).unsqueeze(0)
        # grid = dot_products.view(self.args.grid_size, self.args.grid_size).unsqueeze(0).unsqueeze(0) # [1, 256, 1] -> [1, 1, 14, 14]
        grid = F.interpolate(grid, size=self.args.dino_image_size, mode='bilinear', align_corners=False) # [B, Num_Patches, Feature_Dim]*[B, Feature_Dim, 1] = [B, Num_Patches, 1]
        
        pred = torch.sigmoid(grid) # Logits to probabilities

        # Flatten label and prediction tensors
        pred = pred.view(-1)
        target = gt_mask.view(-1)
        
        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice_coeff

    # def _auth_loss(self, dot_products, secret_key, gt_mask, margin=1.5):
    #     projections = dot_products * secret_key
    #     loss = torch.clamp(margin - projections, min=0)
    #     loss = (loss * gt_mask).mean()
    #     return loss
    
    def _auth_loss(self, dot_products, secret_key, gt_mask):
        TARGET_SCORE = 4.0
        target_scores = secret_key * TARGET_SCORE
        # loss = F.mse_loss(dot_products, target_scores, reduction='none')
        loss = F.l1_loss(dot_products, target_scores, reduction='none')
        loss = (loss * gt_mask).mean()
        return loss
    
    def compute_bit_accuracy(self, original_message: torch.Tensor, 
                           decoded_message: torch.Tensor) -> float:
        """Compute bit accuracy between original and decoded messages"""
        matches = (original_message == decoded_message).float()
        return matches.mean().item()

class WatermarkEvaluator:
    """Evaluation suite for watermark robustness testing"""
    
    def __init__(self):
        pass
    
    def brightness_attack(self, image: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
        """Apply brightness change attack"""
        return torch.clamp(image + factor, 0, 1)
    
    def contrast_attack(self, image: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
        """Apply contrast change attack"""
        mean_val = image.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((image - mean_val) * factor + mean_val, 0, 1)
    
    def gaussian_noise_attack(self, image: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        """Apply Gaussian noise attack"""
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image + noise, 0, 1)
    
    def gaussian_blur_attack(self, image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply Gaussian blur attack"""
        # Simplified blur using average pooling
        padding = kernel_size // 2
        blurred = F.avg_pool2d(image, kernel_size, stride=1, padding=padding)
        return blurred
    
    def jpeg_compression_attack(self, image: torch.Tensor, quality: int = 50) -> torch.Tensor:
        """Simulate JPEG compression attack"""
        # Simplified JPEG simulation using quantization
        image_quantized = torch.round(image * 255) / 255
        noise = torch.randn_like(image) * (100 - quality) / 1000
        return torch.clamp(image_quantized + noise, 0, 1)
    
    def vae_regeneration_attack(self, image: torch.Tensor, vae) -> torch.Tensor:
        """VAE regeneration attack"""
        with torch.no_grad():
            image = F.interpolate(image, size=(args.vae_image_size, args.vae_image_size), mode="bilinear", align_corners=False)
            latent = vae.encode(2*image-1).latent_dist.sample()
            # Add compression noise
            noise = torch.randn_like(latent) * 0.1
            reconstructed = vae.decode(latent + noise).sample
            reconstructed = (reconstructed + 1) / 2
            reconstructed = F.interpolate(reconstructed, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
        return reconstructed

def load_images_from_path(path: str, num_images: int = 1, transform=None) -> torch.Tensor:
    images = []
    file_names = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    for filename in sorted(os.listdir(path)):
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

    return torch.stack(images[1:1+num_images]), file_names[1:1+num_images]


def load_image(path: str, transform=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

def generate_random_messages(batch_size: int, message_bits: int) -> torch.Tensor:
    """Generate random binary messages"""
    return torch.randint(0, 2, (batch_size, message_bits)) * 2 - 1  # Convert to {-1, 1}

def evaluate_robustness(freqmark: FreqMark, test_images: torch.Tensor, 
                       test_messages: torch.Tensor) -> dict:
    """Comprehensive robustness evaluation"""
    evaluator = WatermarkEvaluator()
    results = {}
    
    # Embed watermarks
    print("Embedding watermarks...")
    watermarked_images = []
    for i in range(0, len(test_images), 4):  # Process in small batches
        batch = test_images[i:i+4]
        msg_batch = test_messages[i:i+4]
        watermarked_batch = freqmark.embed_watermark(batch, msg_batch)
        watermarked_images.append(watermarked_batch)
    
    watermarked_images = torch.cat(watermarked_images, dim=0)
    
    # Calculate image quality metrics
    psnr_values = []
    ssim_values = []
    
    for i in range(len(test_images)):
        psnr = freqmark._compute_psnr(watermarked_images[i:i+1], test_images[i:i+1])
        psnr_values.append(psnr)
        
        # Simplified SSIM calculation
        ssim = torch.cosine_similarity(
            watermarked_images[i].flatten(), 
            test_images[i].flatten(), 
            dim=0
        ).item()
        ssim_values.append(ssim)
    
    results['psnr'] = np.mean(psnr_values)
    results['ssim'] = np.mean(ssim_values)
    
    # Test robustness under various attacks
    attacks = {
        'None': lambda x: x,
        'Brightness': evaluator.brightness_attack,
        'Contrast': evaluator.contrast_attack,
        'JPEG': evaluator.jpeg_compression_attack,
        'Gaussian Blur': evaluator.gaussian_blur_attack,
        'Gaussian Noise': evaluator.gaussian_noise_attack,
        'VAE Regen': lambda x: evaluator.vae_regeneration_attack(x, vae=freqmark.vae)
    }
    
    print("Testing robustness under attacks...")
    for attack_name, attack_func in attacks.items():
        print(f"Testing {attack_name} attack...")
        attacked_images = attack_func(watermarked_images)
        
        # Decode messages from attacked images
        bit_accuracies = []
        for i in range(0, len(attacked_images), 4):
            batch = attacked_images[i:i+4]
            msg_batch = test_messages[i:i+4]
            
            decoded = freqmark.decode_watermark(batch)
            accuracy = freqmark.compute_bit_accuracy(msg_batch.device(), decoded)
            bit_accuracies.append(accuracy)
        
        results[f'bit_acc_{attack_name.lower().replace(" ", "_")}'] = np.mean(bit_accuracies)
    
    return results

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

def compute_false_positive_rate(k: int, tau: int) -> float:
    """Compute FPR using formula from paper (Equation 1)"""
    from scipy.special import betainc
    fpr = 0.5 * betainc(tau + 1, k - tau, 0.5)
    return fpr

def run_freqmark_demo(args=None):
    """Run complete FreqMark demonstration"""
    print("=== FreqMark Implementation Demo ===")
    
    print(f"Test setup: {args.num_test_images} images, {args.dino_image_size}x{args.dino_image_size}, {args.message_bits} bits")
    
    # Initialize FreqMark
    freqmark = FreqMark(args=args)
    all_psnrs = []
    # np.save(f'random_vecs', freqmark.direction_vectors.cpu().numpy())

    # Create test dataset
    print("Generating test dataset...")
    # test_images, filenames = load_images_from_path('/mnt/nas5/suhyeon/datasets/DIV2K_train_HR', args.num_test_images, transform=args.transform)
    test_images = load_image(args.image_path, transform=args.transform)

    for i in tqdm(range(args.num_test_images), desc="Embedding Watermarks"):
    # for i in range(args.num_test_images):
        test_batch = test_images[i:i+1].to(args.device)
        # filename = filenames[i]
        filename = '0002.png'

        # Embed watermarks
        print("Embedding watermarks...")
        watermarked_batch = freqmark.embed_watermark(test_batch, img_size=args.dino_image_size)

        # Decode watermarks
        print("Decoding watermarks...")
        watermarked_batch = F.interpolate(watermarked_batch, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
        decoded_batch = freqmark.decode_watermark(watermarked_batch)

        psnr = freqmark._compute_psnr(test_batch, watermarked_batch)
        all_psnrs.append(psnr)

        save_path = os.path.join(args.output_dir, f"{filename}")
        torchvision.utils.save_image(watermarked_batch, save_path)

        predicted_mask = freqmark.decode_watermark(watermarked_batch)
        mask_save_path = os.path.join(args.output_dir, "mask", f"pred_{filename}")
        torchvision.utils.save_image(predicted_mask, mask_save_path)

        # save comparison figures
        # fig, axes = plt.subplots(3, args.batch_size, figsize=(16, 9), squeeze=False)
        # fig.suptitle('FreqMark Watermarking Examples', fontsize=14)
        
        # for i in range(args.batch_size):
        #     # Original
        #     axes[0, i].imshow(test_batch[i].permute(1, 2, 0).cpu().numpy())
        #     axes[0, i].set_title(f'Original {i+1}')
        #     axes[0, i].axis('off')
            
        #     # Watermarked
        #     axes[1, i].imshow(watermarked_batch[i].permute(1, 2, 0).cpu().numpy())
        #     axes[1, i].set_title(f'Watermarked {i+1}')
        #     axes[1, i].axis('off')
            
        #     # Difference (×10)
        #     diff = torch.abs(watermarked_batch[i] - test_batch[i]) * 10
        #     diff = torch.clamp(diff, 0, 1)
        #     axes[2, i].imshow(diff.permute(1, 2, 0).cpu().numpy())
        #     axes[2, i].set_title('Difference ×10')
        #     axes[2, i].axis('off')
        
        # # plt.tight_layout()
        # plt.savefig(f"freqloc_comparison_{filename}.png", dpi=300, bbox_inches='tight')

    # Calculate metrics
    avg_psnr = np.mean(all_psnrs)
    
    print("\n=== Processing Complete ===")
    print(f"Processed {args.num_test_images} images.")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Test robustness
    # evaluator = WatermarkEvaluator()
    
    # # Test individual attacks
    # attacks_results = {}
    
    # print("\nTesting robustness under attacks:")
    
    # Gaussian noise attack
    # attacked = evaluator.gaussian_noise_attack(watermarked_batch)
    # decoded = freqmark.decode_watermark(attacked)
    # attacks_results['gaussian_noise'] = freqmark.compute_bit_accuracy(msg_batch, decoded)
    # print(f"Gaussian Noise: {attacks_results['gaussian_noise']:.3f}")
    
    # # Brightness attack
    # attacked = evaluator.brightness_attack(watermarked_batch)
    # decoded = freqmark.decode_watermark(attacked)
    # attacks_results['brightness'] = freqmark.compute_bit_accuracy(msg_batch, decoded)
    # print(f"Brightness: {attacks_results['brightness']:.3f}")
    
    # # VAE regeneration attack
    # attacked = evaluator.vae_regeneration_attack(watermarked_batch, vae=freqmark.vae)
    # decoded = freqmark.decode_watermark(attacked)
    # attacks_results['vae_regen'] = freqmark.compute_bit_accuracy(msg_batch, decoded)
    # print(f"VAE Regeneration: {attacks_results['vae_regen']:.3f}")
    
    # Visualize results
    # print("\nGenerating visualizations...")
    
    # Show original vs watermarked comparison
    
    # # Performance comparison plot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # # Image quality
    # ax1.bar(['PSNR (dB)', 'Bit Acc (Clean)'], [avg_psnr, bit_accuracy])
    # ax1.set_title('Image Quality & Clean Accuracy')
    # ax1.set_ylim([0, max(avg_psnr, 1) * 1.1])
    
    # for i, v in enumerate([avg_psnr, bit_accuracy]):
    #     ax1.text(i, v + 1, f'{v:.3f}', ha='center', va='bottom')
    
    # # Attack robustness
    # attack_names = list(attacks_results.keys())
    # attack_values = list(attacks_results.values())
    
    # ax2.bar([name.replace('_', ' ').title() for name in attack_names], attack_values)
    # ax2.set_title('Robustness Under Attacks')
    # ax2.set_ylabel('Bit Accuracy')
    # ax2.set_ylim([0, 1.1])
    
    # for i, v in enumerate(attack_values):
    #     ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # plt.tight_layout()
    # plt.savefig("freqmark_attacks.png", dpi=300, bbox_inches='tight')

    # # Calculate theoretical FPR
    # tau = int(args.message_bits * 0.8)  # 80% threshold
    # try:
    #     fpr = compute_false_positive_rate(args.message_bits, tau)
    #     print(f"\nTheoretical Analysis:")
    #     print(f"For {args.message_bits} bits with threshold τ={tau}:")
    #     print(f"False Positive Rate: {fpr:.2e}")
    # except ImportError:
    #     print("scipy not available for FPR calculation")
    
    # # Summary results
    # print(f"\n=== Summary Results ===")
    # print(f"Image Quality:")
    # print(f"  PSNR: {avg_psnr:.2f} dB")
    # print(f"  Bit Accuracy (clean): {bit_accuracy:.3f}")
    # print(f"\nRobustness:")
    # for attack, acc in attacks_results.items():
    #     print(f"  {attack.replace('_', ' ').title()}: {acc:.3f}")
    
    return {
        'psnr': avg_psnr,
        # 'bit_accuracy_clean': bit_accuracy,
        # **attacks_results,
        'watermarked': watermarked_batch,
        'test_images': test_batch,
        'predicted': decoded_batch
        # 'messages': msg_batch
    }

def compare_with_paper_results(results: dict):
    """Compare implementation results with paper benchmarks"""
    print("\n=== Comparison with Paper Results ===")
    
    # Paper results for FreqMark (Table 1)
    paper_results = {
        'psnr': 31.27,
        'ssim': 0.857,
        'bit_acc_none': 1.000,
        'bit_acc_brightness': 0.995,
        'bit_acc_contrast': 1.000,
        'bit_acc_jpeg': 0.991,
        'bit_acc_gaussian_blur': 1.000,
        'bit_acc_gaussian_noise': 0.939,
        'bit_acc_vae_regen': 0.938
    }
    
    print("Metric\t\tPaper\t\tOurs\t\tDiff")
    print("-" * 50)
    
    for metric in paper_results:
        if metric in results:
            paper_val = paper_results[metric]
            our_val = results[metric]
            diff = our_val - paper_val
            print(f"{metric:15s}\t{paper_val:.3f}\t\t{our_val:.3f}\t\t{diff:+.3f}")
    
    # Calculate average bit accuracy
    bit_acc_keys = [k for k in results.keys() if k.startswith('bit_acc_')]
    if bit_acc_keys:
        avg_bit_acc = np.mean([results[k] for k in bit_acc_keys])
        paper_avg = 0.973  # From paper
        print(f"\nAverage Bit Accuracy:")
        print(f"Paper: {paper_avg:.3f}, Ours: {avg_bit_acc:.3f}, Diff: {avg_bit_acc-paper_avg:+.3f}")

def analyze_frequency_domain_advantage(args=None):
    """Analyze why frequency domain optimization works better"""
    print("\n=== Frequency Domain Analysis ===")
    
    # Create a simple test case
    test_images = load_image(args.image_path, transform=args.transform)
    freqmark = FreqMark(args=args)
    
    # Test different optimization spaces
    spaces_to_test = {
        'Pixel Space': 'pixel',
        'Latent Space': 'latent', 
        'Latent Frequency Space': 'latent_freq'
    }
    
    results_by_space = {}
    
    for space_name, space_type in spaces_to_test.items():
        print(f"\nTesting optimization in {space_name}...")
        
        # Simple watermark embedding for comparison
        original = test_images.clone().to(args.device)
        message = generate_random_messages(args.num_test_images, args.message_bits).to(args.device)

        original = F.interpolate(original, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
        
        if space_type == 'pixel':
            # Direct pixel optimization
            perturbation = (torch.randn_like(original) * 0.01).to(args.device)
            perturbation.requires_grad_(True)
            optimizer = optim.Adam([perturbation], lr=0.1)
            
            for step in range(args.steps):
                optimizer.zero_grad()
                watermarked = original + perturbation
                features = freqmark.image_encoder(watermarked)
                dot_products = torch.matmul(features, freqmark.direction_vectors.T) # [1,384] * [384,48]
                loss = torch.clamp(1.0 - dot_products * message, min=0).mean()
                loss += 0.1 * F.mse_loss(watermarked, original)
                loss.backward()
                optimizer.step()
            
            final_watermarked = (original + perturbation).detach()
            
        elif space_type == 'latent':
            # Latent space optimization
            with torch.no_grad():
                original = F.interpolate(original, size=(args.vae_image_size, args.vae_image_size), mode="bilinear", align_corners=False)
                latent = freqmark.vae.encode(2*original-1).latent_dist.sample()

            original = F.interpolate(original, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
            perturbation = (torch.randn_like(latent) * 0.01).to(args.device)
            perturbation.requires_grad_(True)
            optimizer = optim.Adam([perturbation], lr=0.1)
            
            for step in range(args.steps):
                optimizer.zero_grad()
                perturbed_latent = latent + perturbation
                watermarked = freqmark.vae.decode(perturbed_latent).sample
                watermarked = (watermarked + 1) / 2
                watermarked = F.interpolate(watermarked, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)
                features = freqmark.image_encoder(watermarked)
                dot_products = torch.matmul(features, freqmark.direction_vectors.T)
                loss = torch.clamp(1.0 - dot_products * message, min=0).mean()
                loss += 0.1 * F.mse_loss(watermarked, original)
                loss.backward()
                optimizer.step()
            
            final_watermarked = freqmark.vae.decode(latent + perturbation).sample
            final_watermarked = (final_watermarked + 1)/ 2
            final_watermarked = final_watermarked.detach()
            
        else:  # latent_freq (FreqMark)
            final_watermarked = freqmark.embed_watermark(original, message, args.dino_image_size)

        final_watermarked = F.interpolate(final_watermarked, size=(args.dino_image_size, args.dino_image_size), mode="bilinear", align_corners=False)

        # Evaluate robustness
        evaluator = WatermarkEvaluator()
        
        # Test against Gaussian noise
        attacked = evaluator.gaussian_noise_attack(final_watermarked)
        decoded = freqmark.decode_watermark(attacked)
        noise_robustness = freqmark.compute_bit_accuracy(message, decoded)
        
        # Test against VAE regeneration
        attacked = evaluator.vae_regeneration_attack(final_watermarked, freqmark.vae)
        decoded = freqmark.decode_watermark(attacked)
        regen_robustness = freqmark.compute_bit_accuracy(message, decoded)
        
        # Calculate PSNR
        psnr = freqmark._compute_psnr(final_watermarked, original)
        
        results_by_space[space_name] = {
            'psnr': psnr,
            'noise_robustness': noise_robustness,
            'regen_robustness': regen_robustness
        }
        
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Noise Robustness: {noise_robustness:.3f}")
        print(f"  Regen Robustness: {regen_robustness:.3f}")
    
    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    spaces = list(results_by_space.keys())
    metrics = ['psnr', 'noise_robustness', 'regen_robustness']
    metric_names = ['PSNR (dB)', 'Gaussian Noise\nRobustness', 'VAE Regen\nRobustness']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results_by_space[space][metric] for space in spaces]
        bars = ax[i].bar(spaces, values)
        ax[i].set_title(name)
        ax[i].set_ylim([0, max(values) * 1.2])
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'{val:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.suptitle('Comparison of Optimization Spaces', y=1.02)
    plt.savefig("freqmark_comparison.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    return results_by_space

if __name__ == "__main__":
    # Run complete demonstration
    print("Starting FreqMark implementation and evaluation...")

    args = Params()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)

    # Main demo
    results = run_freqmark_demo(args=args)

    # Compare with paper
    # compare_with_paper_results(results)
    
    # Analyze frequency domain advantage
    # freq_analysis = analyze_frequency_domain_advantage(args)
    
    # print(f"\nKey findings:")
    # print(f"- Achieved PSNR: {results['psnr']:.2f} dB (Paper: 31.27 dB)")
    # print(f"- Clean bit accuracy: {demo_results['bit_accuracy_clean']:.3f} (Paper: 1.000)")