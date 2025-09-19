from turtle import distance
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
import timm
from custom_inpaint import CustomInpaintPipeline
warnings.filterwarnings('ignore')


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
        # self.dino_image_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.vae_image_size)),
            transforms.CenterCrop(self.vae_image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # --- FreqMark Core Parameters ---
        self.message_bits = 48
        # self.feature_dim = 384 # DINOv2 small
        self.feature_dim = 192 # DINOv2 small
        self.margin = 1.0
        self.grid_size = 28
        # self.num_patches = self.grid_size*self.grid_size
        self.mask_percentage = 0.3
        self.num_masks = 1
        self.last_grad_steps = 2
        self.num_inference_steps = 50
        self.seed = 42
        self.guidance_scale = 7.5

        # --- Optimization Parameters ---
        self.lr = 2.0
        self.steps = 400
        self.lambda_p = 0.005#0.025#0.5#0.05
        self.lambda_i = 0.001#0.25

        # --- Robustness Parameters --- 
        # self.eps1_std = [0.2, 0.6] # Latent noise
        # self.eps2_std = [0.06, 0.2] # Pixel noise
        # self.eps1_std = 0.25
        # self.eps2_std = 0.06
        # self.eps0_std = 0.01
        self.eps0_std = [0.0, 0.8] # Latent noise
        self.eps1_std = 2.0
        self.eps2_std = 0.06
        self.lambda_reg = 0.01

        self.dino_layer = 1
        
        # --- Demo/Evaluation Parameters ---
        self.batch_size = 1
        self.num_test_images = 1

class FreqMark:
    def __init__(self, args):
        self.args = args

        self.image_encoder = timm.create_model(
            'convnext_tiny',
            pretrained=True,
            features_only=True,
        ).to(self.args.device)
        self.pipe = CustomInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            # torch_dtype=torch.float16,
            cache_dir='/mnt/nas5/suhyeon/caches'
        ).to(self.args.device)

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.pipe.vae.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()

        self.pipe.set_progress_bar_config(disable=True)
    
        self.direction_vectors = torch.load('./random_vec.pt').to(self.args.device)
    
        self.mu = self.args.margin      # Hinge loss margin
        # self.num_patches = (self.args.dino_image_size // 14) ** 2

    def norm_tensor(self, tensor):
        t = tensor.clone().detach()
        
        min_val = t.min()
        max_val = t.max()

        tensor_norm = (tensor - min_val) / (max_val - min_val)

        # print(f"Tensor normalized: min={tensor_norm.min()}, max={tensor_norm.max()}")
        
        return tensor_norm, min_val, max_val

    def denorm_tensor(self, tensor, original_min=None, original_max=None):
        t = tensor.clone().detach()

        return t * (original_max - original_min) + original_min

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

    def embed_watermark(self, original: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            original = original.to(self.args.device)

            prompt_embeds = self.pipe._encode_prompt([""], self.args.device, 1, True, None)
            self.pipe.scheduler.set_timesteps(self.args.num_inference_steps, device=self.args.device)
            timesteps, num_inference_steps = self.pipe.get_timesteps(self.args.num_inference_steps, 1.0, self.args.device)
            
            generator = torch.Generator(self.args.device).manual_seed(self.args.seed)

            latent = self.pipe.vae.encode(2*original-1).latent_dist.sample(generator=generator) # [-1, 1], [B,4,64,64]

            latent = latent * self.pipe.vae.config.scaling_factor

            mask = self._create_random_mask(original, num_masks=1, mask_percentage=self.args.mask_percentage)

            inpaint_mask = mask.to(self.args.device)
            mask_latent = F.interpolate(inpaint_mask, size=(latent.shape[2], latent.shape[3]))

            noise = torch.randn(latent.shape, generator=generator, device=self.args.device, dtype=latent.dtype)
            current_latent = self.pipe.scheduler.add_noise(latent, noise, timesteps[0])
            grad_start_step = len(timesteps) - self.args.last_grad_steps

            latent_history = [current_latent.detach().cpu()]
            noise_pred_history = []
            mse_inside_history = []
            mse_outside_history = []

            # --- 4. Denoising Loop ---
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([current_latent] * 2)
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
                unet_mask = torch.cat([mask_latent] * 2)
                unet_context = torch.cat([latent] * 2)

                latent_model_input = torch.cat([latent_model_input, unet_mask, unet_context], dim=1)

                if i < grad_start_step:
                    with torch.no_grad():
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                else:
                    noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # save for analysis
                noise_pred_history.append(noise_pred.detach().cpu())

                step_output = self.pipe.scheduler.step(noise_pred, t, current_latent, return_dict=False)

                current_latent = step_output[0]

                # save after scaling
                latent_history.append(current_latent.detach().cpu())

                # analysis
                noise_pred_inside = noise_pred * mask_latent
                noise_pred_outside = noise_pred * (1 - mask_latent)
                
                epsilon_inside = noise * mask_latent
                epsilon_outside = noise * (1 - mask_latent)

                mse_inside = F.mse_loss(noise_pred_inside, epsilon_inside)
                mse_outside = F.mse_loss(noise_pred_outside, epsilon_outside)

                mse_inside_history.append(mse_inside.item())
                mse_outside_history.append(mse_outside.item())

                if (i+1) % 5 == 0 or i+1 == len(timesteps):
                    latent_vis = 1 / self.pipe.vae.config.scaling_factor * current_latent
                    image_edit = self.pipe.vae.decode(latent_vis).sample
                    img_edit = (image_edit + 1) / 2
                    img_edit = img_edit.clamp(0, 1)
                    torchvision.utils.save_image(img_edit, os.path.join(self.args.output_dir, f"analysis_timesteps_step_{i+1}.png"))
                    
            current_latent = 1 / self.pipe.vae.config.scaling_factor * current_latent

            analysis_data = {
                'latent_history': latent_history,
                'noise_pred_history': noise_pred_history,
                'timesteps': timesteps.cpu(),
                'mask_latent': mask_latent.cpu(),
                'original_latent': latent.detach().cpu(),
                'original_noise': noise.cpu(), 
                'mse_inside_history': mse_inside_history,
                'mse_outside_history': mse_outside_history,
            }
            torch.save(analysis_data, "inpaint_history.pt")
            print("Analysis data saved to inpaint_history.pt")

            torchvision.utils.save_image(mask, os.path.join(self.args.output_dir, "analysis_timesteps_mask.png"))

            image_edit = self.pipe.vae.decode(current_latent).sample
            img_edit = (image_edit + 1) / 2
            img_edit = img_edit.clamp(0, 1)
        return img_edit.detach()
    
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

def run_freqmark_demo(args=None):
    """Run complete FreqMark demonstration"""
    print("=== FreqMark Implementation Demo ===")
    
    # print(f"Test setup: {args.num_test_images} images, {args.dino_image_size}x{args.dino_image_size}, {args.message_bits} bits")
    
    # Initialize FreqMark
    freqmark = FreqMark(args=args)
    # all_psnrs = []
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
        watermarked_batch = freqmark.embed_watermark(test_batch)
    
if __name__ == "__main__":
    # Run complete demonstration
    print("Starting FreqMark implementation and evaluation...")

    args = Params()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)

    # Main demo
    results = run_freqmark_demo(args=args)