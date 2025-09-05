import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import logging
import os
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

class Params:
    """스크립트 실행을 위한 설정값"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_dir = '/mnt/nas5/suhyeon/datasets/DIV2K_train_HR' # ✅ NSFW를 확인할 이미지 폴더
        self.output_dir = './' # 로그 파일 저장 경로
        self.cache_dir = '/mnt/nas5/suhyeon/caches'
        self.batch_size = 8 # GPU 메모리에 맞춰 조절
        self.num_workers = 4
        self.vae_image_size = 512
        self.transform = transforms.Compose([
            transforms.Resize((self.vae_image_size, self.vae_image_size)),
            transforms.ToTensor(),
        ])

class ImageFolderDataset(Dataset):
    """폴더 내 이미지와 파일명을 함께 로드하는 커스텀 데이터셋"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

def norm_tensor(tensor):
    """텐서를 0과 1 사이로 정규화"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-6)

def scan_for_nsfw(args=None):
    """지정된 폴더의 이미지를 스캔하여 NSFW 파일명을 찾고 출력합니다."""
    print(f"Scanning for NSFW images in: {args.image_dir}")

    # --- 1. 모델 및 데이터 로더 준비 ---
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        dtype=torch.float16,
        cache_dir=args.cache_dir
    ).to(args.device)
    
    # 진행률 표시줄 비활성화 및 로그 최소화
    pipe.set_progress_bar_config(disable=True)
    logging.set_verbosity_error()

    dataset = ImageFolderDataset(args.image_dir, transform=args.transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    nsfw_filenames = []
    
    # --- 2. 이미지 스캔 루프 ---
    pbar = tqdm(dataloader, desc="Scanning images")
    
    for image_batch, filename_batch in pbar:
        batch = image_batch.to(args.device)
        
        with torch.no_grad():
            for i in range(batch.size(0)):
                single_image = batch[i:i+1]
                filename = filename_batch[i]
                
                # Inpainting 파이프라인 실행
                img_norm = norm_tensor(single_image)
                zero_mask = torch.zeros((1,1,512,512)).to(args.device)
                
                pipe_output = pipe(prompt="", image=img_norm, mask_image=zero_mask, num_inference_steps=5)
                is_nsfw = pipe_output.nsfw_content_detected[0]
                
                # ✅ NSFW 탐지 시 파일명 출력 및 리스트에 추가
                if is_nsfw:
                    print(f"\nNSFW Detected: {filename}")
                    if filename not in nsfw_filenames:
                        nsfw_filenames.append(filename)

    # --- 3. 최종 결과 저장 ---
    if nsfw_filenames:
        log_path = os.path.join(args.output_dir, "nsfw_log.txt")
        with open(log_path, "w") as f:
            for name in sorted(nsfw_filenames):
                f.write(f"{name}\n")
        print(f"\n--- Scan Complete ---")
        print(f"Found {len(nsfw_filenames)} NSFW samples. Filenames saved to {log_path}")
    else:
        print("\n--- Scan Complete ---")
        print("No NSFW samples were detected.")
        
    return nsfw_filenames

if __name__ == "__main__":
    args = Params()
    os.makedirs(args.output_dir, exist_ok=True)
    
    nsfw_list = scan_for_nsfw(args=args)