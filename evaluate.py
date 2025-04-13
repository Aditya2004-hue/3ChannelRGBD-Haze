import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Encoder, Decoder
from dataset import DenseHazeNPYDataset
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# -------------------------
# Settings
# -------------------------
MODEL_PATH = "dehaze_autoencoder_rgb.pth"
SAVE_DIR = "results_rgb"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load model
# -------------------------
encoder = Encoder(input_channels=3).to(DEVICE)
decoder = Decoder(output_channels=3).to(DEVICE)

checkpoint = torch.load(MODEL_PATH)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])
encoder.eval()
decoder.eval()

# -------------------------
# Load dataset
# -------------------------
dataset = DenseHazeNPYDataset("hazy.npy", "GT.npy")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

total_psnr = 0.0
total_ssim = 0.0

print("🔍 Evaluating RGB model...")
with torch.no_grad():
    for idx, (gt, hazy) in enumerate(tqdm(loader)):
        gt = gt.to(DEVICE)
        hazy = hazy.to(DEVICE)

        output = decoder(encoder(hazy)).clamp(0, 1)

        # Convert to numpy for metrics
        gt_np = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        psnr_val = psnr(gt_np, out_np, data_range=1)
        ssim_val = ssim(gt_np, out_np, channel_axis=-1, data_range=1)

        total_psnr += psnr_val
        total_ssim += ssim_val

        # Save comparison: [hazy | output | gt]
        comparison = torch.cat([hazy.cpu(), output.cpu(), gt.cpu()], dim=3)  # (B, C, H, 3W)
        save_image(comparison, os.path.join(SAVE_DIR, f"{idx:02d}_compare_rgb.png"))

avg_psnr = total_psnr / len(loader)
avg_ssim = total_ssim / len(loader)

print("\n✅ RGB Evaluation Complete")
print(f"📊 Average PSNR: {avg_psnr:.2f}")
print(f"📊 Average SSIM: {avg_ssim:.4f}")
print(f"🖼️ Comparison images saved in: {SAVE_DIR}/")
