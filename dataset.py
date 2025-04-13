import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DenseHazeNPYDataset(Dataset):
    def __init__(self, hazy_path="hazy.npy", gt_path="GT.npy", transform=None):
        """
        Dataset for RGB dehazing from .npy files.
        Returns images in shape (3, 256, 256), normalized to [0, 1].
        """
        self.hazy_data = np.load(hazy_path, allow_pickle=True)
        self.gt_data = np.load(gt_path, allow_pickle=True)
        assert len(self.hazy_data) == len(self.gt_data), "Hazy and GT data length mismatch."

        self.transform = transform

    def __len__(self):
        return len(self.hazy_data)

    def __getitem__(self, idx):
        hazy = self.hazy_data[idx]
        gt = self.gt_data[idx]

        # Ensure RGB shape: (H, W, 3)
        if hazy.shape[-1] != 3 or gt.shape[-1] != 3:
            raise ValueError("Images must be RGB with shape (H, W, 3)")

        # Normalize and convert to tensor: (3, H, W)
        hazy = torch.tensor(hazy / 255.0, dtype=torch.float32).permute(2, 0, 1)
        gt = torch.tensor(gt / 255.0, dtype=torch.float32).permute(2, 0, 1)

        if self.transform:
            hazy = self.transform(hazy)
            gt = self.transform(gt)

        return gt, hazy  # Ground truth, then hazy input


def build_npy_from_images(input_dir, output_name="GT.npy", img_size=256):
    """
    Converts a folder of RGB images to a .npy file.
    Resizes to (img_size, img_size) if needed.
    """
    image_list = []
    files = sorted(os.listdir(input_dir))

    print(f"🔄 Converting images in: {input_dir}")
    for filename in tqdm(files):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB order
            img = cv2.resize(img, (img_size, img_size))
            image_list.append(img)

    arr = np.array(image_list)
    np.save(output_name, arr)
    print(f"✅ Saved {len(image_list)} RGB images to: {output_name}")


# Optional command-line usage
if __name__ == "__main__":
    build_npy_from_images("GT", "GT.npy")
    build_npy_from_images("hazy", "hazy.npy")

    dataset = DenseHazeNPYDataset("hazy.npy", "GT.npy")
    print(f"Loaded {len(dataset)} RGB samples.")
    sample_gt, sample_hazy = dataset[0]
    print("Sample shape:", sample_gt.shape, sample_hazy.shape)  # (3, 256, 256)
