# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile
from PIL import Image


class RawNoisyDataset(Dataset):
    def __init__(self, raw_dir, patch_size=512, stride=512, augment=True, is_test=False):
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.is_test = is_test
        self.raw_dir = raw_dir

        if not self.raw_dir or not os.path.exists(self.raw_dir):
            raise ValueError(f"Directory not found: {self.raw_dir}")

        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.files = sorted([os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir)
                             if os.path.splitext(f)[1].lower() in valid_exts])

        if self.is_test:
            self.crops = [(i, 0, 0) for i in range(len(self.files))]
        else:
            self.crops = self._prepare_all_grid_crops()

        mode_str = "TEST" if is_test else "TRAIN"
        print(f"[{mode_str}] Dataset loaded successfully. Total valid crops: {len(self.crops)}")

    def _prepare_all_grid_crops(self):
        crop_list = []
        if len(self.files) == 0: return []

        for idx in range(len(self.files)):
            try:
                if self.files[idx].lower().endswith(('.tif', '.tiff')):
                    img_array = tifffile.imread(self.files[idx])
                    if img_array.ndim == 3: img_array = img_array[0]
                else:
                    img_array = np.array(Image.open(self.files[idx]).convert('L'))

                H, W = img_array.shape
                if H < self.patch_size or W < self.patch_size: continue
            except Exception:
                continue

            h_steps = list(range(0, H - self.patch_size + 1, self.stride))
            w_steps = list(range(0, W - self.patch_size + 1, self.stride))

            threshold = np.mean(img_array) * 0.1

            for top in h_steps:
                for left in w_steps:
                    crop_patch = img_array[top:top + self.patch_size, left:left + self.patch_size]
                    if np.mean(crop_patch) > threshold:
                        crop_list.append((idx, top, left))
        return crop_list

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        file_idx, top, left = self.crops[idx]
        noisy_path = self.files[file_idx]
        filename = os.path.basename(noisy_path)
        ps = self.patch_size

        try:
            if noisy_path.lower().endswith(('.tif', '.tiff')):
                n_img = tifffile.imread(noisy_path)
                if n_img.ndim == 3: n_img = n_img[0]
                n_img = n_img.astype(np.float32)
            else:
                with Image.open(noisy_path) as img:
                    if img.mode not in ['L', 'I;16']: img = img.convert('L')
                    n_img = np.array(img).astype(np.float32)

            if self.is_test:
                n = torch.from_numpy(n_img).unsqueeze(0)
            else:
                n_crop = n_img[top:top + ps, left:left + ps]
                n = torch.from_numpy(n_crop).unsqueeze(0)
        except Exception:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        if self.augment and not self.is_test:
            if torch.rand(1) < 0.5: n = torch.flip(n, [2])
            if torch.rand(1) < 0.5: n = torch.flip(n, [1])
            n = torch.rot90(n, torch.randint(0, 4, (1,)).item(), [1, 2])

        return n, filename


def get_datasets(cfg):
    train_dir = cfg.get('train_raw_folder') or "./data/train/raw"
    test_dir = cfg.get('test_raw_folder') or "./data/test/raw"

    train_ds = RawNoisyDataset(raw_dir=train_dir, patch_size=cfg.get('patch_size') or 512,
                               stride=cfg.get('stride') or 512, augment=True, is_test=False)
    test_ds = RawNoisyDataset(raw_dir=test_dir, is_test=True) if test_dir and os.path.exists(test_dir) else None
    return train_ds, test_ds