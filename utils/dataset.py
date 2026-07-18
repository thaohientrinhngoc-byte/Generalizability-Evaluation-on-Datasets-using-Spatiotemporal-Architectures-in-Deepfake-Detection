"""
PyTorch Dataset Loader and Dynamic Balanced Sampler.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.v2 as v2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

cv2.setNumThreads(0)  # Prevent OpenCV CPU multi-threading thrashing


class DeepfakeDataset(Dataset):
    """
    Dataset loader for preprocessed 30-frame face crops.

    Directory structure expected:
        dataset_root/
          ├── split (train/val/test)
          │     ├── real
          │     │     └── video_id_1/*.png
          │     └── fake (or fake_*)
          │           └── video_id_2/*.png
    """

    def __init__(self, dataset_dir: str, split: str = "train", is_train: bool = False):
        self.is_train = is_train
        self.num_frames = config.NUM_FRAMES
        self.frame_size = config.FRAME_SIZE
        self.samples = []

        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            # Fallback if split_dir doesn't exist directly
            split_dir = dataset_dir

        if os.path.exists(split_dir):
            all_subdirs = os.listdir(split_dir)

            # Collect real videos (label 0.0)
            real_dir = os.path.join(split_dir, "real")
            if os.path.exists(real_dir):
                self._collect_from_dir(real_dir, label_val=0.0)

            # Collect fake videos (label 1.0)
            fake_dirs = [
                os.path.join(split_dir, d) for d in all_subdirs
                if d.lower() == "fake" or d.lower().startswith("fake_")
            ]
            for f_dir in fake_dirs:
                if os.path.isdir(f_dir):
                    self._collect_from_dir(f_dir, label_val=1.0)

        if is_train:
            self._jitter = v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08
            )
            self._blur = v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5))

    def _collect_from_dir(self, target_dir: str, label_val: float):
        for video_id in os.listdir(target_dir):
            video_path = os.path.join(target_dir, video_id)
            if not os.path.isdir(video_path):
                continue
            png_paths = sorted([
                os.path.join(video_path, f) for f in os.listdir(video_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])[:self.num_frames]
            if png_paths:
                self.samples.append((png_paths, label_val))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        png_paths, label = self.samples[idx]
        label_t = torch.tensor(label, dtype=torch.float32)

        arr = np.empty((len(png_paths), self.frame_size, self.frame_size, 3), dtype=np.uint8)
        valid = 0
        for p in png_paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            arr[valid] = bgr[..., ::-1]  # BGR to RGB
            valid += 1

        if valid == 0:
            return (
                torch.zeros((self.num_frames, 3, self.frame_size, self.frame_size), dtype=torch.uint8),
                label_t,
            )

        seq = torch.from_numpy(
            np.ascontiguousarray(arr[:valid].transpose(0, 3, 1, 2))
        )  # [T, 3, H, W] uint8

        # Edge-frame padding if missing frames
        if seq.size(0) < self.num_frames:
            shortage = self.num_frames - seq.size(0)
            seq = torch.cat(
                [seq, seq[-1:].expand(shortage, -1, -1, -1).clone()], dim=0
            )

        # Augmentation on uint8 tensors for training set
        if self.is_train:
            if random.random() < 0.5:
                seq = v2.functional.hflip(seq)
            seq = self._jitter(seq)
            if random.random() < 0.4:
                seq = self._blur(seq)

        return seq, label_t


def collate_fn(batch):
    frames = torch.stack([b[0] for b in batch])  # [B, T, C, H, W] uint8
    labels = torch.stack([b[1] for b in batch])  # [B] float32
    return frames, labels


class DynamicBalancedSampler(Sampler):
    """
    Ensures 1:1 balanced sampling between real (0.0) and fake (1.0) classes per batch.
    """

    def __init__(self, labels: list, batch_size: int):
        arr = np.array(labels)
        self.batch_size = batch_size
        self.real_idx = np.where(arr == 0.0)[0]
        self.fake_idx = np.where(arr == 1.0)[0]
        if len(self.real_idx) == 0 or len(self.fake_idx) == 0:
            self.num_batches = 0
        else:
            self.num_batches = max(
                1,
                (min(len(self.real_idx), len(self.fake_idx)) * 2) // batch_size,
            )

    def __iter__(self):
        if self.num_batches == 0:
            return iter([])
        half = self.batch_size // 2
        for _ in range(self.num_batches):
            r = np.random.choice(
                self.real_idx, half, replace=len(self.real_idx) < half
            )
            f = np.random.choice(
                self.fake_idx, half, replace=len(self.fake_idx) < half
            )
            batch = np.concatenate([r, f])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches
