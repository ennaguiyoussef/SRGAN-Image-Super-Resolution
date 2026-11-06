"""PyTorch datasets for aligned DIV2K training and validation crops."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data import paired_crop


class SuperResolutionDataset(Dataset):
    """Return LR float32 in [0, 1] and HR float32 in [-1, 1], in CHW order.

    Training uses random HR crops. Validation uses deterministic center crops.
    Final full-image evaluation will use a separate protocol in phase 5.
    """

    def __init__(self, root, image_paths, hr_size=96, scale=4, training=False):
        self.root = Path(root)
        self.image_paths = list(image_paths)
        self.hr_size, self.scale, self.training = hr_size, scale, training
        if not self.image_paths:
            raise ValueError("Dataset must contain at least one image.")
        if scale <= 0 or hr_size <= 0 or hr_size % scale:
            raise ValueError("HR crop must be positive and divisible by scale.")
        for name in self.image_paths:
            resolved = (self.root / name).resolve()
            if not resolved.is_relative_to(self.root.resolve()) or not resolved.is_file():
                raise FileNotFoundError(f"Missing or invalid dataset image: {name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.image_paths[index]) as image:
            lr, hr = paired_crop(image, self.hr_size, self.scale, self.training)
        lr = torch.from_numpy(np.array(lr, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        hr = torch.from_numpy(np.array(hr, dtype=np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1.0
        return lr, hr
