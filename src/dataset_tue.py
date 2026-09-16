import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class TUEDataset(Dataset):

    def __init__(self, root_dir, files, augment=False, image_size=128):
        self.root_dir = root_dir
        self.files = files
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        path_a = os.path.join(
            self.root_dir, "A", filename
        )

        path_b = os.path.join(
            self.root_dir, "B", filename
        )

        path_label = os.path.join(
            self.root_dir, "label", filename
        )

        # Load images
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        mask = Image.open(path_label)

        # Resize images and mask
        img_a = img_a.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        img_b = img_b.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        # Convert to arrays
        img_a = np.array(img_a, dtype=np.float32) / 255.0
        img_b = np.array(img_b, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        # Binary mask
        mask = (mask > 0).astype(np.float32)

        # -------------------------
        # Training augmentation
        # -------------------------

        if self.augment:

            if random.random() < 0.5:
                img_a = np.fliplr(img_a).copy()
                img_b = np.fliplr(img_b).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.5:
                img_a = np.flipud(img_a).copy()
                img_b = np.flipud(img_b).copy()
                mask = np.flipud(mask).copy()

            k = random.randint(0, 3)

            if k > 0:
                img_a = np.rot90(img_a, k).copy()
                img_b = np.rot90(img_b, k).copy()
                mask = np.rot90(mask, k).copy()

        # HWC → CHW
        img_a = torch.from_numpy(
            img_a
        ).permute(2, 0, 1).float()

        img_b = torch.from_numpy(
            img_b
        ).permute(2, 0, 1).float()

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0).float()

        return img_a, img_b, mask


def create_tue_dataloaders(
    root_dir,
    batch_size=8,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):

    all_files = sorted(
        f for f in os.listdir(
            os.path.join(root_dir, "A")
        )
        if f.lower().endswith(".png")
    )

    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    print(f"Total TUE-CD images : {n}")
    print(f"Training images     : {len(train_files)}")
    print(f"Validation images   : {len(val_files)}")
    print(f"Test images         : {len(test_files)}")

    train_dataset = TUEDataset(
        root_dir,
        train_files,
        augment=True,
        image_size=128
    )

    val_dataset = TUEDataset(
        root_dir,
        val_files,
        augment=False,
        image_size=128
    )

    test_dataset = TUEDataset(
        root_dir,
        test_files,
        augment=False,
        image_size=128
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader