import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OSCDPatchDataset(Dataset):

    def __init__(self, root_dir, split="train", use_indices=True):
        self.root_dir = root_dir
        self.split = split
        self.use_indices = use_indices

        split_dir = os.path.join(root_dir, split)

        self.files = sorted(
            glob.glob(os.path.join(split_dir, "*.npz"))
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found in: {split_dir}"
            )

        print(f"{split.upper()} dataset: {len(self.files)} patches")

    def __len__(self):
        return len(self.files)

    def normalize(self, image):
        """
        Convert Sentinel-2 uint16 values to approximately [0, 1].
        """

        image = image.astype(np.float32)

        image = image / 10000.0

        image = np.clip(image, 0.0, 1.0)

        return image

    def add_spectral_indices(self, image):
        """
        Add NDVI and NDBI.

        Input:
            13 x H x W

        Output:
            15 x H x W
        """

        # Sentinel-2 bands
        B4 = image[3]     # Red
        B8 = image[7]     # NIR
        B11 = image[10]   # SWIR

        eps = 1e-6

        # NDVI
        ndvi = (B8 - B4) / (B8 + B4 + eps)

        # NDBI
        ndbi = (B11 - B8) / (B11 + B8 + eps)

        ndvi = np.clip(ndvi, -1.0, 1.0)
        ndbi = np.clip(ndbi, -1.0, 1.0)

        image = np.concatenate(
            [
                image,
                ndvi[None, :, :],
                ndbi[None, :, :]
            ],
            axis=0
        )

        return image

    def __getitem__(self, idx):

        data = np.load(self.files[idx])

        image1 = data["image1"]
        image2 = data["image2"]
        mask = data["mask"]

        # Normalize Sentinel-2 data
        image1 = self.normalize(image1)
        image2 = self.normalize(image2)

        # Add NDVI and NDBI
        if self.use_indices:
            image1 = self.add_spectral_indices(image1)
            image2 = self.add_spectral_indices(image2)

        # Convert to tensors
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)

        mask = torch.from_numpy(
            mask.astype(np.float32)
        )

        # Mask: H x W → 1 x H x W
        mask = mask.unsqueeze(0)

        return image1, image2, mask


def create_dataloaders(
    dataset_root,
    batch_size=2,
    use_indices=True
):

    train_dataset = OSCDPatchDataset(
        dataset_root,
        split="train",
        use_indices=use_indices
    )

    val_dataset = OSCDPatchDataset(
        dataset_root,
        split="val",
        use_indices=use_indices
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

    return train_loader, val_loader


if __name__ == "__main__":

    DATASET_ROOT = (
        r"C:\Users\SHRISTI\OneDrive\Desktop"
        r"\AI_Urban_Expansion\dataset"
    )

    train_loader, val_loader = create_dataloaders(
        DATASET_ROOT,
        batch_size=2,
        use_indices=True
    )

    print("\nTesting DataLoader...")

    image1, image2, mask = next(iter(train_loader))

    print("T1 shape :", image1.shape)
    print("T2 shape :", image2.shape)
    print("Mask shape:", mask.shape)

    print("\nT1 value range:")
    print("Min:", image1.min().item())
    print("Max:", image1.max().item())

    print("\nMask values:")
    print(torch.unique(mask))