import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class NoiseRobustOSCDDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        use_augmentation=False
    ):

        self.root_dir = root_dir
        self.split = split
        self.use_augmentation = (
            use_augmentation and split == "train"
        )

        split_dir = os.path.join(
            root_dir,
            split
        )

        self.files = sorted(
            glob.glob(
                os.path.join(
                    split_dir,
                    "*.npz"
                )
            )
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No .npz files found in: "
                f"{split_dir}"
            )

        print(
            f"{split.upper()} dataset: "
            f"{len(self.files)} patches"
        )

        print(
            f"Augmentation enabled: "
            f"{self.use_augmentation}"
        )


    def __len__(self):

        return len(self.files)


    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(self, image):

        image = image.astype(
            np.float32
        )

        image = image / 10000.0

        image = np.clip(
            image,
            0.0,
            1.0
        )

        return image


    # ========================================================
    # SPATIAL AUGMENTATION
    # ========================================================

    def spatial_augmentation(
        self,
        image1,
        image2,
        mask
    ):

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if np.random.random() < 0.5:

            image1 = np.flip(
                image1,
                axis=2
            )

            image2 = np.flip(
                image2,
                axis=2
            )

            mask = np.flip(
                mask,
                axis=1
            )


        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if np.random.random() < 0.5:

            image1 = np.flip(
                image1,
                axis=1
            )

            image2 = np.flip(
                image2,
                axis=1
            )

            mask = np.flip(
                mask,
                axis=0
            )


        # ----------------------------------------------------
        # Random 90-degree rotation
        # ----------------------------------------------------

        k = np.random.randint(
            0,
            4
        )

        if k > 0:

            image1 = np.rot90(
                image1,
                k=k,
                axes=(1, 2)
            )

            image2 = np.rot90(
                image2,
                k=k,
                axes=(1, 2)
            )

            mask = np.rot90(
                mask,
                k=k,
                axes=(0, 1)
            )


        return (
            image1.copy(),
            image2.copy(),
            mask.copy()
        )


    # ========================================================
    # SPECTRAL / TEMPORAL PERTURBATION
    # ========================================================

    def spectral_perturbation(
        self,
        image
    ):

        image = image.copy()

        channels = image.shape[0]


        # ----------------------------------------------------
        # Small per-band spectral scaling
        # ----------------------------------------------------

        scale = np.random.uniform(
            0.98,
            1.02,
            size=(channels, 1, 1)
        ).astype(
            np.float32
        )

        image = image * scale


        # ----------------------------------------------------
        # Small additive sensor noise
        # ----------------------------------------------------

        noise = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=image.shape
        ).astype(
            np.float32
        )

        image = image + noise


        # ----------------------------------------------------
        # Small brightness variation
        # ----------------------------------------------------

        brightness = np.random.uniform(
            0.98,
            1.02
        )

        image = image * brightness


        image = np.clip(
            image,
            0.0,
            1.0
        )


        return image


    # ========================================================
    # LOAD PATCH
    # ========================================================

    def __getitem__(self, idx):

        data = np.load(
            self.files[idx]
        )

        image1 = data["image1"]
        image2 = data["image2"]
        mask = data["mask"]


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image1 = self.normalize(
            image1
        )

        image2 = self.normalize(
            image2
        )


        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.use_augmentation:

            # Same spatial transformation for T1,
            # T2 and mask so their alignment is preserved.

            (
                image1,
                image2,
                mask
            ) = self.spatial_augmentation(
                image1,
                image2,
                mask
            )


            # Independent spectral perturbation
            # simulates temporal/acquisition variation.

            image1 = self.spectral_perturbation(
                image1
            )

            image2 = self.spectral_perturbation(
                image2
            )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image1 = torch.from_numpy(
            image1.copy()
        ).float()

        image2 = torch.from_numpy(
            image2.copy()
        ).float()

        mask = torch.from_numpy(
            mask.astype(
                np.float32
            ).copy()
        )

        mask = mask.unsqueeze(0)


        return (
            image1,
            image2,
            mask
        )


# ============================================================
# DATALOADERS
# ============================================================

def create_augmented_dataloaders(
    dataset_root,
    batch_size=2
):

    train_dataset = NoiseRobustOSCDDataset(
        dataset_root,
        split="train",
        use_augmentation=True
    )

    # IMPORTANT:
    # Validation has NO augmentation.

    val_dataset = NoiseRobustOSCDDataset(
        dataset_root,
        split="val",
        use_augmentation=False
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


    return (
        train_loader,
        val_loader
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    DATASET_ROOT = (
        r"C:\Users\SHRISTI\OneDrive\Desktop"
        r"\AI_Urban_Expansion\dataset"
    )

    train_loader, val_loader = (
        create_augmented_dataloaders(
            DATASET_ROOT,
            batch_size=2
        )
    )


    image1, image2, mask = next(
        iter(train_loader)
    )


    print("\nTesting augmented DataLoader...")

    print(
        "T1 shape :",
        image1.shape
    )

    print(
        "T2 shape :",
        image2.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "\nT1 range:"
    )

    print(
        "Min:",
        image1.min().item()
    )

    print(
        "Max:",
        image1.max().item()
    )

    print(
        "\nMask values:"
    )

    print(
        torch.unique(mask)
    )