from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

LATENT_DIM = 64
MASK_SIZE = 64


# ============================================================
# GENERATOR
# ============================================================

class MaskGenerator(nn.Module):

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                latent_dim,
                128 * 8 * 8
            ),

            nn.ReLU(True),

            nn.Unflatten(
                1,
                (128, 8, 8)
            ),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(True),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )


    def forward(self, z):
        return self.network(z)


# ============================================================
# DISCRIMINATOR
# ============================================================

class MaskDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 64x64 -> 32x32
            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 32x32 -> 16x16
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16x16 -> 8x8
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8x8 -> 4x4
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                1
            )
        )


    def forward(self, x):
        return self.network(x)


# ============================================================
# MASK PREPARATION
# ============================================================

def prepare_mask(mask):

    mask = np.asarray(
        mask,
        dtype=np.float32
    )

    mask = (
        mask > 0
    ).astype(
        np.float32
    )

    tensor = torch.from_numpy(
        mask
    )

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = F.interpolate(
        tensor,
        size=(
            MASK_SIZE,
            MASK_SIZE
        ),
        mode="nearest"
    )

    return tensor.squeeze(0)


# ============================================================
# TRAIN GAN
# ============================================================

def train_gan(
    masks,
    epochs=10,
    batch_size=16,
    lr=2e-4,
    device="cpu"
):

    generator = (
        MaskGenerator()
        .to(device)
    )

    discriminator = (
        MaskDiscriminator()
        .to(device)
    )

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    processed_masks = []

    for mask in masks:

        processed_masks.append(
            prepare_mask(mask)
        )

    data = torch.stack(
        processed_masks
    )

    dataset = torch.utils.data.TensorDataset(
        data
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for (real_masks,) in loader:

            real_masks = (
                real_masks
                .to(device)
            )

            current_batch = (
                real_masks.size(0)
            )

            # =================================================
            # DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad()

            real_labels = torch.ones(
                current_batch,
                1,
                device=device
            )

            fake_labels = torch.zeros(
                current_batch,
                1,
                device=device
            )

            real_output = (
                discriminator(
                    real_masks
                )
            )

            real_loss = criterion(
                real_output,
                real_labels
            )

            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                device=device
            )

            fake_masks = (
                generator(noise)
            )

            fake_output = (
                discriminator(
                    fake_masks.detach()
                )
            )

            fake_loss = criterion(
                fake_output,
                fake_labels
            )

            d_loss = (
                real_loss
                +
                fake_loss
            ) / 2.0

            d_loss.backward()

            optimizer_d.step()

            # =================================================
            # GENERATOR
            # =================================================

            optimizer_g.zero_grad()

            noise = torch.randn(
                current_batch,
                LATENT_DIM,
                device=device
            )

            fake_masks = (
                generator(noise)
            )

            fake_output = (
                discriminator(
                    fake_masks
                )
            )

            g_loss = criterion(
                fake_output,
                real_labels
            )

            g_loss.backward()

            optimizer_g.step()

            total_d_loss += (
                d_loss.item()
            )

            total_g_loss += (
                g_loss.item()
            )

        average_d = (
            total_d_loss
            /
            len(loader)
        )

        average_g = (
            total_g_loss
            /
            len(loader)
        )

        print(
            f"GAN Epoch {epoch}/{epochs} | "
            f"D Loss: {average_d:.4f} | "
            f"G Loss: {average_g:.4f}"
        )

    return generator


# ============================================================
# GENERATE MASKS
# ============================================================

def generate_masks(
    generator,
    number,
    device="cpu"
):

    generator.eval()

    with torch.no_grad():

        noise = torch.randn(
            number,
            LATENT_DIM,
            device=device
        )

        masks = generator(
            noise
        )

    masks = (
        masks
        .cpu()
        .numpy()
    )

    masks = (
        masks > 0.5
    ).astype(
        np.uint8
    )

    return masks


# ============================================================
# SAVE GENERATED MASKS
# ============================================================

def save_generated_masks(
    generator,
    output_dir,
    number=200,
    device="cpu"
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    masks = generate_masks(
        generator,
        number,
        device
    )

    for index, mask in enumerate(
        masks
    ):

        np.save(
            output_dir /
            f"gan_mask_{index:04d}.npy",
            mask
        )

    print(
        f"Saved {number} GAN masks to:"
    )

    print(
        output_dir
    )

    return masks