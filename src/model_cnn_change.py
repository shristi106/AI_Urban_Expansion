import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

class CNNEncoder(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        self.encoder = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# CHANGE DETECTION NETWORK
# ============================================================

class ChangeDetectionNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature difference processing
        self.change_block = nn.Sequential(

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.change_block(x)


# ============================================================
# CNN + CHANGE DETECTION NETWORK
# ============================================================

class CNNChangeDetection(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        # Shared CNN encoder
        self.encoder = CNNEncoder(in_channels)

        # Explicit change detection module
        self.change_detector = ChangeDetectionNetwork()

        # ====================================================
        # DECODER
        # ====================================================

        self.decoder = nn.Sequential(

            # 8 -> 16
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 16 -> 32
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(
                16,
                8,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            # Output change map
            nn.Conv2d(
                8,
                1,
                kernel_size=1
            )
        )

    def forward(self, t1, t2):

        # ----------------------------------------------------
        # Extract features from T1 and T2
        # ----------------------------------------------------

        f1 = self.encoder(t1)
        f2 = self.encoder(t2)

        # ----------------------------------------------------
        # Calculate temporal feature difference
        # ----------------------------------------------------

        difference = torch.abs(f2 - f1)

        # ----------------------------------------------------
        # Change detection network
        # ----------------------------------------------------

        change_features = self.change_detector(
            difference
        )

        # ----------------------------------------------------
        # Decode into pixel-level change map
        # ----------------------------------------------------

        output = self.decoder(
            change_features
        )

        # Guarantee exact output size
        output = F.interpolate(
            output,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        return output