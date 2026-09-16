import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic CNN block
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Siamese CNN Encoder
# ============================================================

class SiameseEncoder(nn.Module):

    def __init__(self, in_channels=15):

        super().__init__()

        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        # 128 x 128
        x1 = self.block1(x)

        # 64 x 64
        x2 = self.block2(self.pool(x1))

        # 32 x 32
        x3 = self.block3(self.pool(x2))

        return x1, x2, x3


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv1 = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.conv2 = ConvBlock(64, 32)

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x, skip2, skip1):

        # 32 → 64
        x = self.up1(x)

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.conv1(x)

        # 64 → 128
        x = self.up2(x)

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.conv2(x)

        # Output: 1 x 128 x 128
        x = self.final(x)

        return x


# ============================================================
# Siamese CNN Change Detection Model
# ============================================================

class SiameseCNN(nn.Module):

    def __init__(self, in_channels=15):

        super().__init__()

        # SAME encoder is used for T1 and T2
        self.encoder = SiameseEncoder(
            in_channels=in_channels
        )

        self.decoder = Decoder()

    def forward(self, image1, image2):

        # Extract features from T1
        t1_1, t1_2, t1_3 = self.encoder(image1)

        # Extract features from T2
        t2_1, t2_2, t2_3 = self.encoder(image2)

        # Absolute feature difference
        diff1 = torch.abs(t1_1 - t2_1)
        diff2 = torch.abs(t1_2 - t2_2)
        diff3 = torch.abs(t1_3 - t2_3)

        # Decode difference features
        output = self.decoder(
            diff3,
            diff2,
            diff1
        )

        return output


# ============================================================
# Test model
# ============================================================

if __name__ == "__main__":

    print("Testing Siamese CNN...\n")

    model = SiameseCNN(
        in_channels=15
    )

    # Fake input matching our dataset
    image1 = torch.randn(
        2, 15, 128, 128
    )

    image2 = torch.randn(
        2, 15, 128, 128
    )

    # Forward pass
    output = model(
        image1,
        image2
    )

    print("T1 shape:")
    print(image1.shape)

    print("\nT2 shape:")
    print(image2.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nModel parameters:")
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"{total_params:,}")

    print("\nModel test successful!")