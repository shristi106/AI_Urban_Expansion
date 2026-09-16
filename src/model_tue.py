import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SiameseEncoder(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x1 = self.conv1(x)
        x = self.pool(x1)

        x2 = self.conv2(x)
        x = self.pool(x2)

        x3 = self.conv3(x)
        x = self.pool(x3)

        return x, x1, x2, x3


class VisionTransformerBlock(nn.Module):

    def __init__(
        self,
        channels=128,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        self.position = nn.Parameter(
            torch.zeros(1, 256, channels)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        x = self.pool(x)

        # B,C,H,W -> B,H,W,C
        x = x.permute(0, 2, 3, 1)

        # B,256,C
        x = x.reshape(
            x.size(0),
            256,
            x.size(-1)
        )

        x = x + self.position

        x = self.transformer(x)

        # B,256,C -> B,C,16,16
        x = x.reshape(
            x.size(0),
            16,
            16,
            -1
        )

        x = x.permute(0, 3, 1, 2)

        return x


class HybridTUE(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = SiameseEncoder(in_channels)

        self.vit = VisionTransformerBlock(
            channels=128,
            num_heads=4,
            num_layers=2
        )

        self.decoder1 = ConvBlock(128, 128)
        self.decoder2 = ConvBlock(128 + 128, 64)
        self.decoder3 = ConvBlock(64 + 64, 32)
        self.decoder4 = ConvBlock(32 + 32, 16)

        self.final = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, t1, t2):

        f1, s1, s2, s3 = self.encoder(t1)
        f2, _, _, _ = self.encoder(t2)

        # Temporal difference
        diff = torch.abs(f1 - f2)

        # Transformer
        x = self.vit(diff)

        x = self.decoder1(x)

        # 16 -> 32
        x = F.interpolate(
            x,
            size=s3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = self.decoder2(
            torch.cat([x, s3], dim=1)
        )

        # 32 -> 64
        x = F.interpolate(
            x,
            size=s2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = self.decoder3(
            torch.cat([x, s2], dim=1)
        )

        # 64 -> 128
        x = F.interpolate(
            x,
            size=s1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = self.decoder4(
            torch.cat([x, s1], dim=1)
        )

        # 128 -> 256
        x = F.interpolate(
    x,
    size=(128, 128),
    mode="bilinear",
    align_corners=False
)

        return self.final(x)