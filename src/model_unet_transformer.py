import torch
import torch.nn as nn


# ---------------------------------------------------------
# Basic convolution block
# ---------------------------------------------------------
class DoubleConv(nn.Module):
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


# ---------------------------------------------------------
# U-Net + Transformer Change Detection Network
# ---------------------------------------------------------
class UNetTransformer(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # -------------------------
        # Transformer bottleneck
        # -------------------------
        self.transformer_pool = nn.AdaptiveAvgPool2d((16, 16))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.up3 = nn.ConvTranspose2d(
            128, 128, kernel_size=2, stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        # Final change map
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def encode(self, x):

        e1 = self.enc1(x)       # 128 × 128
        e2 = self.enc2(self.pool1(e1))   # 64 × 64
        e3 = self.enc3(self.pool2(e2))    # 32 × 32

        bottleneck = self.pool3(e3)       # 16 × 16

        return e1, e2, e3, bottleneck

    def transformer_block(self, x):

        # x: B × 128 × 16 × 16

        B, C, H, W = x.shape

        # Convert feature map into tokens
        x = x.flatten(2).transpose(1, 2)

        # Transformer
        x = self.transformer(x)

        # Convert tokens back to feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x

    def decode(self, x, e1, e2, e3):

        # 16 → 32
        x = self.up3(x)

        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        # 32 → 64
        x = self.up2(x)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        # 64 → 128
        x = self.up1(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.final(x)

    def forward(self, t1, t2):

        # Encode both temporal images
        e1_t1, e2_t1, e3_t1, b_t1 = self.encode(t1)
        e1_t2, e2_t2, e3_t2, b_t2 = self.encode(t2)

        # Temporal feature differences
        e1 = torch.abs(e1_t1 - e1_t2)
        e2 = torch.abs(e2_t1 - e2_t2)
        e3 = torch.abs(e3_t1 - e3_t2)

        # Difference at bottleneck
        x = torch.abs(b_t1 - b_t2)

        # Transformer processes temporal difference
        x = self.transformer_block(x)

        # U-Net decoder
        output = self.decode(x, e1, e2, e3)

        return output