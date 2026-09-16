import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SWIN-STYLE TRANSFORMER BLOCK
# ============================================================

class SwinBlock(nn.Module):

    def __init__(self, dim, num_heads=4, window_size=4):
        super().__init__()

        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # Convert feature map to tokens
        tokens = x.flatten(2).transpose(1, 2)

        # Self-attention
        residual = tokens

        tokens = self.norm1(tokens)

        attention_output, _ = self.attention(
            tokens,
            tokens,
            tokens
        )

        tokens = residual + attention_output

        # Feed-forward network
        residual = tokens

        tokens = self.norm2(tokens)

        tokens = self.mlp(tokens)

        tokens = residual + tokens

        # Restore feature map
        x = tokens.transpose(1, 2).reshape(
            B, C, H, W
        )

        return x


# ============================================================
# SWIN ENCODER
# ============================================================

class SwinEncoder(nn.Module):

    def __init__(self, in_channels=13):

        super().__init__()

        # 128 -> 64
        self.enc1 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(2)

        # 64 -> 32
        self.enc2 = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2)

        # 32 -> 16
        self.enc3 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(2)

        # Transformer bottleneck
        self.transformer = nn.Sequential(

            SwinBlock(
                dim=128,
                num_heads=4
            ),

            SwinBlock(
                dim=128,
                num_heads=4
            )
        )

    def forward(self, x):

        x1 = self.enc1(x)

        x2 = self.enc2(
            self.pool1(x1)
        )

        x3 = self.enc3(
            self.pool2(x2)
        )

        bottleneck = self.pool3(x3)

        bottleneck = self.transformer(
            bottleneck
        )

        return x1, x2, x3, bottleneck


# ============================================================
# SWIN TRANSFORMER + U-NET
# ============================================================

class SwinTransformerUNet(nn.Module):

    def __init__(self, in_channels=13):

        super().__init__()

        # Shared encoder for both dates
        self.encoder = SwinEncoder(
            in_channels
        )

        # Combine temporal features
        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # U-NET DECODER
        # ----------------------------------------------------

        # 16 -> 32
        self.up3 = nn.ConvTranspose2d(
            128,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = nn.Sequential(

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 32 -> 64
        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 64 -> 128
        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final segmentation layer
        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, t1, t2):

        # ----------------------------------------------------
        # Extract features from both dates
        # ----------------------------------------------------

        t1_x1, t1_x2, t1_x3, t1_b = self.encoder(t1)

        t2_x1, t2_x2, t2_x3, t2_b = self.encoder(t2)

        # ----------------------------------------------------
        # Temporal feature differences
        # ----------------------------------------------------

        x1 = torch.abs(
            t2_x1 - t1_x1
        )

        x2 = torch.abs(
            t2_x2 - t1_x2
        )

        x3 = torch.abs(
            t2_x3 - t1_x3
        )

        bottleneck = torch.abs(
            t2_b - t1_b
        )

        bottleneck = self.bottleneck(
            bottleneck
        )

        # ----------------------------------------------------
        # U-NET DECODER
        # ----------------------------------------------------

        d3 = self.up3(
            bottleneck
        )

        d3 = torch.cat(
            [d3, x3],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [d2, x2],
            dim=1
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [d1, x1],
            dim=1
        )

        d1 = self.dec1(
            d1
        )

        # Final change map
        output = self.final(
            d1
        )

        # Guarantee exact 128 × 128 output
        output = F.interpolate(
            output,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        return output