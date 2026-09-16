import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CNN BLOCK
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
# SHARED SIAMESE CNN ENCODER
# ============================================================

class SiameseEncoder(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        self.block1 = ConvBlock(
            in_channels,
            32
        )

        self.block2 = ConvBlock(
            32,
            64
        )

        self.block3 = ConvBlock(
            64,
            128
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        # 128 × 128
        x1 = self.block1(x)

        # 64 × 64
        x2 = self.block2(
            self.pool(x1)
        )

        # 32 × 32
        x3 = self.block3(
            self.pool(x2)
        )

        return x1, x2, x3


# ============================================================
# LIGHTWEIGHT VISION TRANSFORMER
# ============================================================

class VisionTransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        # We reduce 32×32 → 16×16 before Transformer
        # so CPU training remains practical.

        self.pool = nn.AdaptiveAvgPool2d(
            (16, 16)
        )

        # 16 × 16 = 256 tokens
        self.num_tokens = 16 * 16

        # Learnable positional embeddings
        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_tokens,
                embed_dim
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # x:
        # [B, 128, 32, 32]

        x = self.pool(x)

        # [B, 128, 16, 16]

        batch_size, channels, height, width = x.shape

        # Convert feature map to tokens
        x = x.flatten(2)

        # [B, 128, 256]

        x = x.transpose(1, 2)

        # [B, 256, 128]

        # Add positional information
        x = x + self.positional_embedding

        # Transformer
        x = self.transformer(x)

        # [B, 256, 128]

        # Convert tokens back to feature map
        x = x.transpose(1, 2)

        x = x.reshape(
            batch_size,
            channels,
            height,
            width
        )

        # [B, 128, 16, 16]

        # Return to 32 × 32
        x = F.interpolate(
            x,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# U-NET DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 32 × 32 → 64 × 64
        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        # Transformer feature + CNN skip
        # 64 + 64 = 128
        self.conv1 = ConvBlock(
            128,
            64
        )

        # 64 × 64 → 128 × 128
        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        # 32 + 32 = 64
        self.conv2 = ConvBlock(
            64,
            32
        )

        # Binary change mask
        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(
        self,
        transformer_features,
        skip2,
        skip1
    ):

        # 32 → 64
        x = self.up1(
            transformer_features
        )

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

        # Final change map
        x = self.final(x)

        return x


# ============================================================
# HYBRID SIAMESE CNN + VISION TRANSFORMER + U-NET
# ============================================================

class HybridSiameseCNNViT(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        # One shared encoder for T1 and T2
        self.encoder = SiameseEncoder(
            in_channels=in_channels
        )

        # Transformer processes temporal difference
        self.transformer = VisionTransformerBlock(
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )

        # U-Net style decoder
        self.decoder = Decoder()

    def forward(self, image1, image2):

        # ====================================================
        # T1
        # ====================================================

        t1_1, t1_2, t1_3 = self.encoder(
            image1
        )

        # ====================================================
        # T2
        # ====================================================

        t2_1, t2_2, t2_3 = self.encoder(
            image2
        )

        # ====================================================
        # Temporal feature differences
        # ====================================================

        diff1 = torch.abs(
            t1_1 - t2_1
        )

        diff2 = torch.abs(
            t1_2 - t2_2
        )

        diff3 = torch.abs(
            t1_3 - t2_3
        )

        # ====================================================
        # Global context using Vision Transformer
        # ====================================================

        transformer_features = self.transformer(
            diff3
        )

        # ====================================================
        # Decode
        # ====================================================

        output = self.decoder(
            transformer_features,
            diff2,
            diff1
        )

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING HYBRID SIAMESE CNN + ViT + U-NET")
    print("=" * 60)

    # CPU
    device = torch.device("cpu")

    # 13-band input
    model = HybridSiameseCNNViT(
        in_channels=13
    ).to(device)

    # Fake T1
    image1 = torch.randn(
        2,
        13,
        128,
        128
    ).to(device)

    # Fake T2
    image2 = torch.randn(
        2,
        13,
        128,
        128
    ).to(device)

    print("\nInput T1:")
    print(image1.shape)

    print("\nInput T2:")
    print(image2.shape)

    print("\nRunning forward pass...")

    with torch.no_grad():

        output = model(
            image1,
            image2
        )

    print("\nOutput:")
    print(output.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print("\n" + "=" * 60)
    print("HYBRID MODEL TEST SUCCESSFUL")
    print("=" * 60)