import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLSTM(nn.Module):

    def __init__(self, in_channels=13):
        super().__init__()

        # -------------------------------------------------
        # ResNet18 spatial feature extractor
        # -------------------------------------------------
        resnet = models.resnet18(weights=None)

        # Modify first convolution for 13-band Sentinel-2 input
        resnet.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove classification layers
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # ResNet18 output: 512 channels
        # Spatial size for 128x128 input: approximately 4x4

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16, 1,
                kernel_size=1
            )
        )

    def extract_features(self, x):

        x = self.feature_extractor(x)

        return x

    def forward(self, t1, t2):

        # ---------------------------------------------
        # Extract spatial features from both dates
        # ---------------------------------------------

        f1 = self.extract_features(t1)
        f2 = self.extract_features(t2)

        # f1/f2:
        # B x 512 x H x W

        B, C, H, W = f1.shape

        # Convert spatial feature maps into sequences.
        # Each spatial location becomes a timestep.
        f1_seq = f1.flatten(2).transpose(1, 2)
        f2_seq = f2.flatten(2).transpose(1, 2)

        # ---------------------------------------------
        # Temporal sequence
        # ---------------------------------------------

        sequence = torch.stack(
            [f1_seq, f2_seq],
            dim=1
        )

        # B x 2 x spatial_locations x 512

        sequence = sequence.permute(
            0, 2, 1, 3
        )

        # B x spatial_locations x 2 x 512

        sequence = sequence.reshape(
            B * H * W,
            2,
            C
        )

        # ---------------------------------------------
        # LSTM
        # ---------------------------------------------

        lstm_out, _ = self.lstm(sequence)

        # Use the temporal difference representation
        temporal_feature = (
            lstm_out[:, -1, :]
            - lstm_out[:, 0, :]
        )

        # Restore spatial structure
        temporal_feature = temporal_feature.reshape(
            B, H, W, 256
        )

        temporal_feature = temporal_feature.permute(
            0, 3, 1, 2
        )

        # ---------------------------------------------
        # Decoder
        # ---------------------------------------------

        output = self.decoder(
            temporal_feature
        )

        # Ensure exact 128x128 output
        output = nn.functional.interpolate(
            output,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        return output