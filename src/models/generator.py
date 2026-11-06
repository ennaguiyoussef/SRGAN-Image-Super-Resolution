"""SRResNet-style 4x generator for SRGAN."""
from torch import nn


class ResidualBlock(nn.Module):
    """Learn a residual correction at constant spatial resolution."""

    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.body(x)


class UpsampleBlock(nn.Module):
    """Learn 4C channels, then rearrange them to C channels at 2x resolution."""

    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return self.body(x)


class Generator(nn.Module):
    """Map RGB LR [0, 1] to RGB SR [-1, 1] at 4x height and width.

    The default architecture uses 16 residual blocks and 64 feature channels.
    Channel-wise PReLU and a final Tanh are explicit implementation choices.
    Spatial dimensions are unrestricted positive integers at inference.
    """

    def __init__(self, num_residual_blocks=16, channels=64):
        super().__init__()
        if type(num_residual_blocks) is not int or num_residual_blocks < 1:
            raise ValueError("num_residual_blocks must be a positive integer.")
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer.")
        self.head = nn.Sequential(nn.Conv2d(3, channels, 9, padding=4), nn.PReLU(channels))
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_residual_blocks))
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.upsample1 = UpsampleBlock(channels)
        self.upsample2 = UpsampleBlock(channels)
        self.output = nn.Sequential(nn.Conv2d(channels, 3, 9, padding=4), nn.Tanh())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Generator expects NCHW RGB input (N, 3, H, W).")
        features = self.head(x)
        features = features + self.trunk(self.residual_blocks(features))
        return self.output(self.upsample2(self.upsample1(features)))
