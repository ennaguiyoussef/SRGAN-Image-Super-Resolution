"""Image-level SRGAN discriminator for 96x96 training crops."""
from torch import nn


class Discriminator(nn.Module):
    """Map RGB 96x96 images in [-1, 1] to one raw logit per image.

    No sigmoid is applied: use BCEWithLogitsLoss for future adversarial training.
    Sigmoid is appropriate only when displaying a probability-like score.
    This classifier is fixed to 96x96 crops; it is not used for SR inference.
    """

    def __init__(self):
        super().__init__()
        channels = [3, 64, 64, 128, 128, 256, 256, 512, 512]
        strides = [1, 2, 1, 2, 1, 2, 1, 2]
        blocks = []
        for index, stride in enumerate(strides):
            layers = [nn.Conv2d(channels[index], channels[index + 1], 3, stride, padding=1)]
            if index:
                layers.append(nn.BatchNorm2d(channels[index + 1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 96, 96):
            raise ValueError("Discriminator expects (N, 3, 96, 96).")
        return self.classifier(self.features(x))
