import torch.nn as nn

from src.utils.model import Conv2dBlockD


class MelDiscriminator(nn.Module):
    """Discriminator model for Mel spectrograms."""

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [
                Conv2dBlockD(1, 32, 3, stride=2, padding=1),
                Conv2dBlockD(32, 64, 3, stride=2, padding=1),
                Conv2dBlockD(64, 128, 3, stride=2, padding=1, use_dropout=True),
                Conv2dBlockD(128, 256, 3, stride=2, padding=1),
                Conv2dBlockD(256, 512, 3, stride=2, padding=1),
                Conv2dBlockD(512, 1024, 3, stride=2, padding=1, use_dropout=True),
            ]
        )
        self.conv_final = nn.Conv2d(1024, 1, kernel_size=1)
        self.lrelu_final = nn.LeakyReLU(0.03)
        self.fully_connected = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        """Initialize weights for all convolutional layers using Xavier Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the Mel Discriminator."""
        ref_means, ref_meansq = [], []
        for block in self.conv_blocks:
            ref_x, mean, meansq = block(ref_x)
            ref_means.append(mean)
            ref_meansq.append(meansq)

        for i, block in enumerate(self.conv_blocks):
            x, _, _ = block(x, ref_means[i], ref_meansq[i])

        x = self.conv_final(x)
        x = self.lrelu_final(x)
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)
