import torch
import torch.nn as nn

from src.model.utils import Conv2dBlockD


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

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lrelu_final = nn.LeakyReLU(0.03)
        self.fully_connected = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper tensor dimensions.

        Args:
            x: Input tensor of shape (batch_size, channels, freq_bins, time_steps)
            ref_x: Reference tensor of same shape as x

        Returns:
            Tensor of shape (batch_size, 1) containing discrimination scores
        """
        ref_means, ref_meansq = [], []

        for block in self.conv_blocks:
            ref_x, mean, meansq = block(ref_x)
            ref_means.append(mean)
            ref_meansq.append(meansq)

        for i, block in enumerate(self.conv_blocks):
            x, _, _ = block(x, ref_means[i], ref_meansq[i])

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.lrelu_final(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)
