"""File with Discriminator model for Fourier Transforms inputs."""

import torch
import torch.nn as nn

from src.model.utils import ConvBlockD


class FourierDiscriminator(nn.Module):
    """Discriminator model for Fourier Transforms."""

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlockD(2, 32, kernel_size=31, stride=2, padding=15),
                ConvBlockD(32, 64, kernel_size=31, stride=2, padding=15),
                ConvBlockD(
                    64, 128, kernel_size=31, stride=2, padding=15, use_dropout=True
                ),
                ConvBlockD(128, 256, kernel_size=31, stride=2, padding=15),
                ConvBlockD(256, 512, kernel_size=31, stride=2, padding=15),
                ConvBlockD(
                    512, 1024, kernel_size=31, stride=2, padding=15, use_dropout=True
                ),
            ]
        )

        # Add adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fully_connected = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reference batch for VBN.

        Args:
            x: Input tensor of shape (batch_size, 2, freq_bins)
            ref_x: Reference tensor for VBN statistics

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
        x = x.squeeze(-1)
        x = self.fully_connected(x)
        return self.sigmoid(x)
