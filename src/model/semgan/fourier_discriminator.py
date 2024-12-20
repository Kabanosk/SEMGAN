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
        self.conv_final = nn.Conv1d(1024, 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(0.03)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x, _, _ = block(x)
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        x = torch.squeeze(x)
        x = self.fully_connected(x)
        return self.sigmoid(x)
