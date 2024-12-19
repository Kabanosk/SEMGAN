"""SEGAN discriminator model."""
import torch
import torch.nn as nn

from src.utils.model import ConvBlockD


class Discriminator(nn.Module):
    """
    SEGAN Discriminator model.

    This model processes input waveforms through a series of convolutional blocks with Virtual Batch Normalization,
    followed by a final fully connected layer and a sigmoid activation to produce the output.
    """

    def __init__(self):
        """
        Initialize the WaveDiscriminator model with convolutional blocks and final layers.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlockD(2, 32, 31, 2, 15),
            ConvBlockD(32, 64, 31, 2, 15),
            ConvBlockD(64, 64, 31, 2, 15, use_dropout=True),
            ConvBlockD(64, 128, 31, 2, 15),
            ConvBlockD(128, 128, 31, 2, 15),
            ConvBlockD(128, 256, 31, 2, 15, use_dropout=True),
            ConvBlockD(256, 256, 31, 2, 15),
            ConvBlockD(256, 512, 31, 2, 15),
            ConvBlockD(512, 512, 31, 2, 15, use_dropout=True),
            ConvBlockD(512, 1024, 31, 2, 15),
            ConvBlockD(1024, 2048, 31, 2, 15)
        ])
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(0.03)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for all convolutional layers using Xavier Normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the discriminator model.

        Args:
            x: Input waveform tensor to classify.
            ref_x: Reference tensor for computing VBN statistics.

        Returns:
            Output tensor with classification scores after sigmoid activation.
        """
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
