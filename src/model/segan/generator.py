"""SEGAN generator model."""
import torch
import torch.nn as nn


class ConvBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class DeconvBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """G"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvBlockG(1, 16, 32, 2, 15),
            ConvBlockG(16, 32, 32, 2, 15),
            ConvBlockG(32, 32, 32, 2, 15),
            ConvBlockG(32, 64, 32, 2, 15),
            ConvBlockG(64, 64, 32, 2, 15),
            ConvBlockG(64, 128, 32, 2, 15),
            ConvBlockG(128, 128, 32, 2, 15),
            ConvBlockG(128, 256, 32, 2, 15),
            ConvBlockG(256, 256, 32, 2, 15),
            ConvBlockG(256, 512, 32, 2, 15),
            ConvBlockG(512, 1024, 32, 2, 15)
        ])
        self.decoder = nn.ModuleList([
            DeconvBlockG(2048, 512, 32, 2, 15),
            DeconvBlockG(1024, 256, 32, 2, 15),
            DeconvBlockG(512, 256, 32, 2, 15),
            DeconvBlockG(512, 128, 32, 2, 15),
            DeconvBlockG(256, 128, 32, 2, 15),
            DeconvBlockG(256, 64, 32, 2, 15),
            DeconvBlockG(128, 64, 32, 2, 15),
            DeconvBlockG(128, 32, 32, 2, 15),
            DeconvBlockG(64, 32, 32, 2, 15),
            DeconvBlockG(64, 16, 32, 2, 15)
        ])
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)
        self.dec_tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, z):
        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)
        c = enc_outputs[-1]
        encoded = torch.cat((c, z), dim=1)
        for i, block in enumerate(self.decoder):
            encoded = block(encoded, enc_outputs[-(i + 2)])
        out = self.dec_tanh(self.dec_final(encoded))
        return out
