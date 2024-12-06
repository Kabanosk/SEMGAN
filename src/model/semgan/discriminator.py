"""SEGAN discriminator model."""
import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.

    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        # define gamma and beta parameters
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.

        Args:
            x: input tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        Result:
            x: normalized batch tensor
            ref_mean: reference mean tensor over features
            ref_mean_sq: reference squared mean tensor over features
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.

        Args:
            x: input tensor
            mean: mean over features
            mean_sq: squared means over features

        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception('Mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception('Squared mean tensor size not equal to number of features : given {}, expected {}'
                            .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


class ConvBlockD(nn.Module):
    """Discriminator convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.03, use_dropout=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.vbn = VirtualBatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout() if use_dropout else nn.Identity()

    def forward(self, x, ref_mean=None, ref_mean_sq=None):
        x = self.conv(x)
        x, mean, mean_sq = self.vbn(x, ref_mean, ref_mean_sq)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x, mean, mean_sq


class WaveDiscriminator(nn.Module):
    """D"""

    def __init__(self):
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
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, ref_x):
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