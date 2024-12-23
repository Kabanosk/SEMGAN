import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(nn.Module):
    """Virtual Batch Normalization (VBN) layer for 1D inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Initialize gamma with normal distribution
        self.gamma = Parameter(
            torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1))
        )
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    @staticmethod
    def get_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and squared mean for 1D case."""
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x**2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for 1D VBN."""
        assert (
            len(x.size()) == 3
        ), "VirtualBatchNorm1d expects 3D input (batch, channels, length)"

        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            batch_size = x.size(0)
            new_coeff = 1.0 / (batch_size + 1.0)
            old_coeff = 1.0 - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(
        self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor
    ) -> torch.Tensor:
        """Normalize using computed statistics."""
        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = (x - mean) / std
        return x * self.gamma + self.beta


class VirtualBatchNorm2d(nn.Module):
    """Virtual Batch Normalization (VBN) layer for 2D inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Initialize gamma with normal distribution for 2D case
        self.gamma = Parameter(
            torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1, 1))
        )
        self.beta = Parameter(torch.zeros(1, num_features, 1, 1))

    @staticmethod
    def get_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and squared mean for 2D case."""
        mean = x.mean([2, 3], keepdim=True).mean(0, keepdim=True)
        mean_sq = (x**2).mean([2, 3], keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for 2D VBN."""
        assert (
            len(x.size()) == 4
        ), "VirtualBatchNorm2d expects 4D input (batch, channels, height, width)"

        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            batch_size = x.size(0)
            new_coeff = 1.0 / (batch_size + 1.0)
            old_coeff = 1.0 - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(
        self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor
    ) -> torch.Tensor:
        """Normalize using computed statistics for 2D case."""
        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = (x - mean) / std
        return x * self.gamma + self.beta


class ConvBlockD(nn.Module):
    """1D Discriminator convolutional block with VBN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        negative_slope: float = 0.03,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.vbn = VirtualBatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout() if use_dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x, mean, mean_sq = self.vbn(x, ref_mean, ref_mean_sq)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x, mean, mean_sq


class Conv2dBlockD(nn.Module):
    """2D Discriminator convolutional block with VBN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        negative_slope: float = 0.03,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.vbn = VirtualBatchNorm2d(out_channels)  # Use VBN2d for 2D features
        self.lrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout() if use_dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x, mean, mean_sq = self.vbn(x, ref_mean, ref_mean_sq)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x, mean, mean_sq


class ConvBlockG(nn.Module):
    """Generator convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.prelu(x)
        return x


class DeconvBlockG(nn.Module):
    """Generator deconvolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.prelu(x)
        return x
