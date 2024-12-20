import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(nn.Module):
    """
    Virtual Batch Normalization (VBN) layer.

    VBN computes batch statistics based on a reference batch and applies
    normalization during the forward pass.

    Implementation inspired by Rafael_Valle and discussion by SimonW.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initialize the VBN layer.

        Args:
            num_features: Number of features in the input tensor.
            eps: Small value to prevent division by zero. Default is 1e-5.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(
            torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1))
        )
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    @staticmethod
    def get_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and squared mean for the given batch.

        Args:
            x: Input tensor containing batch of activations.

        Returns:
            mean: Mean tensor over features.
            mean_sq: Squared mean tensor over features.
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x**2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass of VBN.

        If reference statistics are provided, use them for normalization.
        Otherwise, compute and detach new reference statistics.

        Args:
            x: Input tensor to normalize.
            ref_mean: Reference mean tensor (optional).
            ref_mean_sq: Reference squared mean tensor (optional).

        Returns:
            torch.Tensor: Normalized tensor.
            torch.Tensor: Mean tensor computed from the input batch.
            torch.Tensor: Squared mean tensor computed from the input batch.
        """
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
        """
        Normalize the input tensor using provided statistics.

        Args:
            x: Input tensor.
            mean: Mean tensor over features.
            mean_sq: Squared mean tensor over features.

        Returns:
            x: Normalized tensor.
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                "Mean tensor size not equal to number of features: given {}, expected {}".format(
                    mean.size(1), self.num_features
                )
            )
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                "Squared mean tensor size not equal to number of features: given {}, expected {}".format(
                    mean_sq.size(1), self.num_features
                )
            )

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self) -> str:
        return "{name}(num_features={num_features}, eps={eps})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class ConvBlockD(nn.Module):
    """Discriminator convolutional block with Virtual Batch Normalization.

    This block consists of a convolutional layer, VBN layer, LeakyReLU activation,
    and an optional dropout layer.
    """

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
        """
        Initialize the convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution. Default is 1.
            padding: Padding added to the input. Default is 0.
            negative_slope: Negative slope for LeakyReLU. Default is 0.03.
            use_dropout: Whether to apply dropout. Default is False.
        """
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
        """
        Perform forward pass of the convolutional block.

        Args:
            x: Input tensor.
            ref_mean: Reference mean tensor for VBN (optional).
            ref_mean_sq: Reference squared mean tensor for VBN (optional).

        Returns:
            x: Output tensor after applying all operations.
            mean: Mean tensor computed from the input batch.
            mean_sq: Squared mean tensor computed from the input batch.
        """
        x = self.conv(x)
        x, mean, mean_sq = self.vbn(x, ref_mean, ref_mean_sq)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x, mean, mean_sq


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
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.prelu = nn.PReLU()

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.prelu(x)
        return x


class Conv2dBlockD(nn.Module):
    """Discriminator 2D convolutional block for Mel Spectrogram."""

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
        """
        Initialize the 2D convolutional block for Mel spectrogram.

        Args:
            in_channels: Number of input channels (e.g., 1 for a single-channel mel spectrogram).
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution. Default is 1.
            padding: Padding added to the input. Default is 0.
            negative_slope: Negative slope for LeakyReLU. Default is 0.03.
            use_dropout: Whether to apply dropout. Default is False.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.vbn = VirtualBatchNorm1d(out_channels)  # Use VBN for 2D features
        self.lrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout() if use_dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: torch.Tensor = None,
        ref_mean_sq: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass of the 2D convolutional block.

        Args:
            x: Input tensor (mel spectrogram).
            ref_mean: Reference mean tensor for VBN (optional).
            ref_mean_sq: Reference squared mean tensor for VBN (optional).

        Returns:
            x: Output tensor after applying all operations.
            mean: Mean tensor computed from the input batch.
            mean_sq: Squared mean tensor computed from the input batch.
        """
        x = self.conv(x)
        x, mean, mean_sq = self.vbn(x, ref_mean, ref_mean_sq)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x, mean, mean_sq
