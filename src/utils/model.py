"""Utility functions and classes for the model."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from src.model.segan.discriminator import Discriminator as SeganDiscriminator
from src.model.segan.generator import Generator
from src.model.semgan.fourier_discriminator import FourierDiscriminator
from src.model.semgan.mel_discriminator import MelDiscriminator


def get_model(model_name: str) -> tuple[Generator, list[nn.Module]]:
    """Function to create instance of model with a list of discriminators.

    Args:
        model_name: Name of GAN model to load. This determines which type of discriminators to return.

    Returns:
        Generator: Generator object (same for each model name).
        list[torch.nn.Module]: List of Discriminator objects - one or more depending on the model name.

    Raises:
        ValueError: If parameter model_name is not supported.

    """
    generator = Generator()

    match model_name.lower():
        case "segan":
            discriminators = [SeganDiscriminator()]
        case "semgan":
            discriminators = [
                SeganDiscriminator(),  # Discriminator for waveform input same as the SEGAN model
                MelDiscriminator(),  # Discriminator for mel spectrogram input
                FourierDiscriminator(),  # Discriminator for Fourier transform input
            ]
        case _:
            raise ValueError(
                "Not supported model name. Model name should be from ['segan', 'semgan']."
            )

    return generator, discriminators
