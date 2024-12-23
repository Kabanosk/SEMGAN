"""Utility functions and classes for the model."""

from pathlib import Path

import torch
import torch.nn as nn

from src.model.segan.discriminator import Discriminator as SeganDiscriminator
from src.model.segan.generator import Generator
from src.model.semgan.fourier_discriminator import FourierDiscriminator
from src.model.semgan.mel_discriminator import MelDiscriminator


def load_model(
    generator: nn.Module,
    discriminators: list[nn.Module] | None = None,
    checkpoint_path: str | Path = None,
    g_optimizer: torch.optim.Optimizer | None = None,
    d_optimizers: list[torch.optim.Optimizer] | None = None,
    device: torch.device | str = "cuda",
    generator_only: bool = False,
) -> tuple[nn.Module, list[nn.Module] | None, int]:
    """Load saved model checkpoint states for generator and optionally discriminators.

    This function loads a checkpoint containing states for generator and optionally discriminators
    and their respective optimizers if provided. It handles loading to the specified
    device and ensures compatibility of state dictionaries.

    Args:
        generator: Generator neural network module.
        discriminators: List of discriminator neural network modules. Optional if generator_only=True.
        checkpoint_path: Path to the checkpoint file.
        g_optimizer: Optional generator optimizer. If provided, its state will be loaded.
        d_optimizers: Optional list of discriminator optimizers. If provided, their states will be loaded.
        device: Device to load the models to. Defaults to "cuda".
        generator_only: If True, loads only the generator state. Defaults to False.

    Returns:
        Tuple containing:
            - Loaded generator model
            - List of loaded discriminator models (None if generator_only=True)
            - Epoch number from the checkpoint

    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist.
        RuntimeError: If checkpoint structure doesn't match expected format.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator state
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)

    # Optionally load generator optimizer state
    if g_optimizer is not None and "g_optimizer_state_dict" in checkpoint:
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])

    # If only loading generator, return early
    if generator_only:
        epoch = checkpoint.get("epoch", -1)
        return generator, None, epoch

    # Load discriminator states
    if discriminators is None:
        raise ValueError("discriminators must be provided when generator_only=False")

    if len(discriminators) != len(checkpoint["discriminator_state_dicts"]):
        raise RuntimeError(
            f"Number of discriminators ({len(discriminators)}) doesn't match "
            f"checkpoint ({len(checkpoint['discriminator_state_dicts'])})"
        )

    for discriminator, state_dict in zip(
        discriminators, checkpoint["discriminator_state_dicts"]
    ):
        discriminator.load_state_dict(state_dict)
        discriminator = discriminator.to(device)

    # Load discriminator optimizer states
    if d_optimizers is not None and "d_optimizer_state_dicts" in checkpoint:
        if len(d_optimizers) != len(checkpoint["d_optimizer_state_dicts"]):
            raise RuntimeError(
                f"Number of discriminator optimizers ({len(d_optimizers)}) doesn't match "
                f"checkpoint ({len(checkpoint['d_optimizer_state_dicts'])})"
            )

        for optimizer, state_dict in zip(
            d_optimizers, checkpoint["d_optimizer_state_dicts"]
        ):
            optimizer.load_state_dict(state_dict)

    epoch = checkpoint.get("epoch", -1)

    return generator, discriminators, epoch


from pathlib import Path
from typing import Optional


def get_model(
    model_name: str, checkpoint: Optional[str | Path] = None
) -> tuple[Generator, list[nn.Module]]:
    """Function to create instance of model with a list of discriminators.

    Creates a new model instance and optionally loads weights from checkpoint file.

    Args:
        model_name: Name of GAN model to load. This determines which type of discriminators to return.
        checkpoint: Optional path to checkpoint file containing model states.

    Returns:
        Generator: Generator object (same for each model name).
        list[torch.nn.Module]: List of Discriminator objects - one or more depending on the model name.

    Raises:
        ValueError: If parameter model_name is not supported.
        FileNotFoundError: If checkpoint path is invalid.
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

    if checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Load the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(state_dict["generator_state_dict"])

        if "discriminator_state_dicts" in state_dict:
            if len(discriminators) != len(state_dict["discriminator_state_dicts"]):
                raise RuntimeError(
                    f"Number of discriminators ({len(discriminators)}) doesn't match "
                    f"checkpoint ({len(state_dict['discriminator_state_dicts'])})"
                )

            for d, state in zip(
                discriminators, state_dict["discriminator_state_dicts"]
            ):
                d.load_state_dict(state)

        generator = generator.to(device)
        discriminators = [d.to(device) for d in discriminators]

    return generator, discriminators
