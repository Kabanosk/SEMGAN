import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train script for a SEMGAN model for audio enhancement"
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


def setup_wandb(config: dict) -> None:
    """Initialize Weights & Biases logging.

    Args:
        config: Dictionary containing configuration parameters

    The function expects the following keys in the config dictionary:
        - wandb_project: Name of the W&B project
        - wandb_entity: W&B entity (username or team name)
        - wandb_tags: List of tags for the run
        - wandb_notes: Notes for the run

    Additional configuration parameters will be logged as part of the run config.
    """
    try:
        wandb_config = config["wandb"]
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            config=wandb_config,
            tags=wandb_config["tags"],
            notes=wandb_config["notes"],
        )

        # Log important config parameters explicitly
        wandb.run.summary["batch_size"] = config["train"]["batch_size"]
        wandb.run.summary["learning_rate"] = config["train"]["lr"]
        wandb.run.summary["model_type"] = config["train"]["model_name"]
        wandb.run.summary["beta1"] = config["train"]["beta1"]
        wandb.run.summary["beta2"] = config["train"]["beta2"]

    except Exception as e:
        print(f"Warning: Failed to initialize W&B logging: {str(e)}")
        print("Training will continue without W&B logging")


def process_discriminator_input(
    discriminator: nn.Module,
    clean_wave: torch.Tensor,
    clean_mel: torch.Tensor,
    clean_fourier: torch.Tensor,
    noisy_wave: torch.Tensor,
    fake_wave: torch.Tensor,
    device: torch.device | str,
    is_training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process and prepare input tensors for different types of discriminators.

    This function handles three types of discriminators (wave, mel, and fourier) and prepares
    the appropriate input format for each type. It processes both real (clean) and fake
    (generated) inputs according to the discriminator type.

    Args:
        discriminator: Neural network discriminator module.
        clean_wave: Clean waveform tensor, shape (batch_size, channels, samples).
        clean_mel: Clean mel-spectrogram tensor.
        clean_fourier: Clean Fourier transform tensor.
        noisy_wave: Noisy waveform tensor, shape (batch_size, channels, samples).
        fake_wave: Generated/fake waveform tensor, shape (batch_size, channels, samples).
        device: PyTorch device to use for computations.
        is_training: Whether the model is in training mode. If True, detaches fake wave
            gradient for discriminator training. Defaults to True.

    Returns:
        Tuple:
            - real_input: Processed clean/real input tensor appropriate for the discriminator type.
            - fake_input: Processed fake/generated input tensor appropriate for the discriminator type.
    """
    if isinstance(discriminator, type(discriminators[0])):  # Wave discriminator
        if clean_wave.size(1) == 1:
            clean_wave = clean_wave.expand(-1, 2, -1)
        if noisy_wave.size(1) == 1:
            noisy_wave = noisy_wave.expand(-1, 2, -1)

        fake_wave_temp = fake_wave.detach() if is_training else fake_wave
        if fake_wave_temp.size(1) == 1:
            fake_wave_temp = fake_wave_temp.expand(-1, 2, -1)

        real_input = clean_wave
        fake_input = fake_wave_temp

    elif isinstance(discriminator, type(discriminators[1])):  # Mel discriminator
        real_input = clean_mel
        fake_wave_temp = fake_wave.detach() if is_training else fake_wave

        # Compute STFT
        window = torch.hann_window(1024).to(device)
        stft = torch.stft(
            fake_wave_temp.squeeze(1),
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            window=window,
            return_complex=True,
            normalized=True,
        )
        fake_input = torch.abs(stft).unsqueeze(1)
    else:  # Fourier discriminator
        real_input = clean_fourier
        fake_wave_temp = fake_wave.detach() if is_training else fake_wave

        # Compute FFT and get real/imaginary parts
        fft = torch.fft.fft(fake_wave_temp.squeeze(1))
        fake_input = torch.stack([fft.real, fft.imag], dim=1)

    return real_input, fake_input


def train_gan(
    generator: nn.Module,
    discriminators: list[nn.Module],
    train_dataloader: DataLoader,
    config: dict,
    device: torch.device | str = "cuda",
) -> None:
    generator = generator.to(device)
    discriminators = [d.to(device) for d in discriminators]

    # Setup optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config["train"]["lr"],
        betas=(config["train"]["beta1"], config["train"]["beta2"]),
    )
    d_optimizers = [
        torch.optim.Adam(
            d.parameters(),
            lr=config["train"]["lr"],
            betas=(config["train"]["beta1"], config["train"]["beta2"]),
        )
        for d in discriminators
    ]

    # Loss functions and constants
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    NUM_EPOCHS = config["train"]["epochs"]
    LAMBDA_L1 = 100

    # Create output directory
    output_path = Path(config["train"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        generator.train()
        for d in discriminators:
            d.train()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, (clean_wave, clean_mel, clean_fourier, noisy_wave) in enumerate(
            progress_bar
        ):
            batch_size = clean_wave.size(0)

            # Ensure correct input dimensions
            if clean_wave.dim() == 2:
                clean_wave = clean_wave.unsqueeze(1)
            if noisy_wave.dim() == 2:
                noisy_wave = noisy_wave.unsqueeze(1)

            # Move to device
            clean_wave = clean_wave.to(device)
            clean_mel = clean_mel.to(device)
            clean_fourier = clean_fourier.to(device)
            noisy_wave = noisy_wave.to(device)

            # Labels
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # Generate noise
            z = torch.randn(batch_size, 1024, 8).to(device)

            try:
                # Generate fake samples
                fake_wave = generator(noisy_wave, z)

                # Train Discriminators
                d_losses = []
                for idx, (discriminator, d_optimizer) in enumerate(
                    zip(discriminators, d_optimizers)
                ):
                    d_optimizer.zero_grad()

                    real_input, fake_input = process_discriminator_input(
                        discriminator,
                        clean_wave,
                        clean_mel,
                        clean_fourier,
                        noisy_wave,
                        fake_wave,
                        device,
                    )

                    d_real = discriminator(real_input, real_input)
                    d_fake = discriminator(fake_input, real_input)

                    d_loss_real = adversarial_loss(d_real, real_label)
                    d_loss_fake = adversarial_loss(d_fake, fake_label)
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_losses.append(d_loss)

                    d_loss.backward(
                        retain_graph=True if idx < len(discriminators) - 1 else False
                    )
                    d_optimizer.step()

                # Train Generator
                g_optimizer.zero_grad()

                g_losses = []
                for discriminator in discriminators:
                    _, fake_input = process_discriminator_input(
                        discriminator,
                        clean_wave,
                        clean_mel,
                        clean_fourier,
                        noisy_wave,
                        fake_wave,
                        device,
                        is_training=False,
                    )

                    g_fake = discriminator(fake_input, fake_input)
                    g_loss = adversarial_loss(g_fake, real_label)
                    g_losses.append(g_loss)

                reconstruction_loss = l1_loss(fake_wave, clean_wave)
                g_loss_total = sum(g_losses) + LAMBDA_L1 * reconstruction_loss

                g_loss_total.backward()
                g_optimizer.step()

                # Logging
                if wandb.run is not None:
                    wandb.log(
                        {
                            "d_loss_wave": d_losses[0].item(),
                            "d_loss_mel": d_losses[1].item(),
                            "d_loss_fourier": d_losses[2].item(),
                            "g_loss_wave": g_losses[0].item(),
                            "g_loss_mel": g_losses[1].item(),
                            "g_loss_fourier": g_losses[2].item(),
                            "g_loss_reconstruction": reconstruction_loss.item(),
                            "g_loss_total": g_loss_total.item(),
                        }
                    )

                progress_bar.set_postfix(
                    {
                        "D_loss": sum(d_losses).item() / len(d_losses),
                        "G_loss": g_loss_total.item(),
                    }
                )

            except RuntimeError as e:
                print(f"Error during training: {str(e)}")
                print(f"Batch index: {batch_idx}")
                continue

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dicts": [
                        d.state_dict() for d in discriminators
                    ],
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dicts": [
                        d_opt.state_dict() for d_opt in d_optimizers
                    ],
                },
                output_path / f"checkpoint_epoch_{epoch + 1}.pt",
            )


if __name__ == "__main__":
    from src.config.config import load_config
    from src.utils.data import AudioDataset
    from src.utils.model import get_model

    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    setup_wandb(config)

    generator, discriminators = get_model("semgan")

    dataset = AudioDataset(
        base_path=config["data"]["path"],
        data_name=config["data"]["train"],
        device=device,
        segment_length=16384,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    train_gan(generator, discriminators, dataloader, config, device)
