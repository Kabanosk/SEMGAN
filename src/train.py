import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.config.config import load_config
from src.utils.data import AudioDataset
from src.utils.logger import create_logger
from src.utils.model import get_model


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the configuration file.",
        required=True,
    )
    return parser.parse_args()


def setup_wandb(config: dict) -> None:
    """Initialize Weights and Biases."""
    wandb.init(
        project=config["wandb"].get("project", "semgan_project"),
        tags=config["wandb"].get("tags", []),
        entity=config["wandb"].get("entity", None),
        config=config,
        save_code=True,
    )


def create_dataloaders(config: dict, device: str) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        config: Configuration dictionary
        device: Device to load data to

    Returns:
        tuple: (train_loader, test_loader) containing clean and noisy data loaders
    """
    # Training data
    train_dataset = AudioDataset(
        base_path=config["data"]["path"], data_name=config["data"]["train"], device=device
    )

    # Test data
    test_dataset = AudioDataset(
        base_path=config["data"]["path"], data_name=config["data"]["test"], device=device
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"].get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, test_loader


def evaluate_model(
    generator: nn.Module,
    discriminators: list[nn.Module],
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    z_dim: int,
    lambdas: list[float],
) -> dict:
    """
    Evaluate the model on test data.

    Returns:
        dict: Dictionary containing test metrics
    """
    generator.eval()
    for d in discriminators:
        d.eval()

    test_g_loss = 0.0
    test_d_losses = [0.0] * len(discriminators)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for clean_batch, clean_mel, clean_fourier, noisy_batch in pbar:
            batch_size = clean_batch.size(0)

            # Move data to device
            clean_data = [
                clean_batch.to(device),
                clean_mel.to(device),
                clean_fourier.to(device),
            ]

            # Create labels
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # Generate noise
            z = torch.randn(batch_size, z_dim, 1).to(device)

            # Generate fake samples
            fake_wave = generator(noisy_batch.to(device), z)
            fake_mel = torchaudio.transforms.MelSpectrogram()(fake_wave)
            fake_fourier = torch.fft.fft(fake_wave)
            fake_data = [fake_wave, fake_mel, fake_fourier]

            # Compute discriminator losses
            for i, discriminator in enumerate(discriminators):
                d_real = (
                    discriminator(clean_data[i], clean_data[i])
                    if i == 0
                    else discriminator(clean_data[i])
                )
                d_fake = (
                    discriminator(fake_data[i], clean_data[i])
                    if i == 0
                    else discriminator(fake_data[i])
                )

                d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)
                test_d_losses[i] += d_loss.item()

            # Compute generator loss
            g_loss = 0
            for i, (discriminator, lambda_i) in enumerate(zip(discriminators, lambdas)):
                d_output = (
                    discriminator(fake_data[i], clean_data[i])
                    if i == 0
                    else discriminator(fake_data[i])
                )
                g_loss += lambda_i * criterion(d_output, real_label)

            test_g_loss += g_loss.item()

    # Calculate average losses
    avg_test_g_loss = test_g_loss / len(test_loader)
    avg_test_d_losses = [d_loss / len(test_loader) for d_loss in test_d_losses]

    return {
        "test_g_loss": avg_test_g_loss,
        **{f"test_d{i}_loss": d_loss for i, d_loss in enumerate(avg_test_d_losses)},
    }


def train_speech_enhancement_gan(
    generator: nn.Module,
    discriminators: list[nn.Module],
    config_path: str,
    device: str = "cuda",
    z_dim: int = 1024,
    lambdas: list[float] = None,
):
    """
    Train the speech enhancement GAN with multiple discriminators using config file.
    """
    # Load configuration
    config = load_config(config_path)
    logger = create_logger()

    # Move models to device
    generator = generator.to(device)
    discriminators = [d.to(device) for d in discriminators]

    # Set default lambda weights if not provided
    if lambdas is None:
        lambdas = [1.0 / len(discriminators)] * len(discriminators)

    # Initialize wandb
    setup_wandb(config)

    # Create output directory
    output_path = Path(config["train"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(config, device)

    # Setup optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config["train"]["lr"],
        betas=(config["train"]["beta1"], config["train"]["beta2"]),
    )

    d_optimizers = [
        optim.Adam(
            d.parameters(),
            lr=config["train"]["lr"],
            betas=(config["train"]["beta1"], config["train"]["beta2"]),
        )
        for d in discriminators
    ]

    # Loss function
    criterion = nn.BCELoss()

    # Training loops
    best_g_loss = float("inf")
    for epoch in range(config["train"]["epochs"]):
        generator.train()
        for d in discriminators:
            d.train()

        running_g_loss = 0.0
        running_d_losses = [0.0] * len(discriminators)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['train']['epochs']}")
        for data in pbar:
            clean_batch, clean_mel, clean_fourier, noisy_batch = data
            batch_size = clean_batch.size(0)

            # Move data to device
            clean_data = [
                clean_batch.to(device),
                clean_mel.to(device),
                clean_fourier.to(device),
            ]

            # Create labels
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # Generate noise
            z = torch.randn(batch_size, z_dim, 8).to(device)

            # Train Discriminators
            d_losses = []
            for i, (discriminator, d_optimizer) in enumerate(
                zip(discriminators, d_optimizers)
            ):
                d_optimizer.zero_grad()

                # Generate fake samples
                fake_wave = generator(noisy_batch.to(device), z)
                fake_mel = torchaudio.transforms.MelSpectrogram(sample_rate=48000)(
                    fake_wave
                )
                fake_fourier = torch.fft.fft(fake_wave)
                fake_data = [fake_wave, fake_mel, fake_fourier]

                d_real = (
                    discriminator(clean_data[i], clean_data[i])
                    if i != 1
                    else discriminator(clean_data[i])
                )
                d_fake = (
                    discriminator(fake_data[i].detach(), clean_data[i])
                    if i != 1
                    else discriminator(fake_data[i].detach())
                )

                d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)
                d_losses.append(d_loss)

                d_loss.backward()
                d_optimizer.step()

                running_d_losses[i] += d_loss.item()

            # Train Generator
            g_optimizer.zero_grad()

            # Generate fake samples again
            fake_wave = generator(noisy_batch.to(device), z)
            fake_mel = torchaudio.transforms.MelSpectrogram()(fake_wave)
            fake_fourier = torch.fft.fft(fake_wave)
            fake_data = [fake_wave, fake_mel, fake_fourier]

            # Compute generator loss
            g_loss = 0
            for i, (discriminator, lambda_i) in enumerate(zip(discriminators, lambdas)):
                d_output = (
                    discriminator(fake_data[i], clean_data[i])
                    if i == 0
                    else discriminator(fake_data[i])
                )
                g_loss += lambda_i * criterion(d_output, real_label)

            g_loss.backward()
            g_optimizer.step()

            running_g_loss += g_loss.item()

            # Update progress bar
            loss_dict = {
                f"D{i}_loss": d_loss.item() for i, d_loss in enumerate(d_losses)
            }
            loss_dict["G_loss"] = g_loss.item()
            pbar.set_description(
                f"Epoch [{epoch + 1}/{config['train']['epochs']}] "
                + " ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
            )

            # Log batch metrics
            wandb.log(
                {
                    "batch_g_loss": g_loss.item(),
                    **{
                        f"batch_d{i}_loss": d_loss.item()
                        for i, d_loss in enumerate(d_losses)
                    },
                }
            )

        # Calculate average training losses
        avg_g_loss = running_g_loss / len(train_loader)
        avg_d_losses = [d_loss / len(train_loader) for d_loss in running_d_losses]

        # Evaluate on test set
        test_metrics = evaluate_model(
            generator, discriminators, test_loader, criterion, device, z_dim, lambdas
        )

        # Log metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_g_loss": avg_g_loss,
                **{f"train_d{i}_loss": d_loss for i, d_loss in enumerate(avg_d_losses)},
                **test_metrics,
            }
        )

        # Save best model
        if test_metrics["test_g_loss"] < best_g_loss:
            best_g_loss = test_metrics["test_g_loss"]
            checkpoint = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                **{
                    f"discriminator_{i}_state_dict": d.state_dict()
                    for i, d in enumerate(discriminators)
                },
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                **{
                    f"d_optimizer_{i}_state_dict": opt.state_dict()
                    for i, opt in enumerate(d_optimizers)
                },
                "best_g_loss": best_g_loss,
            }
            best_model_path = output_path / f"{config['train']['model_name']}_best.pt"
            torch.save(checkpoint, best_model_path)

        # Save regular checkpoint
        if (epoch + 1) % config["train"].get("checkpoint_frequency", 10) == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                **{
                    f"discriminator_{i}_state_dict": d.state_dict()
                    for i, d in enumerate(discriminators)
                },
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                **{
                    f"d_optimizer_{i}_state_dict": opt.state_dict()
                    for i, opt in enumerate(d_optimizers)
                },
            }
            save_path = (
                output_path / f"{config['train']['model_name']}_epoch_{epoch + 1}.pt"
            )
            torch.save(checkpoint, save_path)

        logger.info(f"\nEpoch [{epoch + 1}/{config['train']['epochs']}] Metrics:")
        logger.info(f"Train Generator Loss: {avg_g_loss:.4f}")
        logger.info(f"Test Generator Loss: {test_metrics['test_g_loss']:.4f}")
        for i, d_loss in enumerate(avg_d_losses):
            logger.info(f"Train Discriminator {i} Loss: {d_loss:.4f}")
            logger.info(
                f"Test Discriminator {i} Loss: {test_metrics[f'test_d{i}_loss']:.4f}"
            )

    wandb.finish()
    return generator, discriminators


def main():
    args = parse_arguments()
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator, discriminators = get_model(config["train"]["model_name"])

    trained_models = train_speech_enhancement_gan(
        generator=generator,
        discriminators=discriminators,
        config_path=args.config,
        device=device,
    )


if __name__ == "__main__":
    main()
