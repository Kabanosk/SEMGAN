from src.model.segan.generator import Generator
from src.model.segan.discriminator import Discriminator as SeganDiscriminator
from src.model.semgan.discriminator import WaveDiscriminator


def get_model(model_name: str) -> tuple[Generator, SeganDiscriminator | WaveDiscriminator]:
    """Function to create instance of model.

    Args:
        model_name: Name of GAN model to load.

    Returns:
        Generator: Generator object - same for each model name.
        Discriminator: Discriminator object - type depend on selected model name.

    Raises:
        ValueError: If parameter model_name is not supported.

    """
    match model_name.lower():
        case "segan":
            return Generator(), SeganDiscriminator()
        case "semgan":
            return Generator(), WaveDiscriminator()
        case _:
            raise ValueError("Not supported model name. Model name should be from ['segan', 'semgan'].")