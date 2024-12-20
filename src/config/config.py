import logging
from pathlib import Path

import yaml

from src.utils.logger import create_logger


def load_config(path: str | Path) -> dict:
    """Function for loading configuration file.

    Args:
        path: Path to the configuration file.

    Returns:
        dict: Dictionary object with values from the configuration file.
    """
    logger: logging.Logger = create_logger()
    path = Path(path)

    if not path.exists():
        logger.error(f"Config file '{path}' not found.")
        raise ValueError(
            f"Configuration file '{path}' not found, and no default provided."
        )

    try:
        with path.open() as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Loaded configuration is not a dictionary.")

        logger.info(f"Config file '{path}' loaded successfully.")
        return config

    except (FileNotFoundError, PermissionError) as err:
        logger.error(f"Error opening file '{path}': {err}")
        raise err

    except yaml.YAMLError as err:
        logger.error(f"Error parsing YAML file '{path}': {err}")
        raise err

    except ValueError as err:
        logger.error(f"Invalid data format in '{path}': {err}")
        raise err
