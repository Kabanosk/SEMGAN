import sys
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Function for loading configuration file.

    Args:
        path: Path to configuration file.

    Returns:
        dict: Dictionary object with values from configuration file.
    """
    try:
        with open(path) as file:
            config = yaml.safe_load(file)
        return config

    except (FileNotFoundError, PermissionError) as err:
        print(f"Error related to file of {path = } occurred: {repr(err)}.")
        sys.exit(0)

    except yaml.YAMLError as err:
        print(f"Error related to yaml file of {path = } occurred: {repr(err)}.")
        sys.exit(0)
