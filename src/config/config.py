from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as file:
        config = yaml.safe_load(file)
    return config
