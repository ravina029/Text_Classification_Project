# src/utils/config.py
import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(Path(config_path), "r") as file:
        config = yaml.safe_load(file)
    return config
