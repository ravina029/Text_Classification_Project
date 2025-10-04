# src/utils/config.py

from pathlib import Path
import yaml

def load_config(filename="config.yaml"):
    """
    Load the project configuration file.

    Args:
        filename (str): Name of the config file (default: config.yaml).
                        The function looks inside the project_root/config/ folder.

    Returns:
        dict: Parsed YAML configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """

    # Get project root: 2 levels up from this file (utils/ -> src/ -> project_root)
    project_root = Path(__file__).resolve().parents[2]

    # Full path to config file
    config_path = project_root / "config" / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load YAML safely
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
