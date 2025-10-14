import yaml
from pathlib import Path

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
