import logging
from pathlib import Path

def setup_logger(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tc")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)
    return logger
