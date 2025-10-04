# src/main.py
import argparse
import os
import sys
from pathlib import Path


from src.utils.config import load_config
from src.load_data import load_imdb_dataset
from src.preprocess import preprocess_dataset
from src.train import train_model
from src.evaluate import evaluate_model


def main(stage: str, config_path: str):
    config = load_config(config_path)

    if stage == "load":
        data = load_imdb_dataset(config)
        print(f"Loaded {len(data['train']['pos'])} positive train samples")
    
    elif stage == "preprocess":
        processed = preprocess_dataset(config)
        print(f"Preprocessed {len(processed['train']['pos'])} positive train samples")
    
    elif stage == "train":
        train_model(config)
    
    elif stage == "evaluate":
        evaluate_model(config)
    
    elif stage == "full":
        print("🔹 Step 1: Loading dataset...")
        data = load_imdb_dataset(config)
        print(f"Train samples: {len(data['train']['pos']) + len(data['train']['neg'])}")
        
        print("🔹 Step 2: Preprocessing dataset...")
        processed = preprocess_dataset(config)
        print(f"Preprocessed {len(processed['train']['pos']) + len(processed['train']['neg'])} train samples")
        
        print("🔹 Step 3: Training model...")
        train_model(config)
        
        print("🔹 Step 4: Evaluating model...")
        evaluate_model(config)
    
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["load", "preprocess", "train", "evaluate", "full"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    main(args.stage, args.config)
