import os
from pathlib import Path

def load_imdb_dataset(base_path="data/raw/aclImdb"):
    """
    Load IMDB dataset from local raw folder.
    Returns dictionary with train/test splits and labels.
    """
    train_path = Path(base_path) / "train"
    test_path = Path(base_path) / "test"

    datasets = {"train": {"pos": [], "neg": []}, "test": {"pos": [], "neg": []}}

    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            folder = Path(base_path) / split / label
            for file in folder.glob("*.txt"):
                with open(file, encoding="utf-8") as f:
                    datasets[split][label].append(f.read())

    return datasets

if __name__ == "__main__":
    data = load_imdb_dataset()
    print(f"Train pos: {len(data['train']['pos'])}, Train neg: {len(data['train']['neg'])}")
    print(f"Test pos: {len(data['test']['pos'])}, Test neg: {len(data['test']['neg'])}")
