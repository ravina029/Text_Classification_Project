# src/preprocess.py
import re
import string
from pathlib import Path
from src.load_data import load_imdb_dataset
from src.utils.config import load_config

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text, config):
    if config["preprocessing"]["lowercase"]:
        text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    if config["preprocessing"]["remove_punctuation"]:
        text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    if config["preprocessing"]["remove_stopwords"]:
        words = [w for w in text.split() if w not in STOPWORDS]
        text = " ".join(words)
    return text

def preprocess_dataset(config=None):
    if config is None:
        config = load_config()
    data = load_imdb_dataset(config)

    processed = {"train": {"pos": [], "neg": []}, "test": {"pos": [], "neg": []}}
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            processed[split][label] = [clean_text(txt, config) for txt in data[split][label]]
    return processed

if __name__ == "__main__":
    config = load_config()
    processed_data = preprocess_dataset(config)
    print(f"Processed train pos: {len(processed_data['train']['pos'])}")
