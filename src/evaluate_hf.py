# src/evaluate_hf.py

import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from src.utils.config import load_config


def evaluate_model():
    # 🔹 Load configuration
    config = load_config()
    hf_cfg = config["huggingface"]

    # 🔹 Load processed dataset
    data_path = os.path.join(config["paths"]["processed_dir"], "imdb_preprocessed.csv")
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 🔹 Convert to Hugging Face Dataset
    test_ds = Dataset.from_pandas(test_df)

    # 🔹 Load model and tokenizer
    model_path = hf_cfg["model_save_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 🔹 Detect device automatically
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"✅ Using device: {device}")

    # 🔹 Tokenize data
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config["tokenization"]["max_length"],
        )

    test_ds = test_ds.map(tokenize, batched=True)
    test_ds = test_ds.rename_column("label", "labels")
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 🔹 Initialize Trainer for evaluation
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # 🔹 Evaluate
    print("🔍 Evaluating model...")
    results = trainer.evaluate(test_ds)
    print("📊 Evaluation Results:", results)

    # 🔹 Example predictions
    examples = [
        "This movie was absolutely amazing!",
        "The plot was boring and predictable.",
    ]
    for text in examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            sentiment = "positive" if pred == 1 else "negative"
        print(f"Text: {text}\nPrediction: {sentiment}\n")


if __name__ == "__main__":
    evaluate_model()
