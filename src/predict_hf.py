# src/predict_hf.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.utils.yaml_utils import load_config

def predict_with_hf(text: str):
    config = load_config("src/config.yaml")
    model_dir = config["huggingface"]["model_save_dir"]

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()

    label = "positive" if pred == 1 else "negative"
    return label


if __name__ == "__main__":
    print(predict_with_hf("This movie was fantastic!"))
    print(predict_with_hf("The plot was dull and predictable."))
