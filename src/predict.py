import argparse
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_classical(model_path, vectorizer_path, text):
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    X = vec.transform([text])
    pred = clf.predict(X)[0]
    return pred

def predict_hf(model_dir, text, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return int(out.logits.argmax().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--mode", choices=["classical","hf"], default="classical")
    parser.add_argument("--model", type=str, default="results/logreg.joblib")
    parser.add_argument("--vec", type=str, default="results/tfidf_vectorizer.joblib")
    parser.add_argument("--hf_dir", type=str, default="results/hf_outputs/distilbert_imdb")
    args = parser.parse_args()
    if args.mode == "classical":
        print(predict_classical(args.model, args.vec, args.text))
    else:
        print(predict_hf(args.hf_dir, args.text))

if __name__ == "__main__":
    main()
