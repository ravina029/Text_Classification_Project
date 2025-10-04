# src/evaluate.py
import joblib
from pathlib import Path
from preprocess import preprocess_dataset
from utils.config import load_config
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(config=None):
    if config is None:
        config = load_config()

    processed_data = preprocess_dataset(config)

    X_test = processed_data['test']['pos'] + processed_data['test']['neg']
    y_test = [1]*len(processed_data['test']['pos']) + [0]*len(processed_data['test']['neg'])

    # Load model and vectorizer from config paths
    model_dir = Path(config["results"].get("model_dir", "results/models"))
    clf = joblib.load(model_dir / "logreg_model.joblib")
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Test Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Optionally save metrics to JSON
    metrics_file = Path(config["results"].get("metrics_file", "results/metrics.json"))
    import json
    metrics_data = {
        "accuracy": accuracy,
        "classification_report": report
    }
    os.makedirs(metrics_file.parent, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    config = load_config()
    evaluate_model(config)
