# src/train.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_dataset
from utils.config import load_config

def train_model(config=None):
    if config is None:
        config = load_config()

    processed_data = preprocess_dataset(config)

    # Merge positive and negative reviews
    X_train = processed_data['train']['pos'] + processed_data['train']['neg']
    y_train = [1]*len(processed_data['train']['pos']) + [0]*len(processed_data['train']['neg'])

    X_test = processed_data['test']['pos'] + processed_data['test']['neg']
    y_test = [1]*len(processed_data['test']['pos']) + [0]*len(processed_data['test']['neg'])

    # TF-IDF vectorizer
    max_features = config["preprocessing"].get("max_features", 5000)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression parameters from config
    lr_params = config["model"].get("logistic_regression", {})
    clf = LogisticRegression(**lr_params)
    clf.fit(X_train_vec, y_train)

    # Save model and vectorizer
    model_dir = Path(config["results"].get("model_dir", "results/models"))
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, model_dir / "logreg_model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    config = load_config()
    train_model(config)
