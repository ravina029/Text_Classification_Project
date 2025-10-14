import argparse
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.features import build_tfidf, save_vectorizer

def train_classical(cfg):
    logger = setup_logger(cfg["paths"]["logs_dir"] + "/train.log")
    df = __import__("pandas").read_csv(cfg["paths"]["processed_dir"] + "/imdb_preprocessed.csv")
    X = df["final_review"] if "final_review" in df.columns else df["processed_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    vec, X_train_vec = build_tfidf(X_train, max_features=cfg["features"]["tfidf_max_features"], ngram_range=tuple(cfg["features"]["tfidf_ngram_range"]))
    X_test_vec = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    logger.info("Training LogisticRegression")
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    logger.info("Accuracy: %.4f", accuracy_score(y_test, preds))
    logger.info("\n" + classification_report(y_test, preds))
    Path(cfg["paths"]["results_dir"]).mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, Path(cfg["paths"]["results_dir"]) / "logreg.joblib")
    save_vectorizer(vec, Path(cfg["paths"]["results_dir"]) / "tfidf_vectorizer.joblib")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_classical(cfg)

if __name__ == "__main__":
    main()
