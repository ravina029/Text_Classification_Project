from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

def build_tfidf(corpus, max_features=20000, ngram_range=(1,2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(corpus)
    return vec, X

def save_vectorizer(vec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)
