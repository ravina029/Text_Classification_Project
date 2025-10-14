import re, unicodedata
from typing import Dict
from pathlib import Path
try:
    import contractions
except:
    contractions = None
try:
    import emoji
except:
    emoji = None
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def strip_html(text: str) -> str:
    return re.sub(r'<.*?>', ' ', text)

def expand_contractions_safe(text: str) -> str:
    if contractions:
        return contractions.fix(text)
    return text

def full_clean(text: str, cfg: Dict):
    if not isinstance(text, str):
        return ""
    t = normalize_unicode(text)
    t = strip_html(t)
    t = expand_contractions_safe(t)
    if cfg.get("lowercase", True):
        t = t.lower()
    if cfg.get("remove_non_alpha", False):
        t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def lemmatize(text: str, remove_stopwords=True):
    doc = nlp(text)
    toks = [tok.lemma_.lower() for tok in doc if tok.is_alpha and (not tok.is_stop or not remove_stopwords)]
    return " ".join(toks)
