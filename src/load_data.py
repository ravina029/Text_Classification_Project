from pathlib import Path
import pandas as pd

def load_imdb_to_df(base_path: str):
    base = Path(base_path)
    rows = []
    for split in ["train","test"]:
        for lab in ["pos","neg"]:
            folder = base / split / lab
            if not folder.exists():
                continue
            for f in folder.glob("*.txt"):
                rows.append({"text": f.read_text(encoding="utf-8"), "label": 1 if lab=="pos" else 0, "split": split})
    return pd.DataFrame(rows)
