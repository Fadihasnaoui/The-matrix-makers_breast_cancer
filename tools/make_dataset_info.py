import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data") / "data_enriched.csv"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Try common target column names
target_candidates = ["diagnosis", "target", "label", "class", "y"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise ValueError(f"No target column found. Columns: {df.columns.tolist()}")

y = df[target_col]

# Normalize B/M labels to 0/1
if y.dtype == object:
    y = y.astype(str).str.upper().map({"B": 0, "M": 1})
    if y.isna().any():
        raise ValueError("Target column is text but not B/M. Check your labels.")
    y = y.astype(int)

n_samples = int(df.shape[0])
feature_cols = [c for c in df.columns if c != target_col]
n_features = int(len(feature_cols))

counts = y.value_counts().to_dict()
benign = int(counts.get(0, 0))
malignant = int(counts.get(1, 0))

dataset_info = {
    "dataset_name": "Breast Cancer (enriched)",
    "data_file": str(DATA_PATH).replace("\\", "/"),
    "n_samples": n_samples,
    "n_features": n_features,
    "target": target_col,
    "class_names": ["Benign", "Malignant"],
    "class_counts": {"0": benign, "1": malignant},
    "class_balance": {
        "0": round(benign / n_samples, 4) if n_samples else 0.0,
        "1": round(malignant / n_samples, 4) if n_samples else 0.0,
    },
    "split": {"test_ratio": 0.2},
    "positive_class": 1,
}

out = ARTIFACTS_DIR / "dataset_info.json"
out.write_text(json.dumps(dataset_info, indent=2))
print(f"Wrote: {out}")
print(json.dumps(dataset_info, indent=2))
