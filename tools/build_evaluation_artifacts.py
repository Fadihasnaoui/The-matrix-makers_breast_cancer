import os, json, glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.datasets import load_breast_cancer

# -------------------------
# Paths
# -------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

REGISTRY_PATH = os.path.join(ARTIFACTS_DIR, "models_registry.json")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.csv")

OUT_REPORT = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")  # <- single payload for Metrics page

def load_dataset_best_effort():
    """
    Best-effort loader:
    1) If there's a CSV in repo_root/data, try to use it.
       Expected target column: diagnosis OR target OR y
       diagnosis mapping: B/M or benign/malignant
    2) Else fall back to sklearn breast cancer dataset.
    """
    data_dir = os.path.join(REPO_ROOT, "data")
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))

    if csvs:
        # pick the most recently modified csv
        csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        path = csvs[0]
        df = pd.read_csv(path)

        # Find target column
        target_col = None
        for c in ["diagnosis", "target", "y"]:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            raise RuntimeError(f"Found CSV {path} but no target column (diagnosis/target/y).")

        y_raw = df[target_col]

        # Map to 0/1
        if y_raw.dtype == object:
            mapping = {
                "B": 0, "M": 1,
                "benign": 0, "malignant": 1,
                "Benign": 0, "Malignant": 1
            }
            y = y_raw.map(mapping)
        else:
            y = y_raw

        if y.isna().any():
            raise RuntimeError(f"Target mapping produced NaNs for {target_col} in {path}")

        X = df.drop(columns=[target_col])

        return X, y.astype(int), f"csv:{os.path.basename(path)}"

    # fallback dataset
    ds = load_breast_cancer(as_frame=True)
    X = ds.data
    y = pd.Series(ds.target).astype(int)
    return X, y, "sklearn:load_breast_cancer"

def get_score(model, X):
    """Continuous score for ROC"""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return None

def main():
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(f"Missing {REGISTRY_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")

    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    feature_names = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()

    X, y, source = load_dataset_best_effort()

    # Ensure columns match the deployed feature order
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing features used by API: {sorted(list(missing))[:10]} ...")

    X = X[feature_names]

    # Realistic holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rows = []
    roc = {}

    for model_id, info in registry.items():
        model_path = os.path.join(ARTIFACTS_DIR, info["file"])
        if not os.path.exists(model_path):
            print(f"⚠️ missing model file: {model_path} (skipping)")
            continue

        model = joblib.load(model_path)

        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred).astype(int).ravel()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1v = f1_score(y_test, y_pred, zero_division=0)

        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0

        # ROC
        score = get_score(model, X_test)
        auc_val = None
        fpr_list = None
        tpr_list = None
        if score is not None:
            fpr, tpr, _ = roc_curve(y_test, score)
            auc_val = float(auc(fpr, tpr))
            fpr_list = [float(x) for x in fpr]
            tpr_list = [float(x) for x in tpr]
            roc[model_id] = {"fpr": fpr_list, "tpr": tpr_list, "auc": auc_val}

        rows.append({
            "model_id": model_id,
            "model_name": info.get("name", model_id),

            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(specificity),
            "fnr": float(fnr),
            "f1": float(f1v),
            "auc": float(auc_val) if auc_val is not None else None,

            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    # Choose "recommended" by safety-first: lowest FNR, then highest AUC
    def sort_key(r):
        fnr = r["fnr"] if r["fnr"] is not None else 1e9
        aucv = r["auc"] if r["auc"] is not None else -1
        return (fnr, -aucv)

    rows_sorted = sorted(rows, key=sort_key)
    recommended = rows_sorted[0]["model_id"] if rows_sorted else None

    report = {
        "source": source,
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
        "recommended_model_id": recommended,
        "rows": rows_sorted,
        "roc": roc,  # keyed by model_id
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("✅ wrote", OUT_REPORT)
    print("recommended:", recommended)
    print("models:", [r["model_id"] for r in rows_sorted])

if __name__ == "__main__":
    main()
