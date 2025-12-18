from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import json
from fastapi.middleware.cors import CORSMiddleware

import time
from datetime import datetime, timezone, date
from collections import deque
from fastapi.staticfiles import StaticFiles

from API.explainability_routes import router as explainability_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Breast Cancer Multi-Model API",
)

from API.chat_routes import router as chat_router
app.include_router(chat_router)

# ✅ NEW: auth router (you will add API/auth_routes.py next)
from API.auth_routes import router as auth_router, require_role
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
# project root (one level above API/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ---------- Load feature names ----------
feature_names_path = os.path.join(ARTIFACTS_DIR, "feature_names.csv")
if not os.path.exists(feature_names_path):
    raise RuntimeError(f"feature_names.csv not found at {feature_names_path}")

feature_names = pd.read_csv(feature_names_path, header=None)[0].tolist()

# ---------- Load scaler ----------
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
if not os.path.exists(SCALER_PATH):
    raise RuntimeError(f"scaler.joblib not found at {SCALER_PATH}")

SCALER = joblib.load(SCALER_PATH)

# ---------- Load models registry ----------
registry_path = os.path.join(ARTIFACTS_DIR, "models_registry.json")
if not os.path.exists(registry_path):
    raise RuntimeError(f"models_registry.json not found at {registry_path}")

with open(registry_path, "r", encoding="utf-8") as f:
    registry = json.load(f)

# ---------- Load all models ----------
MODELS: Dict[str, Any] = {}
for model_id, info in registry.items():
    model_path = os.path.join(ARTIFACTS_DIR, info["file"])
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    MODELS[model_id] = joblib.load(model_path)

# ---------- Monitoring / Live feed (in-memory) ----------
PRED_LOG = deque(maxlen=200)  # last 200 predictions kept in memory
LAT_LOG = deque(maxlen=500)   # last 500 latencies

METRICS = {
    "predictions_today": 0,
    "avg_latency_ms": 0.0,
    "p95_latency_ms": 0.0,
    "error_rate_1h": 0.0,   # placeholder for now
    "status": "operational",
    "day": date.today().isoformat(),
}

class BreastCancerInput(BaseModel):
    data: Dict[str, float]

# ---------- Static: SHAP images ----------
SHAP_STATIC_DIR = os.path.join(ARTIFACTS_DIR, "shap_summary")
os.makedirs(SHAP_STATIC_DIR, exist_ok=True)
app.mount("/static/shap", StaticFiles(directory=SHAP_STATIC_DIR), name="shap")

# ---------- Routers ----------
app.include_router(explainability_router)

# ---------- Core endpoints ----------
@app.get("/models")
def list_models():
    return registry

@app.post("/predict")
def predict(
    input_data: BreastCancerInput,
    model_id: str = Query("random_forest", description="Which model to use"),
):
    t0 = time.perf_counter()
    now = datetime.now(timezone.utc)

    try:
        if model_id not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

        model = MODELS[model_id]

        # validate features
        provided = set(input_data.data.keys())
        required = set(feature_names)
        missing = required - provided
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {sorted(missing)}")

        # build X in exact order
        x_vals = [input_data.data[f] for f in feature_names]
        X_raw = np.array(x_vals, dtype=float).reshape(1, -1)

        # ✅ scale exactly like training
        try:
            X = SCALER.transform(X_raw)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

        # predict using scaled
        y_pred = int(model.predict(X)[0])

        proba = None
        conf = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0].tolist()
                if proba:
                    conf = float(max(proba))
            except Exception:
                proba = None
                conf = None

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # --- DAILY RESET ---
        today = date.today().isoformat()
        if METRICS["day"] != today:
            METRICS["day"] = today
            METRICS["predictions_today"] = 0
            LAT_LOG.clear()

        # --- UPDATE METRICS ---
        METRICS["predictions_today"] += 1
        LAT_LOG.append(latency_ms)

        latencies = list(LAT_LOG)
        if latencies:
            METRICS["avg_latency_ms"] = float(np.mean(latencies))
            METRICS["p95_latency_ms"] = float(np.percentile(latencies, 95))

        # --- LIVE FEED LOG ---
        PRED_LOG.appendleft({
            "ts": now.isoformat(),
            "model_id": model_id,
            "model_name": registry[model_id].get("name", model_id),
            "prediction": y_pred,
            "confidence": conf,
            "latency_ms": round(latency_ms, 2),
        })

        METRICS["status"] = "operational"

        return {
            "model_id": model_id,
            "model_name": registry[model_id].get("name", model_id),
            "prediction": y_pred,
            "probabilities": proba,
            "confidence": conf,
            "latency_ms": round(latency_ms, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        METRICS["status"] = "degraded"
        raise HTTPException(status_code=500, detail=str(e))

# ✅ UPDATED: only data_scientist can access /metrics
@app.get("/metrics")
def get_metrics(user=Depends(require_role({"data_scientist", "scientist"}))):
    """Return metrics.json (computed in the notebook)."""
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/features")
def get_features():
    """Return the ordered list of feature names."""
    return {"features": feature_names}

@app.get("/feature-ranges")
def get_feature_ranges():
    path = os.path.join(ARTIFACTS_DIR, "random_feature_ranges.json")
    if not os.path.exists(path):
        raise HTTPException(500, "random_feature_ranges.json missing")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/dataset-info")
def get_dataset_info():
    path = os.path.join(ARTIFACTS_DIR, "dataset_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/evaluation-table")
def get_evaluation_table():
    path = os.path.join(ARTIFACTS_DIR, "evaluation_table.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/evaluation-report")
def evaluation_report():
    path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    if not os.path.exists(path):
        return {"rows": [], "roc": {}, "recommended_model_id": None}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Monitoring endpoints ----------
@app.get("/monitoring/summary")
def monitoring_summary():
    return {
        "predictions_today": METRICS["predictions_today"],
        "avg_latency_ms": round(float(METRICS["avg_latency_ms"]), 2),
        "p95_latency_ms": round(float(METRICS["p95_latency_ms"]), 2),
        "error_rate_1h": float(METRICS["error_rate_1h"]),
        "status": METRICS["status"],
        "ts": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/monitoring/live")
def monitoring_live(limit: int = 20):
    limit = max(1, min(limit, 200))
    return list(PRED_LOG)[:limit]

@app.get("/monitoring/ping")
def monitoring_ping():
    t0 = time.perf_counter()
    _ = sum(range(5000))
    ms = (time.perf_counter() - t0) * 1000.0
    return {"ok": True, "server_ms": round(ms, 2)}

@app.get("/debug/sample-payload")
def sample_payload():
    path = os.path.join(ARTIFACTS_DIR, "random_feature_ranges.json")
    if not os.path.exists(path):
        raise HTTPException(500, "random_feature_ranges.json missing")

    with open(path, "r", encoding="utf-8") as f:
        ranges = json.load(f)

    data = {}
    for feat in feature_names:
        r = ranges.get(feat)
        if r is None:
            data[feat] = 0.0
            continue

        if isinstance(r, dict):
            mn, mx = float(r.get("min", 0)), float(r.get("max", 0))
        else:
            mn, mx = float(r[0]), float(r[1])

        data[feat] = (mn + mx) / 2.0

    return {"data": data}

@app.get("/example-patients")
def example_patients():
    path = os.path.join(ARTIFACTS_DIR, "example_patients.json")
    if not os.path.exists(path):
        raise HTTPException(500, "example_patients.json missing (run last notebook cell).")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
