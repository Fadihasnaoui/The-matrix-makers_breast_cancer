import os, json
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()

# Project root: .../ML Project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

FEATIMP_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
LIME_PATH    = os.path.join(ARTIFACTS_DIR, "lime_explanations.json")
SHAP_PATH    = os.path.join(ARTIFACTS_DIR, "shap_summary.json")


def _read_json(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Missing artifact: {os.path.basename(path)}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/feature-importance/{model_id}")
def feature_importance(model_id: str):
    data = _read_json(FEATIMP_PATH)
    if model_id not in data:
        raise HTTPException(status_code=404, detail="Model not found")
    return data[model_id]


@router.get("/lime-explanation/{model_id}")
def lime_explanation(model_id: str, sample_index: int = Query(0, ge=0)):
    data = _read_json(LIME_PATH)

    if model_id not in data:
        raise HTTPException(status_code=404, detail="Model not found")

    model_block = data[model_id]  # expected dict keyed by sample index as string
    key = str(sample_index)

    if key not in model_block:
        # Return a friendly payload instead of crashing the UI
        return {
            "message": f"No LIME explanation saved for model={model_id}, sample_index={sample_index}.",
            "prediction": None,
            "probability": None,
            "feature_weights": {}
        }

    # Must include feature_weights (your frontend expects it)
    return model_block[key]


@router.get("/shap-summary/{model_id}")
def shap_summary(model_id: str, request: Request):
    data = _read_json(SHAP_PATH)

    if model_id not in data:
        raise HTTPException(status_code=404, detail="Model not found")

    entry = data[model_id]
    plot_file = entry.get("plot_file")

    if not plot_file:
        return {
            "message": entry.get("description", "No SHAP plot exported for this model yet."),
            "plot_url": None,
            "description": entry.get("description")
        }

    # This matches the static mount "/static/shap"
    plot_url = str(request.base_url).rstrip("/") + f"/static/shap/{plot_file}"
    return {
        "plot_url": plot_url,
        "plot_file": plot_file,
        "description": entry.get("description")
    }
