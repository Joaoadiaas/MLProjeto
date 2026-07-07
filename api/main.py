"""
FastAPI service that serves the trained churn model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_processing import build_feature_frame  # noqa: E402
from api.schemas import CustomerFeatures, ModelInfoResponse, PredictionResponse  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serves a scikit-learn model that scores telecom customers for churn risk.",
    version="1.0.0",
)

_model = None
_feature_cols = None
_metrics = None


@app.on_event("startup")
def load_artifacts():
    global _model, _feature_cols, _metrics
    model_path = MODELS_DIR / "churn_model.pkl"
    features_path = MODELS_DIR / "feature_list.json"
    metrics_path = MODELS_DIR / "metrics.json"

    if not model_path.exists():
        # Do not crash the app - /health will report the model as unavailable
        # until `python src/train.py` has been run.
        return

    _model = joblib.load(model_path)
    _feature_cols = json.loads(features_path.read_text())
    if metrics_path.exists():
        _metrics = json.loads(metrics_path.read_text())


def _risk_label(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "medium"
    return "high"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Model metrics not available. Run src/train.py first.")
    return _metrics


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` to produce models/churn_model.pkl.",
        )

    row = pd.DataFrame([customer.model_dump()])
    X, _ = build_feature_frame(row, fit_columns=_feature_cols)

    proba = float(_model.predict_proba(X)[0, 1])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_label=_risk_label(proba),
        model_version=(_metrics or {}).get("best_model", "unknown"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
