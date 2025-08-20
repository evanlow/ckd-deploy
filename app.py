

from fastapi import FastAPI, HTTPException
from mangum import Mangum
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import json

app = FastAPI(title="CKD Predictor", version="1.0.0")
base_path = os.getenv("API_GATEWAY_BASE_PATH")  # e.g., "/prod"
if base_path:
    lambda_handler = Mangum(app, api_gateway_base_path=base_path)
else:
    lambda_handler = Mangum(app)

# Model path and threshold from environment
MODEL_PATH = os.getenv("MODEL_PATH", "/var/task/artifacts/model.joblib")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# Try to get expected feature order
EXPECTED = None
if hasattr(model, "feature_names_in_"):
    EXPECTED = list(model.feature_names_in_)
else:
    for cand in ["/var/task/artifacts/config.json", "/var/task/artifacts/metadata.json"]:
        if os.path.exists(cand):
            try:
                with open(cand, "r") as f:
                    meta = json.load(f)
                EXPECTED = meta.get("feature_names") or meta.get("features") or None
                if EXPECTED: EXPECTED = list(EXPECTED)
                break
            except Exception:
                pass

class Rows(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of feature dicts")

@app.get("/health")
async def health():
    return {"status": "ok", "expected_count": len(EXPECTED) if EXPECTED else None}

@app.post("/predict")
async def predict(payload: Rows, threshold: Optional[float] = None):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")

    try:
        X = pd.DataFrame(payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not build DataFrame: {e}")

    # Log column order for traceability
    try:
        print(f"[predict] X.columns: {X.columns.tolist()}")
    except Exception:
        pass

    # Normalize columns
    if EXPECTED:
        missing = [c for c in EXPECTED if c not in X.columns]
        for m in missing:
            X[m] = 0
        extra = [c for c in X.columns if c not in EXPECTED]
        if extra:
            X = X.drop(columns=extra)
        X = X[EXPECTED]

    try:
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
            p1 = proba[:, 1]
            t = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
            pred = (p1 >= t).astype(int).tolist()
            return {"pred": pred, "proba": p1.tolist(), "threshold": t}
        pred = model.predict(X).tolist()
        return {"pred": pred, "proba": proba.tolist() if proba is not None else None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
