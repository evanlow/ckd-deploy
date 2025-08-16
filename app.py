from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json
import joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.50"))
USE_THRESHOLD = True  # set False to always use raw predict()

# ---------- load model ----------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# Try to learn the expected feature order
EXPECTED = None
if hasattr(model, "feature_names_in_"):
    EXPECTED = list(model.feature_names_in_)
else:
    # Fallback: look for a training-time schema file you may have exported
    for cand in ["artifacts/config.json", "artifacts/metadata.json"]:
        if os.path.exists(cand):
            try:
                with open(cand, "r") as f:
                    meta = json.load(f)
                EXPECTED = meta.get("feature_names") or meta.get("features") or None
                if EXPECTED: EXPECTED = list(EXPECTED)
                break
            except Exception:
                pass

app = FastAPI(title="CKD Predictor", version="1.0.0")

class Rows(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of feature dicts")

@app.get("/health")
def health():
    return {"status": "ok", "expected_count": len(EXPECTED) if EXPECTED else None}

@app.get("/schema")
def schema():
    """Expose the expected feature order (nice for debugging)."""
    return {"expected": EXPECTED}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Convert underscores to spaces (handles diet_low_salt -> diet_low salt)
    2) If EXPECTED exists, add any missing cols as 0 and reorder exactly.
    """
    # (1) underscores -> spaces
    rename_map = {c: c.replace("_", " ") for c in df.columns}
    df = df.rename(columns=rename_map)

    # (2) add missing + reorder
    if EXPECTED:
        # add any missing columns with 0 default (works for one-hots and many numerics)
        missing = [c for c in EXPECTED if c not in df.columns]
        if missing:
            for m in missing:
                df[m] = 0
        # drop unexpected extra columns to satisfy strict estimators
        extra = [c for c in df.columns if c not in EXPECTED]
        if extra:
            df = df.drop(columns=extra)
        # reorder to exact training order
        df = df[EXPECTED]
    return df

@app.post("/predict")
def predict(payload: Rows, threshold: Optional[float] = None):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")

    try:
        X = pd.DataFrame(payload.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not build DataFrame: {e}")

    received_cols = list(X.columns)
    X = normalize_columns(X)

    try:
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        if proba is not None and proba.ndim == 2 and proba.shape[1] == 2 and USE_THRESHOLD:
            p1 = proba[:, 1]
            t = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
            pred = (p1 >= t).astype(int).tolist()
            return {
                "pred": pred,
                "proba": p1.tolist(),
                "threshold": t,
                "received_columns": received_cols,
                "served_order": list(X.columns)
            }

        pred = model.predict(X).tolist()
        return {
            "pred": pred,
            "proba": proba.tolist() if proba is not None else None,
            "received_columns": received_cols,
            "served_order": list(X.columns)
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {e}. Received: {received_cols}. Served: {list(X.columns)}"
        )
