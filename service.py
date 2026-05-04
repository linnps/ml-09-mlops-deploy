"""
FastAPI service that serves the trained classifier.

Endpoints:
    GET  /health                — liveness probe
    GET  /info                  — model metadata and metrics
    POST /predict               — single prediction
    POST /predict/batch         — batched prediction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ARTIFACTS = Path("models")


class _State:
    model: Any = None
    schema: dict | None = None


_state = _State()


def load_artifacts() -> None:
    if not (ARTIFACTS / "classifier.joblib").exists():
        raise RuntimeError(
            "Model artifact not found. Run `python train_model.py` first."
        )
    _state.model = joblib.load(ARTIFACTS / "classifier.joblib")
    with open(ARTIFACTS / "schema.json") as f:
        _state.schema = json.load(f)


app = FastAPI(title="ml-09 classifier service",
              description="Tiny FastAPI service serving a synthetic-data classifier.",
              version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Map from feature name to value."
    )

    @field_validator("features")
    @classmethod
    def _required(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("features must not be empty")
        return v


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., description="List of feature dicts.")


def _to_array(rows: list[dict[str, float]]) -> np.ndarray:
    feature_names = _state.schema["feature_names"]
    arr = np.empty((len(rows), len(feature_names)), dtype=float)
    for r, row in enumerate(rows):
        missing = [f for f in feature_names if f not in row]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"missing features in row {r}: {missing}")
        for c, name in enumerate(feature_names):
            arr[r, c] = float(row[name])
    return arr


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state.model is not None}


@app.get("/info")
def info() -> dict:
    if _state.schema is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    s = _state.schema
    return {
        "model_type": s["model_type"],
        "feature_names": s["feature_names"],
        "metrics": s["metrics"],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    X = _to_array([req.features])
    proba = float(_state.model.predict_proba(X)[0, 1])
    return {"prediction": int(proba > 0.5),
            "probability": proba}


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest) -> dict:
    X = _to_array(req.rows)
    proba = _state.model.predict_proba(X)[:, 1]
    preds = (proba > 0.5).astype(int)
    return {"predictions": preds.tolist(), "probabilities": proba.tolist()}
