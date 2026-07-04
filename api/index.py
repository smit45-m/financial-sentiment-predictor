"""
Financial Sentiment Predictor — Vercel Serverless Entry Point.

Lightweight FastAPI application optimised for Vercel's serverless
constraints (250 MB bundle, ~10 s cold start):
  - Only loads Classical ML models (LR + XGBoost via joblib).
  - Skips BiLSTM (PyTorch) and FinBERT (transformers) to stay
    within Vercel's size and cold-start limits.
  - Self-contained: duplicates minimal logic so the function is
    deployable without the full project tree.

All endpoints mirror the main API surface where possible.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("serverless_api")

# ---------------------------------------------------------------------------
# Paths — relative to repo root (Vercel deploys from root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "saved_models"
STATIC_DIR = BASE_DIR / "static"

# Available model types in serverless mode
SERVERLESS_MODELS = ["logistic_regression", "xgboost"]
LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
LABEL_MAP: dict[int, str] = {i: n for i, n in enumerate(LABEL_NAMES)}

# ---------------------------------------------------------------------------
# Lightweight Predictor (Classical ML only)
# ---------------------------------------------------------------------------
_tfidf = None
_models: dict[str, Any] = {}
_loaded = False


def _clean_text(text: str) -> str:
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ensure_loaded() -> None:
    """Lazy-load classical models on first invocation."""
    global _tfidf, _models, _loaded
    if _loaded:
        return

    if not MODELS_DIR.exists():
        logger.warning("Model directory %s not found.", MODELS_DIR)
        _loaded = True
        return

    try:
        import joblib

        vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
        if vec_path.exists():
            _tfidf = joblib.load(vec_path)
            logger.info("Loaded TF-IDF vectorizer.")

        lr_path = MODELS_DIR / "logistic_regression.joblib"
        if lr_path.exists():
            _models["logistic_regression"] = joblib.load(lr_path)
            logger.info("Loaded Logistic Regression model.")

        xgb_path = MODELS_DIR / "xgboost_model.joblib"
        if xgb_path.exists():
            _models["xgboost"] = joblib.load(xgb_path)
            logger.info("Loaded XGBoost model.")

    except Exception as exc:
        logger.error("Error loading models: %s", exc)

    _loaded = True
    logger.info("Serverless models available: %s", list(_models.keys()))


def _predict(text: str, model_name: str) -> dict[str, Any]:
    """Run prediction with a classical model."""
    if _tfidf is None:
        raise ValueError("TF-IDF vectorizer not loaded — models unavailable.")
    if model_name not in _models:
        raise ValueError(
            f"Model '{model_name}' not available. "
            f"Loaded: {list(_models.keys())}. "
            f"Note: BiLSTM and FinBERT are not available in serverless mode."
        )

    cleaned = _clean_text(text)
    features = _tfidf.transform([cleaned])
    model = _models[model_name]
    probas = model.predict_proba(features)[0]
    label = int(np.argmax(probas))
    confidence = float(probas[label])
    sentiment = LABEL_MAP.get(label, "unknown")

    return {
        "sentiment": sentiment,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(probas[i]), 4) for i, name in enumerate(LABEL_NAMES)
        },
    }


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Single-text prediction request."""

    text: str = Field(..., min_length=1)
    model: str = Field(
        default="logistic_regression",
        description="logistic_regression | xgboost (serverless mode)",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    texts: list[str] = Field(..., min_length=1)
    model: str = Field(default="logistic_regression")


class ProbabilityDetail(BaseModel):
    """Per-class probabilities."""

    negative: float
    neutral: float
    positive: float


class PredictionResponse(BaseModel):
    """Standard prediction response."""

    text: str
    model: str
    sentiment: str
    label: int
    confidence: float
    probabilities: ProbabilityDetail


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Sentiment Predictor (Serverless)",
    description=(
        "Lightweight serverless deployment — Classical ML models only. "
        "BiLSTM and FinBERT are available in the full Docker deployment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with response time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---- Routes ---------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the front-end dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            content={
                "message": "Financial Sentiment Predictor API (Serverless)",
                "docs": "/docs",
                "mode": "serverless",
                "available_models": SERVERLESS_MODELS,
                "note": "BiLSTM and FinBERT are only available in the full Docker deployment.",
            }
        )
    return FileResponse(str(index_path))


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Health check with model availability info."""
    _ensure_loaded()
    return {
        "status": "healthy",
        "mode": "serverless",
        "models_loaded": list(_models.keys()),
        "models_count": len(_models),
        "note": "Serverless mode — only classical ML models (LR, XGBoost) are loaded.",
        "unavailable_models": ["bilstm", "finbert"],
        "unavailable_reason": "PyTorch/transformers exceed Vercel's 250 MB limit.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest) -> PredictionResponse:
    """Predict sentiment for a single text."""
    _ensure_loaded()

    if request.model in ("bilstm", "finbert"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' is not available in serverless mode. "
                f"Use 'logistic_regression' or 'xgboost'. "
                f"For BiLSTM/FinBERT, deploy with Docker."
            ),
        )

    try:
        result = _predict(request.text, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        text=request.text,
        model=request.model,
        sentiment=result["sentiment"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=ProbabilityDetail(**result["probabilities"]),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest) -> list[PredictionResponse]:
    """Batch prediction for multiple texts."""
    _ensure_loaded()

    if request.model in ("bilstm", "finbert"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not available in serverless mode.",
        )

    if len(request.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Serverless batch limited to 50 texts per request.",
        )

    results: list[PredictionResponse] = []
    for text in request.texts:
        try:
            result = _predict(text, request.model)
            results.append(
                PredictionResponse(
                    text=text,
                    model=request.model,
                    sentiment=result["sentiment"],
                    label=result["label"],
                    confidence=result["confidence"],
                    probabilities=ProbabilityDetail(**result["probabilities"]),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Batch error for '%s…': %s", text[:50], exc)
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {exc}"
            )

    return results
