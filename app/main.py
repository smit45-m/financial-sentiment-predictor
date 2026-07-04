"""
Financial Sentiment Predictor — FastAPI Application.

Production-quality REST API serving sentiment predictions from
Classical ML, BiLSTM, and FinBERT models with lazy loading,
graceful degradation, and comprehensive error handling.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.predictor import SentimentPredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("financial_sentiment_api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "saved_models"
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Global predictor (loaded lazily)
# ---------------------------------------------------------------------------
predictor = SentimentPredictor()
_models_loaded = False


def _ensure_models_loaded() -> None:
    """Lazy-load models on the first prediction request."""
    global _models_loaded
    if _models_loaded:
        return

    logger.info("Lazy-loading models from %s …", MODELS_DIR)

    if MODELS_DIR.exists():
        try:
            loaded_classical = predictor.load_classical_models(str(MODELS_DIR))
            logger.info("Classical models loaded: %s", loaded_classical)
        except Exception as exc:
            logger.error("Failed to load classical models: %s", exc)

        try:
            predictor.load_bilstm(str(MODELS_DIR))
        except Exception as exc:
            logger.error("Failed to load BiLSTM: %s", exc)

        try:
            predictor.load_finbert()
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
    else:
        logger.warning("Model directory %s does not exist.", MODELS_DIR)

    _models_loaded = True
    logger.info("Available models after loading: %s", predictor.get_available_models())


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting Financial Sentiment Predictor API …")
    logger.info("Model directory: %s (exists=%s)", MODELS_DIR, MODELS_DIR.exists())
    logger.info("Static directory: %s (exists=%s)", STATIC_DIR, STATIC_DIR.exists())

    # Pre-scan for model artefacts
    if MODELS_DIR.exists():
        artefacts = [p.name for p in MODELS_DIR.iterdir()]
        logger.info("Model artefacts found: %s", artefacts)
    else:
        logger.warning("No saved_models/ directory — models will be unavailable.")

    yield  # app is running

    logger.info("Shutting down Financial Sentiment Predictor API.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Sentiment Predictor",
    description=(
        "Predict sentiment of financial earnings-call text using "
        "Classical ML, BiLSTM, or FinBERT models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permissive for demo front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Logging Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request and its response time."""
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


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Single-text prediction request."""

    text: str = Field(..., min_length=1, description="Financial text to analyse.")
    model: str = Field(
        ...,
        description="Model name: logistic_regression | xgboost | bilstm | finbert",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    texts: list[str] = Field(..., min_length=1, description="List of texts.")
    model: str = Field(
        ...,
        description="Model name: logistic_regression | xgboost | bilstm | finbert",
    )


class PredictAllRequest(BaseModel):
    """Predict-all request (single text, all models)."""

    text: str = Field(..., min_length=1, description="Financial text to analyse.")


class ProbabilityDetail(BaseModel):
    """Per-class probability breakdown."""

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
# Routes
# ---------------------------------------------------------------------------

# ---- Static front-end ----------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the front-end dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not built yet. Place index.html in static/.",
        )
    return FileResponse(str(index_path))


# Mount static files (CSS, JS, images) — AFTER the root route
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---- Health Check --------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Health check with model load status."""
    _ensure_models_loaded()
    available = predictor.get_available_models()
    return {
        "status": "healthy",
        "models_loaded": available,
        "models_count": len(available),
        "model_dir_exists": MODELS_DIR.exists(),
    }


# ---- Single Prediction ---------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest) -> PredictionResponse:
    """Predict sentiment for a single text using the specified model."""
    _ensure_models_loaded()

    try:
        result = predictor.predict(request.text, request.model)
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


# ---- Batch Prediction ----------------------------------------------------
@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest) -> list[PredictionResponse]:
    """Predict sentiment for multiple texts using the specified model."""
    _ensure_models_loaded()

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 texts per request.",
        )

    results: list[PredictionResponse] = []
    for text in request.texts:
        try:
            result = predictor.predict(text, request.model)
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
            logger.error("Batch prediction error for text='%s…': %s", text[:50], exc)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for text: {text[:50]}… — {exc}",
            )

    return results


# ---- Model Comparison Results --------------------------------------------
@app.get("/models/compare", tags=["Models"])
async def compare_models() -> dict[str, Any]:
    """Return pre-computed model comparison results."""
    comparison_path = MODELS_DIR / "model_comparison.json"
    if not comparison_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model comparison results not found. Run training first.",
        )

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"status": "success", "comparison": data}


# ---- Predict All Models --------------------------------------------------
@app.post("/predict/all", tags=["Prediction"])
async def predict_all_models(request: PredictAllRequest) -> dict[str, Any]:
    """Predict with ALL available models for side-by-side comparison."""
    _ensure_models_loaded()

    available = predictor.get_available_models()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No models are currently loaded. Check /health for details.",
        )

    all_results = predictor.predict_all(request.text)

    # Shape response
    predictions: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for model_name, result in all_results.items():
        if "error" in result:
            errors[model_name] = result["error"]
        else:
            predictions[model_name] = {
                "sentiment": result["sentiment"],
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }

    return {
        "text": request.text,
        "models_used": list(predictions.keys()),
        "predictions": predictions,
        **({"errors": errors} if errors else {}),
    }
