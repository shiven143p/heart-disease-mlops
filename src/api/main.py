
"""FastAPI service with prediction, logging, and monitoring."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from starlette.responses import Response

from src.api.monitoring import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.api.schemas import PatientInput, PredictionResponse
from src.models.predict import load_model, predict_heart_disease

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ),
)

logger = logging.getLogger("heart_disease_api")

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once during startup."""

    logger.info("Loading trained ML model")

    model_store["model"] = load_model()

    logger.info("Model loaded successfully")

    yield

    model_store.clear()
    logger.info("Application shutdown completed")


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational MLOps classification API. "
        "Not intended for clinical diagnosis."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Record request counts, duration, and status codes."""

    start_time = time.perf_counter()

    method = request.method
    endpoint = request.url.path

    # Use fixed endpoint labels to avoid high-cardinality metrics.
    known_endpoints = {
        "/health",
        "/predict",
        "/metrics",
        "/docs",
        "/openapi.json",
    }

    if endpoint not in known_endpoints:
        endpoint = "other"

    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        logger.exception("Unhandled API request error")
        raise

    finally:
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        logger.info(
            "method=%s endpoint=%s status=%s duration=%.4fs",
            method,
            endpoint,
            status_code,
            duration,
        )


@app.get("/health")
def health():
    """Basic health endpoint."""

    return {
        "status": "healthy",
        "model_loaded": "model" in model_store,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose Prometheus metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    """Generate a heart disease classification prediction."""

    model = model_store.get("model")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        result = predict_heart_disease(
            model,
            patient.model_dump(),
        )

        PREDICTION_COUNT.labels(
            prediction=str(result["prediction"])
        ).inc()

        logger.info(
            "Prediction completed: class=%s",
            result["prediction"],
        )

        return result

    except Exception as exc:
        PREDICTION_ERRORS.inc()

        logger.exception("Prediction processing failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc
