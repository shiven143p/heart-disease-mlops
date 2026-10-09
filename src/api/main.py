
"""FastAPI service for heart disease model inference."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import PatientInput, PredictionResponse
from src.models.predict import load_model, predict_heart_disease

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once during application startup."""
    model_store["model"] = load_model()
    yield
    model_store.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational MLOps classification API. "
        "Not intended for clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in model_store,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    model = model_store.get("model")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        return predict_heart_disease(
            model,
            patient.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc
